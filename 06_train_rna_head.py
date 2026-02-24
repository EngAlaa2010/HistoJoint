#!/usr/bin/env python3
"""
06_train_rna_head.py  (IMPROVED: Residual LayerNorm MLP + Huber loss)

Step 06: Train RNA prediction head
Learn mapping:
  embeddings_spatial (morphology, per spot)  ->  RNA program scores (targets)

Inputs (default under base/outputs):
  - embeddings_spatial.npy          (N x D)   from Step 04
  - embeddings_index.csv            (N rows)  from Step 03 (row_id, barcode, x_pixel, y_pixel, in_tissue, ...)
  - rna_program_scores.npy          (N x P)   from Step 05
  - rna_program_index.csv           (N rows)  from Step 05 (row_id, barcode, in_tissue)

Outputs:
  - outputs/rna_head_model.pt               (torch checkpoint with model + metadata)
  - outputs/rna_head_y_scaler.npz           (train mean/std if standardizing Y)
  - outputs/rna_head_split.csv              (barcode, x_pixel, y_pixel, in_tissue, split)
  - outputs/rna_pred_{train,val,test}.npy   (predictions, original scale)
  - outputs/rna_true_{train,val,test}.npy   (ground truth, original scale)
  - outputs/rna_head_metrics.csv            (per-program + overall metrics for val/test)

Improvements vs baseline:
  - Robust alignment by barcode (prevents accidental row-order mismatch)
  - Spatial tile split to reduce leakage
  - Residual MLP head with LayerNorm (stronger + more stable on embeddings)
  - Huber loss (SmoothL1) for noisy targets
  - Early stopping on validation loss
  - Optional standardization of Y using TRAIN mean/std (recommended)

Run:
  python scripts/06_train_rna_head.py -v --only-in-tissue --standardize-y --device auto

Tune:
  python scripts/06_train_rna_head.py -v --only-in-tissue --standardize-y --device cuda \
    --width 768 --depth 3 --dropout 0.1 --lr 8e-4 --weight-decay 2e-4 --batch-size 256 --epochs 300 --patience 30 \
    --loss huber --huber-beta 1.0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import r2_score
from scipy.stats import pearsonr


# ---------------- logging ----------------

def setup_logger(v: int) -> None:
    level = logging.WARNING
    if v == 1:
        level = logging.INFO
    elif v >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------- helpers ----------------

def load_required(path: Path, kind: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind}: {path}")
    return path


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pearsonr(x, y)[0])


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def spatial_tile_split(
    coords: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    tile_size: float = 500.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Tile-based spatial split:
    - bucket spots into tiles using tile_size in pixel units
    - randomly assign tiles to train/val/test
    Returns split labels array: "train","val","test"
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    x = coords[:, 0]
    y = coords[:, 1]
    tx = np.floor(x / tile_size).astype(int)
    ty = np.floor(y / tile_size).astype(int)

    uniq = pd.DataFrame({"tx": tx, "ty": ty}).drop_duplicates().reset_index(drop=True)
    n_tiles = len(uniq)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_tiles)

    n_train = int(round(train_frac * n_tiles))
    n_val = int(round(val_frac * n_tiles))

    train_tiles = set(map(tuple, uniq.iloc[perm[:n_train]][["tx", "ty"]].values))
    val_tiles = set(map(tuple, uniq.iloc[perm[n_train:n_train + n_val]][["tx", "ty"]].values))
    test_tiles = set(map(tuple, uniq.iloc[perm[n_train + n_val:]][["tx", "ty"]].values))

    split = np.empty(coords.shape[0], dtype=object)
    for i in range(coords.shape[0]):
        t = (int(tx[i]), int(ty[i]))
        if t in train_tiles:
            split[i] = "train"
        elif t in val_tiles:
            split[i] = "val"
        else:
            split[i] = "test"
    return split


def align_by_barcode(
    emb: np.ndarray,
    emb_index: pd.DataFrame,
    rna_scores: np.ndarray,
    rna_index: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Align embeddings and RNA program scores by barcode.
    Returns:
      X (N x D), Y (N x P), idx dataframe with barcode + coords + in_tissue.
    """
    if "barcode" not in emb_index.columns or "barcode" not in rna_index.columns:
        raise ValueError("Both embeddings_index.csv and rna_program_index.csv must contain 'barcode'.")

    emb_df = emb_index.copy()
    emb_df["emb_row"] = emb_df.get("row_id", np.arange(len(emb_df)))

    rna_df = rna_index.copy()
    rna_df["rna_row"] = rna_df.get("row_id", np.arange(len(rna_df)))

    merged = emb_df.merge(
        rna_df[["barcode", "rna_row", "in_tissue"]],
        on="barcode",
        how="inner",
        suffixes=("", "_rna"),
    )
    if len(merged) == 0:
        raise RuntimeError("No overlapping barcodes between embeddings_index and rna_program_index.")

    for col in ["x_pixel", "y_pixel"]:
        if col not in merged.columns:
            raise ValueError(f"embeddings_index.csv must contain '{col}' for spatial split.")

    emb_rows = merged["emb_row"].astype(int).values
    rna_rows = merged["rna_row"].astype(int).values

    X = emb[emb_rows, :].astype(np.float32)
    Y = rna_scores[rna_rows, :].astype(np.float32)

    merged = merged.reset_index(drop=True)
    return X, Y, merged


def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> pd.DataFrame:
    """
    Returns per-program r2/pearson + overall mean row.
    """
    P = Y_true.shape[1]
    rows: List[Dict] = []
    for j in range(P):
        yt = Y_true[:, j]
        yp = Y_pred[:, j]
        rows.append({
            "program": j,
            "r2": float(r2_score(yt, yp)),
            "pearson": safe_pearson(yt, yp),
        })

    df = pd.DataFrame(rows)
    overall = pd.DataFrame([{
        "program": "OVERALL_MEAN",
        "r2": float(np.nanmean(df["r2"].values)),
        "pearson": float(np.nanmean(df["pearson"].values)),
    }])
    return pd.concat([df, overall], ignore_index=True)


# ---------------- Dataset ----------------

class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]


# ---------------- Model (Improved) ----------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(round(dim * hidden_mult))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class BetterMLPHead(nn.Module):
    """
    Residual + LayerNorm MLP head for embedding regression.
    """
    def __init__(self, in_dim: int, out_dim: int, width: int = 512, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[
            ResidualBlock(width, hidden_mult=2.0, dropout=dropout) for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_norm(x)
        return self.out(x)


# ---------------- training utils ----------------

@torch.no_grad()
def predict_array(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = model(xb).detach().cpu().numpy()
        outs.append(yb)
    return np.concatenate(outs, axis=0).astype(np.float32)


def make_loss(loss_name: str, huber_beta: float) -> nn.Module:
    name = loss_name.lower().strip()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        # SmoothL1Loss == Huber; beta controls transition point
        return nn.SmoothL1Loss(beta=float(huber_beta))
    raise ValueError(f"Unknown --loss: {loss_name} (use mse|huber)")


# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 06: Train RNA head (Improved Residual MLP).")
    p.add_argument("--base", type=str, default="/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    p.add_argument("--only-in-tissue", action="store_true")

    # inputs
    p.add_argument("--embeddings", type=str, default="", help="Default: base/outputs/embeddings_spatial.npy")
    p.add_argument("--emb-index", type=str, default="", help="Default: base/outputs/embeddings_index.csv")
    p.add_argument("--rna-scores", type=str, default="", help="Default: base/outputs/rna_program_scores.npy")
    p.add_argument("--rna-index", type=str, default="", help="Default: base/outputs/rna_program_index.csv")

    # split
    p.add_argument("--tile-size", type=float, default=500.0)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)

    # model hyperparams
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.10)

    # optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)

    # loss
    p.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    p.add_argument("--huber-beta", type=float, default=1.0)

    # target standardization
    p.add_argument("--standardize-y", action="store_true", help="Z-score targets using TRAIN mean/std (recommended).")

    # runtime / outputs
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--outdir", type=str, default="", help="Default: base/outputs")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)

    base = Path(args.base)
    outdir = Path(args.outdir) if args.outdir else (base / "outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    emb_path = Path(args.embeddings) if args.embeddings else (base / "outputs" / "embeddings_spatial.npy")
    emb_index_path = Path(args.emb_index) if args.emb_index else (base / "outputs" / "embeddings_index.csv")
    rna_scores_path = Path(args.rna_scores) if args.rna_scores else (base / "outputs" / "rna_program_scores.npy")
    rna_index_path = Path(args.rna_index) if args.rna_index else (base / "outputs" / "rna_program_index.csv")

    load_required(emb_path, "embeddings_spatial.npy")
    load_required(emb_index_path, "embeddings_index.csv")
    load_required(rna_scores_path, "rna_program_scores.npy")
    load_required(rna_index_path, "rna_program_index.csv")

    # outputs
    model_out = outdir / "rna_head_model.pt"
    scaler_out = outdir / "rna_head_y_scaler.npz"
    split_out = outdir / "rna_head_split.csv"
    metrics_out = outdir / "rna_head_metrics.csv"
    pred_train_out = outdir / "rna_pred_train.npy"
    pred_val_out = outdir / "rna_pred_val.npy"
    pred_test_out = outdir / "rna_pred_test.npy"
    true_train_out = outdir / "rna_true_train.npy"
    true_val_out = outdir / "rna_true_val.npy"
    true_test_out = outdir / "rna_true_test.npy"

    if (not args.overwrite) and model_out.exists() and metrics_out.exists() and split_out.exists():
        logging.warning("Step 06 outputs already exist. Use --overwrite to recompute.")
        print("[SKIP] Found existing Step 06 outputs.")
        return

    # load + align
    E = np.load(emb_path)
    emb_index = pd.read_csv(emb_index_path)
    Y = np.load(rna_scores_path)
    rna_index = pd.read_csv(rna_index_path)

    X, T, idx = align_by_barcode(E, emb_index, Y, rna_index)

    # optional filter
    if args.only_in_tissue and "in_tissue" in idx.columns:
        keep = idx["in_tissue"].astype(int).values == 1
        before = len(idx)
        X, T = X[keep], T[keep]
        idx = idx.loc[keep].reset_index(drop=True)
        logging.info("Filtered in_tissue==1: %d -> %d", before, len(idx))

    if len(idx) == 0:
        raise RuntimeError("No data left after alignment/filtering.")

    coords = idx[["x_pixel", "y_pixel"]].values.astype(np.float32)

    # split
    split = spatial_tile_split(
        coords=coords,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        tile_size=float(args.tile_size),
        seed=int(args.seed),
    )
    idx["split"] = split

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    X_train, Y_train = X[train_mask], T[train_mask]
    X_val, Y_val = X[val_mask], T[val_mask]
    X_test, Y_test = X[test_mask], T[test_mask]

    logging.info("Split sizes: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))
    logging.info("X dim: %s | Y dim: %s", X.shape, T.shape)

    # standardize targets using TRAIN stats
    mu: Optional[np.ndarray]
    sd: Optional[np.ndarray]
    if args.standardize_y:
        mu = Y_train.mean(axis=0, keepdims=True).astype(np.float32)
        sd = Y_train.std(axis=0, keepdims=True).astype(np.float32)
        sd = np.maximum(sd, 1e-6)
        Y_train_z = (Y_train - mu) / sd
        Y_val_z = (Y_val - mu) / sd
        Y_test_z = (Y_test - mu) / sd
        np.savez(scaler_out, mean=mu, std=sd)
        logging.info("Standardize Y enabled. Saved scaler: %s", scaler_out)
    else:
        mu, sd = None, None
        Y_train_z, Y_val_z, Y_test_z = Y_train, Y_val, Y_test

    # device
    device = resolve_device(args.device)
    logging.info("Using device: %s", device)

    # loaders
    train_ds = XYDataset(X_train, Y_train_z)
    val_ds = XYDataset(X_val, Y_val_z)
    test_ds = XYDataset(X_test, Y_test_z)

    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=max(0, int(args.workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    in_dim = int(X.shape[1])
    out_dim = int(T.shape[1])

    model = BetterMLPHead(
        in_dim=in_dim,
        out_dim=out_dim,
        width=int(args.width),
        depth=int(args.depth),
        dropout=float(args.dropout),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    loss_fn = make_loss(args.loss, args.huber_beta)

    # training with early stopping
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        train_loss = running / max(1, n)

        # validation
        model.eval()
        v_running = 0.0
        v_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                v_running += float(loss.item()) * xb.size(0)
                v_n += xb.size(0)

        val_loss = v_running / max(1, v_n)

        if epoch == 1 or epoch % 10 == 0:
            logging.info("Epoch %d | train_loss=%.6f | val_loss=%.6f", epoch, train_loss, val_loss)

        # early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                logging.info("Early stopping at epoch %d (best val=%.6f).", epoch, best_val)
                break

    if best_state is None:
        raise RuntimeError("Training failed: best_state is None.")
    model.load_state_dict(best_state)

    # predictions in standardized space
    pred_train_z = predict_array(model, DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=False), device)
    pred_val_z = predict_array(model, val_loader, device)
    pred_test_z = predict_array(model, test_loader, device)

    # un-standardize for saving/metrics
    if args.standardize_y:
        assert mu is not None and sd is not None
        pred_train = (pred_train_z * sd + mu).astype(np.float32)
        pred_val = (pred_val_z * sd + mu).astype(np.float32)
        pred_test = (pred_test_z * sd + mu).astype(np.float32)
    else:
        pred_train, pred_val, pred_test = pred_train_z, pred_val_z, pred_test_z

    # metrics in original scale
    df_val = compute_metrics(Y_val, pred_val)
    df_val.insert(0, "split", "val")
    df_test = compute_metrics(Y_test, pred_test)
    df_test.insert(0, "split", "test")
    metrics = pd.concat([df_val, df_test], ignore_index=True)

    # save checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "in_dim": in_dim,
        "out_dim": out_dim,
        "width": int(args.width),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
        "loss": str(args.loss),
        "huber_beta": float(args.huber_beta),
        "standardize_y": bool(args.standardize_y),
        "y_mean": None if mu is None else mu.astype(np.float32),
        "y_std": None if sd is None else sd.astype(np.float32),
        "split_seed": int(args.seed),
        "tile_size": float(args.tile_size),
    }
    torch.save(ckpt, model_out)

    # save files
    idx[["barcode", "x_pixel", "y_pixel", "in_tissue", "split"]].to_csv(split_out, index=False)
    metrics.to_csv(metrics_out, index=False)

    np.save(pred_train_out, pred_train)
    np.save(pred_val_out, pred_val)
    np.save(pred_test_out, pred_test)
    np.save(true_train_out, Y_train.astype(np.float32))
    np.save(true_val_out, Y_val.astype(np.float32))
    np.save(true_test_out, Y_test.astype(np.float32))

    print(f"[OK] Wrote model:    {model_out}")
    if args.standardize_y:
        print(f"[OK] Wrote scaler:   {scaler_out}")
    print(f"[OK] Wrote split:    {split_out}")
    print(f"[OK] Wrote metrics:  {metrics_out}")
    print(f"[OK] Saved preds:    {pred_train_out.name}, {pred_val_out.name}, {pred_test_out.name}")
    print("[OK] Step 06 complete.")

    overall_test = metrics[(metrics["split"] == "test") & (metrics["program"] == "OVERALL_MEAN")]
    if len(overall_test) == 1:
        r2 = float(overall_test["r2"].values[0])
        pr = float(overall_test["pearson"].values[0])
        print(f"  Test overall mean R2={r2:.3f} | Pearson={pr:.3f}")
    print(f"  N aligned spots: {len(idx)}")
    print("  Model: BetterMLPHead (Residual + LayerNorm)")


if __name__ == "__main__":
    main()
