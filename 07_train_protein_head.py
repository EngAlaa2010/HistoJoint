#!/usr/bin/env python3
"""
07_train_protein_head.py

Step 07: Train Protein prediction head
Learn mapping:
  embeddings_spatial (morphology per spot) -> protein marker vector per spot

Inputs (default under base/outputs):
  - outputs/embeddings_spatial.npy
  - outputs/embeddings_index.csv
  - outputs/adata_prot_processed.h5ad

Outputs (under outputs/):
  - prot_head_model.pt
  - prot_head_y_scaler.npz                 (if --standardize-y)
  - prot_head_split.csv
  - prot_head_metrics.csv
  - prot_pred_{train,val,test}.npy
  - prot_true_{train,val,test}.npy
  - prot_feature_names.csv                 (final targets used, in order)
  - prot_feature_variance.csv              (variance table for transparency)

Key features:
  - Robust alignment by barcode (no row-order assumptions).
  - Tile-based spatial split (reduces leakage).
  - Target selection:
      * default: exclude isotypes (mouse_/rat_ IgG controls)
      * optional: --topk-variance K  (e.g., K=6)
      * optional: --proteins CD3E-1 CD8A-1 ...
      * optional: --proteins-file path/to/list.txt (one protein per line)
  - BetterMLPHead: MLP with LayerNorm + residual blocks.
  - Optional target standardization (recommended): --standardize-y

Examples:
  Train on ALL non-isotype proteins:
    python scripts/07_train_protein_head.py -v --only-in-tissue --standardize-y --device auto

  Train on top-6 variance proteins:
    python scripts/07_train_protein_head.py -v --only-in-tissue --standardize-y --topk-variance 6 --device auto

  Train on a chosen panel:
    python scripts/07_train_protein_head.py -v --only-in-tissue --standardize-y \
      --proteins KRT5-1 EPCAM-1 VIM-1 CD3E-1 CD8A-1 HLA-DRA --device auto
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import scanpy as sc

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


# ---------------- utils ----------------

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


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_to_dense(X) -> np.ndarray:
    if isinstance(X, np.ndarray):
        return X
    try:
        return X.toarray()
    except Exception:
        return np.asarray(X)


def spatial_tile_split(
    coords: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    tile_size: float,
    seed: int,
) -> np.ndarray:
    """
    Tile-based spatial split:
      - bucket spots into tile grid by tile_size pixels
      - assign tiles (not spots) to train/val/test
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

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


def align_embeddings_and_protein(
    E: np.ndarray,
    emb_index: pd.DataFrame,
    prot_adata: sc.AnnData,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Align X and Y by barcode (robust).
    Returns:
      X_all (N x D), Y_all (N x K_all), idx (N rows) with barcode/x/y/in_tissue + prot_row
    """
    if "barcode" not in emb_index.columns:
        raise ValueError("embeddings_index.csv must contain 'barcode' column.")
    for col in ["x_pixel", "y_pixel"]:
        if col not in emb_index.columns:
            raise ValueError(f"embeddings_index.csv must contain '{col}' for spatial split.")

    emb_df = emb_index.copy()
    emb_df["emb_row"] = emb_df.get("row_id", np.arange(len(emb_df)))

    prot_df = pd.DataFrame({"barcode": prot_adata.obs_names})
    prot_df["prot_row"] = np.arange(prot_adata.n_obs, dtype=int)

    merged = emb_df.merge(prot_df, on="barcode", how="inner")
    if len(merged) == 0:
        raise RuntimeError("No overlapping barcodes between embeddings_index.csv and adata_prot_processed.h5ad")

    emb_rows = merged["emb_row"].astype(int).values
    prot_rows = merged["prot_row"].astype(int).values

    X = E[emb_rows, :].astype(np.float32)

    Y_full = l2_to_dense(prot_adata.X).astype(np.float32)
    Y = Y_full[prot_rows, :]

    merged = merged.reset_index(drop=True)
    return X, Y, merged


def is_isotype(name: str) -> bool:
    n = name.lower()
    # Your dataset examples: mouse_IgG2a, mouse_IgG1k, mouse_IgG2bk, rat_IgG2a
    if n.startswith("mouse_") or n.startswith("rat_"):
        return True
    if "igg" in n and ("mouse" in n or "rat" in n):
        return True
    return False


def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    K = Y_true.shape[1]
    rows: List[Dict] = []
    for j in range(K):
        yt = Y_true[:, j]
        yp = Y_pred[:, j]
        rows.append({
            "protein": feature_names[j],
            "protein_idx": j,
            "r2": float(r2_score(yt, yp)),
            "pearson": safe_pearson(yt, yp),
        })
    df = pd.DataFrame(rows)
    overall = pd.DataFrame([{
        "protein": "OVERALL_MEAN",
        "protein_idx": -1,
        "r2": float(np.nanmean(df["r2"].values)),
        "pearson": float(np.nanmean(df["pearson"].values)),
    }])
    return pd.concat([df, overall], ignore_index=True)


# ---------------- dataset ----------------

class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]


# ---------------- model: BetterMLPHead ----------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class BetterMLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(ResidualBlock(h, dropout))
            prev = h
        layers.append(nn.LayerNorm(prev))
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


# ---------------- args ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 07: Train Protein head from spatial morphology embeddings.")
    p.add_argument("--base", type=str, default="/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    p.add_argument("--only-in-tissue", action="store_true", help="Use only in_tissue==1 if available in embeddings_index.csv")

    # inputs
    p.add_argument("--embeddings", type=str, default="", help="Default: base/outputs/embeddings_spatial.npy")
    p.add_argument("--emb-index", type=str, default="", help="Default: base/outputs/embeddings_index.csv")
    p.add_argument("--prot-adata", type=str, default="", help="Default: base/outputs/adata_prot_processed.h5ad")

    # target selection
    p.add_argument("--exclude-isotypes", action="store_true", help="Drop mouse_/rat_ IgG control features (recommended).")
    p.add_argument("--topk-variance", type=int, default=0, help="If >0, keep only top-K variance proteins (after exclusions).")
    p.add_argument("--proteins", nargs="*", default=None, help="Explicit protein list to predict (overrides topk).")
    p.add_argument("--proteins-file", type=str, default="", help="Text file with one protein per line (overrides topk).")

    # split
    p.add_argument("--tile-size", type=float, default=500.0)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)

    # model/training
    p.add_argument("--hidden", nargs="+", type=int, default=[512, 256])
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--standardize-y", action="store_true", help="Z-score protein targets using TRAIN mean/std (recommended).")

    # runtime
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--outdir", type=str, default="", help="Default: base/outputs")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def read_proteins_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--proteins-file not found: {p}")
    items = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            items.append(s)
    return items


# ---------------- main ----------------

def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)
    set_seeds(int(args.seed))

    base = Path(args.base)
    outdir = Path(args.outdir) if args.outdir else (base / "outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    emb_path = Path(args.embeddings) if args.embeddings else (base / "outputs" / "embeddings_spatial.npy")
    emb_index_path = Path(args.emb_index) if args.emb_index else (base / "outputs" / "embeddings_index.csv")
    prot_path = Path(args.prot_adata) if args.prot_adata else (base / "outputs" / "adata_prot_processed.h5ad")

    load_required(emb_path, "embeddings_spatial.npy")
    load_required(emb_index_path, "embeddings_index.csv")
    load_required(prot_path, "adata_prot_processed.h5ad")

    # outputs
    model_out = outdir / "prot_head_model.pt"
    scaler_out = outdir / "prot_head_y_scaler.npz"
    split_out = outdir / "prot_head_split.csv"
    metrics_out = outdir / "prot_head_metrics.csv"
    pred_train_out = outdir / "prot_pred_train.npy"
    pred_val_out = outdir / "prot_pred_val.npy"
    pred_test_out = outdir / "prot_pred_test.npy"
    true_train_out = outdir / "prot_true_train.npy"
    true_val_out = outdir / "prot_true_val.npy"
    true_test_out = outdir / "prot_true_test.npy"
    feat_out = outdir / "prot_feature_names.csv"
    var_out = outdir / "prot_feature_variance.csv"

    if (not args.overwrite) and model_out.exists() and metrics_out.exists() and split_out.exists():
        logging.warning("Step 07 outputs already exist. Use --overwrite to recompute.")
        print("[SKIP] Found existing Step 07 outputs.")
        return

    # load
    E = np.load(emb_path)
    emb_index = pd.read_csv(emb_index_path)
    prot = sc.read_h5ad(prot_path)

    # align by barcode
    X_all, Y_all, idx = align_embeddings_and_protein(E, emb_index, prot)

    # optional in_tissue filter (from embeddings_index)
    if args.only_in_tissue and "in_tissue" in idx.columns:
        keep = idx["in_tissue"].astype(int).values == 1
        before = len(idx)
        X_all = X_all[keep]
        Y_all = Y_all[keep]
        idx = idx.loc[keep].reset_index(drop=True)
        logging.info("Filtered in_tissue==1: %d -> %d", before, len(idx))

    if len(idx) == 0:
        raise RuntimeError("No aligned spots remain after filtering.")

    # ----- target selection -----
    all_names = list(prot.var_names)
    keep_mask = np.ones(len(all_names), dtype=bool)

    if args.exclude_isotypes:
        keep_mask = np.array([not is_isotype(n) for n in all_names], dtype=bool)

    # user-provided explicit protein list overrides topk
    chosen: Optional[List[str]] = None
    if args.proteins_file:
        chosen = read_proteins_file(args.proteins_file)
    elif args.proteins is not None and len(args.proteins) > 0:
        chosen = list(args.proteins)

    if chosen is not None:
        chosen_set = set(chosen)
        keep_mask = keep_mask & np.array([(n in chosen_set) for n in all_names], dtype=bool)

    # compute variance (after current mask)
    Y_masked = Y_all[:, keep_mask]
    names_masked = [n for n, k in zip(all_names, keep_mask) if k]
    if Y_masked.shape[1] == 0:
        raise RuntimeError("No protein targets selected after filtering. Check --exclude-isotypes / --proteins / --proteins-file.")

    variances = Y_masked.var(axis=0).astype(np.float64)
    var_df = pd.DataFrame({"protein": names_masked, "variance": variances}).sort_values("variance", ascending=False)
    var_df.to_csv(var_out, index=False)

    if chosen is None and int(args.topk_variance) > 0:
        k = int(args.topk_variance)
        k = min(k, len(names_masked))
        top_names = var_df.head(k)["protein"].tolist()
        top_set = set(top_names)
        keep_mask = np.array([(n in top_set) for n in all_names], dtype=bool)
        # re-apply exclusion (if requested)
        if args.exclude_isotypes:
            keep_mask = keep_mask & np.array([not is_isotype(n) for n in all_names], dtype=bool)

    # finalize Y + names
    Y = Y_all[:, keep_mask].astype(np.float32)
    feature_names = [n for n, k in zip(all_names, keep_mask) if k]

    if Y.shape[1] == 0:
        raise RuntimeError("Final protein target matrix has 0 columns. Check your selection flags.")

    logging.info("Protein targets selected: K=%d", Y.shape[1])
    logging.info("First targets: %s", ", ".join(feature_names[:min(10, len(feature_names))]))

    pd.DataFrame({"protein": feature_names}).to_csv(feat_out, index=False)

    # ----- split -----
    coords = idx[["x_pixel", "y_pixel"]].values.astype(np.float32)
    split = spatial_tile_split(
        coords,
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

    X_train, Y_train = X_all[train_mask], Y[train_mask]
    X_val, Y_val = X_all[val_mask], Y[val_mask]
    X_test, Y_test = X_all[test_mask], Y[test_mask]

    logging.info("Split sizes: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))
    logging.info("X dim: %s | Y dim: %s", X_all.shape, Y.shape)

    # ----- standardize targets (train stats only) -----
    if args.standardize_y:
        mu = Y_train.mean(axis=0, keepdims=True).astype(np.float32)
        sd = Y_train.std(axis=0, keepdims=True).astype(np.float32)
        sd = np.maximum(sd, 1e-6)
        Y_train_z = (Y_train - mu) / sd
        Y_val_z = (Y_val - mu) / sd
        Y_test_z = (Y_test - mu) / sd
        np.savez(scaler_out, mean=mu, std=sd, feature_names=np.array(feature_names, dtype=object))
        logging.info("Standardize Y enabled. Saved scaler: %s", scaler_out)
    else:
        mu = sd = None
        Y_train_z, Y_val_z, Y_test_z = Y_train, Y_val, Y_test

    # ----- train -----
    device = resolve_device(args.device)
    logging.info("Using device: %s", device)

    train_loader = DataLoader(
        XYDataset(X_train, Y_train_z),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=max(0, int(args.workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        XYDataset(X_val, Y_val_z),
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        XYDataset(X_test, Y_test_z),
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    in_dim = int(X_all.shape[1])
    out_dim = int(Y.shape[1])

    model = BetterMLPHead(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden=list(map(int, args.hidden)),
        dropout=float(args.dropout),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        tr_sum = 0.0
        tr_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            tr_sum += float(loss.item()) * xb.size(0)
            tr_n += xb.size(0)

        train_loss = tr_sum / max(1, tr_n)

        model.eval()
        v_sum = 0.0
        v_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                v_sum += float(loss_fn(model(xb), yb).item()) * xb.size(0)
                v_n += xb.size(0)
        val_loss = v_sum / max(1, v_n)

        if epoch == 1 or epoch % 10 == 0:
            logging.info("Epoch %d | train_loss=%.6f | val_loss=%.6f", epoch, train_loss, val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(args.patience):
                logging.info("Early stopping at epoch %d (best val=%.6f).", epoch, best_val)
                break

    if best_state is None:
        raise RuntimeError("Training failed (best_state is None).")
    model.load_state_dict(best_state)

    # ----- predictions (standardized space) -----
    pred_train_z = predict(model, DataLoader(XYDataset(X_train, Y_train_z), batch_size=int(args.batch_size), shuffle=False), device)
    pred_val_z = predict(model, val_loader, device)
    pred_test_z = predict(model, test_loader, device)

    # unstandardize back to original scale for metrics/saving
    if args.standardize_y:
        pred_train = (pred_train_z * sd + mu).astype(np.float32)
        pred_val = (pred_val_z * sd + mu).astype(np.float32)
        pred_test = (pred_test_z * sd + mu).astype(np.float32)
    else:
        pred_train, pred_val, pred_test = pred_train_z, pred_val_z, pred_test_z

    # ----- metrics -----
    m_val = compute_metrics(Y_val, pred_val, feature_names)
    m_val.insert(0, "split", "val")
    m_test = compute_metrics(Y_test, pred_test, feature_names)
    m_test.insert(0, "split", "test")
    metrics = pd.concat([m_val, m_test], ignore_index=True)

    # ----- save -----
    ckpt = {
        "model_state_dict": model.state_dict(),
        "in_dim": in_dim,
        "out_dim": out_dim,
        "hidden": list(map(int, args.hidden)),
        "dropout": float(args.dropout),
        "standardize_y": bool(args.standardize_y),
        "y_mean": None if mu is None else mu.astype(np.float32),
        "y_std": None if sd is None else sd.astype(np.float32),
        "protein_names": feature_names,
        "exclude_isotypes": bool(args.exclude_isotypes),
        "topk_variance": int(args.topk_variance),
        "seed": int(args.seed),
        "tile_size": float(args.tile_size),
    }
    torch.save(ckpt, model_out)

    idx[["barcode", "x_pixel", "y_pixel", "in_tissue", "split"]].to_csv(split_out, index=False)
    metrics.to_csv(metrics_out, index=False)

    np.save(pred_train_out, pred_train.astype(np.float32))
    np.save(pred_val_out, pred_val.astype(np.float32))
    np.save(pred_test_out, pred_test.astype(np.float32))

    np.save(true_train_out, Y_train.astype(np.float32))
    np.save(true_val_out, Y_val.astype(np.float32))
    np.save(true_test_out, Y_test.astype(np.float32))

    print(f"[OK] Wrote model:    {model_out}")
    if args.standardize_y:
        print(f"[OK] Wrote scaler:   {scaler_out}")
    print(f"[OK] Wrote split:    {split_out}")
    print(f"[OK] Wrote metrics:  {metrics_out}")
    print(f"[OK] Wrote proteins: {feat_out}")
    print(f"[OK] Wrote variance: {var_out}")
    print("[OK] Step 07 complete.")

    overall_test = metrics[(metrics["split"] == "test") & (metrics["protein"] == "OVERALL_MEAN")]
    if len(overall_test) == 1:
        r2 = float(overall_test["r2"].values[0])
        pr = float(overall_test["pearson"].values[0])
        print(f"  Test overall mean R2={r2:.3f} | Pearson={pr:.3f}")
    print(f"  N aligned spots: {len(idx)}")
    print(f"  Targets: K={len(feature_names)}")
    if len(feature_names) <= 12:
        print(f"  Protein names: {feature_names}")
    print("  Model: BetterMLPHead (Residual + LayerNorm)")


if __name__ == "__main__":
    main()
