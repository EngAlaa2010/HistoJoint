#!/usr/bin/env python3
"""
09_train_morph_to_joint.py  (FIXED: early stopping on RETRIEVAL, not loss)

Step 09: Predict JOINT latent from morphology embeddings (and evaluate retrieval)

Goal:
  Learn f(X_morph) -> m (latent_dim) such that m matches the *paired* joint latents
  learned in Step 08 (z for RNA, u for protein).

Key upgrade:
  - Train with CLIP/InfoNCE contrastive loss (retrieval objective)
  - EARLY STOP + BEST CHECKPOINT selected by retrieval metric (Top-1 / Top-5),
    NOT by val_loss (because val_loss and retrieval often de-correlate).

Inputs (default under base/outputs):
  - embeddings_spatial.npy          (N x D)   Step 04
  - embeddings_index.csv            (N rows)  Step 03 (barcode, x_pixel, y_pixel, in_tissue,...)
  - z_rna_latent.npy                (M x L)   Step 08
  - u_prot_latent.npy               (M x L)   Step 08
  - joint_pair_index.csv            (M rows)  Step 08 (barcode)

Outputs (under base/outputs):
  - morph_to_joint_model.pt
  - morph_to_joint_split.csv
  - morph_to_joint_metrics.csv      (epoch log + FINAL_VAL/FINAL_TEST)
  - z_pred_{train,val,test}.npy     (predicted latent vectors; L2-normalized)
  - (optional) u_pred_{train,val,test}.npy   if --save-u-pred

Training objective (default):
  loss = InfoNCE(m_pred, z_true) + InfoNCE(m_pred, u_true)
Optional:
  + mse terms to stabilize: --lambda-mse-z, --lambda-mse-u

Early stopping:
  - score = (Top1 primary) then tie-break by (Top5) then (-mean_rank)
  - primary retrieval target is:
      U if lambda_u > 0, else Z
  - patience counts epochs WITHOUT improving retrieval score

Notes:
  - drop_last=True for train loader (stable negatives)
  - targets are L2-normalized defensively
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- split + alignment ----------------

def spatial_tile_split(
    coords: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    tile_size: float = 500.0,
    seed: int = 0,
) -> np.ndarray:
    """Tile-based spatial split to reduce leakage."""
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


def align_morph_and_joint(
    E: np.ndarray,
    emb_index: pd.DataFrame,
    z: np.ndarray,
    u: np.ndarray,
    pair_index: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Align morphology embeddings (E) with joint latents (z,u) by barcode.
    Returns:
      X (N x D), Z (N x L), U (N x L), idx (N rows with barcode,x_pixel,y_pixel,in_tissue)
    """
    if "barcode" not in emb_index.columns:
        raise ValueError("embeddings_index.csv must contain 'barcode'.")
    if "barcode" not in pair_index.columns:
        raise ValueError("joint_pair_index.csv must contain 'barcode'.")

    for col in ["x_pixel", "y_pixel"]:
        if col not in emb_index.columns:
            raise ValueError(f"embeddings_index.csv must contain '{col}' for spatial split.")

    emb_df = emb_index.copy()
    emb_df["emb_row"] = emb_df.get("row_id", np.arange(len(emb_df)))

    pair_df = pair_index.copy()
    pair_df["pair_row"] = np.arange(len(pair_df))

    merged = emb_df.merge(pair_df[["barcode", "pair_row"]], on="barcode", how="inner")
    if len(merged) == 0:
        raise RuntimeError("No overlapping barcodes between embeddings_index and joint_pair_index.")

    emb_rows = merged["emb_row"].astype(int).values
    pair_rows = merged["pair_row"].astype(int).values

    X = E[emb_rows, :].astype(np.float32)
    Z = z[pair_rows, :].astype(np.float32)
    U = u[pair_rows, :].astype(np.float32)

    merged = merged.reset_index(drop=True)
    return X, Z, U, merged


# ---------------- retrieval metrics ----------------

def retrieval_metrics(pred: np.ndarray, tgt: np.ndarray, k: int = 5) -> Dict[str, float]:
    """
    pred, tgt: (N,L) and (N,L), assumed L2-normalized.
    Evaluate retrieval within this pool: correct match is diagonal.
    Returns top1, top5, mean_rank, median_rank.
    """
    sim = pred @ tgt.T  # (N,N)
    order = np.argsort(-sim, axis=1)  # descending
    gt = np.arange(sim.shape[0])
    pos = (order == gt[:, None]).argmax(axis=1)  # 0-indexed position
    rank = pos + 1

    top1 = float(np.mean(rank == 1))
    topk = float(np.mean(rank <= k))
    mean_rank = float(np.mean(rank))
    median_rank = float(np.median(rank))
    return {"top1": top1, "top5": topk, "mean_rank": mean_rank, "median_rank": median_rank}


def retrieval_score(met: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Score used for early stopping / best checkpoint.
    Higher is better. We maximize:
      (top1, top5, -mean_rank)
    """
    return (float(met["top1"]), float(met["top5"]), -float(met["mean_rank"]))


# ---------------- loss (InfoNCE) ----------------

def clip_loss(a: torch.Tensor, b: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    CLIP-style symmetric InfoNCE.
    a,b shape: (B,L), should be normalized already.
    """
    logits = (a @ b.t()) / tau
    labels = torch.arange(a.size(0), device=a.device)
    loss_ab = nn.functional.cross_entropy(logits, labels)
    loss_ba = nn.functional.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_ab + loss_ba)


# ---------------- dataset ----------------

class MorphJointDataset(Dataset):
    def __init__(self, X: np.ndarray, Z: np.ndarray, U: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Z = torch.from_numpy(Z.astype(np.float32))
        self.U = torch.from_numpy(U.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.Z[i], self.U[i]


# ---------------- model ----------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class MorphToLatent(nn.Module):
    """Maps morphology embedding X -> latent L (L2-normalized)."""
    def __init__(self, in_dim: int, latent_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            layers += [ResidualBlock(h, dropout)]
            prev = h
        layers += [nn.LayerNorm(prev), nn.Linear(prev, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        m = self.net(x)
        return nn.functional.normalize(m, dim=1)


@torch.no_grad()
def encode_all(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i + batch_size].astype(np.float32)).to(device)
        outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Step 09: Morphology -> Joint latent (InfoNCE, early stop on retrieval).")
    p.add_argument("--base", type=str, default="/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    p.add_argument("--outdir", type=str, default="")
    p.add_argument("--only-in-tissue", action="store_true")

    # inputs
    p.add_argument("--embeddings", type=str, default="", help="Default: base/outputs/embeddings_spatial.npy")
    p.add_argument("--emb-index", type=str, default="", help="Default: base/outputs/embeddings_index.csv")
    p.add_argument("--z", type=str, default="", help="Default: base/outputs/z_rna_latent.npy")
    p.add_argument("--u", type=str, default="", help="Default: base/outputs/u_prot_latent.npy")
    p.add_argument("--pair-index", type=str, default="", help="Default: base/outputs/joint_pair_index.csv")

    # split
    p.add_argument("--tile-size", type=float, default=500.0)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)

    # model
    p.add_argument("--hidden", nargs="+", type=int, default=[1024, 512, 256])
    p.add_argument("--dropout", type=float, default=0.10)

    # loss
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--lambda-z", type=float, default=1.0)
    p.add_argument("--lambda-u", type=float, default=1.0)
    p.add_argument("--lambda-mse-z", type=float, default=0.0)
    p.add_argument("--lambda-mse-u", type=float, default=0.0)

    # training
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)

    # early stopping details
    p.add_argument("--early-stop-metric", type=str, default="auto",
                   choices=["auto", "z_top1", "u_top1"],
                   help="Which retrieval to early-stop on. auto: u if lambda_u>0 else z.")
    p.add_argument("--min-delta", type=float, default=1e-6,
                   help="Minimum improvement in retrieval score (top1/top5/mean_rank tie-break) to reset patience.")

    # outputs
    p.add_argument("--save-u-pred", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    # runtime
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--workers", type=int, default=0)
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
    z_path = Path(args.z) if args.z else (base / "outputs" / "z_rna_latent.npy")
    u_path = Path(args.u) if args.u else (base / "outputs" / "u_prot_latent.npy")
    pair_path = Path(args.pair_index) if args.pair_index else (base / "outputs" / "joint_pair_index.csv")

    for pth, nm in [(emb_path, "embeddings_spatial.npy"),
                    (emb_index_path, "embeddings_index.csv"),
                    (z_path, "z_rna_latent.npy"),
                    (u_path, "u_prot_latent.npy"),
                    (pair_path, "joint_pair_index.csv")]:
        if not pth.exists():
            raise FileNotFoundError(f"Missing {nm}: {pth}")

    model_out = outdir / "morph_to_joint_model.pt"
    split_out = outdir / "morph_to_joint_split.csv"
    metrics_out = outdir / "morph_to_joint_metrics.csv"

    z_pred_train_out = outdir / "z_pred_train.npy"
    z_pred_val_out = outdir / "z_pred_val.npy"
    z_pred_test_out = outdir / "z_pred_test.npy"

    u_pred_train_out = outdir / "u_pred_train.npy"
    u_pred_val_out = outdir / "u_pred_val.npy"
    u_pred_test_out = outdir / "u_pred_test.npy"

    if (not args.overwrite) and model_out.exists() and metrics_out.exists() and split_out.exists():
        print("[SKIP] Found existing Step 09 outputs.")
        return

    # load + align
    E = np.load(emb_path).astype(np.float32)
    emb_index = pd.read_csv(emb_index_path)
    z = np.load(z_path).astype(np.float32)
    u = np.load(u_path).astype(np.float32)
    pair_index = pd.read_csv(pair_path)

    X, Z, U, idx = align_morph_and_joint(E, emb_index, z, u, pair_index)

    # optional filter
    if args.only_in_tissue and "in_tissue" in idx.columns:
        keep = idx["in_tissue"].astype(int).values == 1
        before = len(idx)
        X, Z, U = X[keep], Z[keep], U[keep]
        idx = idx.loc[keep].reset_index(drop=True)
        logging.info("Filtered in_tissue==1: %d -> %d", before, len(idx))

    # defensive L2 norm
    def l2n(a: np.ndarray) -> np.ndarray:
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)

    Z = l2n(Z)
    U = l2n(U)

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

    tr = split == "train"
    va = split == "val"
    te = split == "test"

    X_tr, Z_tr, U_tr = X[tr], Z[tr], U[tr]
    X_va, Z_va, U_va = X[va], Z[va], U[va]
    X_te, Z_te, U_te = X[te], Z[te], U[te]

    latent_dim = Z.shape[1]
    logging.info("Aligned spots: N=%d | X=%s | latent=%d", len(idx), X.shape, latent_dim)
    logging.info("Split sizes: train=%d val=%d test=%d", X_tr.shape[0], X_va.shape[0], X_te.shape[0])

    device = resolve_device(args.device)
    logging.info("Using device: %s", device)

    train_loader = DataLoader(
        MorphJointDataset(X_tr, Z_tr, U_tr),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=max(0, int(args.workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = MorphToLatent(
        in_dim=X.shape[1],
        latent_dim=int(latent_dim),
        hidden=list(args.hidden),
        dropout=float(args.dropout),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # choose early-stop target
    if args.early_stop_metric == "auto":
        early_target = "u" if args.lambda_u > 0 else "z"
    elif args.early_stop_metric == "u_top1":
        early_target = "u"
    else:
        early_target = "z"

    best_score: Optional[Tuple[float, float, float]] = None
    best_state = None
    best_epoch = -1
    bad = 0
    rows: List[Dict] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        tr_loss = 0.0
        n = 0

        for xb, zb, ub in train_loader:
            xb = xb.to(device, non_blocking=True)
            zb = zb.to(device, non_blocking=True)
            ub = ub.to(device, non_blocking=True)

            m = model(xb)  # normalized

            loss = 0.0
            if args.lambda_z > 0:
                loss = loss + float(args.lambda_z) * clip_loss(m, zb, tau=float(args.tau))
            if args.lambda_u > 0:
                loss = loss + float(args.lambda_u) * clip_loss(m, ub, tau=float(args.tau))
            if args.lambda_mse_z > 0:
                loss = loss + float(args.lambda_mse_z) * nn.functional.mse_loss(m, zb)
            if args.lambda_mse_u > 0:
                loss = loss + float(args.lambda_mse_u) * nn.functional.mse_loss(m, ub)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        tr_loss /= max(1, n)

        # retrieval on full VAL pool (the thing we care about)
        m_va = encode_all(model, X_va, device=device, batch_size=1024)
        met_va_z = retrieval_metrics(m_va, Z_va, k=5)
        met_va_u = retrieval_metrics(m_va, U_va, k=5)

        met_primary = met_va_u if early_target == "u" else met_va_z
        score = retrieval_score(met_primary)

        # logging
        if epoch == 1 or epoch % 10 == 0:
            logging.info(
                "Epoch %d | train_loss=%.5f | VAL(z): top1=%.3f top5=%.3f mean_rank=%.2f | "
                "VAL(u): top1=%.3f top5=%.3f mean_rank=%.2f | early=%s",
                epoch,
                tr_loss,
                met_va_z["top1"], met_va_z["top5"], met_va_z["mean_rank"],
                met_va_u["top1"], met_va_u["top5"], met_va_u["mean_rank"],
                early_target,
            )

        rows.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_top1_z": met_va_z["top1"],
            "val_top5_z": met_va_z["top5"],
            "val_mean_rank_z": met_va_z["mean_rank"],
            "val_median_rank_z": met_va_z["median_rank"],
            "val_top1_u": met_va_u["top1"],
            "val_top5_u": met_va_u["top5"],
            "val_mean_rank_u": met_va_u["mean_rank"],
            "val_median_rank_u": met_va_u["median_rank"],
            "early_target": early_target,
            "early_score_top1": score[0],
            "early_score_top5": score[1],
            "early_score_neg_mean_rank": score[2],
        })

        # ---- EARLY STOPPING ON RETRIEVAL SCORE ----
        improved = False
        if best_score is None:
            improved = True
        else:
            # lexicographic improvement with small tolerance on top1/top5/mean_rank
            # score = (top1, top5, -mean_rank)
            if score[0] > best_score[0] + args.min_delta:
                improved = True
            elif abs(score[0] - best_score[0]) <= args.min_delta:
                if score[1] > best_score[1] + args.min_delta:
                    improved = True
                elif abs(score[1] - best_score[1]) <= args.min_delta:
                    if score[2] > best_score[2] + args.min_delta:
                        improved = True

        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(args.patience):
                logging.info(
                    "Early stopping at epoch %d (best epoch=%d best_score(top1,top5,-mean_rank)=%s).",
                    epoch, best_epoch, str(best_score)
                )
                break

    if best_state is None:
        raise RuntimeError("Training failed (best_state is None).")
    model.load_state_dict(best_state)

    # final predictions for each split
    m_tr = encode_all(model, X_tr, device=device, batch_size=1024)
    m_va = encode_all(model, X_va, device=device, batch_size=1024)
    m_te = encode_all(model, X_te, device=device, batch_size=1024)

    np.save(z_pred_train_out, m_tr)
    np.save(z_pred_val_out, m_va)
    np.save(z_pred_test_out, m_te)

    if args.save_u_pred:
        np.save(u_pred_train_out, m_tr)
        np.save(u_pred_val_out, m_va)
        np.save(u_pred_test_out, m_te)

    # final retrieval metrics on VAL/TEST pools
    final_val_z = retrieval_metrics(m_va, Z_va, k=5)
    final_val_u = retrieval_metrics(m_va, U_va, k=5)
    final_test_z = retrieval_metrics(m_te, Z_te, k=5)
    final_test_u = retrieval_metrics(m_te, U_te, k=5)

    # save logs + finals
    rows.append({
        "epoch": "FINAL_VAL",
        "train_loss": "",
        "val_top1_z": final_val_z["top1"],
        "val_top5_z": final_val_z["top5"],
        "val_mean_rank_z": final_val_z["mean_rank"],
        "val_median_rank_z": final_val_z["median_rank"],
        "val_top1_u": final_val_u["top1"],
        "val_top5_u": final_val_u["top5"],
        "val_mean_rank_u": final_val_u["mean_rank"],
        "val_median_rank_u": final_val_u["median_rank"],
        "early_target": early_target,
        "best_epoch": best_epoch,
    })
    rows.append({
        "epoch": "FINAL_TEST",
        "train_loss": "",
        "val_top1_z": final_test_z["top1"],
        "val_top5_z": final_test_z["top5"],
        "val_mean_rank_z": final_test_z["mean_rank"],
        "val_median_rank_z": final_test_z["median_rank"],
        "val_top1_u": final_test_u["top1"],
        "val_top5_u": final_test_u["top5"],
        "val_mean_rank_u": final_test_u["mean_rank"],
        "val_median_rank_u": final_test_u["median_rank"],
        "early_target": early_target,
        "best_epoch": best_epoch,
    })

    pd.DataFrame(rows).to_csv(metrics_out, index=False)

    cols = ["barcode", "x_pixel", "y_pixel"]
    if "in_tissue" in idx.columns:
        cols.append("in_tissue")
    cols.append("split")
    idx[cols].to_csv(split_out, index=False)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "in_dim": int(X.shape[1]),
        "latent_dim": int(latent_dim),
        "hidden": list(args.hidden),
        "dropout": float(args.dropout),
        "tau": float(args.tau),
        "lambda_z": float(args.lambda_z),
        "lambda_u": float(args.lambda_u),
        "lambda_mse_z": float(args.lambda_mse_z),
        "lambda_mse_u": float(args.lambda_mse_u),
        "early_stop_metric": args.early_stop_metric,
        "early_target": early_target,
        "best_epoch": int(best_epoch),
        "best_score_top1": None if best_score is None else float(best_score[0]),
        "best_score_top5": None if best_score is None else float(best_score[1]),
        "best_score_neg_mean_rank": None if best_score is None else float(best_score[2]),
    }
    torch.save(ckpt, model_out)

    # print headline numbers based on early_target
    final_val_primary = final_val_u if early_target == "u" else final_val_z
    final_test_primary = final_test_u if early_target == "u" else final_test_z

    print(f"[OK] Wrote model:   {model_out}")
    print(f"[OK] Wrote split:   {split_out}")
    print(f"[OK] Wrote metrics: {metrics_out}")
    print(f"[OK] Saved preds:   {z_pred_train_out.name}, {z_pred_val_out.name}, {z_pred_test_out.name}")
    if args.save_u_pred:
        print(f"[OK] Saved u_preds: {u_pred_train_out.name}, {u_pred_val_out.name}, {u_pred_test_out.name}")
    print("[OK] Step 09 complete.")
    print(f"  Best epoch (by retrieval): {best_epoch} | early_target={early_target}")
    print(f"  FINAL VAL:  top1={final_val_primary['top1']:.3f} top5={final_val_primary['top5']:.3f} mean_rank={final_val_primary['mean_rank']:.2f}")
    print(f"  FINAL TEST: top1={final_test_primary['top1']:.3f} top5={final_test_primary['top5']:.3f} mean_rank={final_test_primary['mean_rank']:.2f}")


if __name__ == "__main__":
    main()