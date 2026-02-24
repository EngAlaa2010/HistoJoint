#!/usr/bin/env python3
"""
08_train_joint_latent.py  (IMPROVED)

Step 08: Joint RNA–Protein latent space (paired alignment)

Learn:
  EncR: RNA programs y -> z (latent_dim)
  EncP: Protein markers q -> u (latent_dim)
  Optional Map: z -> u (supervised latent translation)

Loss:
  - Contrastive InfoNCE (CLIP-style) between z and u for same barcode
  - Optional MSE(Map(z), u)

Inputs:
  - outputs/rna_program_scores.npy
  - outputs/rna_program_index.csv
  - outputs/adata_prot_processed.h5ad
  - optional outputs/prot_feature_names.csv to subset proteins

Outputs:
  - outputs/joint_rna_prot_model.pt
  - outputs/z_rna_latent.npy
  - outputs/u_prot_latent.npy
  - outputs/joint_pair_index.csv     (barcode + split)
  - outputs/joint_metrics.csv        (epoch training curve + final retrieval metrics)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc

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


# ---------------- data loading / alignment ----------------

def load_rna(scores_path: Path, index_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    Y = np.load(scores_path).astype(np.float32)  # (N,P)
    idx = pd.read_csv(index_path)
    if "barcode" not in idx.columns:
        raise ValueError("rna_program_index.csv must contain 'barcode'.")
    idx = idx.copy()
    idx["rna_row"] = idx.get("row_id", np.arange(len(idx)))
    return Y, idx


def load_prot(prot_h5ad: Path, subset_csv: Optional[Path]) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    ad = sc.read_h5ad(prot_h5ad)

    X = ad.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = X.astype(np.float32)
    prot_names = list(ad.var_names)

    if subset_csv is not None and subset_csv.exists():
        keep_names = pd.read_csv(subset_csv).iloc[:, 0].astype(str).tolist()
        keep_idx = [prot_names.index(n) for n in keep_names if n in prot_names]
        if len(keep_idx) == 0:
            raise ValueError("prot_feature_names.csv provided but none matched adata_prot_processed.var_names.")
        X = X[:, keep_idx]
        prot_names = [prot_names[i] for i in keep_idx]

    idx = pd.DataFrame({"barcode": ad.obs_names.astype(str).tolist()})
    idx["prot_row"] = np.arange(len(idx))
    return X, idx, prot_names


def align_by_barcode(
    rna_scores: np.ndarray,
    rna_index: pd.DataFrame,
    prot_mat: np.ndarray,
    prot_index: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    merged = rna_index.merge(
        prot_index[["barcode", "prot_row"]],
        on="barcode",
        how="inner",
    )
    if len(merged) == 0:
        raise RuntimeError("No overlapping barcodes between RNA and protein.")

    r_rows = merged["rna_row"].astype(int).values
    p_rows = merged["prot_row"].astype(int).values

    Y = rna_scores[r_rows, :].astype(np.float32)
    Q = prot_mat[p_rows, :].astype(np.float32)

    merged = merged.reset_index(drop=True)
    return Y, Q, merged


def zscore_train_only(train: np.ndarray, x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = train.mean(axis=0, keepdims=True).astype(np.float32)
    sd = train.std(axis=0, keepdims=True).astype(np.float32)
    sd = np.maximum(sd, eps)
    return (x - mu) / sd, mu, sd


# ---------------- dataset ----------------

class PairDataset(Dataset):
    def __init__(self, Y: np.ndarray, Q: np.ndarray):
        self.Y = torch.from_numpy(Y.astype(np.float32))
        self.Q = torch.from_numpy(Q.astype(np.float32))

    def __len__(self) -> int:
        return self.Y.shape[0]

    def __getitem__(self, i: int):
        return self.Y[i], self.Q[i]


# ---------------- model ----------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class JointAlign(nn.Module):
    def __init__(self, rna_dim: int, prot_dim: int, latent_dim: int,
                 enc_hidden: List[int], dropout: float, use_mapper: bool):
        super().__init__()
        self.enc_r = MLP(rna_dim, enc_hidden, latent_dim, dropout)
        self.enc_p = MLP(prot_dim, enc_hidden, latent_dim, dropout)
        self.use_mapper = use_mapper
        self.map_z_to_u = MLP(latent_dim, [latent_dim], latent_dim, dropout) if use_mapper else None

    def encode_rna(self, y):
        z = self.enc_r(y)
        return nn.functional.normalize(z, dim=1)

    def encode_prot(self, q):
        u = self.enc_p(q)
        return nn.functional.normalize(u, dim=1)


def clip_loss(z: torch.Tensor, u: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    logits = (z @ u.t()) / tau
    labels = torch.arange(z.size(0), device=z.device)
    loss_zu = nn.functional.cross_entropy(logits, labels)
    loss_uz = nn.functional.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_zu + loss_uz)


@torch.no_grad()
def retrieval_metrics(z: torch.Tensor, u: torch.Tensor, ks=(1, 5)) -> Dict[str, float]:
    # cosine similarity (already normalized)
    sim = z @ u.t()  # (N,N)
    ranks = torch.argsort(sim, dim=1, descending=True)  # indices sorted by similarity
    gt = torch.arange(z.size(0), device=z.device).unsqueeze(1)  # (N,1)

    out: Dict[str, float] = {}
    for k in ks:
        topk = ranks[:, :k]
        hit = (topk == gt).any(dim=1).float().mean().item()
        out[f"top{k}"] = float(hit)

    # rank of true match
    # position where ranks == gt
    pos = (ranks == gt).nonzero(as_tuple=False)[:, 1].float()  # (N,)
    out["mean_rank"] = float(pos.mean().item())
    out["median_rank"] = float(pos.median().item())
    return out


# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Step 08: Joint RNA–Protein latent space (contrastive + optional mapper).")
    p.add_argument("--base", type=str, default="/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    p.add_argument("--outdir", type=str, default="")

    p.add_argument("--rna-scores", type=str, default="")
    p.add_argument("--rna-index", type=str, default="")
    p.add_argument("--prot-adata", type=str, default="")
    p.add_argument("--prot-subset", type=str, default="", help="Optional CSV listing proteins to use (one per row).")

    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--enc-hidden", nargs="+", type=int, default=[256, 256])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use-mapper", action="store_true")
    p.add_argument("--lambda-map", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.07)

    p.add_argument("--standardize-inputs", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.80)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)

    base = Path(args.base)
    outdir = Path(args.outdir) if args.outdir else (base / "outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    rna_scores_path = Path(args.rna_scores) if args.rna_scores else (base / "outputs" / "rna_program_scores.npy")
    rna_index_path = Path(args.rna_index) if args.rna_index else (base / "outputs" / "rna_program_index.csv")
    prot_path = Path(args.prot_adata) if args.prot_adata else (base / "outputs" / "adata_prot_processed.h5ad")
    prot_subset = Path(args.prot_subset) if args.prot_subset else None

    for pth, nm in [(rna_scores_path, "rna_program_scores.npy"),
                    (rna_index_path, "rna_program_index.csv"),
                    (prot_path, "adata_prot_processed.h5ad")]:
        if not pth.exists():
            raise FileNotFoundError(f"Missing {nm}: {pth}")

    model_out = outdir / "joint_rna_prot_model.pt"
    metrics_out = outdir / "joint_metrics.csv"
    z_out = outdir / "z_rna_latent.npy"
    u_out = outdir / "u_prot_latent.npy"
    pair_out = outdir / "joint_pair_index.csv"

    if (not args.overwrite) and model_out.exists() and metrics_out.exists() and pair_out.exists():
        print("[SKIP] Found existing Step 08 outputs.")
        return

    # load + align
    Y, rna_idx = load_rna(rna_scores_path, rna_index_path)
    Q, prot_idx, prot_names = load_prot(prot_path, prot_subset)
    Y, Q, merged = align_by_barcode(Y, rna_idx, Q, prot_idx)

    N = Y.shape[0]
    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(N)

    n_train = int(round(args.train_frac * N))
    n_val = int(round(args.val_frac * N))
    tr = perm[:n_train]
    va = perm[n_train:n_train + n_val]
    te = perm[n_train + n_val:]

    split = np.array(["test"] * N, dtype=object)
    split[tr] = "train"
    split[va] = "val"
    split[te] = "test"
    merged["split"] = split

    def take(a, idx): return a[idx]

    Y_tr, Y_va, Y_te = take(Y, tr), take(Y, va), take(Y, te)
    Q_tr, Q_va, Q_te = take(Q, tr), take(Q, va), take(Q, te)

    if args.standardize_inputs:
        Y_tr_z, ymu, ysd = zscore_train_only(Y_tr, Y_tr)
        Y_va_z = (Y_va - ymu) / ysd
        Y_te_z = (Y_te - ymu) / ysd

        Q_tr_z, qmu, qsd = zscore_train_only(Q_tr, Q_tr)
        Q_va_z = (Q_va - qmu) / qsd
        Q_te_z = (Q_te - qmu) / qsd
    else:
        ymu = ysd = qmu = qsd = None
        Y_tr_z, Y_va_z, Y_te_z = Y_tr, Y_va, Y_te
        Q_tr_z, Q_va_z, Q_te_z = Q_tr, Q_va, Q_te

    logging.info("Aligned paired spots: N=%d | RNA dim=%d | Prot dim=%d", N, Y.shape[1], Q.shape[1])
    logging.info("Split sizes: train=%d val=%d test=%d", len(tr), len(va), len(te))
    logging.info("Protein targets: K=%d | first=%s", len(prot_names), ", ".join(prot_names[:min(10, len(prot_names))]))

    device = resolve_device(args.device)
    logging.info("Using device: %s", device)

    train_loader = DataLoader(PairDataset(Y_tr_z, Q_tr_z),
                              batch_size=int(args.batch_size), shuffle=True,
                              num_workers=int(args.workers), pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(PairDataset(Y_va_z, Q_va_z),
                            batch_size=int(args.batch_size), shuffle=False,
                            num_workers=int(args.workers), pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(PairDataset(Y_te_z, Q_te_z),
                             batch_size=int(args.batch_size), shuffle=False,
                             num_workers=int(args.workers), pin_memory=(device.type == "cuda"))

    model = JointAlign(
        rna_dim=Y.shape[1],
        prot_dim=Q.shape[1],
        latent_dim=int(args.latent_dim),
        enc_hidden=list(args.enc_hidden),
        dropout=float(args.dropout),
        use_mapper=bool(args.use_mapper),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # cache full VAL tensors for full-retrieval evaluation each epoch
    Yv_t = torch.from_numpy(Y_va_z.astype(np.float32)).to(device)
    Qv_t = torch.from_numpy(Q_va_z.astype(np.float32)).to(device)

    best_val = float("inf")
    best_state = None
    bad = 0
    curve_rows: List[Dict] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        tr_loss = 0.0
        n = 0

        for yb, qb in train_loader:
            yb = yb.to(device, non_blocking=True)
            qb = qb.to(device, non_blocking=True)

            z = model.encode_rna(yb)
            u = model.encode_prot(qb)

            loss = clip_loss(z, u, tau=float(args.tau))

            if model.use_mapper:
                u_hat = nn.functional.normalize(model.map_z_to_u(z), dim=1)
                loss = loss + float(args.lambda_map) * nn.functional.mse_loss(u_hat, u)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += float(loss.item()) * yb.size(0)
            n += yb.size(0)

        tr_loss /= max(1, n)

        # full-val evaluation
        model.eval()
        with torch.no_grad():
            zv = model.encode_rna(Yv_t)
            uv = model.encode_prot(Qv_t)
            v_loss = clip_loss(zv, uv, tau=float(args.tau)).item()
            mets = retrieval_metrics(zv, uv, ks=(1, 5))
            v_top1 = mets["top1"]

        if epoch == 1 or epoch % 10 == 0:
            logging.info(
                "Epoch %d | train_loss=%.5f | val_loss=%.5f | val_top1=%.3f | val_top5=%.3f",
                epoch, tr_loss, v_loss, mets["top1"], mets["top5"]
            )

        curve_rows.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": float(v_loss),
            "val_top1": float(mets["top1"]),
            "val_top5": float(mets["top5"]),
            "val_mean_rank": float(mets["mean_rank"]),
            "val_median_rank": float(mets["median_rank"]),
        })

        if v_loss < best_val - 1e-6:
            best_val = float(v_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(args.patience):
                logging.info("Early stopping at epoch %d (best val=%.5f).", epoch, best_val)
                break

    if best_state is None:
        raise RuntimeError("Training failed (best_state is None).")
    model.load_state_dict(best_state)

    # encode ALL (in merged order)
    with torch.no_grad():
        if args.standardize_inputs:
            Y_all = (Y - ymu) / ysd
            Q_all = (Q - qmu) / qsd
        else:
            Y_all, Q_all = Y, Q

        y_t = torch.from_numpy(Y_all.astype(np.float32)).to(device)
        q_t = torch.from_numpy(Q_all.astype(np.float32)).to(device)

        z_all = model.encode_rna(y_t).cpu().numpy().astype(np.float32)
        u_all = model.encode_prot(q_t).cpu().numpy().astype(np.float32)

    np.save(z_out, z_all)
    np.save(u_out, u_all)

    # save index WITH split
    merged[["barcode", "split"]].to_csv(pair_out, index=False)

    # final retrieval metrics on VAL and TEST (full sets)
    def final_eval(mask_name: str, rows_idx: np.ndarray) -> Dict[str, float]:
        zt = torch.from_numpy(z_all[rows_idx]).to(device)
        ut = torch.from_numpy(u_all[rows_idx]).to(device)
        return retrieval_metrics(zt, ut, ks=(1, 5))

    val_rows = np.where(merged["split"].values == "val")[0]
    test_rows = np.where(merged["split"].values == "test")[0]

    final_val = final_eval("val", val_rows)
    final_test = final_eval("test", test_rows)

    # write metrics: training curve + final summary rows
    df_curve = pd.DataFrame(curve_rows)

    df_final = pd.DataFrame([
        {"epoch": "FINAL_VAL", **final_val},
        {"epoch": "FINAL_TEST", **final_test},
    ])

    df_out = pd.concat([df_curve, df_final], ignore_index=True)
    df_out.to_csv(metrics_out, index=False)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "latent_dim": int(args.latent_dim),
        "enc_hidden": list(args.enc_hidden),
        "dropout": float(args.dropout),
        "tau": float(args.tau),
        "use_mapper": bool(args.use_mapper),
        "lambda_map": float(args.lambda_map),
        "standardize_inputs": bool(args.standardize_inputs),
        "rna_dim": int(Y.shape[1]),
        "prot_dim": int(Q.shape[1]),
        "protein_names": prot_names,
    }
    if args.standardize_inputs:
        ckpt["rna_mean"] = ymu.astype(np.float32)
        ckpt["rna_std"] = ysd.astype(np.float32)
        ckpt["prot_mean"] = qmu.astype(np.float32)
        ckpt["prot_std"] = qsd.astype(np.float32)

    torch.save(ckpt, model_out)

    print(f"[OK] Wrote model:   {model_out}")
    print(f"[OK] Wrote metrics: {metrics_out}")
    print(f"[OK] Wrote z:       {z_out}  (shape={z_all.shape})")
    print(f"[OK] Wrote u:       {u_out}  (shape={u_all.shape})")
    print(f"[OK] Wrote index:   {pair_out}")
    print("[OK] Step 08 complete.")
    print(f"  FINAL VAL:  top1={final_val['top1']:.3f} top5={final_val['top5']:.3f}")
    print(f"  FINAL TEST: top1={final_test['top1']:.3f} top5={final_test['top5']:.3f}")


if __name__ == "__main__":
    main()
