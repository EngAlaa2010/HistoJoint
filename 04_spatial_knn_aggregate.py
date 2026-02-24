#!/usr/bin/env python3
"""
04_spatial_knn_aggregate.py

Step 04: Spatial kNN aggregation of morphology embeddings.

Input:
  - outputs/embeddings_concat_norm.npy   (N x D)
  - outputs/patch_index.csv              (barcode, x_pixel, y_pixel, in_tissue)

Output:
  - outputs/embeddings_spatial.npy       (N x D)
  - outputs/knn_index.csv                (optional, neighbor graph)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ---------------- logging ----------------

def setup_logger(v: int):
    level = logging.WARNING
    if v == 1:
        level = logging.INFO
    elif v >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------- utils ----------------

def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


# ---------------- main ----------------

def main():
    p = argparse.ArgumentParser(description="Step 04: Spatial kNN aggregation")
    p.add_argument("--base", type=str, default="/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    p.add_argument("--k", type=int, default=6, help="Number of spatial neighbors")
    p.add_argument("--only-in-tissue", action="store_true")
    p.add_argument("--save-knn", action="store_true")
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args()

    setup_logger(args.verbose)

    base = Path(args.base)
    emb_path = base / "outputs" / "embeddings_concat_norm.npy"
    idx_path = base / "outputs" / "embeddings_index.csv"

    if not emb_path.exists() or not idx_path.exists():
        raise FileNotFoundError("Missing embeddings or index file.")

    E = np.load(emb_path)          # (N, D)
    df = pd.read_csv(idx_path)

    if args.only_in_tissue and "in_tissue" in df.columns:
        keep = df["in_tissue"].astype(int) == 1
        df = df.loc[keep].reset_index(drop=True)
        E = E[keep.values]
        logging.info("Filtered to in_tissue==1: N=%d", len(df))

    coords = df[["x_pixel", "y_pixel"]].values.astype(np.float32)
    N, D = E.shape

    logging.info("Building kNN graph (k=%d) for N=%d spots", args.k, N)

    knn = NearestNeighbors(n_neighbors=args.k + 1, metric="euclidean")
    knn.fit(coords)
    dists, nbrs = knn.kneighbors(coords)

    # Drop self (first neighbor)
    nbrs = nbrs[:, 1:]

    M = np.zeros_like(E, dtype=np.float32)

    for i in range(N):
        neigh_idx = nbrs[i]
        M[i] = E[[i, *neigh_idx]].mean(axis=0)

    # Normalize again
    M = l2_normalize(M)

    out_emb = base / "outputs" / "embeddings_spatial.npy"
    np.save(out_emb, M)
    print(f"[OK] Wrote spatial embeddings: {out_emb}")

    if args.save_knn:
        rows = []
        for i in range(N):
            for j in nbrs[i]:
                rows.append({
                    "row_id": i,
                    "neighbor_id": int(j),
                })
        knn_df = pd.DataFrame(rows)
        out_knn = base / "outputs" / "knn_index.csv"
        knn_df.to_csv(out_knn, index=False)
        print(f"[OK] Wrote kNN index: {out_knn}")

    print("[OK] Step 04 complete.")
    print(f"  Output shape: {M.shape}")


if __name__ == "__main__":
    main()
