#!/usr/bin/env python3
"""
05_rna_programs.py

Step 05: Build RNA programs per spot (targets for RNA head)

Default method: NMF on highly-variable genes (HVGs)

Inputs:
  - outputs/adata_rna_processed.h5ad  (log1p normalized in step 01)
  - spatial/tissue_positions.csv      (optional; for in_tissue filtering)

Outputs:
  - outputs/rna_program_scores.npy      (N x P) spot-level program scores
  - outputs/rna_program_gene_loadings.npy (P x G) gene loadings
  - outputs/rna_program_index.csv       (barcode -> row_id -> in_tissue)
  - outputs/rna_program_genes.csv       (list of genes used)
  - outputs/rna_program_top_genes.csv   (top genes per program for interpretation)

Notes:
  - This creates the RNA targets for later supervised learning from morphology embeddings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import NMF


def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 05: Build RNA programs (NMF on HVGs).")
    p.add_argument("--base", type=str, default="/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    p.add_argument("--adata", type=str, default="", help="Default: base/outputs/adata_rna_processed.h5ad")
    p.add_argument("--positions", type=str, default="", help="Default: base/spatial/tissue_positions.csv (optional)")
    p.add_argument("--only-in-tissue", action="store_true", help="Keep only in_tissue==1 if positions exist.")
    p.add_argument("--n-hvg", type=int, default=3000, help="Number of HVGs to use.")
    p.add_argument("--n-programs", type=int, default=32, help="Number of RNA programs (NMF components).")
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--outdir", type=str, default="", help="Default: base/outputs")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p.parse_args()


def load_positions_optional(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    pos = pd.read_csv(path, header=None, index_col=0)
    pos.columns = ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    return pos


def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)

    base = Path(args.base)
    adata_path = Path(args.adata) if args.adata else (base / "outputs" / "adata_rna_processed.h5ad")
    pos_path = Path(args.positions) if args.positions else (base / "spatial" / "tissue_positions.csv")
    if not pos_path.exists():
        pos_path = None

    outdir = Path(args.outdir) if args.outdir else (base / "outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    out_scores = outdir / "rna_program_scores.npy"
    out_loadings = outdir / "rna_program_gene_loadings.npy"
    out_index = outdir / "rna_program_index.csv"
    out_genes = outdir / "rna_program_genes.csv"
    out_top = outdir / "rna_program_top_genes.csv"

    if not args.overwrite and all(p.exists() for p in [out_scores, out_loadings, out_index, out_genes, out_top]):
        logging.warning("Outputs already exist. Use --overwrite to recompute.")
        print("[SKIP] Found existing Step 05 outputs.")
        return

    if not adata_path.exists():
        raise FileNotFoundError(f"Missing: {adata_path}")

    logging.info("Loading AnnData: %s", adata_path)
    adata = sc.read_h5ad(adata_path)

    # Optional in_tissue filtering
    pos_df = load_positions_optional(pos_path)
    if pos_df is not None:
        adata.obs["in_tissue"] = pos_df["in_tissue"].reindex(adata.obs_names).fillna(0).astype(int).values
        if args.only_in_tissue:
            before = adata.n_obs
            adata = adata[adata.obs["in_tissue"] == 1].copy()
            logging.info("Filtered in_tissue==1: %d -> %d", before, adata.n_obs)
    else:
        adata.obs["in_tissue"] = 1

    if adata.n_obs == 0:
        raise RuntimeError("No spots remain after filtering.")

    # HVGs (works on log1p normalized values)
    logging.info("Selecting HVGs: n_hvg=%d", args.n_hvg)
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, flavor="seurat_v3")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    # Get matrix (dense)
    X = adata_hvg.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = X.astype(np.float32)

    # NMF requires non-negative — log1p is non-negative already
    if X.min() < 0:
        raise ValueError("Found negative values in X. NMF requires non-negative data.")

    logging.info("Running NMF: n_programs=%d max_iter=%d", args.n_programs, args.max_iter)
    nmf = NMF(
        n_components=args.n_programs,
        init="nndsvda",
        random_state=args.random_state,
        max_iter=args.max_iter,
    )

    # W: (N x P) spot scores
    # H: (P x G) gene loadings
    W = nmf.fit_transform(X)
    H = nmf.components_

    # Save outputs
    np.save(out_scores, W.astype(np.float32))
    np.save(out_loadings, H.astype(np.float32))

    # Index mapping
    idx = pd.DataFrame({
        "row_id": np.arange(adata_hvg.n_obs, dtype=int),
        "barcode": adata_hvg.obs_names.values,
        "in_tissue": adata_hvg.obs["in_tissue"].astype(int).values,
    })
    idx.to_csv(out_index, index=False)

    # Save gene list used
    genes = pd.DataFrame({"gene": adata_hvg.var_names.values})
    genes.to_csv(out_genes, index=False)

    # Save top genes per program
    top_rows = []
    for k in range(H.shape[0]):
        order = np.argsort(H[k])[::-1]
        top_genes = adata_hvg.var_names.values[order[:30]]
        top_vals = H[k, order[:30]]
        for rank, (g, v) in enumerate(zip(top_genes, top_vals), start=1):
            top_rows.append({"program": k, "rank": rank, "gene": g, "loading": float(v)})
    pd.DataFrame(top_rows).to_csv(out_top, index=False)

    print(f"[OK] Wrote: {out_scores} (shape={W.shape})")
    print(f"[OK] Wrote: {out_loadings} (shape={H.shape})")
    print(f"[OK] Wrote: {out_index}")
    print(f"[OK] Wrote: {out_genes}")
    print(f"[OK] Wrote: {out_top}")
    print("[OK] Step 05 complete.")


if __name__ == "__main__":
    main()
