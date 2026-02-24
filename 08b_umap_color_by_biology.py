#!/usr/bin/env python3
"""
08b_umap_color_by_biology.py

Visualize biological meaning of the joint RNA–Protein latent space:
- Color UMAP by KRT5-1 protein
- Color UMAP by EPCAM-1 protein
- Color UMAP by RNA program 14 score
- Color UMAP by spatial coordinates (x_pixel, y_pixel)

Requires outputs from Steps 03/05/07/08:
  outputs/z_rna_latent.npy
  outputs/u_prot_latent.npy
  outputs/joint_pair_index.csv          (barcode list)
  outputs/adata_prot_processed.h5ad     (protein per barcode)
  outputs/rna_program_scores.npy        (RNA programs)
  outputs/rna_program_index.csv         (barcodes for RNA programs)
  outputs/embeddings_index.csv          (x_pixel,y_pixel for barcodes)

Outputs:
  outputs/joint_umap_color_KRT5-1.png
  outputs/joint_umap_color_EPCAM-1.png
  outputs/joint_umap_color_program14.png
  outputs/joint_umap_color_spatialX.png
  outputs/joint_umap_color_spatialY.png
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def compute_umap(X, n_neighbors=30, min_dist=0.3, seed=0):
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(X)

def zscore(v):
    v = np.asarray(v, dtype=np.float32)
    m = np.nanmean(v)
    s = np.nanstd(v)
    if s < 1e-8:
        return v * 0.0
    return (v - m) / s

def plot_umap(umap_xy, values, title, outpath, cmap=None):
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(umap_xy[:, 0], umap_xy[:, 1], c=values, s=10, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(sc, shrink=0.8)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()

def main():
    args = parse_args()
    base = Path(args.base)
    outdir = base / "outputs"

    # ---- load latents + barcode index ----
    z = np.load(outdir / "z_rna_latent.npy")  # (N, d)
    u = np.load(outdir / "u_prot_latent.npy")  # (N, d)
    idx = pd.read_csv(outdir / "joint_pair_index.csv")  # must contain "barcode"
    barcodes = idx["barcode"].astype(str).values

    assert z.shape[0] == u.shape[0] == len(barcodes), "Mismatch between z/u and joint_pair_index rows."

    # ---- compute UMAP on combined (z and u together), then take z positions only ----
    # We compute on stacked to preserve joint geometry.
    X_all = np.vstack([z, u])  # (2N, d)
    umap_all = compute_umap(X_all, n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed)
    umap_z = umap_all[: len(barcodes)]  # RNA points
    # umap_u = umap_all[len(barcodes):] # protein points (optional)

    # ---- load protein values ----
    prot = sc.read_h5ad(outdir / "adata_prot_processed.h5ad")
    prot_df = pd.DataFrame(
        prot.X.toarray() if hasattr(prot.X, "toarray") else np.asarray(prot.X),
        index=prot.obs_names.astype(str),
        columns=prot.var_names.astype(str),
    )

    # align protein rows to joint barcodes
    prot_df = prot_df.reindex(barcodes)
    if prot_df.isna().any().any():
        missing = prot_df.index[prot_df.isna().any(1)]
        print("[WARN] Missing protein rows for some barcodes:", len(missing))

    # ---- load RNA program scores ----
    rna_scores = np.load(outdir / "rna_program_scores.npy")  # (N, 32)
    rna_idx = pd.read_csv(outdir / "rna_program_index.csv")
    rna_map = pd.DataFrame(rna_scores, index=rna_idx["barcode"].astype(str).values)
    rna_map = rna_map.reindex(barcodes)

    # ---- load spatial coords ----
    emb_idx = pd.read_csv(outdir / "embeddings_index.csv")
    emb_idx["barcode"] = emb_idx["barcode"].astype(str)
    coords = emb_idx.set_index("barcode")[["x_pixel", "y_pixel"]].reindex(barcodes)

    # ---- make plots ----
    # KRT5-1
    if "KRT5-1" in prot_df.columns:
        plot_umap(
            umap_z, prot_df["KRT5-1"].values,
            "Joint latent UMAP (RNA points) colored by protein KRT5-1",
            outdir / "joint_umap_color_KRT5-1.png",
        )
        print("[OK] Wrote joint_umap_color_KRT5-1.png")
    else:
        print("[WARN] KRT5-1 not found in protein markers.")

    # EPCAM-1
    if "EPCAM-1" in prot_df.columns:
        plot_umap(
            umap_z, prot_df["EPCAM-1"].values,
            "Joint latent UMAP (RNA points) colored by protein EPCAM-1",
            outdir / "joint_umap_color_EPCAM-1.png",
        )
        print("[OK] Wrote joint_umap_color_EPCAM-1.png")
    else:
        print("[WARN] EPCAM-1 not found in protein markers.")

    # RNA program 14
    prog14 = rna_map.iloc[:, 14].values  # program index 14
    plot_umap(
        umap_z, prog14,
        "Joint latent UMAP (RNA points) colored by RNA program 14 score",
        outdir / "joint_umap_color_program14.png",
    )
    print("[OK] Wrote joint_umap_color_program14.png")

    # Spatial coords X/Y
    plot_umap(
        umap_z, coords["x_pixel"].values,
        "Joint latent UMAP (RNA points) colored by spatial X (x_pixel)",
        outdir / "joint_umap_color_spatialX.png",
    )
    plot_umap(
        umap_z, coords["y_pixel"].values,
        "Joint latent UMAP (RNA points) colored by spatial Y (y_pixel)",
        outdir / "joint_umap_color_spatialY.png",
    )
    print("[OK] Wrote joint_umap_color_spatialX.png and joint_umap_color_spatialY.png")

    print("\nDone.")

if __name__ == "__main__":
    main()
