#!/usr/bin/env python3
"""
08c_joint_summary_figure.py

Creates ONE combined summary image:
Joint RNA–Protein latent UMAP with multiple biological overlays.

Outputs:
  outputs/joint_summary_figure.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import umap


def compute_umap(X, seed=0):
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(X)


def plot_panel(ax, umap_xy, values, title):
    sca = ax.scatter(
        umap_xy[:, 0],
        umap_xy[:, 1],
        c=values,
        s=8,
        cmap="viridis"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return sca


def main():

    base = Path("/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    outdir = base / "outputs"

    # ---- load latent space ----
    z = np.load(outdir / "z_rna_latent.npy")
    u = np.load(outdir / "u_prot_latent.npy")
    idx = pd.read_csv(outdir / "joint_pair_index.csv")

    barcodes = idx["barcode"].astype(str).values

    # UMAP on joint space
    X_all = np.vstack([z, u])
    umap_all = compute_umap(X_all)
    umap_z = umap_all[:len(barcodes)]

    # ---- load protein ----
    prot = sc.read_h5ad(outdir / "adata_prot_processed.h5ad")
    prot_df = pd.DataFrame(
        prot.X.toarray() if hasattr(prot.X, "toarray") else prot.X,
        index=prot.obs_names.astype(str),
        columns=prot.var_names.astype(str),
    ).reindex(barcodes)

    # ---- load RNA ----
    rna_scores = np.load(outdir / "rna_program_scores.npy")
    rna_idx = pd.read_csv(outdir / "rna_program_index.csv")
    rna_df = pd.DataFrame(rna_scores, index=rna_idx["barcode"].astype(str))
    rna_df = rna_df.reindex(barcodes)

    # ---- spatial ----
    emb_idx = pd.read_csv(outdir / "embeddings_index.csv")
    coords = emb_idx.set_index("barcode")[["x_pixel", "y_pixel"]].reindex(barcodes)

    # ---- create figure ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    # Panel 1: plain structure
    axes[0].scatter(umap_z[:, 0], umap_z[:, 1], s=5)
    axes[0].set_title("Joint Latent Structure", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel 2: KRT5-1
    plot_panel(axes[1], umap_z, prot_df["KRT5-1"], "KRT5-1")

    # Panel 3: EPCAM-1
    plot_panel(axes[2], umap_z, prot_df["EPCAM-1"], "EPCAM-1")

    # Panel 4: RNA Program 14
    plot_panel(axes[3], umap_z, rna_df.iloc[:, 14], "RNA Program 14")

    # Panel 5: Spatial X
    plot_panel(axes[4], umap_z, coords["x_pixel"], "Spatial X")

    # Panel 6: Spatial Y
    plot_panel(axes[5], umap_z, coords["y_pixel"], "Spatial Y")

    plt.suptitle("Joint RNA–Protein Latent Space Captures Biology and Spatial Structure",
                 fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    out_path = outdir / "joint_summary_figure.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Wrote combined figure: {out_path}")


if __name__ == "__main__":
    main()
