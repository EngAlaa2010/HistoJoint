#!/usr/bin/env python3
"""
08c_joint_summary_pubfig.py

Publication-quality summary figure:
Joint RNA–Protein latent UMAP with:
(A) clusters (Leiden) + labels
(B) KRT5-1
(C) EPCAM-1
(D) RNA program 14
(E) Spatial X
(F) Spatial Y

Outputs:
  outputs/joint_summary_pubfig.png
  outputs/joint_summary_pubfig.pdf
  outputs/joint_leiden_clusters.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import scanpy as sc
import umap


# ---------------- style (Nature-ish) ----------------
def set_nature_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",   # use "Arial" if available on your HPC
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
    })


def compute_umap(X, seed=0):
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.30,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(X)


def add_panel_label(ax, letter):
    ax.text(
        0.01, 0.99, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold"
    )


def scatter_colored(ax, xy, values, cmap, vmin=None, vmax=None, s=6, alpha=0.9):
    sca = ax.scatter(
        xy[:, 0], xy[:, 1],
        c=values,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        s=s,
        linewidths=0,
        alpha=alpha
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return sca


def main():
    set_nature_style()

    base = Path("/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    outdir = base / "outputs"

    # ---- load latent + barcode index ----
    z = np.load(outdir / "z_rna_latent.npy").astype(np.float32)
    u = np.load(outdir / "u_prot_latent.npy").astype(np.float32)
    idx = pd.read_csv(outdir / "joint_pair_index.csv")
    barcodes = idx["barcode"].astype(str).values

    # ---- compute ONE UMAP (use z only for cleaner plotting) ----
    umap_z = compute_umap(z, seed=0)

    # ---- clustering (Leiden) on z latent space ----
    ad = sc.AnnData(z)
    sc.pp.neighbors(ad, n_neighbors=20, metric="cosine")
    sc.tl.leiden(ad, resolution=0.8, key_added="leiden")

    clusters = ad.obs["leiden"].astype(str).values
    clust_df = pd.DataFrame({"barcode": barcodes, "leiden": clusters})
    clust_df.to_csv(outdir / "joint_leiden_clusters.csv", index=False)

    # cluster centroids for labels
    uniq = sorted(np.unique(clusters), key=lambda x: int(x) if x.isdigit() else x)
    centroids = {}
    for c in uniq:
        m = clusters == c
        centroids[c] = umap_z[m].mean(axis=0)

    # ---- load protein ----
    prot = sc.read_h5ad(outdir / "adata_prot_processed.h5ad")
    prot_mat = prot.X.toarray() if hasattr(prot.X, "toarray") else prot.X
    prot_df = pd.DataFrame(
        prot_mat,
        index=prot.obs_names.astype(str),
        columns=prot.var_names.astype(str),
    ).reindex(barcodes)

    # ---- load RNA programs ----
    rna_scores = np.load(outdir / "rna_program_scores.npy")
    rna_idx = pd.read_csv(outdir / "rna_program_index.csv")
    rna_df = pd.DataFrame(rna_scores, index=rna_idx["barcode"].astype(str)).reindex(barcodes)

    # ---- load spatial coords ----
    emb_idx = pd.read_csv(outdir / "embeddings_index.csv").set_index("barcode")
    coords = emb_idx[["x_pixel", "y_pixel"]].reindex(barcodes)

    # targets
    krt5 = prot_df["KRT5-1"].values.astype(np.float32)
    epcam = prot_df["EPCAM-1"].values.astype(np.float32)
    prog14 = rna_df.iloc[:, 14].values.astype(np.float32)
    xpix = coords["x_pixel"].values.astype(np.float32)
    ypix = coords["y_pixel"].values.astype(np.float32)

    # ---- shared colorbar ranges ----
    # Proteins share one scale:
    prot_vmin = float(np.nanmin([krt5.min(), epcam.min()]))
    prot_vmax = float(np.nanmax([krt5.max(), epcam.max()]))

    # Spatial share one scale is not meaningful because X and Y differ,
    # but user requested shared colorbar; we do shared based on combined min/max.
    sp_vmin = float(np.nanmin([xpix.min(), ypix.min()]))
    sp_vmax = float(np.nanmax([xpix.max(), ypix.max()]))

    # ---- figure layout (2x3) with dedicated colorbar axes ----
    fig = plt.figure(figsize=(10.5, 6.5))
    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        width_ratios=[1, 1, 1, 0.05],
        height_ratios=[1, 1],
        wspace=0.15, hspace=0.18
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    cax_prot = fig.add_subplot(gs[0, 3])

    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])
    cax_bottom = fig.add_subplot(gs[1, 3])

    # --- (A) clusters ---
    # draw clusters as categorical colors
    # simple stable palette
    palette = plt.cm.tab20(np.linspace(0, 1, max(20, len(uniq))))
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(uniq)}
    colors = np.array([color_map[c] for c in clusters])

    axA.scatter(umap_z[:, 0], umap_z[:, 1], s=6, c=colors, alpha=0.85, linewidths=0)
    axA.set_title("Joint latent space (Leiden clusters)")
    axA.set_xticks([]); axA.set_yticks([])
    for sp in axA.spines.values():
        sp.set_visible(False)

    # label clusters
    for c in uniq:
        cx, cy = centroids[c]
        axA.text(cx, cy, c, ha="center", va="center", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    add_panel_label(axA, "A")

    # --- (B) KRT5-1 ---
    scB = scatter_colored(axB, umap_z, krt5, cmap="magma", vmin=prot_vmin, vmax=prot_vmax)
    axB.set_title("KRT5-1 (protein)")
    add_panel_label(axB, "B")

    # --- (C) EPCAM-1 ---
    scC = scatter_colored(axC, umap_z, epcam, cmap="magma", vmin=prot_vmin, vmax=prot_vmax)
    axC.set_title("EPCAM-1 (protein)")
    add_panel_label(axC, "C")

    # shared protein colorbar
    cb1 = fig.colorbar(scC, cax=cax_prot)
    cb1.set_label("Protein expression (log1p)")
    cb1.outline.set_linewidth(0.6)

    # --- (D) RNA program 14 ---
    scD = scatter_colored(axD, umap_z, prog14, cmap="viridis")
    axD.set_title("RNA program 14 (score)")
    add_panel_label(axD, "D")

    # --- (E) spatial X ---
    scE = scatter_colored(axE, umap_z, xpix, cmap="cividis", vmin=sp_vmin, vmax=sp_vmax)
    axE.set_title("Spatial X (pixels)")
    add_panel_label(axE, "E")

    # --- (F) spatial Y ---
    scF = scatter_colored(axF, umap_z, ypix, cmap="cividis", vmin=sp_vmin, vmax=sp_vmax)
    axF.set_title("Spatial Y (pixels)")
    add_panel_label(axF, "F")

    # bottom shared colorbar: use spatial scale (as requested)
    cb2 = fig.colorbar(scF, cax=cax_bottom)
    cb2.set_label("Spatial coordinate (pixels)")
    cb2.outline.set_linewidth(0.6)

    # add an extra small colorbar for RNA program 14 (placed inside D)
    # (keeps the 2 shared bars requirement intact)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset = inset_axes(axD, width="4%", height="55%", loc="lower right", borderpad=0.8)
    cbD = fig.colorbar(scD, cax=inset)
    cbD.set_label("Score", rotation=90, labelpad=6)
    cbD.outline.set_linewidth(0.6)

    # main title (subtle)
    fig.suptitle("Joint RNA–Protein latent space captures biology and spatial structure", y=0.98, fontsize=12)

    # save
    png_path = outdir / "joint_summary_pubfig.png"
    pdf_path = outdir / "joint_summary_pubfig.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote: {png_path}")
    print(f"[OK] Wrote: {pdf_path}")
    print(f"[OK] Wrote clusters: {outdir / 'joint_leiden_clusters.csv'}")


if __name__ == "__main__":
    main()
