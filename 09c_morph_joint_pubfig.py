#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import umap
import scanpy as sc

def to_dense_col(X, col_idx: int):
    if hasattr(X, "tocsc"):  # sparse
        return np.asarray(X[:, col_idx].toarray()).ravel()
    return np.asarray(X[:, col_idx]).ravel()

def main():
    BASE = Path("/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    OUT = BASE / "outputs"

    split_csv = OUT / "morph_to_joint_split.csv"
    pred_tr = OUT / "z_pred_train.npy"
    pred_va = OUT / "z_pred_val.npy"
    pred_te = OUT / "z_pred_test.npy"

    prot_h5ad = OUT / "adata_prot_processed.h5ad"
    prot_list_csv = OUT / "prot_feature_names.csv"  # your top-6 proteins (one per row)

    assert split_csv.exists(), f"Missing: {split_csv}"
    assert pred_tr.exists() and pred_va.exists() and pred_te.exists(), "Missing Step 09 prediction .npy files"
    assert prot_h5ad.exists(), f"Missing: {prot_h5ad}"

    df = pd.read_csv(split_csv)
    if "barcode" not in df.columns or "split" not in df.columns:
        raise ValueError("morph_to_joint_split.csv must contain columns: barcode, split")

    # Load predicted latents per split
    M_tr = np.load(pred_tr).astype(np.float32)
    M_va = np.load(pred_va).astype(np.float32)
    M_te = np.load(pred_te).astype(np.float32)

    tr_mask = (df["split"].values == "train")
    va_mask = (df["split"].values == "val")
    te_mask = (df["split"].values == "test")

    if tr_mask.sum() != M_tr.shape[0] or va_mask.sum() != M_va.shape[0] or te_mask.sum() != M_te.shape[0]:
        raise RuntimeError(
            "Split counts do not match prediction shapes.\n"
            f"CSV train/val/test = {tr_mask.sum()}/{va_mask.sum()}/{te_mask.sum()}\n"
            f"NPY train/val/test = {M_tr.shape[0]}/{M_va.shape[0]}/{M_te.shape[0]}"
        )

    # Reconstruct in df row order
    m_all = np.zeros((len(df), M_tr.shape[1]), dtype=np.float32)
    m_all[tr_mask] = M_tr
    m_all[va_mask] = M_va
    m_all[te_mask] = M_te

    print("Pred latent shape:", m_all.shape)

    # UMAP on predicted latent
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="cosine",
        random_state=42,
    )
    um = reducer.fit_transform(m_all)  # (N,2)

    # Load TRUE proteins and align by barcode
    ad = sc.read_h5ad(prot_h5ad)
    prot_names_all = list(ad.var_names)

    if prot_list_csv.exists():
        proteins = pd.read_csv(prot_list_csv).iloc[:, 0].astype(str).tolist()
    else:
        # fallback: first 6 proteins
        proteins = prot_names_all[:6]

    # Align df rows to protein rows
    prot_df = pd.DataFrame({"barcode": ad.obs_names.astype(str)})
    prot_df["prot_row"] = np.arange(len(prot_df))

    merged = df.reset_index().rename(columns={"index": "df_row"}).merge(prot_df, on="barcode", how="inner")
    if len(merged) == 0:
        raise RuntimeError("No overlapping barcodes between morph_to_joint_split.csv and adata_prot_processed.h5ad")

    df_rows = merged["df_row"].astype(int).values
    prot_rows = merged["prot_row"].astype(int).values

    um_sub = um[df_rows]

    # Prepare values per protein
    prot_vals = {}
    for p in proteins:
        if p not in prot_names_all:
            print(f"[SKIP] Protein not found: {p}")
            continue
        j = prot_names_all.index(p)
        prot_vals[p] = to_dense_col(ad.X, j)[prot_rows]

    # ---- publication figure layout ----
    # 1 (split) + len(proteins) panels
    panels = 1 + len(prot_vals)
    ncols = 4
    nrows = int(np.ceil(panels / ncols))

    fig = plt.figure(figsize=(4.2 * ncols, 4.0 * nrows))

    # Panel 1: split
    ax0 = fig.add_subplot(nrows, ncols, 1)
    for sp in ["train", "val", "test"]:
        mask = (merged["split"].values == sp)
        ax0.scatter(um_sub[mask, 0], um_sub[mask, 1], s=8, alpha=0.75, label=sp)
    ax0.set_title("UMAP (pred latent)\ncolored by split")
    ax0.set_xlabel("UMAP1")
    ax0.set_ylabel("UMAP2")
    ax0.legend(markerscale=2, frameon=False, loc="best")

    # Protein panels
    i = 2
    for p, vals in prot_vals.items():
        ax = fig.add_subplot(nrows, ncols, i)
        sca = ax.scatter(um_sub[:, 0], um_sub[:, 1], c=vals, s=8, alpha=0.85)
        ax.set_title(f"TRUE {p}")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(p)
        i += 1

    # Clean up empty panels (if any)
    for j in range(i, nrows * ncols + 1):
        ax = fig.add_subplot(nrows, ncols, j)
        ax.axis("off")

    fig.suptitle("Morphology → Joint latent (Step 09) | UMAP colored by TRUE protein expression", y=1.02, fontsize=14)
    fig.tight_layout()

    out_png = OUT / "morph_joint_pubfig.png"
    out_pdf = OUT / "morph_joint_pubfig.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote: {out_png}")
    print(f"[OK] Wrote: {out_pdf}")

if __name__ == "__main__":
    main()