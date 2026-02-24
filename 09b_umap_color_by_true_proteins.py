#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import umap
import scanpy as sc

def to_dense_col(X, col_idx: int):
    """Return one column as a dense 1D numpy array, works for sparse/dense."""
    if hasattr(X, "tocsc"):  # sparse
        return np.asarray(X[:, col_idx].toarray()).ravel()
    return np.asarray(X[:, col_idx]).ravel()

def main():
    BASE = Path("/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    OUT = BASE / "outputs"

    # ---- Step 09 outputs ----
    split_csv = OUT / "morph_to_joint_split.csv"
    pred_tr = OUT / "z_pred_train.npy"
    pred_va = OUT / "z_pred_val.npy"
    pred_te = OUT / "z_pred_test.npy"

    # ---- True protein source (Step 07 processed) ----
    prot_h5ad = OUT / "adata_prot_processed.h5ad"

    # Optional: protein list (top-6 you trained in Step 07)
    prot_list_csv = OUT / "prot_feature_names.csv"  # one protein per row

    assert split_csv.exists(), f"Missing: {split_csv}"
    assert pred_tr.exists() and pred_va.exists() and pred_te.exists(), "Missing Step 09 prediction .npy files"
    assert prot_h5ad.exists(), f"Missing: {prot_h5ad}"

    df = pd.read_csv(split_csv)
    if "barcode" not in df.columns or "split" not in df.columns:
        raise ValueError("morph_to_joint_split.csv must contain columns: barcode, split")

    # Load predicted latents
    M_tr = np.load(pred_tr).astype(np.float32)
    M_va = np.load(pred_va).astype(np.float32)
    M_te = np.load(pred_te).astype(np.float32)

    # Reconstruct full predicted matrix in df row order
    m_all = np.zeros((len(df), M_tr.shape[1]), dtype=np.float32)

    tr_mask = (df["split"].values == "train")
    va_mask = (df["split"].values == "val")
    te_mask = (df["split"].values == "test")

    if tr_mask.sum() != M_tr.shape[0] or va_mask.sum() != M_va.shape[0] or te_mask.sum() != M_te.shape[0]:
        raise RuntimeError(
            "Split counts do not match prediction shapes.\n"
            f"CSV train/val/test = {tr_mask.sum()}/{va_mask.sum()}/{te_mask.sum()}\n"
            f"NPY train/val/test = {M_tr.shape[0]}/{M_va.shape[0]}/{M_te.shape[0]}"
        )

    m_all[tr_mask] = M_tr
    m_all[va_mask] = M_va
    m_all[te_mask] = M_te

    print("Pred latent shape:", m_all.shape)

    # ---- Fit UMAP on predicted latents ----
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="cosine",
        random_state=42,
    )
    emb = reducer.fit_transform(m_all)  # (N,2)

    # Save a base UMAP colored by split
    plt.figure(figsize=(7, 7))
    for sp in ["train", "val", "test"]:
        mask = (df["split"].values == sp)
        plt.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.7, label=sp)
    plt.legend(markerscale=2)
    plt.title("Morph→Joint (predicted latent) UMAP colored by split")
    plt.tight_layout()
    out_base = OUT / "morph_joint_umap_by_split.png"
    plt.savefig(out_base, dpi=300)
    plt.close()
    print(f"[OK] Wrote: {out_base}")

    # ---- Load TRUE protein expression and align by barcode ----
    ad = sc.read_h5ad(prot_h5ad)

    # Choose proteins to plot
    if prot_list_csv.exists():
        proteins = pd.read_csv(prot_list_csv).iloc[:, 0].astype(str).tolist()
    else:
        # fallback: plot ALL proteins in adata (can be many)
        proteins = list(ad.var_names)

    # Align indices by barcode
    prot_df = pd.DataFrame({"barcode": ad.obs_names.astype(str)})
    merged = df.merge(prot_df, on="barcode", how="inner")
    if len(merged) == 0:
        raise RuntimeError("No overlapping barcodes between morph_to_joint_split.csv and adata_prot_processed.h5ad")

    # Row mapping: df row -> protein adata row
    df_row = merged.index.values  # index in merged (not df)
    # safer: compute positions explicitly
    df_pos = merged.merge(df.reset_index().rename(columns={"index": "df_pos"}), on="barcode", how="left")["df_pos"].values
    prot_pos = prot_df.reset_index().rename(columns={"index": "prot_pos"}).merge(
        merged[["barcode"]], on="barcode", how="right"
    )["prot_pos"].values

    # (df_pos, prot_pos) are aligned pairs
    df_pos = df_pos.astype(int)
    prot_pos = prot_pos.astype(int)

    # We'll color only the subset of points that have protein measurements (usually all 4169)
    emb_sub = emb[df_pos]

    # ---- Plot one PNG per protein ----
    for p in proteins:
        if p not in ad.var_names:
            print(f"[SKIP] Protein not found: {p}")
            continue
        j = list(ad.var_names).index(p)
        vals = to_dense_col(ad.X, j)[prot_pos]  # aligned values

        plt.figure(figsize=(7, 7))
        sca = plt.scatter(emb_sub[:, 0], emb_sub[:, 1], c=vals, s=10, alpha=0.8)
        plt.colorbar(sca, fraction=0.046, pad=0.04, label=p)
        plt.title(f"Morph→Joint UMAP colored by TRUE {p}")
        plt.tight_layout()

        out_png = OUT / f"morph_joint_umap_color_{p}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[OK] Wrote: {out_png}")

    print("Done.")

if __name__ == "__main__":
    main()
