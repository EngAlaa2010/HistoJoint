#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scanpy as sc
import umap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge

BASE = Path("/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
OUT  = BASE / "outputs"

# ---- inputs from Step 09 ----
split_csv = OUT / "morph_to_joint_split.csv"
pred_tr = np.load(OUT / "z_pred_train.npy")
pred_va = np.load(OUT / "z_pred_val.npy")
pred_te = np.load(OUT / "z_pred_test.npy")

# ---- true protein matrix ----
prot = sc.read_h5ad(OUT / "adata_prot_processed.h5ad")

# which proteins were used (from Step 07 top-6 run)
prot_list = pd.read_csv(OUT / "prot_feature_names.csv")["protein"].astype(str).tolist()

# load split table
idx = pd.read_csv(split_csv)
barcodes = idx["barcode"].astype(str).values
splits = idx["split"].values

# ---- align protein rows to barcodes ----
# prot.obs_names should be barcodes
prot_obs = pd.Index(prot.obs_names.astype(str))
row_map = prot_obs.get_indexer(barcodes)
if (row_map < 0).any():
    missing = (row_map < 0).sum()
    raise RuntimeError(f"{missing} barcodes in morph_to_joint_split.csv not found in protein AnnData.")

# subset protein targets in same order
prot_names_all = list(prot.var_names.astype(str))
prot_cols = [prot_names_all.index(p) for p in prot_list]
Y_true = prot.X
if not isinstance(Y_true, np.ndarray):
    Y_true = Y_true.toarray()
Y_true = Y_true[row_map, :][:, prot_cols].astype(np.float32)  # (N, K)

# ---- reconstruct FULL predicted latent in idx row order ----
N = len(idx)
L = pred_tr.shape[1]
m_pred = np.zeros((N, L), dtype=np.float32)

tr_idx = np.where(splits == "train")[0]
va_idx = np.where(splits == "val")[0]
te_idx = np.where(splits == "test")[0]

m_pred[tr_idx] = pred_tr
m_pred[va_idx] = pred_va
m_pred[te_idx] = pred_te

# ---- train ridge regression latent -> proteins on TRAIN only ----
X_train = m_pred[tr_idx]
Y_train = Y_true[tr_idx]

model = Ridge(alpha=1.0)
model.fit(X_train, Y_train)

Y_pred = model.predict(m_pred).astype(np.float32)  # (N,K)

# ---- build UMAP on predicted latent (Step 09 space) ----
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.3,
    metric="cosine",
    random_state=42
)
um = reducer.fit_transform(m_pred)  # (N,2)

# ---- plotting ----
K = len(prot_list)
ncols = 3
nrows = K

fig = plt.figure(figsize=(4*ncols, 3*nrows))

# Row 0..K-1: each protein has 3 panels:
# TRUE | PRED | ERROR
for i, pname in enumerate(prot_list):
    yt = Y_true[:, i]
    yp = Y_pred[:, i]
    err = yp - yt

    # TRUE
    ax1 = plt.subplot(nrows, ncols, i*ncols + 1)
    sc1 = ax1.scatter(um[:,0], um[:,1], c=yt, s=8, alpha=0.8)
    ax1.set_title(f"TRUE {pname}")
    ax1.set_xlabel("UMAP1"); ax1.set_ylabel("UMAP2")
    plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)

    # PRED
    ax2 = plt.subplot(nrows, ncols, i*ncols + 2)
    sc2 = ax2.scatter(um[:,0], um[:,1], c=yp, s=8, alpha=0.8)
    ax2.set_title(f"PRED {pname} (from Step09 latent)")
    ax2.set_xlabel("UMAP1"); ax2.set_ylabel("UMAP2")
    plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)

    # ERROR
    ax3 = plt.subplot(nrows, ncols, i*ncols + 3)
    sc3 = ax3.scatter(um[:,0], um[:,1], c=err, s=8, alpha=0.8)
    ax3.set_title(f"ERROR (PRED-TRUE) {pname}")
    ax3.set_xlabel("UMAP1"); ax3.set_ylabel("UMAP2")
    plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)

plt.suptitle("Step 09: Morphology→Latent UMAP colored by TRUE vs PRED protein (Ridge head)", y=0.995, fontsize=14)
plt.tight_layout(rect=[0,0,1,0.98])

out_png = OUT / "step09_true_vs_pred_proteins_pubfig.png"
plt.savefig(out_png, dpi=300)
plt.close()

print(f"[OK] Saved: {out_png}")
print("Proteins:", prot_list)