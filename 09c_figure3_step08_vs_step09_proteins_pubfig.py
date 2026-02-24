#!/usr/bin/env python3
"""
Figure 3 (FINAL, publication-style):
A) Step 08 manifold (true joint latent) colored by TRUE protein
B) Step 09 manifold (morph->latent) colored by TRUE protein
C) Step 09 manifold colored by PRED protein (ridge trained on train split)
D) Error = PRED - TRUE (diverging colormap, centered at 0)

Adds per-protein metrics (computed on TEST split):
- Pearson r (true vs pred)
- RMSE
- R^2

Top-6 proteins come from outputs/prot_feature_names.csv (Step 07).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import scanpy as sc

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from matplotlib.colors import TwoSlopeNorm


BASE = Path("/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
OUT = BASE / "outputs"


# ---------- helpers ----------

def l2norm(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

def safe_pearson(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(pearsonr(a, b)[0])

def fit_umap(X, seed=42):
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.30,
        metric="cosine",
        random_state=seed
    )
    return reducer.fit_transform(X)

def reconstruct_full_pred_latent(split_csv: Path,
                                 pred_train_path: Path,
                                 pred_val_path: Path,
                                 pred_test_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Rebuild full predicted latent (N,L) in the SAME row order as morph_to_joint_split.csv.
    Works because Step09 saved z_pred_{split}.npy in the preserved mask order.
    """
    idx = pd.read_csv(split_csv)
    tr_mask = (idx["split"].values == "train")
    va_mask = (idx["split"].values == "val")
    te_mask = (idx["split"].values == "test")

    m_tr = np.load(pred_train_path).astype(np.float32)
    m_va = np.load(pred_val_path).astype(np.float32)
    m_te = np.load(pred_test_path).astype(np.float32)

    N = len(idx)
    L = m_tr.shape[1]
    m_all = np.zeros((N, L), dtype=np.float32)

    m_all[tr_mask] = m_tr
    m_all[va_mask] = m_va
    m_all[te_mask] = m_te

    return m_all, idx


# ---------- load proteins (TRUE) ----------

prot_h5ad = OUT / "adata_prot_processed.h5ad"
prot = sc.read_h5ad(prot_h5ad)

feat_csv = OUT / "prot_feature_names.csv"
feat_names = pd.read_csv(feat_csv).iloc[:, 0].astype(str).tolist()
top6 = feat_names[:6]
print("Top-6 proteins:", top6)

Q_full = prot.X
if not isinstance(Q_full, np.ndarray):
    Q_full = Q_full.toarray()
Q_full = Q_full.astype(np.float32)

prot_names = list(prot.var_names)
prot_obs_barcodes = prot.obs_names.astype(str).tolist()

k_idx = [prot_names.index(p) for p in top6]
Q6 = Q_full[:, k_idx]  # (N_prot, 6)

prot_df = pd.DataFrame({"barcode": prot_obs_barcodes, "prot_row": np.arange(len(prot_obs_barcodes))})


# ---------- Step 08: TRUE joint latent manifold coords (use u_prot_latent) ----------

pair_csv = OUT / "joint_pair_index.csv"
pair = pd.read_csv(pair_csv).copy()
pair["pair_row"] = np.arange(len(pair))

u_lat = np.load(OUT / "u_prot_latent.npy").astype(np.float32)
u_lat = l2norm(u_lat)

step08 = pair.merge(prot_df, on="barcode", how="inner")
if len(step08) == 0:
    raise RuntimeError("No overlap between joint_pair_index.csv and protein barcodes.")

U08 = u_lat[step08["pair_row"].values.astype(int)]
TRUE6_08 = Q6[step08["prot_row"].values.astype(int)]


# ---------- Step 09: predicted latent manifold coords ----------

split09_csv = OUT / "morph_to_joint_split.csv"
pred_train = OUT / "z_pred_train.npy"
pred_val   = OUT / "z_pred_val.npy"
pred_test  = OUT / "z_pred_test.npy"

m_all, idx09 = reconstruct_full_pred_latent(split09_csv, pred_train, pred_val, pred_test)
m_all = l2norm(m_all)

idx09 = idx09.copy()
idx09["row09"] = np.arange(len(idx09))

step09 = idx09.merge(prot_df, on="barcode", how="inner")
if len(step09) == 0:
    raise RuntimeError("No overlap between morph_to_joint_split.csv and protein barcodes.")

M09 = m_all[step09["row09"].values.astype(int)]
TRUE6_09 = Q6[step09["prot_row"].values.astype(int)]
split09 = step09["split"].values.astype(str)


# ---------- Ridge head: train on TRAIN split only, predict proteins from Step09 latent ----------

train_mask = (split09 == "train")
test_mask  = (split09 == "test")

X_train = M09[train_mask]
Y_train = TRUE6_09[train_mask]

ridge = Ridge(alpha=1.0, random_state=0)
ridge.fit(X_train, Y_train)

PRED6_09 = ridge.predict(M09).astype(np.float32)
ERR6_09 = (PRED6_09 - TRUE6_09).astype(np.float32)


# ---------- Compute TEST metrics per protein ----------

metrics = []
for j, pname in enumerate(top6):
    yt = TRUE6_09[test_mask, j]
    yp = PRED6_09[test_mask, j]
    r = safe_pearson(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    metrics.append({"protein": pname, "pearson_r": r, "rmse": rmse, "r2": r2})

met_df = pd.DataFrame(metrics)
met_out = OUT / "Figure3_step09_test_metrics_top6.csv"
met_df.to_csv(met_out, index=False)
print("[OK] Wrote metrics table:", met_out)


# ---------- UMAP embeddings ----------

UMAP08 = fit_umap(U08, seed=42)
UMAP09 = fit_umap(M09, seed=42)


# ---------- Plot: 6 rows x 4 cols ----------

n_rows = 6
n_cols = 4
fig_w = 17
fig_h = 3.1 * n_rows
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=200)

col_titles = [
    "A  Step 08: TRUE joint manifold\ncolored by TRUE protein",
    "B  Step 09: Morph→latent manifold\ncolored by TRUE protein",
    "C  Step 09: Morph→latent manifold\ncolored by PRED protein (ridge head)",
    "D  Step 09: Error map\n(PRED − TRUE)"
]
for c in range(n_cols):
    axes[0, c].set_title(col_titles[c], fontsize=11, pad=10)

for r, pname in enumerate(top6):
    # robust range for TRUE/PRED (same scale for B and C)
    t_true = TRUE6_09[:, r]
    t_pred = PRED6_09[:, r]
    t_err  = ERR6_09[:, r]

    vmin = float(np.percentile(t_true, 2))
    vmax = float(np.percentile(t_true, 98))

    # metrics for this protein (TEST)
    rowm = met_df.loc[met_df["protein"] == pname].iloc[0]
    r_txt = f"r={rowm['pearson_r']:.2f}  R²={rowm['r2']:.2f}  RMSE={rowm['rmse']:.2f}"
    ylab = f"{pname}\n({r_txt})"

    # A: Step08 TRUE manifold colored by TRUE protein
    ax = axes[r, 0]
    sc0 = ax.scatter(UMAP08[:, 0], UMAP08[:, 1], c=TRUE6_08[:, r], s=6, linewidths=0,
                     vmin=vmin, vmax=vmax)
    ax.set_ylabel(ylab, fontsize=9)
    cb0 = fig.colorbar(sc0, ax=ax, fraction=0.046, pad=0.02)
    cb0.ax.tick_params(labelsize=8)

    # B: Step09 manifold colored by TRUE protein
    ax = axes[r, 1]
    sc1 = ax.scatter(UMAP09[:, 0], UMAP09[:, 1], c=t_true, s=6, linewidths=0,
                     vmin=vmin, vmax=vmax)
    cb1 = fig.colorbar(sc1, ax=ax, fraction=0.046, pad=0.02)
    cb1.ax.tick_params(labelsize=8)

    # C: Step09 manifold colored by PRED protein
    ax = axes[r, 2]
    sc2 = ax.scatter(UMAP09[:, 0], UMAP09[:, 1], c=t_pred, s=6, linewidths=0,
                     vmin=vmin, vmax=vmax)
    cb2 = fig.colorbar(sc2, ax=ax, fraction=0.046, pad=0.02)
    cb2.ax.tick_params(labelsize=8)

    # D: Error map (diverging centered at 0)
    ax = axes[r, 3]
    emax = float(np.percentile(np.abs(t_err), 98))
    norm = TwoSlopeNorm(vmin=-emax, vcenter=0.0, vmax=emax)
    sc3 = ax.scatter(UMAP09[:, 0], UMAP09[:, 1], c=t_err, s=6, linewidths=0,
                     cmap="RdBu_r", norm=norm)
    cb3 = fig.colorbar(sc3, ax=ax, fraction=0.046, pad=0.02)
    cb3.ax.tick_params(labelsize=8)

# Clean publication look
for axrow in axes:
    for ax in axrow:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel(ax.get_ylabel())  # keep only left labels

plt.suptitle(
    "Figure 3 | Step 08 vs Step 09 manifolds with TRUE / PRED protein and error (top-6 proteins)\n"
    "Per-protein metrics computed on TEST split (r, R², RMSE).",
    fontsize=14, y=0.995
)
plt.tight_layout(rect=[0, 0, 1, 0.985])

out_png = OUT / "Figure3_step08_vs_step09_true_pred_error_top6_with_metrics.png"
out_pdf = OUT / "Figure3_step08_vs_step09_true_pred_error_top6_with_metrics.pdf"
plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf)
plt.close()

print("[OK] Wrote:", out_png)
print("[OK] Wrote:", out_pdf)
print("[DONE] This is the final figure (Figure 3).")