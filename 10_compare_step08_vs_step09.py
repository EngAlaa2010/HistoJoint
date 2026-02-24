#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path("/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
OUT = BASE / "outputs"

# Load Step 08 (true joint latent)
z_true = np.load(OUT / "z_rna_latent.npy")
u_true = np.load(OUT / "u_prot_latent.npy")
joint_latent = 0.5 * (z_true + u_true)

# Load Step 09 (predicted latent)
split_df = pd.read_csv(OUT / "morph_to_joint_split.csv")
z_tr = np.load(OUT / "z_pred_train.npy")
z_va = np.load(OUT / "z_pred_val.npy")
z_te = np.load(OUT / "z_pred_test.npy")

# Reassemble predicted latent
m_pred = np.zeros_like(joint_latent)
m_pred[split_df["split"] == "train"] = z_tr
m_pred[split_df["split"] == "val"] = z_va
m_pred[split_df["split"] == "test"] = z_te

# UMAP embedding (fit on true manifold)
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.3,
    metric="cosine",
    random_state=42,
)
emb_true = reducer.fit_transform(joint_latent)
emb_pred = reducer.transform(m_pred)

# Quantitative similarity
cos_sim = np.mean(np.diag(cosine_similarity(joint_latent, m_pred)))

# Plot
fig = plt.figure(figsize=(14, 6))

# Panel A — True manifold
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(emb_true[:, 0], emb_true[:, 1], s=6, alpha=0.8)
ax1.set_title("Step 08: True RNA–Protein Joint Latent")
ax1.set_xlabel("UMAP1")
ax1.set_ylabel("UMAP2")

# Panel B — Morphology projection
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(emb_pred[:, 0], emb_pred[:, 1], s=6, alpha=0.8)
ax2.set_title("Step 09: Morphology → Joint Projection")
ax2.set_xlabel("UMAP1")
ax2.set_ylabel("UMAP2")

fig.suptitle(
    f"Comparison of True vs Morphology-Derived Molecular Manifolds\n"
    f"Mean cosine similarity = {cos_sim:.3f}",
    fontsize=14
)

plt.tight_layout()
out_png = OUT / "figure3_step08_vs_step09.png"
out_pdf = OUT / "figure3_step08_vs_step09.pdf"
plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf)
plt.close()

print(f"[OK] Wrote: {out_png}")
print(f"[OK] Wrote: {out_pdf}")
print(f"Mean latent cosine similarity: {cos_sim:.3f}")