#!/usr/bin/env python3

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
OUT = BASE / "outputs"

# Load latent spaces
z = np.load(OUT / "z_rna_latent.npy")   # RNA latent
u = np.load(OUT / "u_prot_latent.npy")  # Protein latent

print("Shapes:", z.shape, u.shape)

# Stack them for joint UMAP
X = np.vstack([z, u])

labels = np.array(["RNA"] * len(z) + ["Protein"] * len(u))

# Fit UMAP
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.3,
    metric="cosine",
    random_state=42,
)

embedding = reducer.fit_transform(X)

# Split back
emb_z = embedding[:len(z)]
emb_u = embedding[len(z):]

# Plot
plt.figure(figsize=(8, 8))
plt.scatter(emb_z[:, 0], emb_z[:, 1], s=10, alpha=0.6, label="RNA")
plt.scatter(emb_u[:, 0], emb_u[:, 1], s=10, alpha=0.6, label="Protein")
plt.legend()
plt.title("Joint RNA–Protein Latent Space (UMAP)")
plt.tight_layout()

out_png = OUT / "joint_umap.png"
plt.savefig(out_png, dpi=300)
plt.close()

print(f"[OK] Saved UMAP: {out_png}")
