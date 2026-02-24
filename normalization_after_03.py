import numpy as np
from pathlib import Path

def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)

# --- Paths ---
base = Path("/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
in_path = base / "outputs_hires" / "embeddings_concat.npy"
out_path = base / "outputs_hires" / "embeddings_concat_norm.npy"

# --- Load ---
E = np.load(in_path)   # (N, 512)
print("Loaded:", E.shape)

# --- Split ---
e_local = E[:, :256]
e_context = E[:, 256:]

# --- Normalize separately ---
e_local_norm = l2_normalize(e_local)
e_context_norm = l2_normalize(e_context)

# --- Re-concatenate ---
E_norm = np.concatenate([e_local_norm, e_context_norm], axis=1)

# --- Save ---
np.save(out_path, E_norm)

print("Saved:", out_path)
print("Final shape:", E_norm.shape)
