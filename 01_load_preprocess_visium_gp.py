#!/usr/bin/env python3
"""
Load 10x Visium CytAssist FFPE Gene+Protein dataset on OSC and do basic preprocessing + spatial alignment check.

Inputs expected in BASE directory:
  - CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
  - CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_spatial.tar.gz
  - CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_isotype_normalization_factors.csv

Outputs:
  - extracts spatial/ folder (if not already extracted)
  - saves optional processed AnnData objects (if --save is set)
  - writes an alignment plot PNG (if --plot is set)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

import scanpy as sc
from PIL import Image

import matplotlib.pyplot as plt
from scipy import sparse



def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

"""
Extracts a .tar.gz safely
"""

def safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    """
    Extract tar safely to avoid path traversal.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = out_dir / member.name
            resolved_out = out_dir.resolve()
            resolved_member = member_path.resolve()
            if not str(resolved_member).startswith(str(resolved_out)):
                raise RuntimeError(f"Unsafe path detected in tar: {member.name}")
        tar.extractall(path=out_dir)


def ensure_spatial_extracted(spatial_tar: Path, base_dir: Path) -> Path:
    """
    The 10x tar contains paths like 'spatial/tissue_hires_image.png'.
    So we must extract into base_dir (parent), and then use base_dir/'spatial' as the spatial folder.
    Returns the resolved spatial directory (handles accidental nesting).
    """
    spatial_root = base_dir / "spatial"
    needed = [
        spatial_root / "tissue_hires_image.png",
        spatial_root / "tissue_positions.csv",
        spatial_root / "scalefactors_json.json",
    ]

    # If already present, done
    if all(p.exists() for p in needed):
        logging.info("Spatial files already present; skipping extraction.")
        return spatial_root

    # Extract tar into base_dir (NOT into spatial_root)
    logging.info("Extracting spatial tar.gz to base dir: %s", base_dir)
    safe_extract_tar(spatial_tar, base_dir)

    # If still not found, maybe we got spatial/spatial/...
    # Use your resolver to find the correct folder
    spatial_dir = resolve_spatial_dir(spatial_root)

    return spatial_dir



def load_expression(h5_file: Path) -> sc.AnnData:
    logging.info("Loading 10x h5 (gex_only=False): %s", h5_file)

    # IMPORTANT: load Gene Expression + Protein Capture
    adata = sc.read_10x_h5(h5_file, gex_only=False)

    # Fix non-unique feature names (common in 10x)
    adata.var_names_make_unique()

    # Handle scanpy/10x naming differences
    if "feature_type" in adata.var.columns:
        ft_col = "feature_type"
    elif "feature_types" in adata.var.columns:
        ft_col = "feature_types"
        adata.var["feature_type"] = adata.var["feature_types"]  # unify downstream code
    else:
        raise KeyError(
            f"No feature_type/feature_types found. Available var columns: {list(adata.var.columns)}"
        )

    return adata

def split_rna_protein(adata: sc.AnnData) -> tuple[sc.AnnData, sc.AnnData]:
    ft = adata.var["feature_type"]

    gene_mask = ft.isin(["Gene Expression"])
    prot_mask = ft.isin(["Protein Capture", "Protein", "Antibody Capture"])

    adata_rna = adata[:, gene_mask].copy()
    adata_prot = adata[:, prot_mask].copy()

    logging.info("RNA shape: %s", adata_rna.shape)
    logging.info("PROT shape: %s", adata_prot.shape)
    logging.info("Protein feature_type counts:\n%s", ft[prot_mask].value_counts())

    return adata_rna, adata_prot



def normalize_rna(adata_rna: sc.AnnData, target_sum: float = 1e4) -> None:
    logging.info("Normalizing RNA: normalize_total(target_sum=%s) + log1p", target_sum)
    sc.pp.normalize_total(adata_rna, target_sum=target_sum)
    sc.pp.log1p(adata_rna)


def protein_to_dense_df(adata_prot: sc.AnnData) -> pd.DataFrame:
    """
    Convert Protein Capture matrix to a dense DataFrame.
    Protein panels are small (tens of markers), so densifying is fine.
    """
    X = adata_prot.X
    try:
        X_dense = X.toarray()  # scipy sparse
    except AttributeError:
        X_dense = np.asarray(X)  # already dense
    return pd.DataFrame(X_dense, index=adata_prot.obs_names, columns=adata_prot.var_names)


def normalize_protein_isotype_spotwise(
    adata_prot: sc.AnnData,
    iso_csv: Path,
    barcode_col: str = "barcode",
    factor_col: str = "normalization_factor",
    log1p: bool = True,
) -> None:
    """
    This dataset's isotype CSV provides a normalization factor PER SPOT (barcode),
    not per protein. We divide each spot's antibody vector by its factor.
    """
    logging.info("Loading isotype normalization factors: %s", iso_csv)
    iso = pd.read_csv(iso_csv)

    if barcode_col not in iso.columns or factor_col not in iso.columns:
        raise KeyError(
            f"Expected columns '{barcode_col}' and '{factor_col}' in {iso_csv}, "
            f"but got columns: {list(iso.columns)}"
        )

    # Build factor vector aligned to adata_prot.obs_names
    iso = iso.set_index(barcode_col)
    common = adata_prot.obs_names.intersection(iso.index)

    if len(common) == 0:
        raise ValueError("No matching barcodes between adata_prot.obs_names and isotype CSV.")

    # Reindex factors to all spots; missing -> 1.0 (no scaling)
    factors = iso.reindex(adata_prot.obs_names)[factor_col].astype(float)
    factors = factors.fillna(1.0).values.reshape(-1, 1)

    # Scale matrix
    X = adata_prot.X
    try:
        # sparse
        X = X.multiply(1.0 / factors)
        adata_prot.X = X
    except AttributeError:
        # dense
        adata_prot.X = adata_prot.X / factors

    if log1p:
        if sparse.issparse(adata_prot.X):
            adata_prot.X = adata_prot.X.tocsr()
        sc.pp.log1p(adata_prot)

    logging.info("Applied spot-wise isotype normalization (and log1p=%s).", log1p)


def load_spatial(spatial_dir: Path) -> tuple[Image.Image, pd.DataFrame, dict]:
    hires_path = spatial_dir / "tissue_hires_image.png"
    pos_path = spatial_dir / "tissue_positions.csv"
    sf_path = spatial_dir / "scalefactors_json.json"

    logging.info("Loading hires image: %s", hires_path)
    img = Image.open(hires_path)

    logging.info("Loading tissue positions: %s", pos_path)
    pos = pd.read_csv(pos_path, header=None, index_col=0)
    pos.columns = ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]

    logging.info("Loading scalefactors: %s", sf_path)
    with open(sf_path, "r") as f:
        sf = json.load(f)

    return img, pos, sf


def attach_spatial(adata_rna: sc.AnnData, adata_prot: sc.AnnData, pos: pd.DataFrame) -> tuple[sc.AnnData, sc.AnnData, pd.DataFrame]:
    common_barcodes = adata_rna.obs_names.intersection(pos.index)

    if len(common_barcodes) == 0:
        raise ValueError("No common barcodes between expression matrix and tissue_positions.csv.")

    adata_rna = adata_rna[common_barcodes].copy()
    adata_prot = adata_prot[common_barcodes].copy()
    pos = pos.loc[common_barcodes].copy()

    # spatial coords in fullres pixel space: x=col, y=row
    coords = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(np.float32)

    adata_rna.obsm["spatial"] = coords
    adata_prot.obsm["spatial"] = coords

    logging.info("Attached spatial coords to RNA+PROT AnnData for %d spots.", coords.shape[0])
    return adata_rna, adata_prot, pos


def plot_alignment(img: Image.Image, adata_rna: sc.AnnData, out_png: Path, point_size: float = 5.0) -> None:
    logging.info("Saving alignment plot to: %s", out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.scatter(
        adata_rna.obsm["spatial"][:, 0],
        adata_rna.obsm["spatial"][:, 1],
        s=point_size,
    )
    plt.axis("off")
    plt.title("Spot alignment check (fullres pixels)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load 10x CytAssist FFPE Gene+Protein dataset, preprocess, and check spatial alignment."
    )
    p.add_argument(
        "--base",
        type=str,
        default="/fs/scratch/PAS3015/alaa/HE_RNA_Protein",
        help="Base directory containing the 3 dataset files.",
    )
    p.add_argument("--plot", action="store_true", help="Save a spot alignment plot PNG.")
    p.add_argument("--plot-path", type=str, default="", help="Custom output path for the alignment plot.")
    p.add_argument("--save", action="store_true", help="Save processed AnnData objects as .h5ad.")
    p.add_argument("--outdir", type=str, default="", help="Output directory for saved files (plot, h5ad).")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity (-v, -vv).")
    return p.parse_args()


def resolve_spatial_dir(spatial_root: Path) -> Path:
    """
    Find the folder that actually contains:
      tissue_hires_image.png, tissue_positions.csv, scalefactors_json.json
    starting from spatial_root (recursively).
    """
    required = {"tissue_hires_image.png", "tissue_positions.csv", "scalefactors_json.json"}

    # If spatial_root already has them, we're done
    if required.issubset({p.name for p in spatial_root.iterdir()}):
        return spatial_root

    # Otherwise search recursively
    hits = list(spatial_root.rglob("tissue_hires_image.png"))
    if not hits:
        raise FileNotFoundError(f"Could not find tissue_hires_image.png under {spatial_root}")

    candidate = hits[0].parent
    # Validate the candidate folder has all required files
    if not all((candidate / f).exists() for f in required):
        raise FileNotFoundError(
            "Found tissue_hires_image.png but not all required spatial files in the same folder.\n"
            f"Candidate: {candidate}"
        )
    return candidate


def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)

    base = Path(args.base)
    data_dir = base / "data"

    h5_file = data_dir / "CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"
    spatial_tar = data_dir / "CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_spatial.tar.gz"
    iso_csv = data_dir / "CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_isotype_normalization_factors.csv"

    for f in [h5_file, spatial_tar, iso_csv]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}")

    # outputs
    outdir = Path(args.outdir) if args.outdir else base / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    # spatial (extract into base, use base/spatial)
    spatial_dir = ensure_spatial_extracted(spatial_tar, base)
    print(f"[INFO] Using spatial_dir = {spatial_dir}")

    # 1) Load expression
    adata = load_expression(h5_file)
    print(adata)
    print(adata.var["feature_type"].value_counts())

    # 2) Split
    adata_rna, adata_prot = split_rna_protein(adata)

    # 3) Normalize RNA
    normalize_rna(adata_rna, target_sum=1e4)

    # 4) Normalize protein
    normalize_protein_isotype_spotwise(adata_prot, iso_csv, log1p=True)


    # 5) Load spatial + attach
    img, pos, sf = load_spatial(spatial_dir)

    adata_rna, adata_prot, pos = attach_spatial(adata_rna, adata_prot, pos)

    # 6) Plot alignment
    if args.plot:
        plot_path = Path(args.plot_path) if args.plot_path else (outdir / "spot_alignment_check.png")
        plot_alignment(img, adata_rna, plot_path, point_size=5.0)
        print(f"[OK] Saved alignment plot: {plot_path}")

    # 7) Save processed objects
    if args.save:
        rna_out = outdir / "adata_rna_processed.h5ad"
        prot_out = outdir / "adata_prot_processed.h5ad"
        adata_rna.write_h5ad(rna_out)
        adata_prot.write_h5ad(prot_out)
        print(f"[OK] Saved: {rna_out}")
        print(f"[OK] Saved: {prot_out}")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()

