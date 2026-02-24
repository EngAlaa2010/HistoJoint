#!/usr/bin/env python3
"""
02_extract_patches.py

Extract per-spot image patches (local + context) from a 10x Visium CytAssist FFPE dataset.

Inputs:
  - outputs/adata_rna_processed.h5ad   (must contain obsm['spatial'])
  - spatial/aligned_tissue_image.jpg   (preferred for CytAssist morphology)
  - spatial/detected_tissue_image.jpg  (fallback)
  - spatial/tissue_hires_image.png     (fallback, often dim/low-contrast for CytAssist)
  - spatial/tissue_positions.csv       (optional, for in_tissue flag)
  - spatial/scalefactors_json.json     (optional, for explicit scaling)

Outputs:
  - patches_local/       (PNG patches, one per barcode)
  - patches_context/     (PNG patches, one per barcode)
  - outputs/patch_index.csv (barcode -> patch paths + coords + in_tissue + local_std + scale_mode)
  - outputs/patch_montage_local.png (optional QC montage)

Scaling (important):
  Coords from adata.obsm['spatial'] are usually in "fullres Visium space" (as used with tissue_hires_image.png).
  If you extract patches from aligned_tissue_image.jpg/detected_tissue_image.jpg, you typically need to scale
  coords by scalefactors_json.json["regist_target_img_scalef"].

This script supports explicit scale modes:
  --scale-mode {auto,none,regist,hires,lowres}

Quality filter:
  --min-std X : skip patches with grayscale std(local patch) < X
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image


# ---------------- logging ----------------

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


# ---------------- data ----------------

@dataclass(frozen=True)
class Spot:
    barcode: str
    x: float  # pixel x (col)
    y: float  # pixel y (row)


# ---------------- image / patch utils ----------------

def patch_gray_std(patch_rgb: np.ndarray) -> float:
    """Compute grayscale std from uint8 RGB patch (H,W,3)."""
    gray = (
        0.2989 * patch_rgb[..., 0]
        + 0.5870 * patch_rgb[..., 1]
        + 0.1140 * patch_rgb[..., 2]
    ).astype(np.float32)
    return float(gray.std())


def crop_patch_rgb_array(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """
    Center-crop square patch around (cx, cy) from img (uint8 HxWx3).
    Pads out-of-bounds with black.
    Returns uint8 (size,size,3).
    """
    if size <= 0:
        raise ValueError("size must be positive")
    h, w, _ = img.shape
    half = size // 2

    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size
    y1 = y0 + size

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        padded = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
        return padded[y0:y1, x0:x1, :]

    return img[y0:y1, x0:x1, :]


def save_patch_array(patch_rgb: np.ndarray, out_path: Path, overwrite: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return
    Image.fromarray(patch_rgb, mode="RGB").save(out_path, format="PNG", optimize=True)


def make_montage(patch_paths: List[Path], out_path: Path, tile: int = 96, cols: int = 12) -> None:
    """Create a simple montage of patch images for QC."""
    if not patch_paths:
        return

    imgs: List[Image.Image] = []
    for p in patch_paths:
        try:
            im = Image.open(p).convert("RGB")
            if im.size != (tile, tile):
                im = im.resize((tile, tile))
            imgs.append(im)
        except Exception:
            continue

    if not imgs:
        return

    rows = int(np.ceil(len(imgs) / cols))
    canvas = Image.new("RGB", (cols * tile, rows * tile), (255, 255, 255))
    for i, im in enumerate(imgs):
        r = i // cols
        c = i % cols
        canvas.paste(im, (c * tile, r * tile))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)


# ---------------- IO helpers ----------------

def load_positions_optional(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    pos = pd.read_csv(path, header=None, index_col=0)
    pos.columns = ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    return pos


def load_scalefactors_optional(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def choose_image_path(base: Path, explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--hires-image does not exist: {p}")
        return p

    candidates = [
        base / "spatial" / "aligned_tissue_image.jpg",
        base / "spatial" / "detected_tissue_image.jpg",
        base / "spatial" / "tissue_hires_image.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No suitable image found under base/spatial. Tried:\n" + "\n".join(str(p) for p in candidates)
    )



def percent_in_bounds(coords: np.ndarray, img_w: int, img_h: int) -> float:
    #tells us whether coordinates lie inside the image.
    #If in-bounds is low, our coordinates are in a different pixel space than the image.
    x = coords[:, 0]
    y = coords[:, 1]
    ok = (x >= 0) & (x < img_w) & (y >= 0) & (y < img_h)
    return float(ok.mean() * 100.0)


def autoscale_coords_to_image(coords: np.ndarray, img_w: int, img_h: int) -> Tuple[np.ndarray, float]:
    """
    Heuristic fallback: uniformly scale coords so their max fits in the image.
    “If even after chosen scale_mode the coords still don’t fit, apply a last-resort heuristic scaling.”
    Returns (coords_scaled, scale).
    """
    max_x = float(np.nanmax(coords[:, 0]))
    max_y = float(np.nanmax(coords[:, 1]))
    if max_x <= 1 or max_y <= 1:
        return coords, 1.0

    sx = img_w / max_x
    sy = img_h / max_y
    s = min(sx, sy)

    coords2 = coords.copy()
    coords2[:, 0] *= s
    coords2[:, 1] *= s
    return coords2, s


def apply_scale_mode(
    coords: np.ndarray,
    hires_path: Path,
    sf: Optional[dict],
    scale_mode: str,
) -> Tuple[np.ndarray, str, float]:
    """
    Returns (coords_scaled, used_mode, scale_value).

    scale_mode:
      - "none"   : do not scale
      - "regist" : multiply by sf["regist_target_img_scalef"]
      - "hires"  : multiply by sf["tissue_hires_scalef"]
      - "lowres" : multiply by sf["tissue_lowres_scalef"]
      - "auto"   : choose based on image filename; if sf missing, keep coords unchanged
    """
    mode = scale_mode.lower().strip()
    if mode not in {"auto", "none", "regist", "hires", "lowres"}:
        raise ValueError(f"Invalid --scale-mode: {scale_mode}")

    img_name = hires_path.name.lower()

    if mode == "auto":
        # Good default for CytAssist:
        # - aligned/detected images are in registration-target space => use regist_target_img_scalef
        # - tissue_hires/lowres images use their respective scale factors (common Visium behavior)
        if "aligned_tissue_image" in img_name or "detected_tissue_image" in img_name:
            mode = "regist"
        elif "tissue_lowres" in img_name:
            mode = "lowres"
        else:
            # tissue_hires_image.png (or unknown) => hires
            mode = "hires"

    if mode == "none":
        return coords, "none", 1.0

    if sf is None:
        logging.warning("Scale mode '%s' requested but scalefactors_json.json not found; leaving coords unchanged.", mode)
        return coords, f"{mode}_missing_sf", 1.0

    key_map = {
        "regist": "regist_target_img_scalef",
        "hires": "tissue_hires_scalef",
        "lowres": "tissue_lowres_scalef",
    }
    key = key_map[mode]
    s = float(sf.get(key, 1.0))
    if s == 0.0:
        logging.warning("Scale factor %s is 0.0; leaving coords unchanged.", key)
        return coords, f"{mode}_bad_sf", 1.0

    coords2 = coords.copy()
    coords2[:, 0] *= s
    coords2[:, 1] *= s
    return coords2, mode, s


# ---------------- args ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract per-spot local+context image patches from Visium/CytAssist images.")
    p.add_argument("--base", type=str, default="/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    p.add_argument("--adata", type=str, default="", help="Default: base/outputs/adata_rna_processed.h5ad")
    p.add_argument("--hires-image", type=str, default="", help="Explicit image path to use.")
    p.add_argument("--positions", type=str, default="", help="Optional tissue_positions.csv")
    p.add_argument("--scalefactors", type=str, default="", help="Optional scalefactors_json.json")

    p.add_argument("--local-size", type=int, default=96)
    p.add_argument("--context-size", type=int, default=224)

    p.add_argument("--out-local", type=str, default="", help="Default: base/patches_local")
    p.add_argument("--out-context", type=str, default="", help="Default: base/patches_context")
    p.add_argument("--index-out", type=str, default="", help="Default: base/outputs/patch_index.csv")

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--max-spots", type=int, default=0, help="If >0, process only first N spots (testing).")
    p.add_argument("--only-in-tissue", action="store_true")

    p.add_argument("--montage", action="store_true", help="Write montage for QC (up to 200 patches).")
    p.add_argument("--min-std", type=float, default=0.0, help="Skip if local grayscale std < this.")

    p.add_argument(
        "--scale-mode",
        type=str,
        default="auto",
        choices=["auto", "none", "regist", "hires", "lowres"],
        help="How to map obsm['spatial'] coords into the chosen image pixel space.",
    )
    p.add_argument(
        "--allow-autoscale-fallback",
        action="store_true",
        help="If coords are still out-of-bounds after scale-mode, allow heuristic autoscale fallback.",
    )

    p.add_argument("--verbose", "-v", action="count", default=0)
    return p.parse_args()


# ---------------- main ----------------

def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)

    base = Path(args.base)

    adata_path = Path(args.adata) if args.adata else (base / "outputs" / "adata_rna_processed.h5ad")
    if not adata_path.exists():
        raise FileNotFoundError(f"Missing: {adata_path}")

    hires_path = choose_image_path(base, args.hires_image)

    positions_path = Path(args.positions) if args.positions else (base / "spatial" / "tissue_positions.csv")
    if not positions_path.exists():
        positions_path = None

    scalef_path = Path(args.scalefactors) if args.scalefactors else (base / "spatial" / "scalefactors_json.json")
    if not scalef_path.exists():
        scalef_path = None

    out_local = Path(args.out_local) if args.out_local else (base / "patches_local")
    out_ctx = Path(args.out_context) if args.out_context else (base / "patches_context")
    index_out = Path(args.index_out) if args.index_out else (base / "outputs" / "patch_index.csv")

    logging.info("Loading AnnData: %s", adata_path)
    adata = sc.read_h5ad(adata_path)

    if "spatial" not in adata.obsm:
        raise KeyError("AnnData missing obsm['spatial']. Run step 01 first.")

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Expected obsm['spatial'] shape (n_spots, 2).")

    # Load image
    logging.info("Loading image: %s", hires_path)
    img_pil = Image.open(hires_path).convert("RGB")
    img = np.asarray(img_pil, dtype=np.uint8)
    img_h, img_w = img.shape[0], img.shape[1]

    # Load scalefactors
    sf = load_scalefactors_optional(scalef_path)
    if sf is not None:
        logging.info("Loaded scalefactors_json.json (keys=%s)", list(sf.keys()))
    else:
        logging.info("No scalefactors_json.json found (scale-mode may be limited).")

    # Apply explicit scaling mode (deterministic)
    inb0 = percent_in_bounds(coords, img_w, img_h)
    logging.info("Coords in-bounds BEFORE scaling: %.2f%% (image %dx%d)", inb0, img_w, img_h)

    coords, used_scale_mode, used_scale_value = apply_scale_mode(coords, hires_path, sf, args.scale_mode)

    inb1 = percent_in_bounds(coords, img_w, img_h)
    logging.info(
        "Scale-mode applied: %s (scale=%.6f). In-bounds AFTER: %.2f%%",
        used_scale_mode, used_scale_value, inb1
    )

    # Optional heuristic fallback if still not good
    if args.allow_autoscale_fallback and inb1 < 95.0:
        coords2, s_auto = autoscale_coords_to_image(coords, img_w, img_h)
        inb2 = percent_in_bounds(coords2, img_w, img_h)
        logging.warning(
            "Autoscale fallback applied: scale=%.6f. In-bounds AFTER autoscale: %.2f%%",
            s_auto, inb2
        )
        coords = coords2
        used_scale_mode = f"{used_scale_mode}+autoscale"
        used_scale_value *= s_auto

    # Build spots
    barcodes = adata.obs_names.to_list()
    if len(barcodes) != coords.shape[0]:
        raise RuntimeError("Mismatch between obs_names and spatial coords length.")

    spots = [Spot(bc, float(coords[i, 0]), float(coords[i, 1])) for i, bc in enumerate(barcodes)]

    if args.max_spots and args.max_spots > 0:
        spots = spots[: args.max_spots]
        logging.info("Max-spots enabled: processing %d spots", len(spots))

    # Optional in_tissue filtering
    pos_df = load_positions_optional(positions_path)
    in_tissue_map = None
    if pos_df is not None:
        in_tissue_map = pos_df["in_tissue"].to_dict()
        if args.only_in_tissue:
            before = len(spots)
            spots = [s for s in spots if int(in_tissue_map.get(s.barcode, 0)) == 1]
            logging.info("Filtered to in_tissue==1: %d -> %d spots", before, len(spots))

    if not spots:
        raise RuntimeError("No spots to process after filtering.")

    logging.info(
        "Starting patch extraction: %d spots | local=%d | context=%d | workers=%d | min_std=%.3f",
        len(spots), args.local_size, args.context_size, args.workers, float(args.min_std)
    )

    records: List[dict] = []
    montage_candidates_local: List[Path] = []
    skipped_low_std = 0

    def _process_one(s: Spot):
        local = crop_patch_rgb_array(img, s.x, s.y, args.local_size)
        local_std = patch_gray_std(local)
        if args.min_std > 0 and local_std < args.min_std:
            return None

        ctx = crop_patch_rgb_array(img, s.x, s.y, args.context_size)

        local_path = out_local / f"{s.barcode}.png"
        ctx_path = out_ctx / f"{s.barcode}.png"

        save_patch_array(local, local_path, overwrite=args.overwrite)
        save_patch_array(ctx, ctx_path, overwrite=args.overwrite)

        return s, local_path, ctx_path, local_std

    batch_size = max(1, args.batch_size)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for start in range(0, len(spots), batch_size):
            batch = spots[start : start + batch_size]
            futures = {ex.submit(_process_one, s): s for s in batch}

            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    logging.error("Failed spot %s: %s", s.barcode, e)
                    continue

                if out is None:
                    skipped_low_std += 1
                    continue

                s2, local_path, ctx_path, local_std = out

                rec = {
                    "barcode": s2.barcode,
                    "x_pixel": s2.x,
                    "y_pixel": s2.y,
                    "local_patch": str(local_path),
                    "context_patch": str(ctx_path),
                    "local_std": float(local_std),
                    "scale_mode": used_scale_mode,
                    "scale_value": float(used_scale_value),
                    "image_used": str(hires_path),
                }
                if in_tissue_map is not None:
                    rec["in_tissue"] = int(in_tissue_map.get(s2.barcode, 0))
                records.append(rec)

                if args.montage and len(montage_candidates_local) < 200:
                    montage_candidates_local.append(local_path)

            logging.info("Processed %d / %d spots", min(start + batch_size, len(spots)), len(spots))

    # Write index
    index_out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records).sort_values("barcode")
    df.to_csv(index_out, index=False)
    print(f"[OK] Wrote patch index: {index_out}")

    if args.min_std > 0:
        print(f"[INFO] Skipped {skipped_low_std} spots due to local_std < {args.min_std}")

    # Montage
    if args.montage and montage_candidates_local:
        montage_out = base / "outputs" / "patch_montage_local.png"
        make_montage(montage_candidates_local, montage_out, tile=args.local_size, cols=12)
        print(f"[OK] Wrote montage: {montage_out}")

    print("[OK] Patch extraction complete.")
    print(f"  Image used:      {hires_path}")
    print(f"  Local patches:   {out_local}")
    print(f"  Context patches: {out_ctx}")


if __name__ == "__main__":
    main()
