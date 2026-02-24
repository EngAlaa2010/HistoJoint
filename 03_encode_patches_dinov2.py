#!/usr/bin/env python3
"""
03_encode_patches_dinov2.py

Step 03: Encode per-spot image patches using DINOv2 backbone + linear projection head.

Inputs:
  - outputs/patch_index.csv from Step 02 (must have: barcode, local_patch, context_patch)
Outputs:
  - outputs/embeddings_local.npy      (N x proj_dim)
  - outputs/embeddings_context.npy    (N x proj_dim)
  - outputs/embeddings_concat.npy     (N x 2*proj_dim)
  - outputs/embeddings_index.csv      (row_id + patch_index columns)

Notes:
  - No torchvision dependency (HPC-friendly when torchvision is broken).
  - Torch hub loads DINOv2 models: dinov2_vits14 / dinov2_vitb14 / dinov2_vitl14 / dinov2_vitg14
  - Runs on CPU by default; CUDA only if your torch supports it.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


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
class Sample:
    row_id: int
    barcode: str
    local_path: Path
    context_path: Path
    in_tissue: Optional[int] = None
    local_std: Optional[float] = None


def pil_to_tensor_rgb(im: Image.Image) -> torch.Tensor:
    """
    PIL RGB -> float tensor in [0,1], shape (3,H,W)
    No torchvision used.
    """
    arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
    arr = np.transpose(arr, (2, 0, 1))              # (3,H,W)
    return torch.from_numpy(arr)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    x: (3,H,W) float in [0,1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype)[:, None, None]
    return (x - mean) / std


class PatchDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        image_size: int = 224,
        normalize: str = "imagenet",
    ) -> None:
        self.samples = samples
        self.image_size = int(image_size)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, p: Path) -> torch.Tensor:
        im = Image.open(p).convert("RGB")
        im = im.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        t = pil_to_tensor_rgb(im)
        if self.normalize == "imagenet":
            t = imagenet_normalize(t)
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize: {self.normalize}")
        return t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "row_id": torch.tensor(s.row_id, dtype=torch.long),
            "local": self._load(s.local_path),
            "context": self._load(s.context_path),
        }


# ---------------- model ----------------

def build_dinov2_backbone(arch: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    arch must be a hub callable:
      dinov2_vits14 / dinov2_vitb14 / dinov2_vitl14 / dinov2_vitg14
    """
    model = torch.hub.load("facebookresearch/dinov2", arch, pretrained=pretrained)
    model.eval()

    # infer dim
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        out = model(dummy)
        if out.ndim != 2:
            raise RuntimeError(f"DINOv2 backbone output expected (B,D), got {tuple(out.shape)}")
        dim = int(out.shape[1])

    return model, dim


class DinoWithProj(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(backbone_dim, proj_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)          # (B, backbone_dim)
        y = self.proj(z)              # (B, proj_dim)
        return y


# ---------------- utils ----------------

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False (likely CPU-only torch).")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_patch_index(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"barcode", "local_patch", "context_patch"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"patch_index.csv missing columns: {sorted(missing)}")
    return df


# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Step 03: DINOv2 encoding (no torchvision).")
    p.add_argument("--base", type=str, default="/fs/scratch/PAS3015/alaa/HE_RNA_Protein")
    p.add_argument("--patch-index", type=str, default="", help="Default: base/outputs/patch_index.csv")
    p.add_argument("--outdir", type=str, default="", help="Default: base/outputs")
    p.add_argument("--only-in-tissue", action="store_true")
    p.add_argument("--min-std", type=float, default=0.0)

    p.add_argument("--dinov2-arch", type=str, default="dinov2_vitb14",
                   help="torch.hub callable: dinov2_vits14|dinov2_vitb14|dinov2_vitl14|dinov2_vitg14")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--normalize", type=str, default="imagenet", choices=["imagenet", "none"])

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-spots", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--save-local", action="store_true")
    p.add_argument("--save-context", action="store_true")
    p.add_argument("--save-concat", action="store_true")

    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.verbose)

    base = Path(args.base)
    patch_index_path = Path(args.patch_index) if args.patch_index else (base / "outputs" / "patch_index.csv")
    outdir = Path(args.outdir) if args.outdir else (base / "outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    if not patch_index_path.exists():
        raise FileNotFoundError(f"Missing patch index: {patch_index_path}")

    df = load_patch_index(patch_index_path)

    if args.only_in_tissue and "in_tissue" in df.columns:
        before = len(df)
        df = df[df["in_tissue"].astype(int) == 1].copy()
        logging.info("Filtered in_tissue==1: %d -> %d", before, len(df))

    if args.min_std > 0 and "local_std" in df.columns:
        before = len(df)
        df = df[df["local_std"].astype(float) >= float(args.min_std)].copy()
        logging.info("Filtered local_std >= %.3f: %d -> %d", float(args.min_std), before, len(df))

    if args.max_spots and args.max_spots > 0:
        df = df.iloc[: args.max_spots].copy()
        logging.info("Max-spots enabled: encoding %d", len(df))

    if len(df) == 0:
        raise RuntimeError("No rows left after filtering.")

    # default output = concat
    if not (args.save_local or args.save_context or args.save_concat):
        args.save_concat = True

    out_local = outdir / "embeddings_local.npy"
    out_ctx = outdir / "embeddings_context.npy"
    out_cat = outdir / "embeddings_concat.npy"
    out_index = outdir / "embeddings_index.csv"

    if not args.overwrite:
        if out_index.exists() and any(p.exists() for p in [out_local, out_ctx, out_cat]):
            logging.warning("Embedding outputs already exist. Use --overwrite to recompute.")
            print(f"[SKIP] Found existing embeddings under: {outdir}")
            return

    # samples
    samples: List[Sample] = []
    df2 = df.reset_index(drop=True).copy()
    for i, row in df2.iterrows():
        samples.append(Sample(
            row_id=i,
            barcode=str(row["barcode"]),
            local_path=Path(row["local_patch"]),
            context_path=Path(row["context_patch"]),
            in_tissue=int(row["in_tissue"]) if "in_tissue" in df2.columns else None,
            local_std=float(row["local_std"]) if "local_std" in df2.columns else None,
        ))

    device = resolve_device(args.device)
    logging.info("Using device: %s", device)

    backbone, backbone_dim = build_dinov2_backbone(args.dinov2_arch, pretrained=(not args.no_pretrained))
    model = DinoWithProj(backbone, backbone_dim, int(args.proj_dim))
    model.eval()
    model.to(device)
    logging.info("Loaded DINOv2: %s | backbone_dim=%d | proj_dim=%d",
                 args.dinov2_arch, backbone_dim, int(args.proj_dim))

    ds = PatchDataset(samples, image_size=args.image_size, normalize=args.normalize)
    dl = DataLoader(
        ds,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    N = len(samples)
    D = int(args.proj_dim)
    emb_local = np.zeros((N, D), dtype=np.float32)
    emb_ctx = np.zeros((N, D), dtype=np.float32)

    with torch.no_grad():
        for step, batch in enumerate(dl, start=1):
            row_ids = batch["row_id"].numpy()
            local = batch["local"].to(device, non_blocking=True)
            ctx = batch["context"].to(device, non_blocking=True)

            f_local = model(local).detach().cpu().numpy().astype(np.float32)
            f_ctx = model(ctx).detach().cpu().numpy().astype(np.float32)

            emb_local[row_ids, :] = f_local
            emb_ctx[row_ids, :] = f_ctx

            if step % 20 == 0 or step == 1:
                done = int(row_ids.max()) + 1
                logging.info("Encoded %d / %d", done, N)

    if args.save_local:
        np.save(out_local, emb_local)
        print(f"[OK] Wrote: {out_local}")
    if args.save_context:
        np.save(out_ctx, emb_ctx)
        print(f"[OK] Wrote: {out_ctx}")
    if args.save_concat:
        emb_cat = np.concatenate([emb_local, emb_ctx], axis=1)
        np.save(out_cat, emb_cat)
        print(f"[OK] Wrote: {out_cat}")

    out_df = df2.copy()
    out_df.insert(0, "row_id", np.arange(len(out_df), dtype=int))
    out_df.to_csv(out_index, index=False)
    print(f"[OK] Wrote: {out_index}")

    logging.info("Embeddings stats: local mean=%.4f std=%.4f | ctx mean=%.4f std=%.4f",
                 float(emb_local.mean()), float(emb_local.std()),
                 float(emb_ctx.mean()), float(emb_ctx.std()))

    print("[OK] Step 03 complete.")
    print(f"  DINOv2: {args.dinov2_arch} (pretrained={not args.no_pretrained})")
    print(f"  N spots encoded: {N}")
    print(f"  Output dir: {outdir}")


if __name__ == "__main__":
    main()
