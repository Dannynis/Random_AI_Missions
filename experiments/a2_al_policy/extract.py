"""Frozen self-supervised features for CIFAR-10 / CIFAR-100 (DINOv2 CLS + mean patch token, timm weights).

    python extract.py <model> <dataset> [--tiny]
      model:   dinov2-small | dinov2-base
      dataset: cifar10 | cifar100

Writes $LAB_DATA/features/<model>_<dataset>.pt with {"train": (N, 2D) fp16, "test": ..., "ytr", "yte"}.
Resumable: the file is written atomically at the end; an existing file is skipped.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv

DATA = Path(os.environ["LAB_DATA"])
FEAT = DATA / "features"


def load(name: str, train: bool):
    cls = {"cifar10": tv.datasets.CIFAR10, "cifar100": tv.datasets.CIFAR100}[name]
    d = cls(str(DATA), train=train, download=False)
    return torch.from_numpy(d.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(d.targets)


@torch.no_grad()
def embed(model, x_u8: torch.Tensor, res: int, bs: int = 256) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    out = []
    t0 = time.time()
    for i in range(0, len(x_u8), bs):
        x = x_u8[i:i + bs].cuda().float() / 255
        x = F.interpolate(x, size=(res, res), mode="bicubic", align_corners=False)
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.float16):
            h = model.forward_features(x)                    # (B, 1 + patches, D)
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).half().cpu())
        if i // bs % 40 == 0:
            print(f"  {i}/{len(x_u8)} {time.time() - t0:.0f}s", flush=True)
    return torch.cat(out)


def main():
    mname, ds = sys.argv[1], sys.argv[2]
    tiny = "--tiny" in sys.argv
    FEAT.mkdir(parents=True, exist_ok=True)
    path = FEAT / f"{mname}_{ds}{'_tiny' if tiny else ''}.pt"
    if path.exists():
        print("exists", path.name)
        return
    import timm
    res = int(os.environ.get("FEAT_RES", 224))
    arch = {"dinov2-small": "vit_small_patch14_dinov2.lvd142m",
            "dinov2-base": "vit_base_patch14_dinov2.lvd142m"}[mname]
    model = timm.create_model(arch, pretrained=True, img_size=res, num_classes=0).cuda().eval()
    blob = {"model": mname, "dataset": ds, "res": res}
    for split, train in (("train", True), ("test", False)):
        x, y = load(ds, train)
        if tiny:
            x, y = x[:512], y[:512]
        print(mname, ds, split, tuple(x.shape), flush=True)
        blob[split] = embed(model, x, res)
        blob["y" + split[:2]] = y
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)
    print("wrote", path.name, tuple(blob["train"].shape))


if __name__ == "__main__":
    main()
