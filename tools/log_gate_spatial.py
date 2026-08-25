#!/usr/bin/env python3
"""log_gate_spatial.py — one-scene voxel-level gate dump for Fig 4.

Runs AMGF3D forward on a single val sample and saves per-voxel (xyz, gate_early).
Captures coords from the SparseTensor output of backbone.layer1 and the gate
output from fusion_early.gate; both fire in voxel-order so they pair 1-to-1.

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/log_gate_spatial.py \
      --config configs/.../sunrgbd.py \
      --checkpoint work_dirs/.../latest.pth \
      --img-id 4594 \
      --output results/gate_spatial/sunrgbd_4594.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--img-id", type=int, required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    cfg.gpu_ids = [0]
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    # Build a one-sample loader pointing at args.img_id
    subset = torch.utils.data.Subset(dataset, [args.img_id])
    # build_dataloader expects a Custom3DDataset; for a Subset, fall back to
    # a plain DataLoader using the dataset's collate.
    from mmcv.parallel import collate
    from functools import partial
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, num_workers=0, shuffle=False,
        collate_fn=partial(collate, samples_per_gpu=1))

    model = build_detector(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    inner = model.module
    captured = {"coords_layer1": None, "gate_early": None, "voxel_size": inner.voxel_size}

    def hook_layer1(module, input, output):
        # output is a SparseTensor
        coords = output.C[:, 1:].detach().cpu().numpy().astype(np.float32) * inner.voxel_size
        captured["coords_layer1"] = coords

    def hook_gate(module, input, output):
        captured["gate_early"] = output.detach().cpu().numpy().squeeze().astype(np.float32)

    h1 = inner.backbone.layer1.register_forward_hook(hook_layer1)
    if hasattr(inner, "fusion_early") and inner.fusion_early is not None:
        h2 = inner.fusion_early.gate.register_forward_hook(hook_gate)
    else:
        raise RuntimeError("Model has no fusion_early; can't log early-gate spatial overlay.")

    with torch.no_grad():
        for data in loader:
            _ = model(return_loss=False, rescale=True, **data)

    h1.remove(); h2.remove()

    coords = captured["coords_layer1"]
    gate = captured["gate_early"]
    if coords is None or gate is None:
        raise RuntimeError(f"hooks didn't fire (coords={coords is None}, gate={gate is None})")
    assert coords.shape[0] == gate.shape[0], \
        f"voxel mismatch: coords {coords.shape} vs gate {gate.shape}"
    np.savez(out_path, coords=coords, gate=gate, voxel_size=inner.voxel_size)
    print(f"[log_gate_spatial] saved {coords.shape[0]} voxels -> {out_path}")
    print(f"  gate range = [{gate.min():.4f}, {gate.max():.4f}], mean={gate.mean():.4f}")


if __name__ == "__main__":
    main()
