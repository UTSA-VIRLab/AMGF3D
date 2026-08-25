#!/usr/bin/env python3
"""log_gate_distribution.py — Step 8 per-dataset gate analysis.

Loads an AMGF3D checkpoint, attaches non-invasive forward hooks to
`model.fusion_early.gate` and `model.fusion_late.gate`, runs the val
dataloader, and writes per-dataset gate-value distribution statistics.

Usage:
  CUDA_VISIBLE_DEVICES=5 python tools/log_gate_distribution.py \
      --config configs/tr3d/ff_crossattn/tr3d-ff_crossattn_dual_sunrgbd.py \
      --checkpoint work_dirs/tr3d-ff_crossattn_dual_sunrgbd/latest.pth \
      --output-dir results/gate_distribution/sunrgbd

Output: a directory containing
  - gate_hist_early.npy   (1000-bin histogram of fusion_early gate values)
  - gate_hist_late.npy    (1000-bin histogram of fusion_late gate values)
  - gate_stats.json       (mean, std, entropy, suppressed_frac per site)
  - sample_gate_values.npz  (gate values for 50 random val samples; for figures)

Does NOT modify any model code. Hooks are attached only to the local model
instance, so this is safe to run while other trainers are using the same
detector class.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


HIST_BINS = np.linspace(0.0, 1.0, 1001)  # 1000 equal-width bins on [0, 1]
SUPPRESS_THRESHOLD = 0.1  # gate < 0.1 = "image suppressed for this voxel"
SAMPLE_RAW_K = 50  # number of val frames whose raw gate tensors we keep


class GateLogger:
    """Maintains running histogram + summary stats for one gate site."""

    def __init__(self):
        self.hist = np.zeros(len(HIST_BINS) - 1, dtype=np.int64)
        self.n = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.n_suppressed = 0
        self.raw_samples = []  # list of np.ndarray for ~50 random batches

    def update(self, tensor, keep_raw=False):
        v = tensor.detach().cpu().float().reshape(-1).numpy()
        if v.size == 0:
            return
        h, _ = np.histogram(v, bins=HIST_BINS)
        self.hist += h
        self.n += v.size
        self.sum += float(v.sum())
        self.sum_sq += float((v * v).sum())
        self.n_suppressed += int((v < SUPPRESS_THRESHOLD).sum())
        if keep_raw and len(self.raw_samples) < SAMPLE_RAW_K:
            self.raw_samples.append(v.astype(np.float32))

    def stats(self):
        if self.n == 0:
            return {"n": 0}
        mean = self.sum / self.n
        var = max(self.sum_sq / self.n - mean * mean, 0.0)
        std = float(var ** 0.5)
        # Entropy in nats of the histogram (proxy for gate-value diversity)
        probs = self.hist / max(self.hist.sum(), 1)
        nz = probs[probs > 0]
        entropy = float(-(nz * np.log(nz)).sum())
        return {
            "n_voxels": int(self.n),
            "mean": float(mean),
            "std": std,
            "entropy_nats": entropy,
            "suppressed_frac": float(self.n_suppressed / self.n),
            "suppress_threshold": SUPPRESS_THRESHOLD,
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples-per-gpu", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=None,
                   help="Optional cap on number of val batches (default = full val set)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)

    # Force test_mode dataset, eval-only, batch=1 for memory safety
    cfg.data.test.test_mode = True
    cfg.gpu_ids = [0]  # CUDA_VISIBLE_DEVICES handles real GPU
    cfg.dist_params = None
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.val)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
    )

    model = build_detector(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # Attach forward hooks on the gate Sequential of each fusion stage.
    inner = model.module
    site_loggers = {}

    def make_hook(site, keep_raw_indices):
        log = GateLogger()
        site_loggers[site] = log

        def hook(_mod, _inp, output):
            keep = (state["batch_idx"] in keep_raw_indices)
            log.update(output, keep_raw=keep)
        return hook

    # Pick random batch indices to dump raw gate values from (for figures later)
    rng = random.Random(0)
    n_val = len(loader)
    keep_set = set(rng.sample(range(n_val), min(SAMPLE_RAW_K, n_val)))
    state = {"batch_idx": -1}

    hook_handles = []
    if hasattr(inner, "fusion_early") and inner.fusion_early is not None:
        hook_handles.append(inner.fusion_early.gate.register_forward_hook(
            make_hook("early", keep_set)))
    if hasattr(inner, "fusion_late") and inner.fusion_late is not None:
        hook_handles.append(inner.fusion_late.gate.register_forward_hook(
            make_hook("late", keep_set)))

    if not hook_handles:
        raise RuntimeError("Model has no fusion_early/fusion_late.gate -- wrong detector?")

    print(f"[log_gate] dataset={Path(args.config).stem} n_val={n_val} sites={list(site_loggers)}")

    with torch.no_grad():
        for i, data in enumerate(loader):
            state["batch_idx"] = i
            try:
                _ = model(return_loss=False, rescale=True, **data)
            except Exception as e:  # eg. corrupted sample
                print(f"  batch {i}: error {type(e).__name__}: {e}")
                continue
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n_val}] processed")
            if args.max_batches is not None and (i + 1) >= args.max_batches:
                break

    for h in hook_handles:
        h.remove()

    # Persist results
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "n_val_batches_processed": state["batch_idx"] + 1,
        "samples_per_gpu": args.samples_per_gpu,
        "sites": {site: logger.stats() for site, logger in site_loggers.items()},
        "histogram_bin_edges": HIST_BINS.tolist(),
    }
    (out / "gate_stats.json").write_text(json.dumps(summary, indent=2))

    raw_dump = {}
    for site, logger in site_loggers.items():
        np.save(out / f"gate_hist_{site}.npy", logger.hist)
        if logger.raw_samples:
            raw_dump[site] = np.array(logger.raw_samples, dtype=object)
    if raw_dump:
        np.savez(out / "sample_gate_values.npz",
                 **{k: np.asarray(v, dtype=object) for k, v in raw_dump.items()})

    print(f"[log_gate] DONE -> {out}")
    for site, st in summary["sites"].items():
        print(f"  {site}: mean={st['mean']:.4f} std={st['std']:.4f} "
              f"suppressed_frac={st['suppressed_frac']:.4f} entropy={st['entropy_nats']:.4f}")


if __name__ == "__main__":
    main()
