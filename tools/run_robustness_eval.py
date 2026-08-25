#!/usr/bin/env python3
"""run_robustness_eval.py — Step 7 of the ACCV revision plan.

Runs eval over the val set with one perturbation applied to the test pipeline,
then writes the headline mAP@.25 / mAP@.50 to a CSV row plus a JSON summary.

Usage:
  CUDA_VISIBLE_DEVICES=5 python tools/run_robustness_eval.py \
      --config configs/tr3d/ff_crossattn/tr3d-ff_crossattn_dual_sunrgbd.py \
      --checkpoint work_dirs/tr3d-ff_crossattn_dual_sunrgbd/latest.pth \
      --method amgf3d --dataset sunrgbd \
      --perturbation points_subsample --level 0.5 \
      --output-csv results/robustness_eval/results.csv

The perturbation is spliced into the test_pipeline in this process only;
the on-disk config is not modified.

Perturbations and their canonical levels:
    points_xyz_noise        sigma in metres        {0.01, 0.05, 0.1}
    points_subsample        keep_ratio             {0.75, 0.5, 0.25}
    rgb_gaussian            sigma in norm pixels   {0.05, 0.1, 0.2}
    rgb_dropout             patch_frac of area     {0.25, 0.5, 0.75}
    calibration_rotation    sigma in degrees       {1.0, 3.0, 5.0}
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

# Force-register robustness pipeline transforms in PIPELINES registry of THIS process
from mmdet3d.datasets.pipelines import robustness_transforms  # noqa: F401
from mmdet3d.datasets.pipelines.robustness_transforms import perturbation_spec

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


def splice_perturbation(test_pipeline, pert_name, level):
    """Insert the perturbation transforms into the right slot of a test pipeline.

    Convention:
      - point-space perturbations are inserted BEFORE MultiScaleFlipAug3D
        (so they operate on the freshly-loaded Points object, before any
        flip/scale augmentation).
      - image-space perturbations are inserted INSIDE MultiScaleFlipAug3D,
        right after the Normalize transform (so sigma is in normalized-pixel units).
    """
    spec = perturbation_spec(pert_name, level)
    out = []
    point_perts = {'PointXYZNoise', 'PointSubsample', 'CalibrationRotation',
                   'PointColorDrop', 'ExtrinsicRotationNoise', 'RayDepthNoise'}
    image_perts = {'RgbGaussianNoise', 'RgbDropout'}
    # A spec may mix sites (e.g. rgb_dropout_pointcolor): image transforms go
    # inside MultiScaleFlipAug3D after Normalize, point transforms before it.
    img_spec = [s for s in spec if s['type'] in image_perts]
    pts_spec = [s for s in spec if s['type'] in point_perts]

    for step in test_pipeline:
        if step.get('type') == 'MultiScaleFlipAug3D':
            if pts_spec:
                # point perts go BEFORE MultiScaleFlipAug3D
                out.extend(pts_spec)
            if img_spec:
                # Insert after Normalize in the inner transforms
                inner = list(step.get('transforms', []))
                new_inner = []
                inserted = False
                for s in inner:
                    new_inner.append(s)
                    if not inserted and s.get('type') == 'Normalize':
                        new_inner.extend(img_spec)
                        inserted = True
                if not inserted:
                    # Fallback: prepend to inner if Normalize not found
                    new_inner = img_spec + new_inner
                step = copy.deepcopy(step)
                step['transforms'] = new_inner
        out.append(step)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--method', required=True, help='label written to the output CSV (amgf3d | tr3d_ff); must match the labels crossmethod_robustness_stats.py expects')
    p.add_argument('--dataset', required=True, help='label only (sunrgbd | scannet | arkitscenes)')
    p.add_argument('--perturbation', required=True)
    p.add_argument('--level', type=float, required=True)
    p.add_argument('--output-csv', required=True)
    p.add_argument('--samples-per-gpu', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tag', default='', help='extra column for distinguishing runs')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None
    cfg.gpu_ids = [0]

    # Splice the perturbation into the test pipeline.
    # mmdet3d configs typically put it on cfg.data.test.pipeline OR cfg.test_pipeline.
    if hasattr(cfg.data.test, 'pipeline'):
        cfg.data.test.pipeline = splice_perturbation(
            cfg.data.test.pipeline, args.perturbation, args.level)
    else:
        raise RuntimeError('cfg.data.test has no pipeline attribute')

    print(f'[robust] cfg={Path(args.config).stem} pert={args.perturbation} level={args.level}')
    print('[robust] perturbed test pipeline:')
    for s in cfg.data.test.pipeline:
        print('  ', s.get('type'), {k: v for k, v in s.items() if k != 'type'} if isinstance(s, dict) else s)

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
    )

    model = build_detector(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    t0 = time.time()
    outputs = single_gpu_test(model, loader, show=False)
    elapsed = time.time() - t0

    # Evaluate
    eval_kwargs = {}
    if hasattr(cfg, 'evaluation'):
        eval_kwargs = {k: v for k, v in dict(cfg.evaluation).items()
                       if k not in ('interval', 'save_best', 'rule', 'pipeline')}
    eval_results = dataset.evaluate(outputs, **eval_kwargs)
    # mAP@.25 and mAP@.50 may surface under different keys; collect both.
    flat = {}
    for k, v in eval_results.items():
        if isinstance(v, (int, float)):
            flat[k] = float(v)
    map25 = (flat.get('mAP_0.25') or flat.get('mAP@0.25')
             or flat.get('Overall_0.25') or flat.get('pts_bbox_mAP_0.25'))
    map50 = (flat.get('mAP_0.50') or flat.get('mAP@0.50')
             or flat.get('Overall_0.50') or flat.get('pts_bbox_mAP_0.50'))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        'method': args.method,
        'dataset': args.dataset,
        'perturbation': args.perturbation,
        'level': args.level,
        'mAP_0.25': map25 if map25 is not None else '',
        'mAP_0.50': map50 if map50 is not None else '',
        'n_val': len(dataset),
        'elapsed_sec': round(elapsed, 1),
        'checkpoint': args.checkpoint,
        'config': args.config,
        'tag': args.tag,
    }
    # Append row (header if file new)
    new_file = not out_path.exists()
    with out_path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)

    # Also dump a JSON sibling with full eval_results for inspection
    json_dir = out_path.parent / 'json_per_run'
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f'{args.method}_{args.dataset}_{args.perturbation}_{args.level}.json'
    json_path.write_text(json.dumps({**row, 'eval_results': flat}, indent=2))

    print(f'[robust] DONE mAP@.25={map25} mAP@.50={map50} '
          f'elapsed={elapsed:.0f}s -> {out_path}')


if __name__ == '__main__':
    main()
