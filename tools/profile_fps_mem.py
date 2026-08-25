# Paired FPS + peak-GPU-memory profiler for AMGF3D vs TR3D+FF.
# Mirrors tools/analysis_tools/benchmark.py loading so FPS and memory come from
# one consistent run on the same hardware (samples_per_gpu=1, batch-1 inference).
import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


def parse_args():
    p = argparse.ArgumentParser(description='Paired FPS + peak-memory profiler')
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--samples', type=int, default=300)
    p.add_argument('--warmup', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False)

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    pure_inf_time = 0.0
    counted = 0
    torch.cuda.empty_cache()

    for i, data in enumerate(data_loader):
        if i == args.warmup:
            # start clean peak-memory accounting after warmup
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if i >= args.warmup:
            pure_inf_time += elapsed
            counted += 1
        if (i + 1) == args.samples:
            break

    fps = counted / pure_inf_time if pure_inf_time > 0 else float('nan')
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_resv = torch.cuda.max_memory_reserved() / (1024 ** 3)
    print('==== PROFILE RESULT ====')
    print(f'config         : {args.config}')
    print(f'checkpoint     : {args.checkpoint}')
    print(f'samples counted: {counted} (warmup {args.warmup})')
    print(f'FPS            : {fps:.2f} img/s')
    print(f'peak allocated : {peak_alloc:.3f} GB')
    print(f'peak reserved  : {peak_resv:.3f} GB')


if __name__ == '__main__':
    main()
