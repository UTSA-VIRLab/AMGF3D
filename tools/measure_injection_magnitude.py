#!/usr/bin/env python
# ============================================================================
# Measure the relative appearance-injection magnitude at the fusion stage(s):
#     r = mean_voxels  || f_out - f_in || / || f_in ||
# where f_in is the 3D feature entering a fusion module and f_out is the fused
# output. For AMGF3D this is exactly || lambda*g*Phi || / || f ||, i.e., how much
# the (bounded, gated) appearance residual moves the geometric feature; for the
# additive baseline it is the full-strength injection.
#
# LOCAL-SAFE: this is an architectural property (residual scaling), independent of
# the robustness AP, so it is valid to run on local checkpoints. Clean SUN only.
#
# Usage:
#   python tools/measure_injection_magnitude.py --config <cfg> --checkpoint <ckpt> [--n 30]
# ============================================================================
import argparse, numpy as np, torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_detector

FUSION_HINTS = ('fusion_early', 'fusion_late', '.fusion', 'crossattn', 'amgf', 'feature_fusion')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n', type=int, default=30, help='number of clean val samples')
    ap.add_argument('--list-modules', action='store_true', help='just print module names and exit')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None

    model = build_detector(cfg.model)
    if args.list_modules:
        for nm, _ in model.named_modules():
            if any(h in nm.lower() for h in FUSION_HINTS):
                print('  candidate:', nm)
        return
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0]); model.eval()

    stats = {}
    def mk_hook(nm):
        def hook(mod, inp, out):
            try:
                fin = inp[0]
                fout = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(fin) or not torch.is_tensor(fout):
                    return
                if fin.shape != fout.shape or fin.dim() != 2:
                    return
                res = (fout - fin).norm(dim=-1)
                base = fin.norm(dim=-1).clamp_min(1e-6)
                stats.setdefault(nm, []).append((res / base).mean().item())
            except Exception:
                pass
        return hook

    hooked = []
    for nm, mod in model.module.named_modules():
        low = nm.lower()
        # top-level fusion modules only (their output is the residual-added feature);
        # exclude sub-modules like 'fusion_early.fusion' (the inner Phi MLP).
        if nm and '.' not in nm and ('fusion' in low or low.endswith('crossattn') or low.endswith('amgf')):
            mod.register_forward_hook(mk_hook(nm)); hooked.append(nm)
    print('hooked fusion modules:', hooked if hooked else 'NONE (run with --list-modules to inspect)')

    ds = build_dataset(cfg.data.test)
    loader = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= args.n:
                break
            model(return_loss=False, rescale=True, **data)

    print(f'\n=== relative appearance-injection ||f_out - f_in|| / ||f_in|| (clean SUN, n<={args.n}) ===')
    allv = []
    for nm, v in stats.items():
        print(f'  {nm:28s}: {100*np.mean(v):5.1f}%   (samples={len(v)})')
        allv += v
    if allv:
        print(f'  {"OVERALL (mean over stages)":28s}: {100*np.mean(allv):5.1f}%')
    else:
        print('  no residual-form fusion captured; use --list-modules to find the right names')

if __name__ == '__main__':
    main()
