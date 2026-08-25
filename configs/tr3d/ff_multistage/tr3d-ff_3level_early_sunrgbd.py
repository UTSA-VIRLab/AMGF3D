# TR3D+FF 3-Level with Early Fusion (after Layer 1)
_base_ = ['./tr3d-ff_3level_baseline_sunrgbd.py']
model = dict(fusion_mode='early')
