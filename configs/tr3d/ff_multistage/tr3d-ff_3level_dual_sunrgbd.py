# TR3D+FF 3-Level with Dual Fusion (Layer 1 + Layer 3)
_base_ = ['./tr3d-ff_3level_baseline_sunrgbd.py']
model = dict(fusion_mode='dual')
