# TR3D+FF 4-Level Rebalanced with Dual Fusion (Layer 1 + Layer 3)
_base_ = ['./tr3d-ff_4level_rebalanced_baseline_sunrgbd.py']
model = dict(fusion_mode='dual')
