# TR3D+FF 4-Level Rebalanced with Late Fusion (after Layer 3)
_base_ = ['./tr3d-ff_4level_rebalanced_baseline_sunrgbd.py']
model = dict(fusion_mode='late')
