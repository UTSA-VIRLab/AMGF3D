# TR3D+FF 4-Level More Points with Dual Fusion (Layer 1 + Layer 3)
_base_ = ['./tr3d-ff_4level_more_pts_baseline_sunrgbd.py']
model = dict(fusion_mode='dual')
