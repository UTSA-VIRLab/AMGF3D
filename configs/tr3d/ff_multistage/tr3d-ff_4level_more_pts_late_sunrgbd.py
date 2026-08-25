# TR3D+FF 4-Level More Points with Late Fusion (after Layer 3)
_base_ = ['./tr3d-ff_4level_more_pts_baseline_sunrgbd.py']
model = dict(fusion_mode='late')
