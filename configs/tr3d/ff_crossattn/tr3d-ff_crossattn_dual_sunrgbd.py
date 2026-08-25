# TR3D+FF with Cross-Attention Fusion (Dual Stage - Early + Late)
_base_ = ['./tr3d-ff_crossattn_late_sunrgbd.py']
model = dict(fusion_stage='dual')  # Fuse at both layer1 and layer3
