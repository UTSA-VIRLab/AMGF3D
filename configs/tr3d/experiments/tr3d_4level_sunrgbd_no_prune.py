# Experiment: 4-Level TR3D WITHOUT pruning
# Hypothesis: Pruning removes potential positive samples, hurting training

_base_ = ['../tr3d_4level_sunrgbd-3d-10class.py']

model = dict(
    neck=dict(
        pts_threshold=-1),  # DISABLE PRUNING
)
