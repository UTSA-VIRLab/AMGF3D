# Experiment: 4-Level TR3D with HIGHER top_pts_threshold
# Hypothesis: With 4 levels, points are spread thinner, need more positives per box

_base_ = ['../tr3d_4level_sunrgbd-3d-10class.py']

model = dict(
    neck=dict(
        pts_threshold=-1),  # Also disable pruning
    head=dict(
        assigner=dict(
            top_pts_threshold=12,  # Doubled from 6
            label2level=[3, 2, 2, 0, 0, 1, 1, 0, 3, 2]),  # Also use rebalanced mapping
    ),
)
