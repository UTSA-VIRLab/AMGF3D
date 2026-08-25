# Experiment: 4-Level TR3D with REBALANCED label2level
# Hypothesis: Better distribution of classes across levels improves performance
#
# SUN RGB-D classes and their typical sizes:
# bed(0): very large → 3
# table(1): large → 2
# sofa(2): large → 2
# chair(3): small → 0  (moved from 1)
# toilet(4): small → 0  (moved from 1)
# desk(5): medium → 1  (moved from 2)
# dresser(6): medium → 1  (moved from 2)
# night_stand(7): small → 0  (moved from 1)
# bookshelf(8): very large → 3
# bathtub(9): large → 2

_base_ = ['../tr3d_4level_sunrgbd-3d-10class.py']

model = dict(
    neck=dict(
        pts_threshold=-1),  # Also disable pruning
    head=dict(
        assigner=dict(
            # Rebalanced: use ALL 4 levels
            # Level 0: chair, toilet, night_stand (3 classes) - small
            # Level 1: desk, dresser (2 classes) - medium
            # Level 2: table, sofa, bathtub (3 classes) - large
            # Level 3: bed, bookshelf (2 classes) - very large
            label2level=[3, 2, 2, 0, 0, 1, 1, 0, 3, 2]),
    ),
)
