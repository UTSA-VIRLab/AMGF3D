# Geometry-only-residual baseline (ARKitScenes), requested by BMVC'26 reviewer R1-W3:
# full AMGF3D architecture (multi-scale + gate + dual-stage) but every projected 2D
# descriptor zeroed, so the model keeps the added capacity while receiving no appearance
# information. If its accuracy falls back to geometry-only TR3D rather than tracking full
# AMGF3D, the fusion gains are appearance-driven, not a parameter-count artifact.
# Used two ways: (a) trained from scratch (seeds 0-2); (b) as an eval config on the
# trained FULL checkpoints for the test-time zero-image control.
_base_ = ['../tr3d-ff_crossattn_dual_arkitscenes.py']
model = dict(
    zero_image_features=True,
)
