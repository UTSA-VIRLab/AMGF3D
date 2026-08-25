# Geometry-only-residual control (SUN RGB-D), BMVC'26 rebuttal R1-W3 companion to
# zero_image_arkitscenes.py. SUN RGB-D is the dense-RGB-coverage regime (every point has a
# valid projection), so test-time appearance reliance is expected to be much higher than on
# scene-level ARKitScenes. Used as an eval config on trained FULL SUN checkpoints.
_base_ = ['../tr3d-ff_crossattn_dual_sunrgbd.py']
model = dict(
    zero_image_features=True,
)
