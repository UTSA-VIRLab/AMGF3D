# Zero-image control for the ADDITIVE baseline TR3D+FF (SUN RGB-D), BMVC'26 rebuttal.
# Companion to zero_image_sunrgbd.py: shows that appearance dependence at test time is a
# property of RGB-D fusion per se (both models collapse), so the question R1-W3 raises is
# not WHETHER images are used but whether AMGF3D's ADVANTAGE is capacity — it is not.
_base_ = ['../../tr3d-ff_sunrgbd-3d-10class.py']
model = dict(
    zero_image_features=True,
)
