# Ablation B: single-stage fusion at layer1 only (early). Multi-scale and gate kept.
# Question: is dual-stage injection necessary, or does early-only suffice?
_base_ = ['../tr3d-ff_crossattn_dual_sunrgbd.py']
model = dict(
    fusion_stage='early',
    use_multiscale=True,
)
