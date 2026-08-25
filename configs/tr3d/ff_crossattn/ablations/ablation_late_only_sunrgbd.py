# Ablation C: single-stage fusion at layer3 only (late). Multi-scale and gate kept.
# Question: is dual-stage injection necessary, or does late-only suffice?
_base_ = ['../tr3d-ff_crossattn_dual_sunrgbd.py']
model = dict(
    fusion_stage='late',
    use_multiscale=True,
)
