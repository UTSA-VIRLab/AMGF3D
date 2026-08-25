# Ablation A: drop the multi-scale FPN aggregation; keep gate and dual-stage.
# Uses CrossAttentionFusion (single-scale) at both layer1 and layer3.
# Question: does multi-scale FPN aggregation matter, or is single-scale fusion enough?
_base_ = ['../tr3d-ff_crossattn_dual_sunrgbd.py']
model = dict(
    use_multiscale=False,
    fusion_stage='dual',
)
