# Ablation D: drop the consistency-aware gate; keep multi-scale and dual-stage.
# Replaces `g * fusion(...)` with `fusion(...)` in MultiScaleFusion.
# Question: does the per-voxel adaptive gate matter, or is unweighted multi-scale fusion enough?
_base_ = ['../tr3d-ff_crossattn_dual_sunrgbd.py']
model = dict(
    use_gate=False,
    use_multiscale=True,
    fusion_stage='dual',
)
