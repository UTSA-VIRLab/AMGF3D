# Final model (single-scale): drop multi-scale FPN aggregation; keep gate + dual-stage.
# ARKitScenes counterpart of ablation_no_multiscale_sunrgbd.py.
_base_ = ['../tr3d-ff_crossattn_dual_arkitscenes.py']
model = dict(
    use_multiscale=False,
    fusion_stage='dual',
)
