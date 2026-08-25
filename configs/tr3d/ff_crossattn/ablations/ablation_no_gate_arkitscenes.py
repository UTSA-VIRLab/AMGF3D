# Gate ablation (ARKitScenes): drop the feature-conditioned gate; keep multi-scale + dual-stage.
# Sets use_gate=False, so MultiScaleFusion uses `feat_3d + 0.1 * fusion(...)` instead of
# `feat_3d + 0.1 * gate * fusion(...)` (see tr3d_ff_crossattn.py, MultiScaleFusion.forward).
#
# Purpose: isolate the per-voxel gate's contribution to ROBUSTNESS. ARKitScenes is where the
# full model shows the significant gains under calibration rotation and RGB dropout
# (Sec. Robustness). Train this no-gate variant over the SAME training seeds as the full model,
# then evaluate clean and under each perturbation; compare against the full AMGF3D to show
# whether the gate is responsible for the graceful degradation claimed in Sec. Methodology.
_base_ = ['../tr3d-ff_crossattn_dual_arkitscenes.py']
model = dict(
    use_gate=False,
    use_multiscale=True,
    fusion_stage='dual',
)
