# Stage ablation (ARKitScenes): EARLY-only fusion (layer1 only). Keep multi-scale + gate.
# Mirrors ablation_early_only_sunrgbd.py; isolates dual-stage placement's contribution to
# ROBUSTNESS on ARKitScenes (where the full model shows the significant calib/dropout gains).
# Train over the same seeds as the full model, then eval clean + under perturbation and compare.
_base_ = ['../tr3d-ff_crossattn_dual_arkitscenes.py']
model = dict(
    fusion_stage='early',
    use_multiscale=True,
    use_gate=True,
)
