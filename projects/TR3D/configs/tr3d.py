_base_ = ['../../../configs/_base_/default_runtime.py']
# TR3DPointSample is now in mmdet3d.datasets.pipelines - no custom_imports needed
# TR3DNeck and TR3DHead are already in mmdet3d.models - no custom_imports needed
# TR3DMinkResNet replaced with MinkResNet (max_channels=128 gives same output channels)
# custom_imports = dict(imports=['projects.TR3D.tr3d'])  # Not needed - causes registry import error

model = dict(
    type='MinkSingleStage3DDetector',
    voxel_size=0.01,
    # FIX: Use MinkResNet instead of TR3DMinkResNet (same output channels with max_channels)
    backbone=dict(
        type='MinkResNet',
        in_channels=3,
        depth=34,
        norm='batch',
        max_channels=128),  # Outputs (64, 128, 128, 128) - same as TR3DMinkResNet with num_planes=(64, 128, 128, 128)
    neck=dict(
        type='TR3DNeck', in_channels=(64, 128, 128, 128), out_channels=128),
    head=dict(  # FIX: Use 'head' not 'bbox_head' (older API)
        type='TR3DHead',
        in_channels=128,
        n_reg_outs=6,
        n_classes=18,  # Will be overridden by dataset config
        voxel_size=0.01,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6),  # label2level will be overridden by dataset-specific config
        bbox_loss=dict(type='AxisAlignedIoULoss', mode='diou', reduction='none')),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0.01))

# FIX: Use older API format (optimizer/optimizer_config/lr_config/runner)
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
