# TR3D Official Config - Fixed for compatibility
# This is a standalone version that works without base config namespace resolution
# Based on: projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py

_base_ = ['../../../configs/_base_/default_runtime.py']
# TR3DPointSample is now registered in mmdet3d.datasets.pipelines
# No need for custom_imports
# custom_imports = dict(imports=['projects.TR3D.tr3d'])

# Model definition
model = dict(
    type='MinkSingleStage3DDetector',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='TR3DMinkResNet',
        in_channels=3,
        depth=34,
        norm='batch',
        num_planes=(64, 128, 128, 128)),
    neck=dict(
        type='TR3DNeck', 
        in_channels=(64, 128, 128, 128), 
        out_channels=128),
    bbox_head=dict(
        type='TR3DHead',
        in_channels=128,
        voxel_size=0.01,
        pts_center_threshold=6,
        num_reg_outs=6,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            label2level=[0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0]),
        bbox_loss=dict(type='AxisAlignedIoULoss', mode='diou', reduction='none')),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0.01))

# Optimizer and training schedule
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=12,
    by_epoch=True,
    milestones=[8, 11],
    gamma=0.1)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Dataset configuration
dataset_type = 'ScanNetDataset'
data_root = './data/scannet/'
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

# Training pipeline - Uses TR3DPointSample (random 33-100% sampling)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='GlobalAlignment', rotation_axis=2),
    # CRITICAL: TR3DPointSample samples random fraction from 33% to 100%
    dict(type='TR3DPointSample', num_points=0.33),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.02, 0.02],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    # Use older API format for compatibility
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# Test pipeline
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # No point sampling for test - use all points
            dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

# Data configuration - Use older format for compatibility
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=15,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
