# Experiment: 3-Level TR3D (skips only level 0, keeps level 3)
# This is the RECOMMENDED configuration for better performance than 4-level
#
# Rationale:
# - Level 0 (finest) consumes 1/3 of memory but adds noise
# - Level 3 (coarsest) is useful for large objects like beds
# - This configuration saves memory while keeping large object detection
#
# Architecture:
#   Backbone: [f0, f1, f2, f3] (4 levels, 128ch cap)
#        ↓
#   Neck: Skip f0, process f1→f2→f3
#        ↓
#   Head: 3 outputs (levels 1, 2, 3 mapped to indices 0, 1, 2)

voxel_size = .01
n_points = 100000

model = dict(
    type='MinkSingleStage3DDetector',
    voxel_size=voxel_size,
    backbone=dict(
        type='MinkResNet',
        in_channels=3,
        depth=34,
        max_channels=128,
        norm='batch'),
    neck=dict(
        type='TR3DNeck3Level',  # NEW: 3-level neck
        in_channels=(64, 128, 128, 128),
        out_channels=128),
    head=dict(
        type='TR3DHead',
        in_channels=128,
        n_reg_outs=8,  # Rotated boxes for SUN RGB-D
        n_classes=10,
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            # 3-level mapping (output indices 0, 1, 2 = backbone levels 1, 2, 3)
            # Level 0 (output idx 0, backbone level 1): medium objects
            # Level 1 (output idx 1, backbone level 2): large objects
            # Level 2 (output idx 2, backbone level 3): very large objects
            #
            # Classes: bed, table, sofa, chair, toilet, desk, dresser, night_stand, bookshelf, bathtub
            # bed: very large → 2
            # table: large → 1
            # sofa: large → 1
            # chair: medium → 0
            # toilet: medium → 0
            # desk: large → 1
            # dresser: medium-large → 1
            # night_stand: medium → 0
            # bookshelf: very large → 2
            # bathtub: large → 1
            label2level=[2, 1, 1, 0, 0, 1, 1, 0, 2, 1]),
        bbox_loss=dict(type='RotatedIoU3DLoss', reduction='none')),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))

optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='PointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=.5,
        flip_ratio_bev_vertical=.0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.523599, .523599],
        scale_ratio_range=[.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointSample', num_points=n_points),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            modality=dict(use_camera=False, use_lidar=True),
            data_root=data_root,
            ann_file=data_root + 'sunrgbd_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
