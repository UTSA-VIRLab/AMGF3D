# Base TR3D model for ARKitScenes dataset
# Similar structure to tr3d_sunrgbd-3d-10class.py
# TR3DPointSample is now registered in mmdet3d.datasets.pipelines.transforms_3d

voxel_size = .01
n_points = 100000

model = dict(
    type='MinkSingleStage3DDetector',
    voxel_size=voxel_size,
    backbone=dict(type='MinkResNet', in_channels=3, max_channels=128, depth=34, norm='batch'),
    neck=dict(
        type='TR3DNeck',
        in_channels=(64, 128, 128, 128),
        out_channels=128),
    head=dict(
        type='TR3DHead',
        in_channels=128,
        n_reg_outs=6,  # Axis-aligned boxes (ARKitScenes)
        n_classes=17,  # ARKitScenes has 17 classes
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            # ARKitScenes label2level: Assign classes to 2 levels (0=large, 1=small)
            # Based on typical object sizes: large (cabinet, bed, sofa, table, refrigerator, bathtub) -> 0
            # Small objects (others) -> 1
            # Classes: cabinet, refrigerator, shelf, stove, bed, sink, washer, toilet, bathtub,
            #          oven, dishwasher, fireplace, stool, chair, table, tv_monitor, sofa
            label2level=[0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0]),
        bbox_loss=dict(type='AxisAlignedIoULoss', mode='diou', reduction='none')),
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
        # dict(type='TensorboardLoggerHook')
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# ARKitScenes Dataset Configuration
dataset_type = 'ARKitScenesDataset'
data_root = './data/arkitscenes/'
class_names = ('cabinet', 'refrigerator', 'shelf', 'stove', 'bed',
               'sink', 'washer', 'toilet', 'bathtub', 'oven',
               'dishwasher', 'fireplace', 'stool', 'chair', 'table',
               'tv_monitor', 'sofa')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    # ARKitScenes doesn't use GlobalAlignment (no axis alignment like ScanNet)
    dict(type='PointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=.5,
        flip_ratio_bev_vertical=.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.02, .02],  # Same as ScanNet/SUN RGB-D
        scale_ratio_range=[.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
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
            dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=16,  # Same as ScanNet/SUN RGB-D
    workers_per_gpu=8,  # Match TR3D official config
    train=dict(
        type='RepeatDataset',
        times=5,  # ARKitScenes typically uses 5x (similar to SUN RGB-D)
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'arkitscenes_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'arkitscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'arkitscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
