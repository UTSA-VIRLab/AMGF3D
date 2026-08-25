# TR3D for ARKitScenes - Point Cloud Only (No Fusion)
# This is a stable baseline to verify the model works on ARKitScenes
#
# ARKitScenes: 17 classes

voxel_size = .01
n_points = 100000

model = dict(
    type='MinkSingleStage3DDetector',
    backbone=dict(
        type='MinkResNet',
        in_channels=3,
        max_channels=128,
        depth=34,
        norm='batch'),
    neck=dict(
        type='TR3DNeck',
        in_channels=(64, 128, 128, 128),
        out_channels=128),
    head=dict(
        type='TR3DHead',
        in_channels=128,
        n_reg_outs=6,
        n_classes=17,
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            # All classes at level 0 for simplicity
            label2level=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Standard IoU loss (stable)
        bbox_loss=dict(type='AxisAlignedIoULoss', mode='iou', reduction='none')),
    voxel_size=voxel_size,
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))

# Conservative training settings
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
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

# Point cloud only pipeline (no images)
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False,
         use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='PointSample', num_points=n_points),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=.5, flip_ratio_bev_vertical=.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-.087266, .087266],
         scale_ratio_range=[.95, 1.05], translation_std=[.1, .1, .1], shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False,
         use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='MultiScaleFlipAug3D', img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='PointSample', num_points=n_points),
             dict(type='NormalizePointsColor', color_mean=None),
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='Collect3D', keys=['points'])
         ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            modality=dict(use_camera=False, use_lidar=True),
            data_root=data_root,
            ann_file=data_root + 'arkitscenes_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'arkitscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'arkitscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))

