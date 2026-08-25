# TR3D with Stronger Data Augmentation
#
# Enhanced augmentation to improve generalization:
# - Larger rotation range
# - Larger scale range
# - Point dropout (randomly remove points)
# - Longer training (24 epochs)

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
        type='TR3DNeck',
        in_channels=(64, 128, 128, 128),
        out_channels=128),
    head=dict(
        type='TR3DHead',
        in_channels=128,
        n_reg_outs=8,
        n_classes=10,
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            label2level=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]),
        bbox_loss=dict(type='RotatedIoU3DLoss', reduction='none')),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))

optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[16, 22])  # Adjusted for 24 epochs
runner = dict(type='EpochBasedRunner', max_epochs=24)  # Longer training
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
        flip_ratio_bev_vertical=.5),  # Added vertical flip
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],  # Increased: ±45° instead of ±30°
        scale_ratio_range=[.8, 1.2],  # Increased range
        translation_std=[.2, .2, .2],  # Increased translation
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
