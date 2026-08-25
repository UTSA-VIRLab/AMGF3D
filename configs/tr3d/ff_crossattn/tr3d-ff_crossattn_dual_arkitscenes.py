# TR3D+FF with Cross-Attention Fusion for ARKitScenes
# ARKitScenes has RGB images + 3D point clouds, enabling cross-attention fusion.
#
# ARKitScenes: 17 classes
# cabinet, refrigerator, shelf, stove, bed, sink, washer, toilet, bathtub,
# oven, dishwasher, fireplace, stool, chair, table, tv_monitor, sofa

voxel_size = .01
n_points = 100000

model = dict(
    type='TR3DFFCrossAttn3DDetector',
    fusion_stage='dual',            # Fuse at both layer1 and layer3
    use_multiscale=True,            # Use all 4 FPN levels
    num_heads=4,                    # Multi-head attention
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
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
        n_reg_outs=6,  # Axis-aligned boxes
        n_classes=17,  # ARKitScenes has 17 classes
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            # label2level for 17 classes (0=large, 1=medium based on typical sizes)
            # cabinet, refrigerator, shelf, stove, bed, sink, washer, toilet, bathtub,
            # oven, dishwasher, fireplace, stool, chair, table, tv_monitor, sofa
            # Aligned with TR3D+FF baseline (tr3d-ff_arkitscenes.py):
            label2level=[0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0]),
        bbox_loss=dict(type='AxisAlignedIoULoss', mode='diou', reduction='none')),
    voxel_size=voxel_size,
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))

optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
evaluation = dict(interval=1, save_best='mAP_0.25', rule='greater')
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210323_173222-cad62aeb.pth'
resume_from = None
workflow = [('train', 1)]

# ARKitScenes Dataset Configuration
dataset_type = 'ARKitScenesDataset'  # Custom dataset (or use ScanNetDataset with modifications)
data_root = './data/arkitscenes/'
class_names = ('cabinet', 'refrigerator', 'shelf', 'stove', 'bed',
               'sink', 'washer', 'toilet', 'bathtub', 'oven',
               'dishwasher', 'fireplace', 'stool', 'chair', 'table',
               'tv_monitor', 'sofa')

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False,
         use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(type='Resize', img_scale=[(256, 192), (320, 240), (384, 288)],
         multiscale_mode='value', keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PointSample', num_points=n_points),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=.5, flip_ratio_bev_vertical=.5),
    # Aligned with TR3D+FF baseline: rot ±30°, scale 0.85-1.15
    dict(type='GlobalRotScaleTrans', rot_range=[-.523599, .523599],
         scale_ratio_range=[.85, 1.15], translation_std=[.1, .1, .1], shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False,
         use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug3D', img_scale=(256, 192), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='Resize', multiscale_mode='value', keep_ratio=True),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='PointSample', num_points=n_points),
             dict(type='NormalizePointsColor', color_mean=None),
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='Collect3D', keys=['points', 'img'])
         ])
]

data = dict(
    samples_per_gpu=16,  # was 8; aligned with TR3D+FF baseline
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            modality=dict(use_camera=True, use_lidar=True),
            data_root=data_root,
            ann_file=data_root + 'arkitscenes_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=True, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'arkitscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=True, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'arkitscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
