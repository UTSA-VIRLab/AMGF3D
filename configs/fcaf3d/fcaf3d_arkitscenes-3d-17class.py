_base_ = ['fcaf3d.py']
n_points = 100000

model = dict(
    head=dict(
        n_classes=17,  # ARKitScenes has 17 classes
        n_reg_outs=6,  # Axis-aligned boxes (like ScanNet, not rotated like SUN RGB-D)
        bbox_loss=dict(type='AxisAlignedIoULoss')))  # Use axis-aligned IoU loss

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
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.087266, 0.087266],  # ±5 degrees (similar to ScanNet)
        scale_ratio_range=[0.9, 1.1],
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
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(type='PointSample', num_points=n_points),
            dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=8,  # Same as SUN RGB-D/ScanNet
    workers_per_gpu=4,  # Same as SUN RGB-D/ScanNet
    train=dict(
        type='RepeatDataset',
        times=5,  # Similar to SUN RGB-D (3x) but more for ARKitScenes
        dataset=dict(
            type=dataset_type,
            modality=dict(use_camera=False, use_lidar=True),  # Point cloud only
            data_root=data_root,
            ann_file=data_root + 'arkitscenes_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
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
