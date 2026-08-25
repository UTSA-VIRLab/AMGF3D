_base_ = ['./tr3d.py', '../../../configs/_base_/datasets/scannet-3d-18class.py']
# TR3DPointSample is now registered in mmdet3d.datasets.pipelines - no custom_imports needed
# TR3DMinkResNet is replaced with MinkResNet (max_channels=128 gives same output channels)
# Pack3DDetInputs replaced with DefaultFormatBundle3D + Collect3D (older API)

model = dict(
    # Override head configuration for ScanNet 18 classes
    # CRITICAL FIX: label2level must be inside assigner dict, NOT directly in head dict
    # TR3DHead.__init__() does NOT accept label2level - only TR3DAssigner.__init__() does
    head=dict(
        n_classes=18,  # ScanNet has 18 classes (override base config)
        assigner=dict(  # Override assigner to add label2level for ScanNet
            # Base config has: type='TR3DAssigner', top_pts_threshold=6
            # We merge by including all required fields explicitly
            type='TR3DAssigner',
            top_pts_threshold=6,
            label2level=[0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0])))  # ScanNet 18 classes -> 2 levels

dataset_type = 'ScanNetDataset'
data_root = './data/scannet/'
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

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
    # We do not sample 100k points for ScanNet, as very few scenes have
    # significantly more then 100k points. So we sample 33 to 100% of them.
    # FIX: Use TR3DPointSample (random 33-100% sampling) instead of PointSample (fixed 33%)
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
    # FIX: Use older API format (Pack3DDetInputs doesn't exist)
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
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # We do not sample 100k points for ScanNet, as very few scenes have
            # significantly more then 100k points. So it doesn't affect
            # inference time and we can accept all points.
            dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

# FIX: Use older API format (data dict instead of train_dataloader/val_dataloader)
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,  # Match TR3D official config
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
