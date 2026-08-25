# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from os import path as osp

from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.core.visualizer import show_result_v2
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class ARKitScenesDataset(Custom3DDataset):
    """ARKitScenes Dataset for 3D Object Detection.

    This class serves as the API for ARKitScenes 3DOD dataset.
    Similar to ScanNet but with different data organization.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used.
            Defaults to None.
        box_type_3d (str, optional): Type of 3D box. Defaults to 'Depth'.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether in test mode. Defaults to False.
    """
    CLASSES = ('cabinet', 'refrigerator', 'shelf', 'stove', 'bed',
               'sink', 'washer', 'toilet', 'bathtub', 'oven',
               'dishwasher', 'fireplace', 'stool', 'chair', 'table',
               'tv_monitor', 'sofa')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=True, use_lidar=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert 'use_camera' in self.modality and 'use_lidar' in self.modality
        assert self.modality['use_camera'] or self.modality['use_lidar']

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_lidar']:
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename

        if self.modality['use_camera']:
            img_paths = info.get('img_paths', [])
            if len(img_paths) > 0:
                # Use first image for now (can be extended for multi-view)
                img_info = dict(filename=osp.join(self.data_root, img_paths[0]))
                input_dict['img_prefix'] = None
                input_dict['img_info'] = img_info

                # Camera parameters
                intrinsics = info.get('intrinsics', np.eye(4))
                extrinsics = info.get('extrinsics', [])
                if len(extrinsics) > 0:
                    depth2img = intrinsics @ np.linalg.inv(extrinsics[0])
                    input_dict['depth2img'] = depth2img
            else:
                # No images available - set placeholder for pipeline compatibility
                input_dict['img_prefix'] = None
                input_dict['img_info'] = dict(filename='')

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information.
        """
        info = self.data_infos[index]

        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(np.float32)
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0,), dtype=np.int64)

        # Convert to DepthInstance3DBoxes (axis-aligned, no yaw)
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        axis_align_matrix = info['annos'].get(
            'axis_align_matrix', np.eye(4, dtype=np.float32))

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            axis_align_matrix=axis_align_matrix)
        return anns_results

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        input_dict['ann_info'] = dict(
            axis_align_matrix=self.data_infos[index]['annos'].get(
                'axis_align_matrix', np.eye(4, dtype=np.float32)))
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def _build_default_pipeline(self):
        """Build the default pipeline for visualization."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            # Sample points for visualization (all points too much)
            dict(type='PointSample', num_points=200000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._build_default_pipeline() if pipeline is None else Compose(pipeline)
        
        for i, result in enumerate(results):
            # Handle nested results (e.g., 'pts_bbox')
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.splitext(osp.basename(pts_path))[0]
            
            # Extract points (load_annos=True to ensure we have annotations)
            points = self._extract_data(i, pipeline, 'points', load_annos=True).numpy()
            
            # Extract ground truth bounding boxes
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_corners = gt_bboxes.corners.numpy() if len(gt_bboxes) > 0 else None
            # gt_labels_3d is already a numpy array from get_ann_info (no .numpy() needed)
            gt_labels = self.get_ann_info(i)['gt_labels_3d'] if gt_corners is not None else None
            
            # Extract predicted bounding boxes
            pred_bboxes = result['boxes_3d']
            pred_corners = pred_bboxes.corners.numpy() if len(pred_bboxes) > 0 else None
            # pred_labels from result might be numpy or torch tensor - convert if needed
            pred_labels_raw = result['labels_3d']
            if pred_corners is not None:
                # Check if it's a torch tensor and convert to numpy
                if hasattr(pred_labels_raw, 'cpu'):  # torch tensor
                    pred_labels = pred_labels_raw.cpu().numpy()
                elif isinstance(pred_labels_raw, np.ndarray):  # already numpy
                    pred_labels = pred_labels_raw
                else:
                    pred_labels = np.array(pred_labels_raw)
            else:
                pred_labels = None
            
            # Visualize with show_result_v2 (similar to ScanNet/S3DIS)
            # Enable show=True and snapshot=True to generate PNG images
            show_result_v2(
                points,
                gt_corners,
                gt_labels,
                pred_corners,
                pred_labels,
                out_dir,
                file_name,
                show=show,
                snapshot=True)  # Always save PNG images
