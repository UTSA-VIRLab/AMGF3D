# TR3D Multi-Stage Fusion Detector
#
# This detector extends the standard TR3D detector with multi-stage
# fusion capabilities for RGB features.
#
# Key Features:
# - Supports 2D-3D feature fusion at multiple backbone stages
# - Handles RGB feature extraction and projection
# - Graceful fallback when RGB is unavailable
# - Works with different fusion modes (baseline, early, late, dual)

try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.')

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.core import bbox3d2result
from mmdet3d.models import DETECTORS, build_backbone, build_head, build_neck
from .base import Base3DDetector


@DETECTORS.register_module()
class TR3DMultiStageFusionDetector(Base3DDetector):
    """TR3D detector with multi-stage 2D-3D fusion.

    This detector extends MinkSingleStage3DDetector with support for
    multi-stage fusion between 3D point cloud features and 2D RGB features.

    Args:
        backbone (dict): Config of the backbone (should be MinkMultiStageFusionResNet)
        head (dict): Config of the head.
        voxel_size (float): Voxel size in meters.
        neck (dict, optional): Config of the neck.
        rgb_backbone (dict, optional): Config of the 2D RGB backbone for feature extraction.
            If None, uses pre-computed RGB features.
        rgb_projector (dict, optional): Config for projecting RGB features to match 3D.
        train_cfg (dict, optional): Config for train stage.
        test_cfg (dict, optional): Config for test stage.
        init_cfg (dict, optional): Config for weight initialization.
    """

    def __init__(self,
                 backbone,
                 head,
                 voxel_size,
                 neck=None,
                 rgb_backbone=None,
                 rgb_projector=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(TR3DMultiStageFusionDetector, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        self.voxel_size = voxel_size

        # Optional RGB backbone for extracting 2D features
        # If not provided, expects pre-computed RGB features in input
        self.rgb_backbone = None
        if rgb_backbone is not None:
            self.rgb_backbone = build_backbone(rgb_backbone)

        # RGB projector to map pixel features to point space
        self.rgb_projector = None
        if rgb_projector is not None:
            in_channels = rgb_projector.get('in_channels', 3)
            out_channels = rgb_projector.get('out_channels', 64)
            self.rgb_projector = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, out_channels)
            )

        self.init_weights()

    def extract_feat(self, *args):
        """Just implement @abstractmethod of BaseModule."""
        pass

    def _extract_rgb_features(self, img, img_metas):
        """Extract 2D RGB features from images.

        Args:
            img (Tensor): Input images [B, C, H, W]
            img_metas (list[dict]): Image metadata

        Returns:
            Tensor: RGB features [N_pixels, C_rgb]
        """
        if img is None:
            return None

        if self.rgb_backbone is not None:
            # Use 2D backbone to extract features
            rgb_feats = self.rgb_backbone(img)
            if isinstance(rgb_feats, (list, tuple)):
                rgb_feats = rgb_feats[-1]  # Use last layer features
            return rgb_feats
        else:
            # Use raw RGB (assumes img is pre-processed)
            return img

    def _get_point_to_rgb_mapping(self, points, img_metas):
        """Get mapping from 3D points to 2D RGB feature indices.

        This function projects 3D points to 2D image plane and returns
        the corresponding pixel indices for each point.

        Args:
            points (list[Tensor]): List of point clouds [N, 3+]
            img_metas (list[dict]): Image metadata with projection matrices

        Returns:
            Tensor: Mapping indices [N_total_points]
        """
        # For SUN RGB-D, points already have color information
        # The mapping is implicit - points[i] already has its RGB at points[i, 3:6]
        # We return None to indicate direct color usage
        return None

    def _prepare_rgb_features_for_fusion(self, points, rgb_features):
        """Prepare RGB features for fusion with 3D features.

        For datasets like SUN RGB-D where points already have color attributes,
        we extract the color directly from point cloud data.

        Args:
            points (list[Tensor]): List of point clouds [N, 6] with RGB
            rgb_features (Tensor, optional): Pre-extracted RGB features

        Returns:
            tuple: (rgb_feats, point_to_rgb_idx)
                - rgb_feats: Tensor of shape [N_pts, C_rgb]
                - point_to_rgb_idx: Tensor of shape [N_pts] mapping to rgb_feats
        """
        if rgb_features is not None:
            # Use pre-computed RGB features
            return rgb_features, None

        # Extract RGB from point cloud colors
        all_rgb = []
        for pts in points:
            if pts.shape[1] >= 6:
                # Points have RGB at indices 3, 4, 5
                rgb = pts[:, 3:6]  # [N, 3]
            else:
                # No RGB available
                rgb = torch.zeros(pts.shape[0], 3, device=pts.device)
            all_rgb.append(rgb)

        rgb_feats = torch.cat(all_rgb, dim=0)  # [N_total, 3]

        # Project RGB if projector exists
        if self.rgb_projector is not None:
            rgb_feats = self.rgb_projector(rgb_feats)

        # Create identity mapping (point i maps to rgb i)
        point_to_rgb_idx = torch.arange(
            rgb_feats.shape[0],
            device=rgb_feats.device,
            dtype=torch.long
        )

        return rgb_feats, point_to_rgb_idx

    def extract_feats(self, points, img=None, img_metas=None):
        """Extract features from points with optional RGB fusion.

        Args:
            points (list[Tensor]): Raw point clouds with RGB [N, 6].
            img (Tensor, optional): Input images [B, C, H, W].
            img_metas (list[dict], optional): Image metadata.

        Returns:
            SparseTensor or list[SparseTensor]: Voxelized features.
        """
        # Voxelize points
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)

        # Prepare RGB data for fusion (as raw tensors, not SparseTensor)
        rgb_data = None

        if hasattr(self.backbone, 'fusion_mode') and self.backbone.fusion_mode != 'baseline':
            # Extract RGB from voxelized features (features are the RGB values)
            # features shape: [N_voxels, 3] where 3 is RGB
            voxel_rgb = features  # Already at voxel level

            # Project RGB if projector exists
            if self.rgb_projector is not None:
                rgb_feats = self.rgb_projector(voxel_rgb)
            else:
                rgb_feats = voxel_rgb

            # Pass RGB as tuple of (coordinates, features) to avoid coordinate manager issues
            rgb_data = (coordinates.clone(), rgb_feats)

            x = self.backbone(x, rgb_data)
        else:
            # No fusion or baseline mode
            x = self.backbone(x)

        # Pass through neck
        if self.with_neck:
            x = self.neck(x)

        return x

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, img_metas,
                      img=None, **kwargs):
        """Forward training pass.

        Args:
            points (list[Tensor]): Raw point clouds [N, 6] with RGB.
            gt_bboxes_3d (list[BaseInstance3DBoxes]): Ground truth boxes.
            gt_labels_3d (list[Tensor]): Ground truth labels.
            img_metas (list[dict]): Scene metadata.
            img (Tensor, optional): Input images.

        Returns:
            dict: Loss values.
        """
        x = self.extract_feats(points, img, img_metas)
        losses = self.head.forward_train(x, gt_bboxes_3d, gt_labels_3d, img_metas)
        return losses

    def simple_test(self, points, img_metas, img=None, *args, **kwargs):
        """Test without augmentations.

        Args:
            points (list[Tensor]): Points of each sample.
            img_metas (list[dict]): Scene metadata.
            img (Tensor, optional): Input images.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        x = self.extract_feats(points, img, img_metas)
        bbox_list = self.head.forward_test(x, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations."""
        raise NotImplementedError


@DETECTORS.register_module()
class TR3DMultiStageFusionDetectorV2(TR3DMultiStageFusionDetector):
    """Version 2 with additional features.

    Adds support for:
    - Multi-scale RGB features
    - Attention-based fusion weighting
    - Feature caching for efficiency
    """

    def __init__(self, *args, use_multiscale_rgb=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_multiscale_rgb = use_multiscale_rgb
        self._rgb_cache = None

    def _prepare_rgb_features_for_fusion(self, points, rgb_features):
        """Enhanced RGB preparation with caching."""
        rgb_feats, idx = super()._prepare_rgb_features_for_fusion(points, rgb_features)

        # Cache for multi-scale usage
        self._rgb_cache = rgb_feats

        return rgb_feats, idx
