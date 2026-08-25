# TR3D+FF Multi-Stage Fusion Detector
#
# This detector extends TR3D+FF with multi-stage fusion capabilities.
# It uses a pretrained 2D backbone (ResNet) to extract semantic image features
# and fuses them at multiple points in the 3D backbone.
#
# Fusion Modes:
# - baseline: Original TR3D+FF (single fusion after initial pooling)
# - early: Fuse after Layer 1 only
# - late: Fuse after Layer 3 only
# - dual: Fuse at both Layer 1 and Layer 3
#
# Key difference from our previous multi-stage fusion:
# - Uses ACTUAL 2D image features from pretrained ResNet (not just point cloud RGB)
# - This provides rich semantic features that complement 3D geometry

try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.')

import torch
from torch import nn
from functools import partial

from mmdet3d.core import bbox3d2result
from mmdet3d.models import DETECTORS, build_backbone, build_neck, build_head
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type
from .base import Base3DDetector


class DualStageCombiner(nn.Module):
    """Combiner module for dual-stage fusion.

    Combines information from early (layer1) and late (layer3) fusion stages
    to produce refined fusion outputs. Uses attention mechanism to weight
    the contribution of each stage.

    Args:
        early_channels (int): Channels after early fusion (64)
        late_channels (int): Channels after late fusion (128)
    """

    def __init__(self, early_channels=64, late_channels=128):
        super(DualStageCombiner, self).__init__()

        # Store early fusion state for later combination
        self.early_state = None

        # Project early features to late channel dimension for combination
        self.early_proj = nn.Sequential(
            ME.MinkowskiConvolution(early_channels, late_channels, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(late_channels),
            ME.MinkowskiReLU(inplace=True))

        # Attention weights for combining early and late
        self.attn_conv = nn.Sequential(
            ME.MinkowskiConvolution(late_channels * 2, late_channels, kernel_size=1, dimension=3),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(late_channels, 2, kernel_size=1, dimension=3),
            ME.MinkowskiSoftmax(dim=1))

        # Output refinement
        self.refine = nn.Sequential(
            ME.MinkowskiConvolution(late_channels, late_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(late_channels),
            ME.MinkowskiReLU(inplace=True))

    def store_early(self, x):
        """Store early fusion output for later combination."""
        self.early_state = x

    def combine(self, late_x):
        """Combine early and late fusion outputs.

        Args:
            late_x (SparseTensor): Features after late fusion

        Returns:
            SparseTensor: Combined features
        """
        if self.early_state is None:
            return late_x

        # Project early to match late channels
        early_proj = self.early_proj(self.early_state)

        # Need to handle coordinate mismatch between early and late
        # Late has lower resolution, so we use late coordinates as base
        # and sample early features at those locations
        early_feats_at_late = self._sample_at_coords(early_proj, late_x)

        # Concatenate for attention computation
        concat_feats = torch.cat([early_feats_at_late, late_x.F], dim=1)
        concat_sparse = ME.SparseTensor(
            concat_feats,
            coordinate_map_key=late_x.coordinate_map_key,
            coordinate_manager=late_x.coordinate_manager)

        # Compute attention weights
        attn = self.attn_conv(concat_sparse)  # [N, 2]

        # Weighted combination
        combined = attn.F[:, 0:1] * early_feats_at_late + attn.F[:, 1:2] * late_x.F

        combined_sparse = ME.SparseTensor(
            combined,
            coordinate_map_key=late_x.coordinate_map_key,
            coordinate_manager=late_x.coordinate_manager)

        # Refinement
        output = self.refine(combined_sparse)

        # Residual connection
        output = ME.SparseTensor(
            output.F + late_x.F,
            coordinate_map_key=late_x.coordinate_map_key,
            coordinate_manager=late_x.coordinate_manager)

        # Clear stored state
        self.early_state = None

        return output

    def _sample_at_coords(self, src_sparse, tgt_sparse):
        """Sample source features at target coordinates using nearest neighbor."""
        src_coords = src_sparse.C  # [N_src, 4]
        tgt_coords = tgt_sparse.C  # [N_tgt, 4]
        src_feats = src_sparse.F   # [N_src, C]

        device = tgt_coords.device
        n_tgt = tgt_coords.shape[0]
        n_channels = src_feats.shape[1]

        output_feats = torch.zeros(n_tgt, n_channels, device=device)

        # Process per batch
        batch_indices = tgt_coords[:, 0].unique()

        for batch_idx in batch_indices:
            tgt_mask = tgt_coords[:, 0] == batch_idx
            src_mask = src_coords[:, 0] == batch_idx

            tgt_xyz = tgt_coords[tgt_mask, 1:].float()
            src_xyz = src_coords[src_mask, 1:].float()
            src_f = src_feats[src_mask]

            if src_xyz.shape[0] == 0 or tgt_xyz.shape[0] == 0:
                continue

            # Find nearest neighbor
            dists = torch.cdist(tgt_xyz, src_xyz)
            min_indices = dists.argmin(dim=1)

            tgt_indices = torch.where(tgt_mask)[0]
            output_feats[tgt_indices] = src_f[min_indices]

        return output_feats


class MultiStageFusionModule(nn.Module):
    """Fusion module for a specific stage with gating and learnable mixing.

    Args:
        img_channels (int): Number of 2D image feature channels (e.g., 256)
        sparse_channels (int): Number of 3D sparse feature channels
        use_gating (bool): Whether to use reliability gating
        gate_threshold (float): Threshold for gating
    """

    def __init__(self, img_channels, sparse_channels, use_gating=True, gate_threshold=0.1):
        super(MultiStageFusionModule, self).__init__()

        self.use_gating = use_gating
        self.gate_threshold = gate_threshold

        # Project 2D image features to match 3D channels
        self.img_proj = nn.Sequential(
            ME.MinkowskiConvolution(img_channels, sparse_channels, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(sparse_channels),
            ME.MinkowskiReLU(inplace=True))

        # Learnable mixing coefficient
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Gating network (operates on projected features)
        if use_gating:
            self.gate_conv = nn.Sequential(
                ME.MinkowskiConvolution(sparse_channels, sparse_channels // 4, kernel_size=1, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(sparse_channels // 4, 1, kernel_size=1, dimension=3),
                ME.MinkowskiSigmoid())

    def forward(self, sparse_3d, img_sparse):
        """Forward fusion.

        Args:
            sparse_3d (SparseTensor): 3D sparse features
            img_sparse (SparseTensor): Projected 2D image features as SparseTensor

        Returns:
            SparseTensor: Fused sparse features
        """
        # Project image features to match 3D channels
        projected_img = self.img_proj(img_sparse)

        # Compute gating weight
        if self.use_gating:
            gate = self.gate_conv(projected_img)
            # Apply threshold
            gate_feats = torch.where(
                gate.F > self.gate_threshold,
                gate.F,
                torch.zeros_like(gate.F))
            gate = ME.SparseTensor(
                gate_feats,
                coordinate_map_key=gate.coordinate_map_key,
                coordinate_manager=gate.coordinate_manager)
        else:
            gate_feats = torch.ones(projected_img.F.shape[0], 1, device=projected_img.F.device)
            gate = ME.SparseTensor(
                gate_feats,
                coordinate_map_key=projected_img.coordinate_map_key,
                coordinate_manager=projected_img.coordinate_manager)

        # Clamp alpha
        alpha = torch.clamp(self.alpha, 0.0, 1.0)

        # Fused features: 3D + alpha * gate * projected_2D
        fused_feats = sparse_3d.F + alpha * gate.F * projected_img.F

        return ME.SparseTensor(
            fused_feats,
            coordinate_map_key=sparse_3d.coordinate_map_key,
            coordinate_manager=sparse_3d.coordinate_manager)


@DETECTORS.register_module()
class TR3DFFMultiStage3DDetector(Base3DDetector):
    """TR3D+FF with Multi-Stage Fusion.

    This detector extends TR3D+FF to support fusion at multiple backbone stages
    using pretrained 2D image features.

    Args:
        img_backbone (dict): Config of the 2D backbone.
        img_neck (dict): Config of the 2D neck.
        backbone (dict): Config of the 3D backbone.
        neck (dict): Config of the 3D neck.
        head (dict): Config of the 3D head.
        voxel_size (float): Voxel size in meters.
        fusion_mode (str): Fusion mode - 'baseline', 'early', 'late', or 'dual'.
        use_gating (bool): Whether to use reliability gating.
        gate_threshold (float): Threshold for gating.
        train_cfg (dict, optional): Config for train stage.
        test_cfg (dict, optional): Config for test stage.
        init_cfg (dict, optional): Config for weight initialization.
    """

    VALID_FUSION_MODES = ('baseline', 'early', 'late', 'dual')

    def __init__(self,
                 img_backbone,
                 img_neck,
                 backbone,
                 neck,
                 head,
                 voxel_size,
                 fusion_mode='baseline',
                 use_gating=True,
                 gate_threshold=0.1,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(TR3DFFMultiStage3DDetector, self).__init__(init_cfg)

        if fusion_mode not in self.VALID_FUSION_MODES:
            raise ValueError(f'Invalid fusion_mode: {fusion_mode}. '
                           f'Must be one of {self.VALID_FUSION_MODES}')

        self.fusion_mode = fusion_mode
        self.use_gating = use_gating
        self.gate_threshold = gate_threshold
        self.voxel_size = voxel_size

        # 2D backbone and neck
        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck)

        # 3D backbone and neck
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)

        # Head
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)

        # Fusion modules based on mode
        # Image features from FPN are 256 channels
        # 3D backbone channels: layer1=64, layer2=128, layer3=128, layer4=128
        img_channels = 256

        if fusion_mode == 'baseline':
            # Original TR3D+FF: single fusion after initial pooling (before layers)
            # At this point, features are 64 channels
            self.fusion_initial = nn.Sequential(
                ME.MinkowskiConvolution(img_channels, 64, kernel_size=1, dimension=3),
                ME.MinkowskiBatchNorm(64),
                ME.MinkowskiReLU(inplace=True))
            self.fusion_after_layer1 = None
            self.fusion_after_layer3 = None

        elif fusion_mode == 'early':
            # Fuse after layer1 only (64 channels)
            self.fusion_initial = None
            self.fusion_after_layer1 = MultiStageFusionModule(
                img_channels=img_channels,
                sparse_channels=64,
                use_gating=use_gating,
                gate_threshold=gate_threshold)
            self.fusion_after_layer3 = None

        elif fusion_mode == 'late':
            # Fuse after layer3 only (128 channels)
            self.fusion_initial = None
            self.fusion_after_layer1 = None
            self.fusion_after_layer3 = MultiStageFusionModule(
                img_channels=img_channels,
                sparse_channels=128,
                use_gating=use_gating,
                gate_threshold=gate_threshold)

        elif fusion_mode == 'dual':
            # Fuse at both layer1 and layer3 with combiner
            self.fusion_initial = None
            self.fusion_after_layer1 = MultiStageFusionModule(
                img_channels=img_channels,
                sparse_channels=64,
                use_gating=use_gating,
                gate_threshold=gate_threshold)
            self.fusion_after_layer3 = MultiStageFusionModule(
                img_channels=img_channels,
                sparse_channels=128,
                use_gating=use_gating,
                gate_threshold=gate_threshold)
            # Combiner to combine early and late fusion outputs
            self.dual_combiner = DualStageCombiner(
                early_channels=64,
                late_channels=128)

    def init_weights(self, pretrained=None):
        # Freeze 2D backbone and neck (use pretrained features)
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        for param in self.img_neck.parameters():
            param.requires_grad = False
        self.img_backbone.eval()
        self.img_neck.eval()

        # Initialize 3D components
        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def _project_img_to_points(self, x, img_features, img_metas, img_shape):
        """Project 2D image features to 3D points.

        Args:
            x (SparseTensor): 3D sparse tensor with coordinates
            img_features (Tensor): 2D image features [B, C, H, W]
            img_metas (list[dict]): Image metadata
            img_shape (tuple): Image shape

        Returns:
            SparseTensor: Projected features as SparseTensor
        """
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size

        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):
            coord_type = 'DEPTH'
            img_scale_factor = (
                point.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                point.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_shape[-2:],
                img_shape=img_shape[-2:],
                aligned=True,
                padding_mode='zeros',
                align_corners=True))

        projected_features = torch.cat(projected_features, dim=0)
        return ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)

    def _baseline_fusion(self, x, img_features, img_metas, img_shape):
        """Original TR3D+FF fusion after initial pooling."""
        img_sparse = self._project_img_to_points(x, img_features, img_metas, img_shape)
        projected = self.fusion_initial(img_sparse)
        return ME.SparseTensor(
            projected.F + x.F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)

    def extract_feat(self, *args):
        """Just implement @abstractmethod of BaseModule."""
        pass

    def extract_feats(self, points, img, img_metas):
        """Extract features from points with multi-stage fusion.

        Args:
            points (list[Tensor]): Raw point clouds.
            img (Tensor): Input images.
            img_metas (list[dict]): Image metadata.

        Returns:
            list[SparseTensor]: Multi-level features.
        """
        # Extract 2D image features (frozen)
        with torch.no_grad():
            img_feat = self.img_backbone(img)
            img_features = self.img_neck(img_feat)[0]  # [B, 256, H, W]

        # Voxelize points
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)

        # Initial 3D processing
        x = self.backbone.conv1(x)
        x = self.backbone.norm1(x)
        x = self.backbone.relu(x)
        if self.backbone.pool:
            x = self.backbone.maxpool(x)

        # Baseline mode: fuse after initial pooling
        if self.fusion_mode == 'baseline':
            x = self._baseline_fusion(x, img_features, img_metas, img.shape)

        # Process through backbone layers with optional fusion
        outs = []
        for i in range(self.backbone.num_stages):
            x = getattr(self.backbone, f'layer{i + 1}')(x)

            # Early fusion after layer1
            if i == 0 and self.fusion_after_layer1 is not None:
                img_sparse = self._project_img_to_points(x, img_features, img_metas, img.shape)
                x = self.fusion_after_layer1(x, img_sparse)
                # Store for dual combiner
                if self.fusion_mode == 'dual' and hasattr(self, 'dual_combiner'):
                    self.dual_combiner.store_early(x)

            # Late fusion after layer3
            if i == 2 and self.fusion_after_layer3 is not None:
                img_sparse = self._project_img_to_points(x, img_features, img_metas, img.shape)
                x = self.fusion_after_layer3(x, img_sparse)
                # Apply combiner for dual mode
                if self.fusion_mode == 'dual' and hasattr(self, 'dual_combiner'):
                    x = self.dual_combiner.combine(x)

            outs.append(x)

        # Pass through neck
        x = self.neck(outs)
        return x

    def forward_train(self, points, img, gt_bboxes_3d, gt_labels_3d, img_metas):
        """Forward of training."""
        x = self.extract_feats(points, img, img_metas)
        losses = self.head.forward_train(x, gt_bboxes_3d, gt_labels_3d, img_metas)
        return losses

    def simple_test(self, points, img_metas, img, *args, **kwargs):
        """Test without augmentations."""
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
