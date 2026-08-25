# TR3D+FF with Cross-Attention Fusion
#
# Key improvements over simple additive fusion:
# 1. Cross-attention: 3D features query 2D features selectively
# 2. Multi-scale 2D features: Use all FPN levels, not just one
# 3. Position-aware attention: Include spatial relationships
#
# This should exceed TR3D+FF baseline by learning WHAT 2D info is relevant

try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn('Please follow `getting_started.md` to install MinkowskiEngine.')

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.core import bbox3d2result
from mmdet3d.models import DETECTORS, build_backbone, build_neck, build_head
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type
from .base import Base3DDetector


class CrossAttentionFusion(nn.Module):
    """Cross-Attention based fusion: 3D features query 2D features.

    Unlike simple addition, this learns WHICH 2D features are relevant
    for each 3D point through attention mechanism.

    Args:
        d_3d (int): 3D feature dimension
        d_2d (int): 2D feature dimension (typically 256 from FPN)
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """

    def __init__(self, d_3d, d_2d=256, num_heads=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()

        self.d_3d = d_3d
        self.d_2d = d_2d
        self.num_heads = num_heads
        self.head_dim = d_3d // num_heads

        assert d_3d % num_heads == 0, f'd_3d ({d_3d}) must be divisible by num_heads ({num_heads})'

        # Query projection (from 3D features)
        self.q_proj = nn.Linear(d_3d, d_3d)
        # Key projection (from 2D features)
        self.k_proj = nn.Linear(d_2d, d_3d)
        # Value projection (from 2D features)
        self.v_proj = nn.Linear(d_2d, d_3d)
        # Output projection
        self.out_proj = nn.Linear(d_3d, d_3d)

        # Layer norm and dropout
        self.norm1 = nn.LayerNorm(d_3d)
        self.norm2 = nn.LayerNorm(d_3d)
        self.dropout = nn.Dropout(dropout)

        # FFN for post-attention processing
        self.ffn = nn.Sequential(
            nn.Linear(d_3d, d_3d * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_3d * 2, d_3d),
            nn.Dropout(dropout))

        # Learnable gate to control fusion strength
        self.gate = nn.Sequential(
            nn.Linear(d_3d * 2, d_3d),
            nn.ReLU(inplace=True),
            nn.Linear(d_3d, 1),
            nn.Sigmoid())

    def forward(self, feat_3d, feat_2d):
        """
        Args:
            feat_3d (Tensor): [N, d_3d] 3D point features
            feat_2d (Tensor): [N, d_2d] projected 2D features at same points

        Returns:
            Tensor: [N, d_3d] fused features
        """
        N = feat_3d.shape[0]

        # Save original for NaN fallback
        feat_3d_orig = feat_3d.clone()

        # Clamp inputs to prevent extreme values
        feat_3d = torch.clamp(feat_3d, -100, 100)
        feat_2d = torch.clamp(feat_2d, -100, 100)

        # Normalize inputs
        feat_3d_norm = self.norm1(feat_3d)

        # Compute Q, K, V
        Q = self.q_proj(feat_3d_norm)  # [N, d_3d]
        K = self.k_proj(feat_2d)        # [N, d_3d]
        V = self.v_proj(feat_2d)        # [N, d_3d]

        # Reshape for multi-head attention: [N, num_heads, head_dim]
        Q = Q.view(N, self.num_heads, self.head_dim)
        K = K.view(N, self.num_heads, self.head_dim)
        V = V.view(N, self.num_heads, self.head_dim)

        # Self-attention per point (each point attends to its own 2D projection)
        # This computes a weighted combination based on Q-K similarity
        attn_weights = (Q * K).sum(dim=-1, keepdim=True) / (self.head_dim ** 0.5)  # [N, num_heads, 1]
        # Clamp before sigmoid to prevent saturation
        attn_weights = torch.clamp(attn_weights, -10, 10)
        attn_weights = torch.sigmoid(attn_weights)  # Use sigmoid for per-point gating

        # Apply attention
        attended = attn_weights * V  # [N, num_heads, head_dim]
        attended = attended.view(N, self.d_3d)  # [N, d_3d]

        # Output projection with dropout
        attended = self.out_proj(attended)
        attended = self.dropout(attended)

        # Residual connection with scaling for stability
        feat_3d = feat_3d + 0.1 * attended

        # FFN with residual (scaled)
        feat_3d = feat_3d + 0.1 * self.ffn(self.norm2(feat_3d))

        # Final clamp
        feat_3d = torch.clamp(feat_3d, -100, 100)
        
        # NaN check - if NaN detected, return original features
        if torch.any(torch.isnan(feat_3d)):
            return torch.clamp(feat_3d_orig, -100, 100)

        return feat_3d


class MultiScaleFusion(nn.Module):
    """Fuse 2D features from multiple FPN scales with 3D features.

    This uses features from all FPN levels and learns to weight them
    based on the 3D feature context.

    Args:
        d_3d (int): 3D feature dimension
        d_2d (int): 2D feature dimension per scale (256)
        num_scales (int): Number of FPN scales to use
    """

    def __init__(self, d_3d, d_2d=256, num_scales=4, use_gate=True):
        super(MultiScaleFusion, self).__init__()

        self.num_scales = num_scales
        self.use_gate = use_gate

        # Project each scale to 3D dimension
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_2d, d_3d),
                nn.ReLU(inplace=True))
            for _ in range(num_scales)])

        # Learn scale weights based on 3D context
        self.scale_attn = nn.Sequential(
            nn.Linear(d_3d, d_3d // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_3d // 2, num_scales),
            nn.Softmax(dim=-1))

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_3d * 2, d_3d),
            nn.ReLU(inplace=True),
            nn.Linear(d_3d, d_3d))

        # Gate for adaptive fusion. Kept as a module even when disabled, so the
        # checkpoint loader's state_dict shapes do not surprise an unsuspecting
        # caller; just bypassed in forward when use_gate=False.
        self.gate = nn.Sequential(
            nn.Linear(d_3d * 2, 1),
            nn.Sigmoid())

    def forward(self, feat_3d, feat_2d_list):
        """
        Args:
            feat_3d (Tensor): [N, d_3d] 3D features
            feat_2d_list (list[Tensor]): List of [N, d_2d] 2D features per scale

        Returns:
            Tensor: [N, d_3d] fused features
        """
        # Save original for NaN fallback  
        feat_3d_orig = feat_3d.clone()
        
        # Clamp input for stability
        feat_3d = torch.clamp(feat_3d, -100, 100)
        
        # Project each scale
        projected = []
        for i, (feat_2d, proj) in enumerate(zip(feat_2d_list, self.scale_projs)):
            feat_2d = torch.clamp(feat_2d, -100, 100)
            projected.append(proj(feat_2d))

        # Stack: [N, num_scales, d_3d]
        projected = torch.stack(projected, dim=1)

        # Compute scale attention weights based on 3D context
        scale_weights = self.scale_attn(feat_3d)  # [N, num_scales]
        scale_weights = scale_weights.unsqueeze(-1)  # [N, num_scales, 1]

        # Weighted sum of scales
        fused_2d = (projected * scale_weights).sum(dim=1)  # [N, d_3d]

        # Gate for adaptive fusion (bypassed when use_gate=False for ablation).
        combined = torch.cat([feat_3d, fused_2d], dim=-1)
        if self.use_gate:
            gate = self.gate(combined)
            output = feat_3d + 0.1 * gate * self.fusion(combined)
        else:
            output = feat_3d + 0.1 * self.fusion(combined)

        # Final clamp
        output = torch.clamp(output, -100, 100)
        
        # NaN check - if NaN detected, return original features
        if torch.any(torch.isnan(output)):
            return torch.clamp(feat_3d_orig, -100, 100)

        return output


@DETECTORS.register_module()
class TR3DFFCrossAttn3DDetector(Base3DDetector):
    """TR3D+FF with Cross-Attention Fusion.

    Key differences from TR3DFFMultiStage3DDetector:
    1. Uses cross-attention instead of simple addition
    2. Uses multi-scale 2D features from all FPN levels
    3. Designed to exceed TR3D+FF baseline (69.4%)

    Args:
        img_backbone (dict): Config of the 2D backbone.
        img_neck (dict): Config of the 2D neck.
        backbone (dict): Config of the 3D backbone.
        neck (dict): Config of the 3D neck.
        head (dict): Config of the 3D head.
        voxel_size (float): Voxel size in meters.
        fusion_stage (str): Where to fuse - 'early' (after layer1), 'late' (after layer3), 'dual'.
        use_multiscale (bool): Whether to use multi-scale 2D features.
        num_heads (int): Number of attention heads.
    """

    def __init__(self,
                 img_backbone,
                 img_neck,
                 backbone,
                 neck,
                 head,
                 voxel_size,
                 fusion_stage='late',
                 use_multiscale=True,
                 num_heads=4,
                 use_gate=True,
                 zero_image_features=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(TR3DFFCrossAttn3DDetector, self).__init__(init_cfg)

        self.fusion_stage = fusion_stage
        self.use_multiscale = use_multiscale
        self.voxel_size = voxel_size
        self.num_heads = num_heads
        self.use_gate = use_gate
        # Geometry-only-residual control (BMVC'26 rebuttal, reviewer R1-W3):
        # zero the FPN maps before sampling, so every projected 2D descriptor is
        # exactly zero. The fusion modules keep their full architecture/capacity
        # but receive no appearance information -- isolating "extra parameters"
        # from "effective RGB-D fusion".
        self.zero_image_features = zero_image_features

        # 2D backbone and neck (frozen)
        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck)

        # 3D backbone and neck
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)

        # Head
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)

        # Fusion modules
        # 3D channel dimensions: layer1=64, layer2=128, layer3=128, layer4=128
        if use_multiscale:
            if fusion_stage in ['early', 'dual']:
                self.fusion_early = MultiScaleFusion(d_3d=64, d_2d=256, num_scales=4, use_gate=use_gate)
            if fusion_stage in ['late', 'dual']:
                self.fusion_late = MultiScaleFusion(d_3d=128, d_2d=256, num_scales=4, use_gate=use_gate)
        else:
            if fusion_stage in ['early', 'dual']:
                self.fusion_early = CrossAttentionFusion(d_3d=64, d_2d=256, num_heads=num_heads)
            if fusion_stage in ['late', 'dual']:
                self.fusion_late = CrossAttentionFusion(d_3d=128, d_2d=256, num_heads=num_heads)

    def init_weights(self, pretrained=None):
        # Freeze 2D backbone and neck
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

        Returns regular tensor (not SparseTensor) for cross-attention.
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

        return torch.cat(projected_features, dim=0)

    def _project_multiscale(self, x, img_feat_list, img_metas, img_shape):
        """Project multi-scale 2D features to 3D points.

        Returns list of tensors, one per FPN scale.
        """
        projected_list = []
        for img_features in img_feat_list:
            projected = self._project_img_to_points(x, img_features, img_metas, img_shape)
            projected_list.append(projected)
        return projected_list

    def extract_feat(self, *args):
        """Just implement @abstractmethod of BaseModule."""
        pass

    def extract_feats(self, points, img, img_metas):
        """Extract features with cross-attention fusion."""
        # Extract 2D features (frozen)
        with torch.no_grad():
            img_feat = self.img_backbone(img)
            img_neck_out = self.img_neck(img_feat)  # List of [B, 256, H_i, W_i]
            # Use first 4 FPN levels for multi-scale fusion
            img_feat_list = img_neck_out[:4]
            if self.zero_image_features:
                # Geometry-only-residual control: bilinear sampling of an
                # all-zero map returns exactly zero descriptors everywhere.
                img_feat_list = [torch.zeros_like(f) for f in img_feat_list]

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

        # Process through backbone layers with fusion
        outs = []
        for i in range(self.backbone.num_stages):
            x = getattr(self.backbone, f'layer{i + 1}')(x)

            # Early fusion after layer1
            if i == 0 and self.fusion_stage in ['early', 'dual']:
                if self.use_multiscale:
                    feat_2d_list = self._project_multiscale(x, img_feat_list, img_metas, img.shape)
                    fused = self.fusion_early(x.F, feat_2d_list)
                else:
                    feat_2d = self._project_img_to_points(x, img_feat_list[0], img_metas, img.shape)
                    fused = self.fusion_early(x.F, feat_2d)
                x = ME.SparseTensor(
                    fused,
                    coordinate_map_key=x.coordinate_map_key,
                    coordinate_manager=x.coordinate_manager)

            # Late fusion after layer3
            if i == 2 and self.fusion_stage in ['late', 'dual']:
                if self.use_multiscale:
                    feat_2d_list = self._project_multiscale(x, img_feat_list, img_metas, img.shape)
                    fused = self.fusion_late(x.F, feat_2d_list)
                else:
                    feat_2d = self._project_img_to_points(x, img_feat_list[0], img_metas, img.shape)
                    fused = self.fusion_late(x.F, feat_2d)
                x = ME.SparseTensor(
                    fused,
                    coordinate_map_key=x.coordinate_map_key,
                    coordinate_manager=x.coordinate_manager)

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
