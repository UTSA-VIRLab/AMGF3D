# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from functools import partial

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from ..builder import DETECTORS, build_backbone, build_neck
from .single_stage import SingleStage3DDetector
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type


@DETECTORS.register_module()
class VoteNet(SingleStage3DDetector):
    r"""`VoteNet <https://arxiv.org/pdf/1904.09664.pdf>`_ for 3D detection."""

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoteNet, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=None,
            pretrained=pretrained)

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
            pts_semantic_mask (list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (list[torch.Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses.
        """
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test with augmentation."""
        points_cat = [torch.stack(pts) for pts in points]
        feats = self.extract_feats(points_cat, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, pts_cat, img_meta in zip(feats, points_cat, img_metas):
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
            bbox_list = self.bbox_head.get_bboxes(
                pts_cat, bbox_preds, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]


@DETECTORS.register_module()
class VoteNetFF(VoteNet):
    def __init__(self,
                 img_backbone,
                 img_neck,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoteNetFF, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=None,
            pretrained=pretrained)
        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck)
        self.conv = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
    
    def init_weights(self, pretrained=None):
        # self.img_backbone.init_weights()
        # self.img_neck.init_weights()
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        for param in self.img_neck.parameters():
            param.requires_grad = False
        self.img_backbone.eval()
        self.img_neck.eval()
        self.backbone.init_weights()
        self.bbox_head.init_weights()
    
    def _f(self, xyz, features, img_features, img_metas, img_shape):
        projected_features = []
        for point, img_feature, img_meta in zip(xyz, img_features, img_metas):
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
        projected_features = projected_features.transpose(0, 1).unsqueeze(0)
        projected_features = self.conv(projected_features)
        batch_size, n_features, n_points = features.shape
        projected_features = projected_features[0].reshape(
            n_features, batch_size, n_points).transpose(0, 1)
        return projected_features + features
    
    def extract_feat(self, points, img, img_metas=None):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.
        """
        with torch.no_grad():
            x = self.img_backbone(img)
            img_features = self.img_neck(x)[0]

        x = self.backbone(points, partial(
            self._f, img_features=img_features, img_metas=img_metas, img_shape=img.shape))
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feats(self, points, imgs, img_metas):
        """Extract features of multiple samples."""
        return [
            self.extract_feat(pts, img, img_meta)
            for pts, img, img_meta in zip(points, imgs, img_metas)
        ]

    def forward_train(self,
                      points,
                      img,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat, img, img_metas)
        bbox_preds = self.bbox_head(x, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
    
    def simple_test(self, points, img_metas, img, imgs=None, rescale=False):
        points_cat = torch.stack(points)
        
        x = self.extract_feat(points_cat, img, img_metas)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results


class PointWiseCrossAttentionFusion(nn.Module):
    """Cross-Attention Fusion for Point Cloud Features.
    
    Adapted from TR3D's cross-attention but works with [B, C, N] format
    instead of [N, C] format.
    
    Args:
        d_3d (int): 3D feature dimension (e.g., 128)
        d_2d (int): 2D feature dimension (e.g., 256 from FPN)
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    
    def __init__(self, d_3d, d_2d=256, num_heads=4, dropout=0.1):
        super(PointWiseCrossAttentionFusion, self).__init__()
        
        self.d_3d = d_3d
        self.d_2d = d_2d
        self.num_heads = num_heads
        self.head_dim = d_3d // num_heads
        
        assert d_3d % num_heads == 0, f'd_3d ({d_3d}) must be divisible by num_heads ({num_heads})'
        
        # Project 2D features to 3D dimension
        self.proj_2d = nn.Conv1d(d_2d, d_3d, kernel_size=1)
        
        # Query projection (from 3D features)
        self.q_proj = nn.Conv1d(d_3d, d_3d, kernel_size=1)
        # Key projection (from 2D features)
        self.k_proj = nn.Conv1d(d_3d, d_3d, kernel_size=1)
        # Value projection (from 2D features)
        self.v_proj = nn.Conv1d(d_3d, d_3d, kernel_size=1)
        # Output projection
        self.out_proj = nn.Conv1d(d_3d, d_3d, kernel_size=1)
        
        # Layer norm
        self.norm1 = nn.BatchNorm1d(d_3d)
        self.norm2 = nn.BatchNorm1d(d_3d)
        self.dropout = nn.Dropout(dropout)
        
        # FFN for post-attention processing
        self.ffn = nn.Sequential(
            nn.Conv1d(d_3d, d_3d * 2, kernel_size=1),
            nn.BatchNorm1d(d_3d * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_3d * 2, d_3d, kernel_size=1),
            nn.Dropout(dropout))
        
        # Learnable gate to control fusion strength
        self.gate = nn.Sequential(
            nn.Conv1d(d_3d * 2, d_3d, kernel_size=1),
            nn.BatchNorm1d(d_3d),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_3d, 1, kernel_size=1),
            nn.Sigmoid())
        
        # Initialize gate bias to favor fusion (sigmoid will be ~0.88 with bias=2.0)
        nn.init.constant_(self.gate[-2].bias, 2.0)
    
    def forward(self, feat_3d, feat_2d):
        """
        Args:
            feat_3d (Tensor): [B, d_3d, N] 3D point features
            feat_2d (Tensor): [B, d_2d, N] projected 2D features at same points
            
        Returns:
            Tensor: [B, d_3d, N] fused features
        """
        B, C, N = feat_3d.shape
        
        # Save original for NaN fallback
        feat_3d_orig = feat_3d.clone()
        
        # Clamp inputs to prevent extreme values (widened range for better expressiveness)
        feat_3d = torch.clamp(feat_3d, -1000, 1000)
        feat_2d = torch.clamp(feat_2d, -1000, 1000)
        
        # Project 2D features to 3D dimension
        feat_2d_proj = self.proj_2d(feat_2d)  # [B, d_3d, N]
        
        # Normalize inputs
        feat_3d_norm = self.norm1(feat_3d)
        
        # Compute Q, K, V
        Q = self.q_proj(feat_3d_norm)  # [B, d_3d, N]
        K = self.k_proj(feat_2d_proj)   # [B, d_3d, N]
        V = self.v_proj(feat_2d_proj)   # [B, d_3d, N]
        
        # Reshape for multi-head attention: [B, num_heads, head_dim, N]
        Q = Q.view(B, self.num_heads, self.head_dim, N)
        K = K.view(B, self.num_heads, self.head_dim, N)
        V = V.view(B, self.num_heads, self.head_dim, N)
        
        # Point-wise attention (each point attends to its own 2D projection)
        # Compute attention weights: [B, num_heads, N]
        attn_weights = (Q * K).sum(dim=2) / (self.head_dim ** 0.5)  # [B, num_heads, N]
        attn_weights = torch.clamp(attn_weights, -10, 10)
        attn_weights = torch.sigmoid(attn_weights).unsqueeze(2)  # [B, num_heads, 1, N]
        
        # Apply attention
        attended = attn_weights * V  # [B, num_heads, head_dim, N]
        attended = attended.view(B, self.d_3d, N)  # [B, d_3d, N]
        
        # Output projection with dropout
        attended = self.out_proj(attended)
        attended = self.dropout(attended)
        
        # Residual connection with scaling (0.1 is optimal for VoteNet's selective fusion)
        # Weaker fusion allows model to be selective and only use 2D when beneficial
        feat_3d = feat_3d + 0.1 * attended
        
        # FFN with residual (conservative scaling for stability)
        feat_3d = feat_3d + 0.1 * self.ffn(self.norm2(feat_3d))
        
        # Final clamp (widened range)
        feat_3d = torch.clamp(feat_3d, -1000, 1000)
        
        # NaN check - if NaN detected, return original features
        if torch.any(torch.isnan(feat_3d)):
            return torch.clamp(feat_3d_orig, -1000, 1000)
        
        return feat_3d


class PointWiseMultiScaleFusion(nn.Module):
    """Fuse 2D features from multiple FPN scales with 3D point features.
    
    Args:
        d_3d (int): 3D feature dimension (e.g., 128)
        d_2d (int): 2D feature dimension per scale (256)
        num_scales (int): Number of FPN scales to use
    """
    
    def __init__(self, d_3d, d_2d=256, num_scales=4):
        super(PointWiseMultiScaleFusion, self).__init__()
        
        self.num_scales = num_scales
        
        # Project each scale to 3D dimension
        self.scale_projs = nn.ModuleList([
            nn.Conv1d(d_2d, d_3d, kernel_size=1)
            for _ in range(num_scales)])
        
        # Learn scale weights based on 3D context
        self.scale_attn = nn.Sequential(
            nn.Conv1d(d_3d, d_3d // 2, kernel_size=1),
            nn.BatchNorm1d(d_3d // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_3d // 2, num_scales, kernel_size=1),
            nn.Softmax(dim=1))
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(d_3d * 2, d_3d, kernel_size=1),
            nn.BatchNorm1d(d_3d),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_3d, d_3d, kernel_size=1))
        
        self.gate = nn.Sequential(
            nn.Conv1d(d_3d * 2, d_3d, kernel_size=1),
            nn.BatchNorm1d(d_3d),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_3d, 1, kernel_size=1),
            nn.Sigmoid())
        
        # Initialize gate bias to favor fusion (sigmoid will be ~0.88 with bias=2.0)
        nn.init.constant_(self.gate[-2].bias, 2.0)
    
    def forward(self, feat_3d, feat_2d_list):
        """
        Args:
            feat_3d (Tensor): [B, d_3d, N] 3D features
            feat_2d_list (list[Tensor]): List of [B, d_2d, N] 2D features per scale
            
        Returns:
            Tensor: [B, d_3d, N] fused features
        """
        B, C, N = feat_3d.shape
        
        # Save original for NaN fallback
        feat_3d_orig = feat_3d.clone()
        
        # Clamp input for stability (widened range)
        feat_3d = torch.clamp(feat_3d, -1000, 1000)
        
        # Project each scale
        projected = []
        for feat_2d, proj in zip(feat_2d_list, self.scale_projs):
            feat_2d = torch.clamp(feat_2d, -1000, 1000)
            projected.append(proj(feat_2d))  # [B, d_3d, N]
        
        # Stack: [B, num_scales, d_3d, N]
        projected = torch.stack(projected, dim=1)
        
        # Compute scale attention weights based on 3D context: [B, num_scales, N]
        scale_weights = self.scale_attn(feat_3d)  # [B, num_scales, N]
        scale_weights = scale_weights.unsqueeze(2)  # [B, num_scales, 1, N]
        
        # Weighted sum of scales: [B, d_3d, N]
        fused_2d = (projected * scale_weights).sum(dim=1)
        
        # Gate for adaptive fusion
        gate = self.gate(torch.cat([feat_3d, fused_2d], dim=1))  # [B, 1, N]
        
        # Final fusion with gating (0.1 is optimal for VoteNet's selective fusion)
        # Conservative scaling allows selective 2D integration without interfering with geometry
        combined = torch.cat([feat_3d, fused_2d], dim=1)  # [B, d_3d*2, N]
        output = feat_3d + 0.1 * gate * self.fusion(combined)
        
        # Final clamp (widened range)
        output = torch.clamp(output, -1000, 1000)
        
        # NaN check - if NaN detected, return original features
        if torch.any(torch.isnan(output)):
            return torch.clamp(feat_3d_orig, -1000, 1000)
        
        return output


@DETECTORS.register_module()
class VoteNetCrossAttn(VoteNet):
    """VoteNet with Cross-Attention Fusion.
    
    Uses cross-attention mechanism to selectively fuse 2D image features
    with 3D point cloud features, similar to TR3D-CA.
    """
    
    def __init__(self,
                 img_backbone,
                 img_neck,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None,
                 fusion_stage='early',
                 use_multiscale=True,
                 num_heads=4):
        super(VoteNetCrossAttn, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=None,
            pretrained=pretrained)
        
        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck)
        self.fusion_stage = fusion_stage
        self.use_multiscale = use_multiscale
        self.num_heads = num_heads
        
        # Fusion module - works at SA0 output (128 channels)
        d_3d = 128  # SA0 output dimension
        d_2d = 256  # FPN output dimension
        
        if use_multiscale:
            self.fusion = PointWiseMultiScaleFusion(d_3d=d_3d, d_2d=d_2d, num_scales=4)
        else:
            self.fusion = PointWiseCrossAttentionFusion(d_3d=d_3d, d_2d=d_2d, num_heads=num_heads)
    
    def init_weights(self, pretrained=None):
        # Freeze 2D backbone and neck
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        for param in self.img_neck.parameters():
            param.requires_grad = False
        self.img_backbone.eval()
        self.img_neck.eval()
        
        # Initialize 3D backbone and head
        self.backbone.init_weights()
        self.bbox_head.init_weights()
    
    def _f(self, xyz, features, img_features_list, img_metas, img_shape):
        """
        Fusion function called during PointNet2 forward pass.
        
        Args:
            xyz (Tensor): [B, N, 3] point coordinates
            features (Tensor): [B, C, N] 3D point features (C=128 after SA0)
            img_features_list (list[Tensor]): List of [B, C, H, W] 2D features from FPN
            img_metas (list): Image metadata
            img_shape (tuple): Image shape
            
        Returns:
            Tensor: [B, C, N] fused features
        """
        batch_size, n_features, n_points = features.shape
        
        # Project 2D features to 3D points for each FPN scale
        # feat_2d_batch will be list of [B, C, N] tensors, one per FPN scale
        feat_2d_batch = []
        
        for img_feature in img_features_list:
            # img_feature is [B, C, H, W]
            # Process each batch item separately (matching VoteNetFF pattern)
            projected_features = []
            for point, img_feat_single, img_meta in zip(xyz, img_feature, img_metas):
                coord_type = 'DEPTH'
                img_scale_factor = (
                    point.new_tensor(img_meta['scale_factor'][:2])
                    if 'scale_factor' in img_meta.keys() else 1)
                img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
                img_crop_offset = (
                    point.new_tensor(img_meta['img_crop_offset'])
                    if 'img_crop_offset' in img_meta.keys() else 0)
                proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
                
                # img_feat_single is [C, H, W], unsqueeze to [1, C, H, W]
                # point is [N, 3]
                projected = point_sample(
                    img_meta=img_meta,
                    img_features=img_feat_single.unsqueeze(0),  # [1, C, H, W]
                    points=point,  # [N, 3]
                    proj_mat=point.new_tensor(proj_mat),
                    coord_type=coord_type,
                    img_scale_factor=img_scale_factor,
                    img_crop_offset=img_crop_offset,
                    img_flip=img_flip,
                    img_pad_shape=img_shape[-2:],
                    img_shape=img_shape[-2:],
                    aligned=True,
                    padding_mode='zeros',
                    align_corners=True)  # Returns [N, C]
                
                projected_features.append(projected)  # List of [N, C] tensors
            
            # Concatenate across batch: [B*N, C] -> reshape to [B, N, C] -> transpose to [B, C, N]
            projected_features = torch.cat(projected_features, dim=0)  # [B*N, C]
            projected_features = projected_features.view(batch_size, n_points, -1)  # [B, N, C]
            projected_features = projected_features.transpose(1, 2)  # [B, C, N]
            
            feat_2d_batch.append(projected_features)
        
        # Apply fusion
        if self.use_multiscale:
            # Multi-scale fusion
            fused = self.fusion(features, feat_2d_batch)
        else:
            # Single-scale fusion (use P2 level, index 0)
            fused = self.fusion(features, feat_2d_batch[0])
        
        return fused
    
    def extract_feat(self, points, img, img_metas=None):
        """Extract features from backbone+neck with fusion.
        
        Args:
            points (torch.Tensor): Input points [B, N, 3+features]
            img (torch.Tensor): Input images [B, 3, H, W]
            img_metas (list): Image metadata
        """
        # Extract 2D features (frozen)
        with torch.no_grad():
            x = self.img_backbone(img)
            img_features_list = self.img_neck(x)  # List of [B, C, H, W] from FPN
        
        # Extract 3D features with fusion
        x = self.backbone(points, partial(
            self._f, 
            img_features_list=img_features_list, 
            img_metas=img_metas, 
            img_shape=img.shape))
        
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def extract_feats(self, points, imgs, img_metas):
        """Extract features of multiple samples."""
        return [
            self.extract_feat(pts, img, img_meta)
            for pts, img, img_meta in zip(points, imgs, img_metas)
        ]
    
    def forward_train(self,
                      points,
                      img,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        points_cat = torch.stack(points)
        
        x = self.extract_feat(points_cat, img, img_metas)
        bbox_preds = self.bbox_head(x, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
    
    def simple_test(self, points, img_metas, img, imgs=None, rescale=False):
        points_cat = torch.stack(points)
        
        x = self.extract_feat(points_cat, img, img_metas)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results