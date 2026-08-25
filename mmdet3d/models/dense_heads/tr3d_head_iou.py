"""TR3D Head with IoU-Aware Scoring.

This head predicts IoU quality instead of centerness, which directly aligns
with the evaluation metric (mAP uses IoU thresholds).

Key differences from centerness:
- Centerness: geometric measure of how centered a point is
- IoU: actual overlap between predicted and GT box (evaluation metric)

The IoU prediction is used to:
1. Weight the classification score at inference: score = cls * iou
2. Optionally weight the bbox loss during training
"""

try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner import BaseModule
from torch import nn

from mmdet3d.models.builder import HEADS, build_loss
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner
from mmdet3d.core.bbox.iou_calculators import BboxOverlaps3D


@HEADS.register_module()
class TR3DHeadIoU(BaseModule):
    """TR3D Head with IoU prediction for quality-aware scoring.

    Args:
        n_classes (int): Number of classes.
        in_channels (int): Input channels from neck.
        n_reg_outs (int): Number of regression outputs (6 or 8).
        voxel_size (float): Voxel size for coordinate conversion.
        assigner (dict): Config for assigner.
        bbox_loss (dict): Config for bbox loss.
        cls_loss (dict): Config for classification loss.
        iou_loss_weight (float): Weight for IoU prediction loss.
        use_iou_weight (bool): Whether to weight bbox loss by IoU.
        train_cfg (dict): Training config.
        test_cfg (dict): Testing config.
    """

    def __init__(self,
                 n_classes,
                 in_channels,
                 n_reg_outs,
                 voxel_size,
                 assigner,
                 bbox_loss=dict(type='AxisAlignedIoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 iou_loss_weight=1.0,
                 use_iou_weight=False,
                 train_cfg=None,
                 test_cfg=None):
        super(TR3DHeadIoU, self).__init__()
        self.voxel_size = voxel_size
        self.assigner = build_assigner(assigner)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.iou_loss_weight = iou_loss_weight
        self.use_iou_weight = use_iou_weight
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.iou_calculator = BboxOverlaps3D(coordinate='depth')
        self._init_layers(n_classes, in_channels, n_reg_outs)

    def _init_layers(self, n_classes, in_channels, n_reg_outs):
        self.bbox_conv = ME.MinkowskiConvolution(
            in_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            in_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        # IoU prediction layer (predicts IoU between predicted and GT box)
        self.iou_conv = ME.MinkowskiConvolution(
            in_channels, 1, kernel_size=1, bias=True, dimension=3)

    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.normal_(self.iou_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

    def _forward_single(self, x):
        """Forward for a single level."""
        reg_final = self.bbox_conv(x).features
        reg_distance = torch.exp(reg_final[:, 3:6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
        cls_pred = self.cls_conv(x).features
        iou_pred = self.iou_conv(x).features

        bbox_preds, cls_preds, iou_preds, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            iou_preds.append(iou_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

        return bbox_preds, cls_preds, iou_preds, points

    def forward(self, x):
        bbox_preds, cls_preds, iou_preds, points = [], [], [], []
        for i in range(len(x)):
            bbox_pred, cls_pred, iou_pred, point = self._forward_single(x[i])
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            iou_preds.append(iou_pred)
            points.append(point)
        return bbox_preds, cls_preds, iou_preds, points

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format."""
        if bbox.shape[-1] != 6:
            return bbox
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox."""
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center, y_center, z_center,
            bbox_pred[:, 3], bbox_pred[:, 4], bbox_pred[:, 5]], -1)

        if bbox_pred.shape[1] == 6:
            return base_bbox

        # Rotated case
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale / (1 + q) * q,
             bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)

    def _loss_single(self, bbox_preds, cls_preds, iou_preds, points,
                     gt_bboxes, gt_labels, img_meta):
        """Compute loss for a single image."""
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)

        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        iou_preds = torch.cat(iou_preds)
        points = torch.cat(points)

        # Classification loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
        cls_loss = self.cls_loss(cls_preds, cls_targets)

        # Bbox loss and IoU targets
        pos_bbox_preds = bbox_preds[pos_mask]
        pos_iou_preds = iou_preds[pos_mask]

        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]

            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]

            # Compute predicted boxes
            pred_boxes = self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)

            # Compute IoU between predicted and target boxes for IoU targets
            with torch.no_grad():
                # For rotated boxes, compute 3D IoU
                if pred_boxes.shape[1] == 7:
                    iou_targets = self._compute_iou_rotated(pred_boxes, pos_bbox_targets)
                else:
                    iou_targets = self._compute_iou_aligned(pred_boxes, pos_bbox_targets)
                iou_targets = iou_targets.clamp(min=0, max=1)

            # Bbox loss
            bbox_loss_raw = self.bbox_loss(
                self._bbox_to_loss(pred_boxes),
                self._bbox_to_loss(pos_bbox_targets))

            if self.use_iou_weight:
                iou_weight = iou_targets.detach()
                iou_denorm = max(iou_weight.sum().item(), 1e-6)
                bbox_loss = (bbox_loss_raw * iou_weight).sum() / iou_denorm
            else:
                bbox_loss = bbox_loss_raw.mean()

            # IoU prediction loss (BCE)
            iou_loss = self.iou_loss_weight * torch.nn.functional.binary_cross_entropy_with_logits(
                pos_iou_preds.squeeze(-1),
                iou_targets,
                reduction='mean')
        else:
            bbox_loss = None
            iou_loss = pos_iou_preds.sum() * 0

        return bbox_loss, cls_loss, iou_loss, pos_mask

    def _compute_iou_aligned(self, pred_boxes, target_boxes):
        """Compute axis-aligned 3D IoU."""
        # Convert to corner format: (x1, y1, z1, x2, y2, z2)
        pred_corners = self._bbox_to_loss(pred_boxes)
        target_corners = self._bbox_to_loss(target_boxes)

        # Compute intersection
        inter_min = torch.max(pred_corners[:, :3], target_corners[:, :3])
        inter_max = torch.min(pred_corners[:, 3:], target_corners[:, 3:])
        inter_dims = (inter_max - inter_min).clamp(min=0)
        inter_vol = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]

        # Compute volumes
        pred_dims = pred_corners[:, 3:] - pred_corners[:, :3]
        target_dims = target_corners[:, 3:] - target_corners[:, :3]
        pred_vol = pred_dims[:, 0] * pred_dims[:, 1] * pred_dims[:, 2]
        target_vol = target_dims[:, 0] * target_dims[:, 1] * target_dims[:, 2]

        # IoU
        union_vol = pred_vol + target_vol - inter_vol
        iou = inter_vol / union_vol.clamp(min=1e-6)
        return iou

    def _compute_iou_rotated(self, pred_boxes, target_boxes):
        """Compute rotated 3D IoU (approximate using aligned IoU for simplicity)."""
        # For rotated boxes, we use an approximation based on aligned IoU
        # A more accurate method would use proper rotated IoU calculation
        # but this is computationally expensive

        # Use center distance and size similarity as proxy
        center_dist = torch.sqrt(torch.sum((pred_boxes[:, :3] - target_boxes[:, :3])**2, dim=1))
        size_pred = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
        size_target = target_boxes[:, 3] * target_boxes[:, 4] * target_boxes[:, 5]
        size_ratio = torch.min(size_pred, size_target) / torch.max(size_pred, size_target).clamp(min=1e-6)

        # Approximate IoU based on center distance and size similarity
        max_dim = torch.max(torch.max(pred_boxes[:, 3:6], dim=1)[0],
                           torch.max(target_boxes[:, 3:6], dim=1)[0])
        center_score = torch.exp(-center_dist / max_dim.clamp(min=1e-6))

        # Combine center score and size ratio
        approx_iou = center_score * size_ratio
        return approx_iou.clamp(min=0, max=1)

    def _loss(self, bbox_preds, cls_preds, iou_preds, points,
              gt_bboxes, gt_labels, img_metas):
        bbox_losses, cls_losses, iou_losses, pos_masks = [], [], [], []
        for i in range(len(img_metas)):
            bbox_loss, cls_loss, iou_loss, pos_mask = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                iou_preds=[x[i] for x in iou_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i])
            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            iou_losses.append(iou_loss)
            pos_masks.append(pos_mask)

        return dict(
            bbox_loss=torch.mean(torch.stack(bbox_losses)) if bbox_losses else torch.tensor(0.0),
            cls_loss=torch.sum(torch.cat(cls_losses)) / max(torch.sum(torch.cat(pos_masks)).item(), 1),
            iou_loss=torch.mean(torch.stack(iou_losses)) if iou_losses else torch.tensor(0.0))

    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        bbox_preds, cls_preds, iou_preds, points = self(x)
        return self._loss(bbox_preds, cls_preds, iou_preds, points,
                          gt_bboxes, gt_labels, img_metas)

    def _nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene."""
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores, self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes, box_dim=box_dim, with_yaw=with_yaw, origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels

    def _get_bboxes_single(self, bbox_preds, cls_preds, iou_preds, points, img_meta):
        """Get bboxes for a single image with IoU-weighted scores."""
        cls_scores = torch.cat(cls_preds).sigmoid()
        iou_scores = torch.cat(iou_preds).sigmoid()

        # KEY: Multiply classification scores by IoU prediction
        scores = cls_scores * iou_scores

        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        boxes, scores, labels = self._nms(boxes, scores, img_meta)
        return boxes, scores, labels

    def _get_bboxes(self, bbox_preds, cls_preds, iou_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                iou_preds=[x[i] for x in iou_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results

    def forward_test(self, x, img_metas):
        bbox_preds, cls_preds, iou_preds, points = self(x)
        return self._get_bboxes(bbox_preds, cls_preds, iou_preds, points, img_metas)
