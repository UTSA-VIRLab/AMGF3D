# Research Report: Quality-Aware Detection Heads for TR3D+FF Cross-Attention Model

## Abstract

This report presents two novel extensions to the TR3D+FF Cross-Attention Dual Fusion architecture for indoor 3D object detection: (1) Centerness-aware detection head and (2) IoU-aware detection head. Both approaches introduce auxiliary prediction branches that estimate localization quality, enabling quality-weighted scoring during inference. We hypothesize that these additions will improve detection performance, particularly at stricter IoU thresholds (mAP@0.50), by better aligning classification confidence with localization accuracy.

---

## 1. Introduction

### 1.1 Background

The TR3D+FF Cross-Attention Dual model combines sparse 3D convolutions with 2D RGB features through a learned cross-attention fusion mechanism. While this architecture effectively leverages multi-modal information, the detection head follows a conventional design where classification and regression branches operate independently. This independence creates a misalignment problem: high classification scores do not necessarily correspond to well-localized predictions.

### 1.2 Problem Statement

In anchor-free 3D object detection, points near object boundaries often produce:
- High classification scores (they are inside the object)
- Poor localization quality (they are far from the object center)

This misalignment leads to:
1. Suboptimal NMS ranking, where poorly localized boxes with high scores suppress better alternatives
2. Degraded performance at stricter IoU thresholds (AP@0.50 vs AP@0.25)
3. Inconsistent training gradients between classification and regression tasks

### 1.3 Proposed Solutions

We propose two complementary approaches to address this misalignment:

1. **Centerness-Aware Head**: Predicts geometric centerness (how centered a point is within the bounding box)
2. **IoU-Aware Head**: Directly predicts the IoU between the predicted box and ground truth

Both approaches modulate the final detection score as:

```
final_score = classification_score × quality_score
```

where `quality_score` is either centerness or predicted IoU.

---

## 2. Related Work

### 2.1 Centerness in 2D Detection

FCOS (Tian et al., 2019) introduced centerness for 2D anchor-free detection, demonstrating significant improvements by downweighting predictions at object boundaries. The centerness target is computed as:

```
centerness = sqrt(min(l,r)/max(l,r) × min(t,b)/max(t,b))
```

where l, r, t, b are distances from the point to box boundaries.

### 2.2 Centerness in 3D Detection

FCAF3D (Rukhovich et al., 2022) extended centerness to 3D indoor detection with the formulation:

```
centerness_3d = sqrt(min(dx)/max(dx) × min(dy)/max(dy) × min(dz)/max(dz))
```

This approach achieved state-of-the-art results on ScanNet and SUN RGB-D benchmarks.

### 2.3 IoU-Aware Detection

IoU-Net (Jiang et al., 2018) proposed learning to predict IoU directly, arguing that it provides a more direct quality measure than geometric proxies. This approach has been adopted in various 2D detectors including ATSS and PAA.

---

## 3. Methodology

### 3.1 Baseline Architecture

The baseline TR3D+FF Cross-Attention Dual model consists of:

```
Input: RGB Image + Point Cloud
         │                │
         ▼                ▼
    ResNet50+FPN     MinkResNet34
         │                │
         └───► Cross-Attention Fusion (Dual Stage)
                          │
                          ▼
                     TR3DNeck (2-level FPN)
                          │
                          ▼
                     TR3DHead
                    ┌────┴────┐
                    ▼         ▼
               bbox_conv  cls_conv
               (8 dims)   (10 cls)
```

### 3.2 Centerness-Aware Head Architecture

We extend the baseline head with a centerness prediction branch:

```
                     TR3DHeadWithCenterness
                    ┌────┬────┬────┐
                    ▼    ▼    ▼    │
               bbox   cls   centerness
              (8dim) (10cls)  (1dim)
                              │
                              ▼
              final_score = sigmoid(cls) × sigmoid(centerness)
```

#### 3.2.1 Centerness Target Computation

For a point P inside a 3D bounding box, we compute distances to each face after rotating to the box's local coordinate frame:

```python
# Rotate point to box-local coordinates
shift = point - box_center
shift_rotated = rotate_z(shift, -box_angle)

# Compute distances to faces
dx_min = half_width + shift_rotated_x
dx_max = half_width - shift_rotated_x
dy_min = half_length + shift_rotated_y
dy_max = half_length - shift_rotated_y
dz_min = half_height + shift_rotated_z
dz_max = half_height - shift_rotated_z

# Centerness target
centerness = sqrt(
    min(dx_min, dx_max) / max(dx_min, dx_max) ×
    min(dy_min, dy_max) / max(dy_min, dy_max) ×
    min(dz_min, dz_max) / max(dz_min, dz_max)
)
```

#### 3.2.2 Loss Function

The total loss becomes:

```
L_total = L_cls + L_bbox + λ_cent × L_centerness
```

Where:
- `L_cls`: Focal Loss for classification
- `L_bbox`: DIoU Loss for bounding box regression (optionally weighted by centerness)
- `L_centerness`: Binary Cross-Entropy for centerness prediction
- `λ_cent = 1.0`: Centerness loss weight

#### 3.2.3 Centerness-Weighted Bbox Loss

Following FCAF3D, we optionally weight the bbox loss by centerness targets:

```python
bbox_loss = (raw_bbox_loss × centerness_targets).sum() / centerness_targets.sum()
```

This focuses training on high-quality predictions near object centers.

### 3.3 IoU-Aware Head Architecture

```
                        TR3DHeadIoU
                    ┌────┬────┬────┐
                    ▼    ▼    ▼    │
               bbox   cls    iou
              (8dim) (10cls) (1dim)
                              │
                              ▼
              final_score = sigmoid(cls) × sigmoid(iou)
```

#### 3.3.1 IoU Target Computation

Unlike centerness (computed from GT boxes), IoU targets are computed from predictions:

```python
# Forward pass produces bbox predictions
pred_boxes = bbox_pred_to_bbox(points, bbox_pred)

# Compute IoU between predictions and assigned GT boxes
with torch.no_grad():
    iou_targets = compute_3d_iou(pred_boxes, gt_boxes)
    iou_targets = iou_targets.clamp(0, 1)
```

#### 3.3.2 IoU Computation for Rotated Boxes

For rotated 3D boxes, exact IoU computation is expensive. We use an approximation:

```python
def approximate_rotated_iou(pred_boxes, target_boxes):
    # Center distance score
    center_dist = ||pred_center - target_center||_2
    max_dim = max(pred_dims, target_dims)
    center_score = exp(-center_dist / max_dim)

    # Size similarity
    size_pred = prod(pred_dims)
    size_target = prod(target_dims)
    size_ratio = min(size_pred, size_target) / max(size_pred, size_target)

    # Approximate IoU
    approx_iou = center_score × size_ratio
    return approx_iou
```

#### 3.3.3 Loss Function

```
L_total = L_cls + L_bbox + λ_iou × L_iou
```

Where:
- `L_iou`: Binary Cross-Entropy between predicted and target IoU
- `λ_iou = 1.0`: IoU loss weight

---

## 4. Implementation Details

### 4.1 Network Architecture

| Component | Specification |
|-----------|---------------|
| 2D Backbone | ResNet-50 (frozen, ImageNet pretrained) |
| 2D Neck | FPN (4 levels, 256 channels) |
| 3D Backbone | MinkResNet-34 (max_channels=128) |
| 3D Neck | TR3DNeck (2 levels, 128 channels) |
| Fusion | Cross-Attention Dual (layer1 + layer3) |
| Quality Branch | 1×1 MinkowskiConvolution |

### 4.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| LR Schedule | Step decay at epochs 8, 11 |
| Total Epochs | 12 |
| Batch Size | 8 per GPU |
| Voxel Size | 0.01m |
| Points per Scene | 100,000 |

### 4.3 Assigner Configuration

| Parameter | Value |
|-----------|-------|
| Type | TR3DAssignerWithCenterness / TR3DAssigner |
| top_pts_threshold | 6 |
| label2level | [1,1,1,0,0,1,0,0,1,0] |

---

## 5. Theoretical Analysis

### 5.1 Centerness vs IoU: Comparison

| Aspect | Centerness | IoU |
|--------|------------|-----|
| **Target Source** | Ground truth geometry | Predicted boxes |
| **Stability** | Stable (fixed targets) | Dynamic (changes with predictions) |
| **Semantic Meaning** | Geometric quality | Evaluation metric |
| **Computation** | O(1) per point | O(n) intersection volume |
| **Training Difficulty** | Easier (stable targets) | Harder (moving targets) |

### 5.2 Expected Benefits

#### 5.2.1 Centerness Benefits
1. **Boundary suppression**: Points at object edges produce lower scores
2. **Better NMS**: High-quality predictions ranked higher
3. **Focused training**: Centerness-weighted loss prioritizes center points

#### 5.2.2 IoU Benefits
1. **Direct metric alignment**: Predicts what evaluation measures
2. **Quality awareness**: Can learn complex quality patterns beyond geometry
3. **Adaptive weighting**: IoU targets adapt to prediction quality

### 5.3 Limitations

#### 5.3.1 Centerness Limitations
- Geometric assumption may not hold for all object shapes
- Does not account for regression error, only position
- Fixed targets ignore prediction quality

#### 5.3.2 IoU Limitations
- Moving targets complicate optimization
- Approximate IoU for rotated boxes introduces error
- May require more training iterations to converge

---

## 6. Experimental Setup

### 6.1 Dataset

**SUN RGB-D**:
- 10,335 RGB-D images for indoor scene understanding
- 10 object categories: bed, table, sofa, chair, toilet, desk, dresser, night_stand, bookshelf, bathtub
- Train/Val split: 5,285/5,050 images

### 6.2 Evaluation Metrics

- **mAP@0.25**: Mean Average Precision at IoU threshold 0.25
- **mAP@0.50**: Mean Average Precision at IoU threshold 0.50
- **Per-class AP**: Individual class performance analysis

### 6.3 Baselines

| Model | mAP@0.25 | mAP@0.50 |
|-------|----------|----------|
| TR3D | 67.1% | 50.4% |
| TR3D+FF | 69.4% | ~51% |
| CrossAttn Dual (2-level) | TBD | TBD |

---

## 7. Expected Results

### 7.1 Hypothesis

We hypothesize that:

1. **Centerness-aware head** will improve both mAP@0.25 and mAP@0.50, with larger gains at the stricter threshold
2. **IoU-aware head** will show improvements primarily at mAP@0.50 due to direct metric alignment
3. Both approaches will particularly benefit classes with diverse object sizes

### 7.2 Predicted Performance

| Model | Expected mAP@0.25 | Expected mAP@0.50 |
|-------|-------------------|-------------------|
| CrossAttn Dual (baseline) | 69-70% | 49-51% |
| CrossAttn Dual + Centerness | **70-72%** | **52-55%** |
| CrossAttn Dual + IoU | 70-71% | 51-53% |

### 7.3 Per-Class Predictions

Classes expected to benefit most from quality-aware heads:

| Class | Reason |
|-------|--------|
| table | Large, flat objects with ambiguous boundaries |
| desk | Similar to table, boundary uncertainty |
| bookshelf | Elongated shape, many boundary points |
| bathtub | Complex geometry |

---

## 8. Ablation Studies (Proposed)

### 8.1 Centerness Weight Ablation

| use_centerness_weight | Description |
|-----------------------|-------------|
| True | Bbox loss weighted by centerness targets |
| False | Unweighted bbox loss |

### 8.2 Loss Weight Ablation

| λ_cent / λ_iou | Description |
|----------------|-------------|
| 0.5 | Lower auxiliary loss weight |
| 1.0 | Equal weight (default) |
| 2.0 | Higher auxiliary loss weight |

### 8.3 Combined Approach (Future Work)

Potentially combine both centerness and IoU in a single head:

```
final_score = cls × (α × centerness + (1-α) × iou)
```

---

## 9. Conclusion

This report presents two quality-aware extensions to the TR3D+FF Cross-Attention Dual model. The centerness-aware approach follows proven techniques from FCAF3D, while the IoU-aware approach explores direct metric alignment. Both methods address the fundamental misalignment between classification confidence and localization quality in anchor-free 3D detection.

We recommend prioritizing the **centerness-aware approach** for initial experiments due to its proven effectiveness and training stability. The IoU-aware approach offers an interesting alternative that directly optimizes for the evaluation metric but may require additional tuning.

---

## 10. References

1. Rukhovich, D., Vorontsova, A., & Konushin, A. (2022). FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection. ECCV 2022.

2. Rukhovich, D., Vorontsova, A., & Konushin, A. (2023). TR3D: Towards Real-Time Indoor 3D Object Detection. arXiv:2302.02858.

3. Tian, Z., Shen, C., Chen, H., & He, T. (2019). FCOS: Fully Convolutional One-Stage Object Detection. ICCV 2019.

4. Jiang, B., Luo, R., Mao, J., Xiao, T., & Jiang, Y. (2018). Acquisition of Localization Confidence for Accurate Object Detection. ECCV 2018.

5. Zhang, S., Chi, C., Yao, Y., Lei, Z., & Li, S. Z. (2020). Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection. CVPR 2020.

---

## Appendix A: Configuration Files

### A.1 Centerness Configuration

```python
# configs/tr3d/ff_crossattn/tr3d-ff_crossattn_dual_centerness_sunrgbd.py
head=dict(
    type='TR3DHeadWithCenterness',
    in_channels=128,
    n_reg_outs=8,
    n_classes=10,
    voxel_size=0.01,
    assigner=dict(
        type='TR3DAssignerWithCenterness',
        top_pts_threshold=6,
        label2level=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]),
    bbox_loss=dict(type='RotatedIoU3DLoss', mode='diou', reduction='none'),
    cls_loss=dict(type='FocalLoss', reduction='none'),
    centerness_loss_weight=1.0,
    use_centerness_weight=True)
```

### A.2 IoU Configuration

```python
# configs/tr3d/ff_crossattn/tr3d-ff_crossattn_dual_iou_sunrgbd.py
head=dict(
    type='TR3DHeadIoU',
    in_channels=128,
    n_reg_outs=8,
    n_classes=10,
    voxel_size=0.01,
    assigner=dict(
        type='TR3DAssigner',
        top_pts_threshold=6,
        label2level=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]),
    bbox_loss=dict(type='RotatedIoU3DLoss', mode='diou', reduction='none'),
    cls_loss=dict(type='FocalLoss', reduction='none'),
    iou_loss_weight=1.0,
    use_iou_weight=False)
```

---

## Appendix B: Training Commands

```bash
# Centerness variant
python tools/train.py configs/tr3d/ff_crossattn/tr3d-ff_crossattn_dual_centerness_sunrgbd.py \
    --work-dir work_dirs/crossattn_dual_centerness

# IoU variant
python tools/train.py configs/tr3d/ff_crossattn/tr3d-ff_crossattn_dual_iou_sunrgbd.py \
    --work-dir work_dirs/crossattn_dual_iou
```

---

*Report prepared for TR3D+FF Cross-Attention research experiments*
*Date: December 2024*
