#!/usr/bin/env python3
"""per_scene_eval.py — Stage B for Figure 3.

For each (method, dataset), reads the prediction pickle dumped by tools/test.py
and emits per-scene TP/FP/FN counts at IoU=0.25 using the same matching logic
as mmdet3d's `indoor_eval`. Used for Fig 3's scene-ranking + caption numbers.

Usage:
  python tools/per_scene_eval.py \
      --config configs/.../some_config.py \
      --predictions Paper_Submission_and_Reviews/revision_artifacts/figure_assets/predictions/amcgf3d/sunrgbd.pkl \
      --output      Paper_Submission_and_Reviews/revision_artifacts/figure_assets/per_scene_stats/amcgf3d_sunrgbd.json \
      --score-thr 0.3 \
      --iou-thr 0.25

Audit: also prints aggregate TP / (TP+FN) (i.e. recall@iou_thr@score_thr)
and TP / (TP+FP) (precision) over the whole val set. Compare to mmdet3d's
own evaluate() output to confirm matching parity.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from mmcv import Config

from mmdet3d.datasets import build_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    # mmdet3d test.py --out writes a list[dict] with the same length as the val set.
    with open(args.predictions, "rb") as f:
        dt_annos = pickle.load(f)
    assert len(dt_annos) == len(dataset), \
        f"predictions ({len(dt_annos)}) vs dataset ({len(dataset)}) length mismatch"

    box_type_3d = dataset.box_type_3d
    box_mode_3d = dataset.box_mode_3d
    label2cat = {i: c for i, c in enumerate(dataset.CLASSES)}

    per_scene = []
    agg_tp = 0
    agg_fp = 0
    agg_fn = 0

    for img_id in range(len(dataset)):
        det = dt_annos[img_id]
        # tools/test.py wraps SUN RGB-D / ScanNet / ARKit predictions either flat
        # or under 'pts_bbox'. Handle both.
        if "pts_bbox" in det:
            det = det["pts_bbox"]

        # Predictions in this scene
        pred_boxes = det["boxes_3d"]                       # BaseInstance3DBoxes
        pred_scores = det["scores_3d"].cpu().numpy() if torch.is_tensor(det["scores_3d"]) else np.asarray(det["scores_3d"])
        pred_labels = det["labels_3d"].cpu().numpy() if torch.is_tensor(det["labels_3d"]) else np.asarray(det["labels_3d"])
        # Apply score threshold
        keep = pred_scores >= args.score_thr
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        # GT for this scene
        gt_anno = dataset.data_infos[img_id]["annos"]
        gt_num = gt_anno.get("gt_num", 0)
        if gt_num > 0:
            gt_boxes_np = gt_anno["gt_boxes_upright_depth"]
            gt_boxes = box_type_3d(
                gt_boxes_np,
                box_dim=gt_boxes_np.shape[-1],
                origin=(0.5, 0.5, 0.5),
            ).convert_to(box_mode_3d)
            gt_labels = np.asarray(gt_anno["class"])
        else:
            gt_boxes = box_type_3d(np.zeros((0, 7), dtype=np.float32))
            gt_labels = np.zeros(0, dtype=np.int64)

        scene_tp = 0
        scene_fp = 0
        scene_fn = 0
        per_class = {}

        # Convert prediction boxes to the dataset's box_mode_3d so IoU is in
        # the right coord. system.
        if len(pred_boxes):
            pred_boxes = pred_boxes.convert_to(box_mode_3d)

        for cls_id in set(np.concatenate([pred_labels, gt_labels]).astype(np.int64).tolist()):
            p_mask = pred_labels == cls_id
            g_mask = gt_labels == cls_id
            n_p = int(p_mask.sum())
            n_g = int(g_mask.sum())

            if n_p == 0 and n_g == 0:
                continue
            # FN-only or FP-only fast paths
            if n_p == 0:
                scene_fn += n_g
                per_class[label2cat[cls_id]] = {"tp": 0, "fp": 0, "fn": n_g, "n_gt": n_g, "n_pred": 0}
                continue
            if n_g == 0:
                scene_fp += n_p
                per_class[label2cat[cls_id]] = {"tp": 0, "fp": n_p, "fn": 0, "n_gt": 0, "n_pred": n_p}
                continue

            p_boxes_c = pred_boxes[p_mask]
            g_boxes_c = gt_boxes[g_mask]
            # Sort predictions in this class by score (high -> low)
            order = np.argsort(-pred_scores[p_mask])
            ious = p_boxes_c.overlaps(p_boxes_c, g_boxes_c).cpu().numpy()  # [n_p, n_g]
            matched = np.zeros(n_g, dtype=bool)
            cls_tp = cls_fp = 0
            for p_idx in order:
                # Find best unmatched GT of this class with IoU >= threshold
                best = -1; best_iou = args.iou_thr - 1e-9
                for g_idx in range(n_g):
                    if matched[g_idx]:
                        continue
                    if ious[p_idx, g_idx] > best_iou:
                        best_iou = ious[p_idx, g_idx]; best = g_idx
                if best >= 0:
                    cls_tp += 1
                    matched[best] = True
                else:
                    cls_fp += 1
            cls_fn = int((~matched).sum())
            scene_tp += cls_tp; scene_fp += cls_fp; scene_fn += cls_fn
            per_class[label2cat[cls_id]] = {
                "tp": cls_tp, "fp": cls_fp, "fn": cls_fn, "n_gt": n_g, "n_pred": n_p,
            }

        # Resolve a stable scene identifier — different datasets store it differently.
        info = dataset.data_infos[img_id]
        scene_id = (info.get("point_cloud", {}).get("lidar_idx")
                    or info.get("pts_path")
                    or str(img_id))
        per_scene.append({
            "img_id": img_id,
            "scene_id": str(scene_id),
            "tp": scene_tp,
            "fp": scene_fp,
            "fn": scene_fn,
            "gt_num": int(gt_num),
            "per_class": per_class,
        })
        agg_tp += scene_tp; agg_fp += scene_fp; agg_fn += scene_fn

    summary = {
        "config": args.config,
        "predictions": args.predictions,
        "iou_thr": args.iou_thr,
        "score_thr": args.score_thr,
        "n_scenes": len(per_scene),
        "agg_tp": agg_tp,
        "agg_fp": agg_fp,
        "agg_fn": agg_fn,
        "precision": agg_tp / max(agg_tp + agg_fp, 1),
        "recall":    agg_tp / max(agg_tp + agg_fn, 1),
        "per_scene": per_scene,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[per_scene_eval] scenes={summary['n_scenes']} "
          f"TP={agg_tp} FP={agg_fp} FN={agg_fn} "
          f"prec={summary['precision']:.4f} rec={summary['recall']:.4f}")
    print(f"[per_scene_eval] -> {out_path}")


if __name__ == "__main__":
    main()
