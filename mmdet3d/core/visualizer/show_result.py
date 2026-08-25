# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import mmcv
import numpy as np
import trimesh
import matplotlib.pyplot as plt

from .image_vis import (draw_camera_bbox3d_on_img, draw_depth_bbox3d_on_img,
                        draw_lidar_bbox3d_on_img)


def _write_obj(points, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


def _write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes.

    Args:
        scene_bbox(list[ndarray] or ndarray): xyz pos of center and
            3 lengths (x_size, y_size, z_size) and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename(str): Filename.
    """

    def heading2rotmat(heading_angle):
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    if len(scene_bbox) == 0:
        scene_bbox = np.zeros((1, 7))
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to obj file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='obj')

    return


def _write_oriented_bbox_v2(corners, labels, out_filename):
    colors = np.multiply([
        plt.cm.get_cmap('nipy_spectral', 19)((i * 5 + 11) % 18 + 1)[:3] for i in range(18)
    ], 255).astype(np.uint8).tolist()
    with open(out_filename, 'w') as file:
        for i, (corner, label) in enumerate(zip(corners, labels)):
            c = colors[label]
            for p in corner:
                file.write(f'v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n')
            j = i * 8 + 1
            for k in [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                      [2, 3, 7, 6], [3, 0, 4, 7], [1, 2, 6, 5]]:
                file.write('f')
                for l in k:
                    file.write(f' {j + l}')
                file.write('\n')


def show_result(points,
                gt_bboxes,
                pred_bboxes,
                out_dir,
                filename,
                show=False,
                snapshot=False,
                pred_labels=None):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results.
            Defaults to False.
        pred_labels (np.ndarray, optional): Predicted labels of boxes.
            Defaults to None.
    """
    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if show:
        from .open3d_vis import Visualizer

        vis = Visualizer(points)
        if pred_bboxes is not None:
            if pred_labels is None:
                vis.add_bboxes(bbox3d=pred_bboxes)
            else:
                palette = np.random.randint(
                    0, 255, size=(pred_labels.max() + 1, 3)) / 256
                labelDict = {}
                for j in range(len(pred_labels)):
                    i = int(pred_labels[j].numpy())
                    if labelDict.get(i) is None:
                        labelDict[i] = []
                    labelDict[i].append(pred_bboxes[j])
                for i in labelDict:
                    vis.add_bboxes(
                        bbox3d=np.array(labelDict[i]),
                        bbox_color=palette[i],
                        points_in_box_color=palette[i])

        if gt_bboxes is not None:
            vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_bboxes is not None:
        # bottom center to gravity center
        gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2

        _write_oriented_bbox(gt_bboxes,
                             osp.join(result_path, f'{filename}_gt.obj'))

    if pred_bboxes is not None:
        # bottom center to gravity center
        pred_bboxes[..., 2] += pred_bboxes[..., 5] / 2

        _write_oriented_bbox(pred_bboxes,
                             osp.join(result_path, f'{filename}_pred.obj'))


def show_result_v2(points,
                   gt_corners,
                   gt_labels,
                   pred_corners,
                   pred_labels,
                   out_dir,
                   filename,
                   show=False,
                   snapshot=False):
    """Visualize detection results with ground truth and predictions using corners format.
    
    Args:
        points (np.ndarray): Point cloud.
        gt_corners (np.ndarray): Ground truth boxes in corners format [N, 8, 3].
        gt_labels (np.ndarray): Ground truth labels.
        pred_corners (np.ndarray): Predicted boxes in corners format [M, 8, 3].
        pred_labels (np.ndarray): Predicted labels.
        out_dir (str): Output directory.
        filename (str): Filename.
        show (bool, optional): Visualize online. Defaults to False.
        snapshot (bool, optional): Save PNG snapshot. Defaults to False.
    """
    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    # Generate PNG images if show or snapshot is enabled
    if show or snapshot:
        try:
            from .open3d_vis import Visualizer
            import os
            
            # Check if we have a display (for headless environments)
            has_display = os.environ.get('DISPLAY') is not None
            
            vis = Visualizer(points)
            
            # Add predicted bounding boxes with color per class
            if pred_corners is not None and len(pred_corners) > 0:
                # Convert corners [N, 8, 3] to bbox format [N, 7] where 7 = [x, y, z, x_size, y_size, z_size, yaw]
                # For axis-aligned boxes (ARKitScenes uses axis-aligned boxes)
                pred_bboxes = np.zeros((len(pred_corners), 7))
                for i in range(len(pred_corners)):
                    corners = pred_corners[i]  # [8, 3]
                    # Get min and max along each axis to find bounding box
                    min_vals = corners.min(axis=0)  # [x_min, y_min, z_min]
                    max_vals = corners.max(axis=0)  # [x_max, y_max, z_max]
                    # Center (gravity center)
                    center = (min_vals + max_vals) / 2.0
                    # Dimensions (width, height, depth)
                    dims = max_vals - min_vals
                    pred_bboxes[i, :3] = center
                    pred_bboxes[i, 3:6] = dims
                    pred_bboxes[i, 6] = 0.0  # No rotation for axis-aligned boxes
                
                if pred_labels is not None and len(pred_labels) > 0:
                    # Create color palette for different classes
                    # Ensure colors are in [0, 1] range for Open3D (strictly less than 1.0)
                    max_label = int(pred_labels.max())
                    # Generate colors in [0.1, 0.9] range to avoid edge cases
                    np.random.seed(42)
                    palette = np.random.rand(max_label + 1, 3) * 0.8 + 0.1
                    
                    # Group boxes by label
                    label_dict = {}
                    for j in range(len(pred_labels)):
                        label = int(pred_labels[j])
                        if label < 0 or label > max_label:
                            continue  # Skip invalid labels
                        if label not in label_dict:
                            label_dict[label] = []
                        label_dict[label].append(pred_bboxes[j])
                    
                    # Add boxes with colors per class
                    for label, boxes in label_dict.items():
                        if label >= 0 and label <= max_label:
                            color = tuple(palette[label].clip(0.0, 0.99))  # Ensure [0, 0.99]
                            vis.add_bboxes(
                                bbox3d=np.array(boxes),
                                bbox_color=color,
                                points_in_box_color=color)
                else:
                    vis.add_bboxes(bbox3d=pred_bboxes)
            
            # Add ground truth bounding boxes in blue
            if gt_corners is not None and len(gt_corners) > 0:
                # Convert corners [N, 8, 3] to bbox format [N, 7]
                gt_bboxes = np.zeros((len(gt_corners), 7))
                for i in range(len(gt_corners)):
                    corners = gt_corners[i]  # [8, 3]
                    min_vals = corners.min(axis=0)
                    max_vals = corners.max(axis=0)
                    center = (min_vals + max_vals) / 2.0
                    dims = max_vals - min_vals
                    gt_bboxes[i, :3] = center
                    gt_bboxes[i, 3:6] = dims
                    gt_bboxes[i, 6] = 0.0
                
                vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 0.99))  # Blue for GT (clamped)
            
                # Save PNG if snapshot is enabled
            if snapshot:
                show_path = osp.join(result_path, f'{filename}_online.png')
                try:
                    # Try to render and capture image
                    # For headless mode, create_window() might fail, so we handle it
                    vis.o3d_visualizer.poll_events()
                    vis.o3d_visualizer.update_renderer()
                    # Capture image (this should work if window was created)
                    vis.o3d_visualizer.capture_screen_image(show_path, do_render=True)
                except Exception as e:
                    # If capture fails, try using show() method (might open window briefly)
                    try:
                        # For headless: try to capture without running window
                        if has_display:
                            vis.show(show_path)
                        else:
                            # Headless mode: try to render anyway
                            vis.o3d_visualizer.poll_events()
                            vis.o3d_visualizer.update_renderer()
                            vis.o3d_visualizer.capture_screen_image(show_path, do_render=True)
                    except Exception as e2:
                        print(f"Warning: Could not render PNG for {filename}: {e2}. Skipping PNG.")
                finally:
                    # Clean up window
                    try:
                        vis.o3d_visualizer.destroy_window()
                    except:
                        pass
            elif show:
                vis.show()
            else:
                # Just clean up if no show or snapshot
                try:
                    vis.o3d_visualizer.destroy_window()
                except:
                    pass
        except Exception as e:
            # If visualization fails, continue with .obj files only
            print(f"Warning: Visualization failed for {filename}: {e}. Continuing with .obj files only.")

    # Always write .obj files for MeshLab
    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_corners is not None:
        _write_oriented_bbox_v2(gt_corners, gt_labels,
                                osp.join(result_path, f'{filename}_gt.obj'))

    if pred_corners is not None:
        _write_oriented_bbox_v2(pred_corners, pred_labels,
                                osp.join(result_path, f'{filename}_pred.obj'))


def show_seg_result(points,
                    gt_seg,
                    pred_seg,
                    out_dir,
                    filename,
                    palette,
                    ignore_index=None,
                    show=False,
                    snapshot=False):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_seg (np.ndarray): Ground truth segmentation mask.
        pred_seg (np.ndarray): Predicted segmentation mask.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        palette (np.ndarray): Mapping between class labels and colors.
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. Defaults to None.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results.
            Defaults to False.
    """
    # we need 3D coordinates to visualize segmentation mask
    if gt_seg is not None or pred_seg is not None:
        assert points is not None, \
            '3D coordinates are required for segmentation visualization'

    # filter out ignored points
    if gt_seg is not None and ignore_index is not None:
        if points is not None:
            points = points[gt_seg != ignore_index]
        if pred_seg is not None:
            pred_seg = pred_seg[gt_seg != ignore_index]
        gt_seg = gt_seg[gt_seg != ignore_index]

    if gt_seg is not None:
        gt_seg_color = palette[gt_seg]
        gt_seg_color = np.concatenate([points[:, :3], gt_seg_color], axis=1)
    if pred_seg is not None:
        pred_seg_color = palette[pred_seg]
        pred_seg_color = np.concatenate([points[:, :3], pred_seg_color],
                                        axis=1)

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    # online visualization of segmentation mask
    # we show three masks in a row, scene_points, gt_mask, pred_mask
    if show:
        from .open3d_vis import Visualizer
        mode = 'xyzrgb' if points.shape[1] == 6 else 'xyz'
        vis = Visualizer(points, mode=mode)
        if gt_seg is not None:
            vis.add_seg_mask(gt_seg_color)
        if pred_seg is not None:
            vis.add_seg_mask(pred_seg_color)
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_seg is not None:
        _write_obj(gt_seg_color, osp.join(result_path, f'{filename}_gt.obj'))

    if pred_seg is not None:
        _write_obj(pred_seg_color, osp.join(result_path,
                                            f'{filename}_pred.obj'))


def show_multi_modality_result(img,
                               gt_bboxes,
                               pred_bboxes,
                               proj_mat,
                               out_dir,
                               filename,
                               box_mode='lidar',
                               img_metas=None,
                               show=False,
                               gt_bbox_color=(61, 102, 255),
                               pred_bbox_color=(241, 101, 72)):
    """Convert multi-modality detection results into 2D results.

    Project the predicted 3D bbox to 2D image plane and visualize them.

    Args:
        img (np.ndarray): The numpy array of image in cv2 fashion.
        gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
        pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
        proj_mat (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        out_dir (str): Path of output directory.
        filename (str): Filename of the current frame.
        box_mode (str, optional): Coordinate system the boxes are in.
            Should be one of 'depth', 'lidar' and 'camera'.
            Defaults to 'lidar'.
        img_metas (dict, optional): Used in projecting depth bbox.
            Defaults to None.
        show (bool, optional): Visualize the results online. Defaults to False.
        gt_bbox_color (str or tuple(int), optional): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61).
        pred_bbox_color (str or tuple(int), optional): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241).
    """
    if box_mode == 'depth':
        draw_bbox = draw_depth_bbox3d_on_img
    elif box_mode == 'lidar':
        draw_bbox = draw_lidar_bbox3d_on_img
    elif box_mode == 'camera':
        draw_bbox = draw_camera_bbox3d_on_img
    else:
        raise NotImplementedError(f'unsupported box mode {box_mode}')

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if show:
        show_img = img.copy()
        if gt_bboxes is not None:
            show_img = draw_bbox(
                gt_bboxes, show_img, proj_mat, img_metas, color=gt_bbox_color)
        if pred_bboxes is not None:
            show_img = draw_bbox(
                pred_bboxes,
                show_img,
                proj_mat,
                img_metas,
                color=pred_bbox_color)
        mmcv.imshow(show_img, win_name='project_bbox3d_img', wait_time=0)

    if img is not None:
        mmcv.imwrite(img, osp.join(result_path, f'{filename}_img.png'))

    if gt_bboxes is not None:
        gt_img = draw_bbox(
            gt_bboxes, img, proj_mat, img_metas, color=gt_bbox_color)
        mmcv.imwrite(gt_img, osp.join(result_path, f'{filename}_gt.png'))

    if pred_bboxes is not None:
        pred_img = draw_bbox(
            pred_bboxes, img, proj_mat, img_metas, color=pred_bbox_color)
        mmcv.imwrite(pred_img, osp.join(result_path, f'{filename}_pred.png'))
