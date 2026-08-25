"""Convert Omni3D-annotated Hypersim frames into SUN-RGB-D-format data (BMVC'26 rebuttal
follow-up, R2-W2 synthetic testbed / R1-W2 exact-calibration perturbations).

Emits the EXACT layout SUNRGBDDataset expects (zero pipeline-code changes):
    data/hypersim10/points/{idx:06d}.bin              float32 [x y z r g b], rgb in [0,1]
    data/hypersim10/sunrgbd_trainval/image/{idx:06d}.jpg
    data/hypersim10/hypersim_infos_{split}.pkl        SUN pkl schema

Geometry (all EXACT, synthetic):
  - camera coords (Omni3D): x right, y down, z forward; per-image pinhole K.
  - per-frame gravity from box verticals: up_cam = sign-fixed mean of R_cam[:,1]
    (empirically 0.998 mean-resultant-length consistency).
  - depth frame: z=up, y=camera-forward projected horizontal, x=y×z (SUN convention).
  - Hypersim depth_meters is Euclidean distance along the ray -> planar conversion
    via x_cam = dist * K^-1 p / ||K^-1 p||.
  - boxes: corners = R@(±w/2,±h/2,±l/2)+c with dims=(w,h,l) (verified exact);
    7-DoF depth box = [R_g c, w, l, h, yaw=atan2 of R_g R[:,0]]; boxes tilted >15° dropped.
  - calib for the fusion projection: SUNRGBDDataset builds depth2img = K @ FLIP @ Rt^T
    with FLIP=[[1,0,0],[0,0,-1],[0,1,0]]; we store Rt = R_g @ FLIP so depth2img = K @ R_g^T.
"""
import os, sys, json, pickle, argparse
import numpy as np
import h5py
from PIL import Image
from multiprocessing import Pool

CLASSES = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
           'night_stand', 'bookshelf', 'bathtub')  # SUN RGB-D order
NAME_MAP = {'night stand': 'night_stand', 'bookcase': 'bookshelf'}
FLIP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
RAW = 'data/hypersim_omni3d/raw'
OUT = 'data/hypersim10'
MAX_PTS = 130000
TILT_DROP_DEG = 15.0

def frame_paths(file_path):
    # 'hypersim/ai_X/images/scene_cam_YY_final_preview/frame.ZZZZ.tonemap.jpg'
    parts = file_path.split('/')
    scene, rel_rgb = parts[1], '/'.join(parts[2:])
    rel_dep = rel_rgb.replace('_final_preview', '_geometry_hdf5').replace(
        '.tonemap.jpg', '.depth_meters.hdf5')
    return (os.path.join(RAW, scene, rel_rgb), os.path.join(RAW, scene, rel_dep))

def gravity_frame(anns):
    ups = np.stack([np.array(a['R_cam'])[:, 1] for a in anns])
    ups *= np.sign(ups @ ups[0])[:, None]
    up = ups.mean(0); up /= np.linalg.norm(up)
    if up @ np.array([0., -1., 0.]) < 0:      # object 'up' should oppose camera +y (down)
        up = -up
    spread = np.degrees(np.arccos(np.clip(ups @ up * np.sign(ups @ up), -1, 1))).max()
    fwd = np.array([0., 0., 1.]); fwd = fwd - (fwd @ up) * up
    if np.linalg.norm(fwd) < 0.1:             # camera looking straight up/down
        fwd = np.array([0., -1., 0.]); fwd = fwd - (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    R_g = np.stack([right, fwd, up])          # rows: depth axes in camera coords
    return R_g, spread

def convert_one(job):
    idx, img_info, anns = job
    rgb_p, dep_p = frame_paths(img_info['file_path'])
    if not (os.path.isfile(rgb_p) and os.path.isfile(dep_p)):
        return None
    K = np.array(img_info['K'])
    R_g, spread = gravity_frame(anns)

    # ---- points ----
    with h5py.File(dep_p, 'r') as f:
        dist = np.array(f['dataset']).astype(np.float64)      # 768x1024 ray distance
    H, W = dist.shape
    img = np.asarray(Image.open(rgb_p).convert('RGB'), dtype=np.float32) / 255.0
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(H * W)])
    rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)
    d = dist.ravel()
    ok = np.isfinite(d) & (d > 0.05) & (d < 30.0)
    x_cam = (rays[:, ok] * d[ok]).T
    col = img.reshape(-1, 3)[ok]
    if len(x_cam) > MAX_PTS:
        sel = np.random.RandomState(idx).choice(len(x_cam), MAX_PTS, replace=False)
        x_cam, col = x_cam[sel], col[sel]
    x_depth = (R_g @ x_cam.T).T
    pts = np.concatenate([x_depth, col], 1).astype(np.float32)

    # ---- boxes ----
    boxes, names, boxes2d, dropped = [], [], [], 0
    for a in anns:
        R = np.array(a['R_cam']); c = np.array(a['center_cam'])
        w, h, l = a['dimensions']
        up_axis = R_g @ R[:, 1]
        tilt = np.degrees(np.arccos(np.clip(abs(up_axis[2]), -1, 1)))
        if tilt > TILT_DROP_DEG:
            dropped += 1; continue
        c_d = R_g @ c
        a0 = R_g @ R[:, 0]
        yaw = float(np.arctan2(a0[1], a0[0]))
        boxes.append([c_d[0], c_d[1], c_d[2], w, l, h, yaw])
        names.append(NAME_MAP.get(a['category_name'], a['category_name']))
        bb = np.array(a['bbox2D_proj'], dtype=np.float64)
        boxes2d.append([max(bb[0], 0), max(bb[1], 0), min(bb[2], W), min(bb[3], H)])
    if not boxes:
        return None

    pts.tofile(f'{OUT}/points/{idx:06d}.bin')
    if not os.path.isfile(f'{OUT}/sunrgbd_trainval/image/{idx:06d}.jpg'):
        os.link(rgb_p, f'{OUT}/sunrgbd_trainval/image/{idx:06d}.jpg')

    boxes = np.array(boxes, dtype=np.float64)
    info = dict(
        point_cloud=dict(num_features=6, lidar_idx=idx),
        pts_path=f'points/{idx:06d}.bin',
        image=dict(image_idx=idx, image_shape=np.array([H, W], dtype=np.int32),
                   image_path=f'image/{idx:06d}.jpg'),
        calib=dict(K=K.astype(np.float32), Rt=(R_g @ FLIP).astype(np.float32)),
        annos=dict(
            gt_num=len(boxes),
            name=np.array(names),
            bbox=np.array(boxes2d, dtype=np.float64),
            location=boxes[:, :3].copy(),
            dimensions=boxes[:, 3:6].copy(),
            rotation_y=boxes[:, 6].copy(),
            index=np.arange(len(boxes), dtype=np.int32),
            **{'class': np.array([CLASSES.index(n) for n in names], dtype=np.int64)},
            gt_boxes_upright_depth=boxes,
        ),
    )
    return idx, info, spread, dropped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['train', 'val'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    os.makedirs(f'{OUT}/points', exist_ok=True)
    os.makedirs(f'{OUT}/sunrgbd_trainval/image', exist_ok=True)
    d = pickle.load(open(f'data/hypersim_omni3d/compact_{args.split}.pkl', 'rb'))
    items = sorted(d['images'].items())
    if args.limit: items = items[:args.limit]
    base = 0 if args.split == 'train' else 500000   # disjoint idx ranges
    jobs = [(base + n, im, d['annos'][i]) for n, (i, im) in enumerate(items)]

    infos, spreads, dropped, skipped = [], [], 0, 0
    with Pool(args.workers) as p:
        for r in p.imap_unordered(convert_one, jobs, chunksize=8):
            if r is None: skipped += 1; continue
            idx, info, spread, drp = r
            infos.append(info); spreads.append(spread); dropped += drp
    infos.sort(key=lambda x: x['image']['image_idx'])
    with open(f'{OUT}/hypersim_infos_{args.split}.pkl', 'wb') as f:
        pickle.dump(infos, f)
    print(f'{args.split}: wrote {len(infos)} frames (skipped {skipped}, boxes dropped {dropped})')
    print(f'gravity spread deg: median {np.median(spreads):.2f}  p95 {np.percentile(spreads,95):.2f}  max {np.max(spreads):.2f}')

if __name__ == '__main__':
    main()
