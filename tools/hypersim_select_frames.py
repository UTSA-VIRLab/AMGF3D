"""Select Hypersim frames for the synthetic robustness benchmark (R2-W2 / R1-W2).

- Eval  = ALL Omni3D val frames with >=1 usable SUN-10 box  (~5.5k, SUN-val scale)
- Train = subsample of Omni3D train frames-with-SUN10 to ~6k (SUN-train scale):
    * keep every frame containing a rare class (bed, night stand, bathtub, toilet, dresser)
    * fill remaining quota scene-balanced, per-camera stride to cut view overlap
Writes compact pickles (images + filtered annos) so the 2.6GB json is loaded once,
plus a download manifest {scene: [zip-member suffixes]}.
"""
import json, pickle, collections, random

SUN10 = {'bed','table','sofa','chair','toilet','desk','dresser','night stand','bookcase','bathtub'}
RARE  = {'bed','night stand','bathtub','toilet','dresser'}
TRAIN_QUOTA = 6000
random.seed(0)

def usable(a):
    return (a.get('valid3D', True) and not a.get('behind_camera', False)
            and a['category_name'] in SUN10
            and a.get('visibility', 1) >= 0.25 and a.get('segmentation_pts', 999) >= 50)

out_manifest = collections.defaultdict(set)
for split in ['val','train']:
    d = json.load(open(f'data/hypersim_omni3d/datasets/Omni3D/Hypersim_{split}.json'))
    imgs = {im['id']: im for im in d['images']}
    by_img = collections.defaultdict(list)
    for a in d['annotations']:
        if usable(a):
            by_img[a['image_id']].append(a)

    if split == 'val':
        chosen = set(by_img)
    else:
        rare_frames = {i for i, anns in by_img.items()
                       if any(a['category_name'] in RARE for a in anns)}
        chosen = set(rare_frames)
        # scene-balanced fill with per-camera stride 3
        by_scene = collections.defaultdict(list)
        for i in by_img:
            if i in chosen: continue
            fp = imgs[i]['file_path'].split('/')          # hypersim/ai_X/images/scene_cam_YY_final_preview/frame.ZZZZ...
            scene, cam, frame = fp[1], fp[3], int(fp[4].split('.')[1])
            if frame % 3 == 0:
                by_scene[scene].append(i)
        for s in by_scene: random.shuffle(by_scene[s])
        scenes = sorted(by_scene)
        k = 0
        while len(chosen) < TRAIN_QUOTA and any(by_scene.values()):
            s = scenes[k % len(scenes)]; k += 1
            if by_scene[s]: chosen.add(by_scene[s].pop())

    # compact pickle for the converter
    keep_imgs = {i: imgs[i] for i in chosen}
    keep_anns = {i: by_img[i] for i in chosen}
    with open(f'data/hypersim_omni3d/compact_{split}.pkl','wb') as f:
        pickle.dump({'images': keep_imgs, 'annos': keep_anns}, f)

    # manifest members
    hist = collections.Counter()
    for i in chosen:
        fp = imgs[i]['file_path']                          # hypersim/ai_X/images/scene_cam_YY_final_preview/frame.ZZZZ.tonemap.jpg
        parts = fp.split('/'); scene = parts[1]
        rel_rgb = '/'.join(parts[2:])                      # images/scene_cam_YY_final_preview/frame.ZZZZ.tonemap.jpg
        rel_dep = rel_rgb.replace('_final_preview','_geometry_hdf5').replace('.tonemap.jpg','.depth_meters.hdf5')
        out_manifest[scene].update([rel_rgb, rel_dep])
        for a in keep_anns[i]: hist[a['category_name']] += 1
    print(f"{split}: frames={len(chosen)} scenes={len({imgs[i]['file_path'].split('/')[1] for i in chosen})}")
    print("  ", dict(sorted(hist.items(), key=lambda x:-x[1])))

manifest = {s: sorted(v) for s, v in out_manifest.items()}
json.dump(manifest, open('data/hypersim_omni3d/download_manifest.json','w'))
n_files = sum(len(v) for v in manifest.values())
print(f"manifest: {len(manifest)} scenes, {n_files} files (~{n_files*1.4/1024:.0f} GB rough)")
