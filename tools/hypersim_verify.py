"""Verify converted Hypersim frames: BEV overlay, fusion-projection check, point-in-box stats."""
import pickle, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

OUT = 'data/hypersim10'
FLIP = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
infos = pickle.load(open(f'{OUT}/hypersim_infos_val.pkl','rb'))
os.makedirs(f'{OUT}/_verify', exist_ok=True)
print(f"{len(infos)} frames")

def corners7(b):
    cx,cy,cz,dx,dy,dz,yaw = b
    c,s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    k = np.array([[x,y,z] for x in (-dx/2,dx/2) for y in (-dy/2,dy/2) for z in (-dz/2,dz/2)])
    return (R@k.T).T + [cx,cy,cz]

ious, inbox = [], []
for n,info in enumerate(infos[:8]):
    pts = np.fromfile(f"{OUT}/{info['pts_path']}",dtype=np.float32).reshape(-1,6)
    K, Rt = info['calib']['K'].astype(np.float64), info['calib']['Rt'].astype(np.float64)
    depth2img = K @ (FLIP @ Rt.T)              # exactly as SUNRGBDDataset builds it
    img = np.asarray(Image.open(f"{OUT}/sunrgbd_trainval/{info['image']['image_path']}"))
    H,W = info['image']['image_shape']
    fig,ax = plt.subplots(1,2,figsize=(16,7))
    ax[0].imshow(img)
    boxes = info['annos']['gt_boxes_upright_depth']
    for bi,b in enumerate(boxes):
        cs = corners7(b)
        # point-in-box (axis-aligned in yaw frame)
        cx,cy,cz,dx,dy,dz,yaw = b
        c,s = np.cos(-yaw), np.sin(-yaw)
        local = (pts[:,:3]-[cx,cy,cz]) @ np.array([[c,-s,0],[s,c,0],[0,0,1]]).T
        nin = np.sum((np.abs(local[:,0])<dx/2)&(np.abs(local[:,1])<dy/2)&(np.abs(local[:,2])<dz/2))
        inbox.append(nin)
        # project corners through the SAME matrix the fusion uses
        h = (depth2img @ cs.T)
        uv = (h[:2]/h[2]).T
        x0,y0,x1,y1 = uv[:,0].min(),uv[:,1].min(),uv[:,0].max(),uv[:,1].max()
        gt = info['annos']['bbox'][bi]
        ix0,iy0,ix1,iy1 = max(x0,gt[0]),max(y0,gt[1]),min(x1,gt[2]),min(y1,gt[3])
        inter = max(0,ix1-ix0)*max(0,iy1-iy0)
        union = (x1-x0)*(y1-y0)+(gt[2]-gt[0])*(gt[3]-gt[1])-inter
        ious.append(inter/max(union,1e-9))
        for (a,bidx) in [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]:
            ax[0].plot(uv[[a,bidx],0],uv[[a,bidx],1],'lime',lw=1)
        poly = cs[[0,2,6,4,0]][:, :2]
        ax[1].plot(poly[:,0],poly[:,1],'r-')
    ax[0].set_xlim(0,W); ax[0].set_ylim(H,0); ax[0].set_title('projected 3D boxes (fusion path)')
    ss = pts[::15]
    ax[1].scatter(ss[:,0],ss[:,1],s=0.3,c=ss[:,3:6]); ax[1].set_aspect('equal'); ax[1].set_title('BEV + boxes')
    fig.savefig(f'{OUT}/_verify/frame{n}.png',dpi=90,bbox_inches='tight'); plt.close(fig)
print(f"projected-vs-Omni3D 2D IoU: median={np.median(ious):.3f} min={np.min(ious):.3f}")
print(f"points-in-box: median={np.median(inbox):.0f} min={np.min(inbox)} (of ~130k)")
print("renders in", f'{OUT}/_verify/')
