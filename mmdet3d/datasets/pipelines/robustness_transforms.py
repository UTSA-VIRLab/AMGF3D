"""Robustness perturbation transforms for Step 7 of the ACCV revision plan.

These transforms degrade one modality at eval time to measure how much
AMGF3D vs TR3D+FF rely on each. They register into mmdet3d's PIPELINES
registry when this module is imported.

NOTE: this file is intentionally side-effect-isolated. The currently-running
trainers will not pick up these transforms because their imports are already
resolved in memory; they only run when this file is explicitly imported
(typically from `tools/run_robustness_eval.py`).

Perturbations exposed:
    - PointXYZNoise(sigma)        : Gaussian noise on point XYZ ("depth noise" proxy)
    - PointSubsample(keep_ratio)  : random uniform downsample of the point cloud
    - RgbGaussianNoise(sigma)     : Gaussian noise on the (normalized) image tensor
    - RgbDropout(patch_frac)      : zero out a square patch of the image
    - CalibrationRotation(sigma_deg): small random rotation of points (simulates
        per-frame extrinsic miscalibration; cheaper than rewriting proj_mat)
"""

import numpy as np
import torch

from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class PointXYZNoise:
    """Add Gaussian noise to the XYZ coordinates of each point.

    Args:
        sigma (float): standard deviation in metres. Levels per plan: 0.01, 0.05, 0.1.
    """

    def __init__(self, sigma: float):
        self.sigma = float(sigma)

    def __call__(self, results):
        if 'points' not in results:
            return results
        points = results['points']
        coords = points.coord.clone() if hasattr(points, 'coord') else None
        # mmdet3d's BasePoints exposes .tensor (the underlying torch.Tensor)
        t = points.tensor
        noise = torch.randn(t.shape[0], 3, device=t.device, dtype=t.dtype) * self.sigma
        t[:, :3] = t[:, :3] + noise
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(sigma={self.sigma})'


@PIPELINES.register_module()
class PointSubsample:
    """Randomly subsample the point cloud (without replacement).

    Args:
        keep_ratio (float): fraction of points to keep, in (0, 1]. Levels per plan: 0.75, 0.5, 0.25.
    """

    def __init__(self, keep_ratio: float):
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio = float(keep_ratio)

    def __call__(self, results):
        if 'points' not in results:
            return results
        points = results['points']
        n = points.tensor.shape[0]
        k = max(int(n * self.keep_ratio), 1)
        idx = torch.randperm(n)[:k]
        results['points'] = points[idx]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keep_ratio={self.keep_ratio})'


@PIPELINES.register_module()
class RgbGaussianNoise:
    """Add Gaussian noise to the image tensor AFTER Normalize (normalized space).

    Args:
        sigma (float): standard deviation in normalized-pixel units. Levels per plan: 0.05, 0.1, 0.2.
    """

    def __init__(self, sigma: float):
        self.sigma = float(sigma)

    def __call__(self, results):
        if 'img' not in results:
            return results
        img = results['img']
        # mmdet3d may store list of images for multi-view; handle both.
        if isinstance(img, list):
            results['img'] = [self._perturb(x) for x in img]
        else:
            results['img'] = self._perturb(img)
        return results

    def _perturb(self, img):
        if isinstance(img, np.ndarray):
            noise = np.random.randn(*img.shape).astype(img.dtype) * self.sigma
            return img + noise
        elif isinstance(img, torch.Tensor):
            return img + torch.randn_like(img) * self.sigma
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(sigma={self.sigma})'


@PIPELINES.register_module()
class RgbDropout:
    """Zero out a square patch of the image.

    Args:
        patch_frac (float): fraction of image area covered by the dropout patch
            (so patch side length = sqrt(patch_frac) * min(H, W)).
            Levels per plan: 0.25, 0.5, 0.75.
    """

    def __init__(self, patch_frac: float):
        assert 0.0 < patch_frac <= 1.0
        self.patch_frac = float(patch_frac)

    def __call__(self, results):
        if 'img' not in results:
            return results
        img = results['img']
        if isinstance(img, list):
            results['img'] = [self._perturb(x) for x in img]
        else:
            results['img'] = self._perturb(img)
        return results

    def _perturb(self, img):
        if isinstance(img, np.ndarray):
            H, W = img.shape[:2]
            patch = int((self.patch_frac ** 0.5) * min(H, W))
            if patch <= 0:
                return img
            y0 = np.random.randint(0, max(H - patch, 1))
            x0 = np.random.randint(0, max(W - patch, 1))
            img = img.copy()
            img[y0:y0 + patch, x0:x0 + patch, ...] = 0
            return img
        elif isinstance(img, torch.Tensor):
            # tensor is CHW
            C, H, W = img.shape if img.ndim == 3 else img.shape[-3:]
            patch = int((self.patch_frac ** 0.5) * min(H, W))
            if patch <= 0:
                return img
            y0 = np.random.randint(0, max(H - patch, 1))
            x0 = np.random.randint(0, max(W - patch, 1))
            img = img.clone()
            img[..., y0:y0 + patch, x0:x0 + patch] = 0
            return img
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(patch_frac={self.patch_frac})'


@PIPELINES.register_module()
class ExtrinsicRotationNoise:
    """TRUE extrinsic-calibration rotation error (Hypersim synthetic testbed).

    Right-multiplies the depth->image projection by a random small rotation
    (axis uniform on the sphere, angle ~ N(0, sigma_deg)), so the projection
    is computed as if the depth-to-RGB extrinsics were mis-calibrated.
    Points and ground-truth boxes are UNTOUCHED: unlike the main paper's
    point-rotation proxy (which R1-W2 noted also shifts the geometry-to-label
    relation), this perturbs only the 2D-3D correspondence. Full SO(3), not
    just yaw. Requires 'depth2img' in results (SUNRGBD-style datasets).
    """

    def __init__(self, sigma_deg: float):
        self.sigma_deg = float(sigma_deg)

    def __call__(self, results):
        if 'depth2img' not in results:
            return results
        axis = np.random.randn(3)
        axis /= (np.linalg.norm(axis) + 1e-12)
        ang = np.random.randn() * np.deg2rad(self.sigma_deg)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        dR = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        results['depth2img'] = np.asarray(results['depth2img']) @ dR
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(sigma_deg={self.sigma_deg})'


@PIPELINES.register_module()
class RayDepthNoise:
    """Sensor-model depth noise: distance error along the per-point ray.

    Points are in the depth frame with the camera at the origin, so the ray
    direction is x/||x||. Distance noise eps ~ N(0, sigma0 + sigma1 * d^2)
    (quadratic growth mimics stereo/ToF error); x' = x * (d + eps) / d.
    More physical than isotropic XYZ noise: errors are radial, growing with
    range, exactly the "errors in the depth map" R2 highlights.
    """

    def __init__(self, sigma0: float, sigma1: float = 0.0):
        self.sigma0, self.sigma1 = float(sigma0), float(sigma1)

    def __call__(self, results):
        if 'points' not in results:
            return results
        t = results['points'].tensor
        d = torch.norm(t[:, :3], dim=1).clamp(min=1e-6)
        sigma = self.sigma0 + self.sigma1 * d * d
        eps = torch.randn_like(d) * sigma
        t[:, :3] = t[:, :3] * ((d + eps) / d).unsqueeze(1)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(sigma0={self.sigma0}, sigma1={self.sigma1})'


@PIPELINES.register_module()
class PointColorDrop:
    """Zero the point-cloud color channels (dims 3:6).

    BMVC'26 rebuttal (R1-W2): companion to RgbDropout so that appearance is
    removed from BOTH branches -- the image fed to the 2D backbone AND the
    per-point color that forms the initial voxel features. Zeroing all point
    color is a strictly harsher variant than patch-consistent removal.
    """

    def __call__(self, results):
        if 'points' not in results:
            return results
        t = results['points'].tensor
        if t.shape[1] >= 6:
            t[:, 3:6] = 0.0
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@PIPELINES.register_module()
class CalibrationRotation:
    """Apply a small random rotation about the world z-axis to the point cloud.

    Used as a proxy for extrinsic miscalibration. Levels per plan: 1, 3, 5 degrees.
    The rotation is sampled isotropically around z; an x/y component would require
    deeper integration with the dataset's box transformation, which we avoid here.

    Args:
        sigma_deg (float): standard deviation of the rotation in degrees.
    """

    def __init__(self, sigma_deg: float):
        self.sigma_deg = float(sigma_deg)

    def __call__(self, results):
        if 'points' not in results:
            return results
        theta = np.random.randn() * np.deg2rad(self.sigma_deg)
        c, s = float(np.cos(theta)), float(np.sin(theta))
        # Rotate XY about origin
        points = results['points']
        t = points.tensor
        xy = t[:, :2].clone()
        t[:, 0] = c * xy[:, 0] - s * xy[:, 1]
        t[:, 1] = s * xy[:, 0] + c * xy[:, 1]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(sigma_deg={self.sigma_deg})'


def perturbation_spec(name: str, level: float):
    """Map (perturbation name, level) → list of pipeline-transform dicts to splice into a test pipeline.

    Insert these right after the natural loading position for the modality
    they perturb. The caller knows where (see tools/run_robustness_eval.py).
    """
    if name == 'points_xyz_noise':
        return [dict(type='PointXYZNoise', sigma=level)]
    if name == 'points_subsample':
        return [dict(type='PointSubsample', keep_ratio=level)]
    if name == 'rgb_gaussian':
        return [dict(type='RgbGaussianNoise', sigma=level)]
    if name == 'rgb_dropout':
        return [dict(type='RgbDropout', patch_frac=level)]
    if name == 'rgb_dropout_pointcolor':
        # Harsher R1-W2 variant: image patch dropout AND all point color zeroed,
        # so the 3D stream's color channels are not left clean.
        return [dict(type='RgbDropout', patch_frac=level),
                dict(type='PointColorDrop')]
    if name == 'calibration_rotation':
        return [dict(type='CalibrationRotation', sigma_deg=level)]
    if name == 'extrinsic_rotation':
        # true 2D-3D extrinsic error (projection-only; points/GT untouched)
        return [dict(type='ExtrinsicRotationNoise', sigma_deg=level)]
    if name == 'ray_depth_noise':
        # level = sigma0 in metres; quadratic term fixed to 0.002/m
        return [dict(type='RayDepthNoise', sigma0=level, sigma1=0.002)]
    raise ValueError(f'unknown perturbation: {name}')
