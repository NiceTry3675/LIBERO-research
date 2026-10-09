"""Camera geometry: depth -> world points, world points -> pixels, link frames.

Cameras follow OpenCV conventions with metres: x_cam = R x_world + t (E = [R t; 0 1]). Everything returned
here is in centimetres, the unit of the harness's state and of every fact shown to a model.
"""
from __future__ import annotations

import numpy as np


def backproject(depth_mm: np.ndarray, K: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(world points in cm, H x W x 3; valid mask H x W). Depth 0 marks pixels without a surface."""
    h, w = depth_mm.shape
    v, u = np.mgrid[0:h, 0:w]
    d = depth_mm.astype(np.float64) / 1000.0
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(d)], axis=-1)
    cam = rays * d[..., None]
    R, t = E[:3, :3], E[:3, 3]
    world = (cam - t) @ R          # R^T (x - t), row-vector form
    return world * 100.0, depth_mm > 0


def project(points_cm: np.ndarray, K: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pixel coordinates N x 2, depth in cm N) of world points in cm."""
    p = np.asarray(points_cm, dtype=np.float64).reshape(-1, 3) / 100.0
    cam = p @ E[:3, :3].T + E[:3, 3]
    uv = cam[:, :2] / cam[:, 2:3] @ np.diag([K[0, 0], K[1, 1]]) + K[:2, 2]
    return uv, cam[:, 2] * 100.0


def quat_to_matrix(q_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in q_wxyz)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def link_frame(pose) -> tuple[np.ndarray, np.ndarray]:
    """(rotation 3x3, origin in cm) of a link pose [x, y, z (m), qw, qx, qy, qz]."""
    return quat_to_matrix(pose[3:7]), np.asarray(pose[:3], dtype=np.float64) * 100.0


def fit_table_plane(points_cm: np.ndarray, table_z: float, band: float = 1.5, iters: int = 200,
                    inlier: float = 0.2, seed: int = 0) -> tuple[np.ndarray, float]:
    """RANSAC plane (unit normal n with n_z > 0, offset c: n.x = c) through points near the given table height."""
    near = points_cm[np.abs(points_cm[:, 2] - table_z) < band]
    if len(near) < 50:
        return np.array([0.0, 0.0, 1.0]), table_z
    rng = np.random.default_rng(seed)
    best, best_n, best_c = -1, None, None
    for _ in range(iters):
        a, b, c = near[rng.choice(len(near), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        n = n if n[2] > 0 else -n
        off = n @ a
        count = int((np.abs(near @ n - off) < inlier).sum())
        if count > best:
            best, best_n, best_c = count, n, off
    inl = near[np.abs(near @ best_n - best_c) < inlier]
    centroid = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    n = vt[2] if vt[2][2] > 0 else -vt[2]
    return n, float(n @ centroid)


def height_above(points_cm: np.ndarray, plane: tuple[np.ndarray, float]) -> np.ndarray:
    n, c = plane
    return points_cm @ n - c
