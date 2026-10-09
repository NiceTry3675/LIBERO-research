"""A height map of the table from the top and agent depth cameras, with the robot removed.

World grid of CELL cm cells. A cell's height is the highest non-robot top-camera point above the table plane in
it (the straight-down view sees tops); cells the top camera does not see (under an arm, or tall far objects
outside its view) take the agent camera's height, after dropping its silhouette pixels. Cells covered only by
robot points are UNKNOWN (NaN), never empty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import robot_model
from .capture import Capture
from .geometry import backproject, fit_table_plane, height_above

CELL = 0.5                       # cm
X_RANGE = (-60.0, 60.0)
Y_RANGE = (-52.0, 36.0)
BODY_Y = -45.0                   # the robot stands at y = -65; objects spawn at y >= -35


@dataclass
class HeightMap:
    height: np.ndarray           # cm above the table plane; NaN = unknown
    source: np.ndarray           # 0 unknown, 1 top camera, 2 agent camera
    robot: np.ndarray            # cell had robot points and no object point
    rgb: np.ndarray              # colour of the point that gave the height (top camera raw image), uint8
    plane: tuple                 # (normal, offset) of the table plane
    table_z: float

    def cell_centre(self, iy, ix) -> tuple[float, float]:
        return X_RANGE[0] + (ix + 0.5) * CELL, Y_RANGE[0] + (iy + 0.5) * CELL

    def world_xy(self) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.height.shape
        xs = X_RANGE[0] + (np.arange(nx) + 0.5) * CELL
        ys = Y_RANGE[0] + (np.arange(ny) + 0.5) * CELL
        return np.meshgrid(xs, ys)


def grid_shape() -> tuple[int, int]:
    return int(round((Y_RANGE[1] - Y_RANGE[0]) / CELL)), int(round((X_RANGE[1] - X_RANGE[0]) / CELL))


def _cells(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.floor((P[:, 0] - X_RANGE[0]) / CELL).astype(int)
    iy = np.floor((P[:, 1] - Y_RANGE[0]) / CELL).astype(int)
    ny, nx = grid_shape()
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return iy, ix, ok


def _silhouette(depth_mm: np.ndarray, jump_mm: float = 20.0) -> np.ndarray:
    """Pixels whose depth jumps by more than jump_mm to a 4-neighbour: mixed foreground/background silhouettes."""
    d = depth_mm.astype(np.float64)
    s = np.zeros(d.shape, dtype=bool)
    s[:, 1:] |= np.abs(d[:, 1:] - d[:, :-1]) > jump_mm
    s[:, :-1] |= np.abs(d[:, 1:] - d[:, :-1]) > jump_mm
    s[1:] |= np.abs(d[1:] - d[:-1]) > jump_mm
    s[:-1] |= np.abs(d[1:] - d[:-1]) > jump_mm
    return s


def build(cap: Capture, boxes: dict, plane=None, robot_margin: float = 1.0, with_rgb: bool = True) -> HeightMap:
    ny, nx = grid_shape()
    height = np.full((ny, nx), np.nan)
    source = np.zeros((ny, nx), dtype=np.uint8)
    robot = np.zeros((ny, nx), dtype=bool)
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    P_top, valid_top = backproject(cap.depth("top_camera"), cap.K("top_camera"), cap.E("top_camera"))
    if plane is None:
        plane = fit_table_plane(P_top[valid_top], cap.table_z)
    image = cap.rgb("top_camera")[::2, ::2] if with_rgb else None
    for cam, code in (("top_camera", 1), ("agent_camera", 2)):
        if cam == "top_camera":
            P, valid = P_top, valid_top
        else:
            P, valid = backproject(cap.depth(cam), cap.K(cam), cap.E(cam))
            valid &= ~_silhouette(cap.depth(cam))
        is_robot = robot_model.mask(P, cap, boxes, robot_margin) | (P[..., 1] < BODY_Y)
        h = height_above(P.reshape(-1, 3), plane).reshape(P.shape[:2])
        sel = valid & (h > -3.0)                       # below the table: floor or fallen objects, not table cells
        pts, hs, rob = P[sel], h[sel], is_robot[sel]
        iy, ix, ok = _cells(pts)
        iy, ix, hs, rob = iy[ok], ix[ok], hs[ok], rob[ok]
        # robot-only cells
        robot_cells = np.zeros((ny, nx), dtype=bool)
        robot_cells[iy[rob], ix[rob]] = True
        obj = ~rob
        if code == 1:
            np.fmax.at(height, (iy[obj], ix[obj]), hs[obj])
            source[iy[obj], ix[obj]] = 1
            robot |= robot_cells & (source == 0)
            if image is not None:
                rows, cols = np.nonzero(sel)
                rows, cols = rows[ok][obj], cols[ok][obj]
                order = np.argsort(hs[obj])            # the highest point in a cell wins (written last)
                rgb[iy[obj][order], ix[obj][order]] = image[rows[order], cols[order]]
        else:
            fill = (source[iy[obj], ix[obj]] == 0)
            tmp = np.full((ny, nx), np.nan)
            np.fmax.at(tmp, (iy[obj][fill], ix[obj][fill]), hs[obj][fill])
            new = np.isfinite(tmp) & (source == 0)
            height[new] = tmp[new]
            source[new] = 2
            robot |= robot_cells & (source == 0)
            if with_rgb:                                # colour for cells only the agent camera sees
                agent_img = cap.rgb("agent_camera")[::2, ::2]
                rows, cols = np.nonzero(sel)
                rows, cols = rows[ok][obj][fill], cols[ok][obj][fill]
                cy, cx = iy[obj][fill], ix[obj][fill]
                take = new[cy, cx]
                order = np.argsort(hs[obj][fill][take])
                rgb[cy[take][order], cx[take][order]] = agent_img[rows[take][order], cols[take][order]]
    return HeightMap(height, source, robot & (source == 0), rgb, plane, cap.table_z)
