"""Objects on the table as connected raised regions of the height map, with their measurements.

Every measurement is in world units (cm, degrees) and computed only from the height map and its colours,
i.e. from depth, the top camera's raw image and the robot's own body.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from .heightmap import CELL, HeightMap

RAISED = 0.5           # cm above the table: depth is exact to 1 mm, so coasters (~0.9 cm) and cup floors stay in
MIN_AREA = 4.0         # cm2


@dataclass
class Blob:
    id: int
    iy: np.ndarray = field(repr=False)
    ix: np.ndarray = field(repr=False)
    centroid: tuple[float, float]          # x, y cm
    area: float                            # cm2
    length: float                          # long side of the minimum-area rectangle, cm
    width: float
    yaw: float                             # long-side direction, degrees in [0, 180)
    fill: float                            # area / (length * width)
    circularity: float
    h95: float                             # cm above the table
    hmed: float
    top_tilt: float | None                 # degrees from horizontal of the plane fitted to the top cells
    hollow: float                          # share of the eroded footprint lower than half of h95
    lab: tuple[float, float, float]        # median CIE Lab of the top cells (top camera raw image)
    robot_gap: float                       # cm to the nearest robot-only cell
    unknown_edge: float                    # share of the outline bordering unknown cells (hidden parts)
    flat: bool = False                     # found by colour only: a mat or pad thinner than the depth step

    def signature(self) -> dict:
        return {"h95": self.h95, "length": self.length, "width": self.width, "area": self.area,
                "hollow": self.hollow, "top_tilt": self.top_tilt, "lab": self.lab}


def _lab(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    return np.stack([lab[:, 0] * 100 / 255, lab[:, 1] - 128, lab[:, 2] - 128], axis=1)


def extract(hm: HeightMap, raised: float = RAISED, min_area: float = MIN_AREA, flat: np.ndarray | None = None) -> list[Blob]:
    """Raised regions, plus (if given) the flat-object mask from colour.flat_mask as separate flat blobs."""
    H = hm.height
    up = np.nan_to_num(H, nan=-1.0) > raised
    up = ndimage.binary_opening(up, structure=np.ones((2, 2)))       # one-cell opening only: keeps thin wires
    labels, n = ndimage.label(up, structure=np.ones((3, 3)))
    if flat is not None:
        flat_only = flat & ~ndimage.binary_dilation(up, iterations=2)
        fl, fn = ndimage.label(flat_only, structure=np.ones((3, 3)))
        labels = np.where(fl > 0, fl + n, labels)
        n += fn
    unknown = ~np.isfinite(H)
    robot_dist = ndimage.distance_transform_edt(~hm.robot) * CELL if hm.robot.any() else np.full(H.shape, np.inf)
    X, Y = hm.world_xy()
    out = []
    for k in range(1, n + 1):
        iy, ix = np.nonzero(labels == k)
        area = len(iy) * CELL * CELL
        if area < min_area:
            continue
        h = np.nan_to_num(H[iy, ix], nan=0.0)
        is_flat = flat is not None and k > n - (fn if flat is not None else 0)
        xs, ys = X[iy, ix], Y[iy, ix]
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        (cx, cy), (w, l), ang = cv2.minAreaRect(pts)
        w, l = w + CELL, l + CELL
        if w > l:
            w, l, ang = l, w, ang + 90
        mask = (labels == k).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = max(sum(cv2.arcLength(c, True) for c in contours) * CELL, 1e-6)
        h95 = float(np.percentile(h, 95))
        top = (h >= 0.85 * h95) if not is_flat else np.zeros(len(h), dtype=bool)
        tilt = None
        if top.sum() >= 12 and np.ptp(xs[top]) + np.ptp(ys[top]) >= 3:
            A = np.stack([xs[top], ys[top], np.ones(top.sum())], axis=1)
            coef, *_ = np.linalg.lstsq(A, h[top], rcond=None)
            tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
        core = ndimage.binary_erosion(mask.astype(bool), iterations=3)
        hollow = float((H[core] < 0.5 * h95).mean()) if core.any() else 0.0
        colour_cells = (h >= 0.5 * h95) if not is_flat else np.ones(len(h), dtype=bool)
        lab = np.median(_lab(hm.rgb[iy[colour_cells], ix[colour_cells]]), axis=0)
        ring = ndimage.binary_dilation(mask.astype(bool), structure=np.ones((3, 3))) & ~mask.astype(bool)
        out.append(Blob(
            id=len(out) + 1, iy=iy, ix=ix, centroid=(float(xs.mean()), float(ys.mean())), area=area,
            length=float(l), width=float(w), yaw=float(ang % 180), fill=float(area / max(l * w, 1e-6)),
            circularity=float(4 * np.pi * area / perim ** 2), h95=h95, hmed=float(np.median(h)),
            top_tilt=tilt, hollow=hollow, lab=tuple(float(v) for v in lab),
            robot_gap=float(robot_dist[iy, ix].min()), unknown_edge=float(unknown[ring].mean()) if ring.any() else 0.0,
            flat=bool(is_flat)))
    return out


def shape_change(now: Blob, ref: Blob) -> float:
    """The largest normalised change of a blob's shape signature from its reference (the J1 code rule):
    above about 2.3 the object has fallen over; standing objects stay below about 1."""
    terms = [abs(1 - now.h95 / max(ref.h95, 0.5)) / 0.30, (now.length / ref.length - 1) / 0.35,
             (now.width / ref.width - 1) / 0.50, (now.area / ref.area - 1) / 0.60]
    if ref.hollow >= 0.3:
        terms.append((ref.hollow - now.hollow) / 0.30)
    if now.top_tilt is not None and ref.top_tilt is not None:
        terms.append((now.top_tilt - ref.top_tilt) / 20.0)
    return float(max(terms))
