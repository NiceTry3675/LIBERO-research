"""Colour names and flat objects (mats, pads) from the top camera's raw image.

Names are fixed a priori (no tuning on labels): achromatic by chroma and lightness, otherwise by hue, with a
wood-like brown band. Flat objects are about 2 mm thick, at the limit of depth, so they are found as table
cells whose colour differs from the table's own colour around them.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .heightmap import CELL, HeightMap

# CIELab hue angles (not HSV): sRGB red sits near 40 deg, orange 55-65, yellow ~100, green ~135, cyan ~195,
# blue ~305, magenta ~330
HUES = [(50, "red"), (75, "orange"), (105, "yellow"), (130, "lime"), (170, "green"), (230, "teal"), (315, "blue"),
        (345, "purple"), (360, "red")]
ACHROMATIC = 18.0      # chroma below this is grey: randomized lighting tints neutral objects by up to ~15
SYNONYMS = {"wooden": {"brown", "light brown"}, "wood": {"brown", "light brown"}, "silver": {"light grey", "grey", "white"},
            "metal": {"grey", "light grey"}, "steel": {"grey", "light grey"}, "beige": {"white", "light brown"},
            "maroon": {"red", "brown"}, "olive": {"lime", "yellow", "green"}, "gray": {"grey"}, "cyan": {"teal"},
            "violet": {"purple"}, "golden": {"yellow", "orange"}, "gold": {"yellow", "orange"}}


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (..., 3) -> CIE Lab (..., 3) with L in 0-100."""
    shape = rgb.shape
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    lab = np.stack([lab[:, 0] * 100 / 255, lab[:, 1] - 128, lab[:, 2] - 128], axis=1)
    return lab.reshape(shape)


def name(lab, white_l: float = 95.0) -> str:
    """One colour name for a median Lab colour; white_l is the image's bright reference (lighting varies)."""
    L, a, b = (float(v) for v in lab)
    if L == 0 and a == 0 and b == 0:
        return "unknown colour"
    C = np.hypot(a, b)
    rel = 100 * L / max(white_l, 1.0)
    if C < ACHROMATIC:
        return "white" if rel > 85 else "light grey" if rel > 65 else "grey" if rel > 30 else "black"
    hue = (np.degrees(np.arctan2(b, a)) + 360) % 360
    if 40 <= hue <= 95 and C <= 40:
        return "brown" if L < 55 else "light brown" if L < 78 else "beige"
    for limit, label in HUES:
        if hue < limit:
            return "pink" if label in ("red", "purple") and L > 68 and C < 60 else label
    return "red"


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIE76 colour difference (simple and monotone enough for thresholds)."""
    return np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), axis=-1)


def table_model(hm: HeightMap, tile_cm: float = 10.0) -> np.ndarray:
    """Per-cell expected table colour: the median Lab of nearby bare-table cells (h < 0.15 cm)."""
    lab = to_lab(hm.rgb)
    bare = np.isfinite(hm.height) & (np.abs(np.nan_to_num(hm.height)) < 0.15) & (hm.source == 1)
    t = int(round(tile_cm / CELL))
    ny, nx = bare.shape
    model = np.full((ny, nx, 3), np.nan)
    for y0 in range(0, ny, t):
        for x0 in range(0, nx, t):
            ys, xs = slice(max(0, y0 - t), min(ny, y0 + 2 * t)), slice(max(0, x0 - t), min(nx, x0 + 2 * t))
            sel = bare[ys, xs]
            if sel.sum() >= 20:
                model[y0:y0 + t, x0:x0 + t] = np.median(lab[ys, xs][sel], axis=0)
    return model


def flat_mask(hm: HeightMap, model: np.ndarray, threshold: float = 18.0, max_height: float = 0.5,
              min_area: float = 20.0) -> np.ndarray:
    """Cells of thin objects: lower than max_height cm but coloured unlike the table there."""
    lab = to_lab(hm.rgb)
    h = np.nan_to_num(hm.height, nan=np.inf)
    cand = (hm.source == 1) & (h < max_height) & np.isfinite(model[..., 0]) & (delta_e(lab, model) > threshold)
    cand = ndimage.binary_opening(cand, structure=np.ones((3, 3)))
    labels, n = ndimage.label(cand)
    keep = np.zeros_like(cand)
    for k in range(1, n + 1):
        region = labels == k
        if region.sum() * CELL * CELL < min_area:
            continue
        # mats and pads are compact; table edges, seams and shadows are long thin strips
        thin = ndimage.binary_erosion(region, iterations=int(round(1.5 / CELL)))
        if thin.sum() * CELL * CELL < 0.3 * region.sum() * CELL * CELL:
            continue
        keep |= ndimage.binary_closing(region, structure=np.ones((3, 3)))
    return keep
