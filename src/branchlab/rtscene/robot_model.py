"""The robot's own body as boxes fixed to its links, used to remove robot pixels from depth at run time.

A real robot knows its joint angles and therefore every link's pose (forward kinematics); what it needs from
calibration is each link's extent, as a CAD model would give. calibrate() learns that extent once, offline,
as an axis-aligned box in each link's own frame (robust percentiles of the link's segmented points over many
captures). At run time mask() needs only the link poses of the current capture: no segmentation, no object
information.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .capture import Capture
from .geometry import backproject, link_frame

CALIBRATION = Path(__file__).with_name("robot_boxes.json")


def calibrate(samples, low: float = 0.5, high: float = 99.5, min_points: int = 30) -> dict:
    """samples: iterable of (capture, camera, segmentation map, {seg id: link name}). Returns link -> [lo, hi] (cm, link frame)."""
    local: dict[str, list[np.ndarray]] = {}
    for cap, cam, seg, link_of in samples:
        P, valid = backproject(cap.depth(cam), cap.K(cam), cap.E(cam))
        for sid, name in link_of.items():
            if name not in cap.links:
                continue
            m = (seg == sid) & valid
            if m.sum() < 3:
                continue
            R, o = link_frame(cap.links[name])
            local.setdefault(name, []).append((P[m] - o) @ R)
    boxes = {}
    for name, chunks in local.items():
        pts = np.concatenate(chunks)
        if len(pts) >= min_points:
            boxes[name] = [np.percentile(pts, low, axis=0).round(2).tolist(), np.percentile(pts, high, axis=0).round(2).tolist()]
    return boxes


def save(boxes: dict, path: Path = CALIBRATION) -> None:
    path.write_text(json.dumps(boxes, indent=1))


def load(path: Path = CALIBRATION) -> dict:
    return json.loads(path.read_text())


def mask(points_cm: np.ndarray, cap: Capture, boxes: dict, margin: float = 1.0) -> np.ndarray:
    """Boolean mask of points (..., 3) inside any link box (inflated by margin cm) at this capture's link poses."""
    flat = points_cm.reshape(-1, 3)
    hit = np.zeros(len(flat), dtype=bool)
    for name, (lo, hi) in boxes.items():
        if name not in cap.links:
            continue
        R, o = link_frame(cap.links[name])
        lo_m, hi_m = np.asarray(lo) - margin, np.asarray(hi) + margin
        # quick reject by the box's world bounding sphere
        centre_local = (lo_m + hi_m) / 2
        radius = np.linalg.norm(hi_m - lo_m) / 2
        centre = o + R @ centre_local
        near = np.where(~hit)[0]
        near = near[np.sum((flat[near] - centre) ** 2, axis=1) <= radius ** 2]
        if not len(near):
            continue
        loc = (flat[near] - o) @ R
        inside = np.all((loc >= lo_m) & (loc <= hi_m), axis=1)
        hit[near[inside]] = True
    return hit.reshape(points_cm.shape[:-1])
