"""Offline-only ground truth from the simulator: segmentation, object poses and outcomes.

This is the only rtscene module that may read segmentation, state.objects, functional points, divergence or
success. Everything here is for calibration (the robot's own body model) and for scoring; none of it may
reach a model or a run-time decision.
"""
from __future__ import annotations

import re

import numpy as np

from .capture import Capture, Episode
from .geometry import quat_to_matrix

ROBOT_PATTERNS = re.compile(r"^(fl|fr|lr|rr)_(link\d|base_link|castor_link|wheel_link)$|^(box\d_Link|base_link|footprint|"
                            r"inertial_link|camera_base_link|camera_link\d|left_camera|right_camera|"
                            r"left_wheel_link|right_wheel_link)$")
SCENE = {"table", "ground", "wall", ""}


def segmentation(ep: Episode, cam: str) -> np.ndarray:
    import numpy as _np
    return _np.load(ep.dir/"captures.npz")[f"{cam}_seg"]


def entity_names(ep: Episode) -> dict[int, str]:
    return {int(k): v for k, v in ep.meta["entity_names"].items()}


def robot_ids(ep: Episode) -> dict[int, str]:
    return {i: n for i, n in entity_names(ep).items() if ROBOT_PATTERNS.match(n)}


def objects(ep: Episode, cap: Capture) -> dict:
    return ep.meta["captures"][cap.idx]["state"]["objects"]


def object_ids(ep: Episode) -> dict[str, list[int]]:
    """Tracked object role (e.g. 'cup') -> segmentation ids whose entity name is the role's model name."""
    names = entity_names(ep)
    first = ep.meta["captures"][0]["state"]["objects"]
    out = {}
    for role, o in first.items():
        if role in ("wall", "table"):
            continue
        model = str(o.get("model", ""))
        exact = [i for i, n in names.items() if n == model]
        # articulated objects (laptop, ...) name their links after the model; never take the robot's own links
        out[role] = exact or [i for i, n in names.items()
                              if model and n.startswith(model) and not ROBOT_PATTERNS.match(n)]
        if not out[role] and "position_cm" in o:
            out[role] = _nearest_ids(ep, o["position_cm"], exclude=set(robot_ids(ep)))
    return out


def _nearest_ids(ep: Episode, position_cm, exclude: set, radius: float = 6.0) -> list[int]:
    """Segmentation ids (generic link names such as link_0) whose top-view points lie within radius cm of
    an object's true origin in xy at the first capture: the links of an articulated object."""
    from .geometry import backproject
    cap = ep.captures()[0]
    seg = segmentation(ep, "top_camera")[cap.idx]
    P, valid = backproject(cap.depth("top_camera"), cap.K("top_camera"), cap.E("top_camera"))
    names = entity_names(ep)
    d = np.hypot(P[..., 0] - position_cm[0], P[..., 1] - position_cm[1])
    near = valid & (d < radius) & (P[..., 2] > cap.table_z + 0.3)
    ids, counts = np.unique(seg[near], return_counts=True)
    return [int(i) for i, c in zip(ids, counts)
            if c >= 20 and int(i) not in exclude and names.get(int(i), "") not in SCENE]


def tilt_deg(q_start, q_now) -> float:
    """Angle between the object's start up-axis (the local axis most aligned with world z) then and now."""
    R0, R1 = quat_to_matrix(q_start), quat_to_matrix(q_now)
    k = int(np.argmax(np.abs(R0[2])))
    u0 = R0[:, k] * np.sign(R0[2, k])
    u1 = R1[:, k] * np.sign(R0[2, k])
    return float(np.degrees(np.arccos(np.clip(u0 @ u1, -1.0, 1.0))))


def blob_actors(ep: Episode, cap: Capture, blobs) -> dict[int, tuple[int | None, float]]:
    """blob id -> (majority segmentation id over the top-camera points in its cells, that id's share)."""
    from .geometry import backproject
    from .heightmap import _cells, grid_shape
    grid = np.zeros(grid_shape(), dtype=np.int32)
    for b in blobs:
        grid[b.iy, b.ix] = b.id
    seg = segmentation(ep, "top_camera")[cap.idx]
    P, valid = backproject(cap.depth("top_camera"), cap.K("top_camera"), cap.E("top_camera"))
    iy, ix, ok = _cells(P[valid])
    bid = grid[iy[ok], ix[ok]]
    sid = seg[valid][ok]
    out = {}
    for b in blobs:
        ids = sid[bid == b.id]
        if not len(ids):
            out[b.id] = (None, 0.0)
            continue
        vals, counts = np.unique(ids, return_counts=True)
        k = int(np.argmax(counts))
        out[b.id] = (int(vals[k]), float(counts[k] / counts.sum()))
    return out


def object_state(ep: Episode, cap: Capture, role: str) -> dict:
    """True tilt from the start pose and height change of a tracked object at a capture (labels only)."""
    o0 = ep.meta["captures"][0]["state"]["objects"][role]
    o = objects(ep, cap)[role]
    return {"tilt": tilt_deg(o0["quat_wxyz"], o["quat_wxyz"]), "dz": o["position_cm"][2] - o0["position_cm"][2],
            "z": o["position_cm"][2], "xy_moved": float(np.hypot(o["position_cm"][0] - o0["position_cm"][0],
                                                                 o["position_cm"][1] - o0["position_cm"][1]))}
