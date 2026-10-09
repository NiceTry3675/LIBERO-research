#!/usr/bin/env python3
"""J3 pre-close: will closing the gripper now catch something between the fingers? Code geometry first.

At every replayed pre-close capture, depth points from the closing arm's wrist camera and the top/agent cameras
are moved into the gripper frame (origin between the fingertips; a = approach, f = finger axis, n = a x f).
The robot's own links and the table are removed, and the points inside the closing zone (between the finger
pads: a in [-PAD, +0.5] cm, |f| < gap/2 - 0.3, |n| <= SLAB) are counted. The label is the close's own
result in the replay: the fingers stopped on something, or closed fully on nothing.

    PYTHONPATH=src python scripts/rtscene_j3.py [--roots DIR ...] [--out outputs/analysis/j3_rows.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from branchlab.rtscene import robot_model
from branchlab.rtscene.capture import episodes
from branchlab.rtscene.geometry import backproject

PROJECT = Path(__file__).resolve().parents[1]
ROOTS = [PROJECT/"outputs/replay_local/replay", *sorted((PROJECT/"outputs/robodawn_colab").glob("*/replay"))]
PAD = 4.0          # cm of finger pad behind the fingertips along the approach
SLAB = 1.5         # cm half-thickness of the pads across the finger axis
GAP_PER_OPENING = 9.0


def zone_counts(cap, arm: str, boxes: dict) -> dict:
    st = cap.arms[arm]
    a = np.asarray(st["approach"], float)
    f = np.asarray(st["finger_axis"], float)
    a, f = a / np.linalg.norm(a), f / np.linalg.norm(f)
    n = np.cross(a, f)
    tcp = np.asarray(st["position_cm"], float)
    gap = GAP_PER_OPENING * float(st["gripper_real"])
    out = {"gap": round(gap, 2)}
    for cam in (f"{arm}_camera", "top_camera", "agent_camera"):
        P, valid = backproject(cap.depth(cam), cap.K(cam), cap.E(cam))
        pts = P[valid]
        near = np.sum((pts - tcp) ** 2, axis=1) < 15.0 ** 2
        pts = pts[near]
        if not len(pts):
            out[cam] = {"points": 0}
            continue
        rob = robot_model.mask(pts, cap, boxes, 0.5)
        pts = pts[~rob & (pts[:, 2] > cap.table_z + 0.3)]
        loc = (pts - tcp) @ np.stack([a, f, n], axis=1)
        inside = (loc[:, 0] >= -PAD) & (loc[:, 0] <= 0.5) & (np.abs(loc[:, 1]) < max(gap / 2 - 0.3, 0.2)) & (np.abs(loc[:, 2]) <= SLAB)
        ahead = (loc[:, 0] > 0.5) & (loc[:, 0] <= 4.0) & (np.abs(loc[:, 1]) < gap / 2) & (np.abs(loc[:, 2]) <= SLAB)
        span = float(np.ptp(loc[inside, 1])) if inside.sum() >= 3 else 0.0
        out[cam] = {"points": int(inside.sum()), "ahead": int(ahead.sum()), "span": round(span, 2),
                    "offset": round(float(np.median(loc[inside, 1])), 2) if inside.sum() >= 3 else None}
    return out


def memory_features(ep, cap, arm: str, boxes: dict, cache: dict, start=None) -> dict:
    """Geometry of the closing fingers against the object as last seen before this turn's motions.

    The gripper hides the object at close time, so the object is taken from the turn's start capture: the blob
    whose footprint is nearest to the fingertips. Features: the fingertips' distance to that footprint (0 =
    over it), their height relative to its top, the footprint's extent along the finger axis around the
    fingertips against the gap, and the centring offset along the finger axis."""
    from branchlab.rtscene import blobs as blobs_mod, heightmap
    if start is None:
        start = max((c for c in ep.captures() if c.kind == "turn" and c.idx < cap.idx), key=lambda c: c.idx, default=None)
    if start is None:
        return {}
    if start.idx not in cache:
        cache[start.idx] = blobs_mod.extract(heightmap.build(start, boxes, with_rgb=False))
    bl = cache[start.idx]
    st = cap.arms[arm]
    tcp = np.asarray(st["position_cm"], float)
    f = np.asarray(st["finger_axis"], float); f /= np.linalg.norm(f)
    a = np.asarray(st["approach"], float); a /= np.linalg.norm(a)
    gap = GAP_PER_OPENING * float(st["gripper_real"])
    from branchlab.rtscene.heightmap import X_RANGE, Y_RANGE, CELL
    best = None
    for b in bl:
        xs = X_RANGE[0] + (b.ix + 0.5) * CELL
        ys = Y_RANGE[0] + (b.iy + 0.5) * CELL
        d = np.hypot(xs - tcp[0], ys - tcp[1])
        if best is None or d.min() < best[0]:
            best = (float(d.min()), b, xs, ys)
    if best is None:
        return {"mem_dist": 99.0}
    dist, b, xs, ys = best
    rel = np.stack([xs - tcp[0], ys - tcp[1]], axis=1)
    f2 = f[:2] / max(np.linalg.norm(f[:2]), 1e-6)
    along_f = rel @ f2
    across = rel @ np.array([-f2[1], f2[0]])
    band = np.abs(across) <= 1.5                        # the strip the finger pads sweep
    extent = float(np.ptp(along_f[band])) if band.sum() >= 3 else 0.0
    centre = float(np.median(along_f[band])) if band.sum() >= 3 else None
    return {"mem_dist": round(dist, 2), "mem_h95": round(b.h95, 1), "mem_z_rel": round(tcp[2] - start.table_z - b.h95, 2),
            "mem_extent_f": round(extent, 2), "mem_centre_f": centre and round(centre, 2), "gap": round(gap, 2),
            "mem_fits": bool(extent > 0.3 and extent < gap - 0.3), "approach_z": round(float(a[2]), 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", type=Path, nargs="+", default=ROOTS)
    parser.add_argument("--out", type=Path, default=PROJECT/"outputs/analysis/j3_rows.jsonl")
    args = parser.parse_args()
    boxes = robot_model.load()
    seen = set()
    with open(args.out, "w") as fh:
        for root in args.roots:
            if not root.is_dir():
                continue
            for ep in episodes(root):
                if ep.key in seen or ep.task == "click_bell":     # click_bell closes on purpose to press
                    continue
                seen.add(ep.key)
                log = {(c["turn"], c["cmd_index"]): c for c in ep.meta["command_log"]}
                cache = {}
                for cap in ep.captures():
                    if cap.kind != "preclose":
                        continue
                    arm = cap.command.split()[0]
                    result = log.get((cap.turn, cap.cmd_index)) or {}
                    note = str(result.get("note") or "")
                    label = "grasp" if note.startswith("fingers stopped") else "empty" if note.startswith("fingers closed fully") else "other"
                    row = {"key": ep.key, "task": ep.task, "capture": cap.idx, "arm": arm, "label": label, "note": note[:60],
                           **zone_counts(cap, arm, boxes), **memory_features(ep, cap, arm, boxes, cache)}
                    fh.write(json.dumps(row) + "\n")
                print(ep.key, flush=True)


if __name__ == "__main__":
    main()
