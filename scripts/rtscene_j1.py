#!/usr/bin/env python3
"""J1 dataset: is a task object still standing as at the start and on the table? One row per tracked object per
turn-start (and final) capture of the replayed episodes, with the parser's measurements and the true label.

Binding: at the first capture each tracked task object is pinned to the blob its segmentation dominates
(oracle binding, labelled as such; at run time Gemini or a matcher names it). From then on the object is
followed by code only: the blob nearest to its last position, scored by distance, colour, area and height.
Nothing after the first capture reads segmentation or true poses except the label columns.

    PYTHONPATH=src python scripts/rtscene_j1.py build [--roots DIR ...]   # rows -> outputs/analysis/j1_rows.jsonl
    PYTHONPATH=src python scripts/rtscene_j1.py report

report: flat objects (start height < 1.5 cm: mats, pads, coasters) cannot topple and are left out; checks
while the object is lifted (true rise > 2 cm, i.e. held) are left out; tilts of 15-60 degrees are reported
apart. The code rule's AUC and its detection rate at 0, 1 and 2 false alarms are given for all checks and for
checks with the arm at least 2 cm away (an adjacent arm hides part of the footprint).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from branchlab.rtscene import blobs as blobs_mod
from branchlab.rtscene import heightmap, labels, robot_model
from branchlab.rtscene.blobs import shape_change as score_c0
from branchlab.rtscene.capture import Episode, episodes

PROJECT = Path(__file__).resolve().parents[1]
ROOTS = [PROJECT/"outputs/replay_local/replay", *sorted((PROJECT/"outputs/robodawn_colab").glob("*/replay"))]
SKIP_ROLES = {"wall", "table", "cluttered_obj"}
SKIP_TASKS = {"open_laptop", "adjust_bottle"}        # their goal changes the object's pose on purpose
GATE = 12.0                                          # cm from the last position


def lab_distance(a, b) -> float:
    return float(np.linalg.norm(np.subtract(a, b)))


def cost(blob, last) -> float:
    d = math.dist(blob.centroid, last.centroid)
    return (d / 5 + lab_distance(blob.lab, last.lab) / 20 + abs(math.log(blob.area / last.area)) / 0.5
            + abs(blob.h95 - last.h95) / max(2.0, 0.3 * last.h95))


def run_episode(ep: Episode, boxes: dict) -> list[dict]:
    if ep.task in SKIP_TASKS:
        return []
    caps = [c for c in ep.captures() if c.kind in ("turn", "final")]
    first = caps[0]
    hm0 = heightmap.build(first, boxes)
    b0 = blobs_mod.extract(hm0)
    actor = labels.blob_actors(ep, first, b0)
    roles = {r: ids for r, ids in labels.object_ids(ep).items() if r not in SKIP_ROLES and ids}
    tracks = {}
    for role, ids in roles.items():
        cands = [b for b in b0 if actor[b.id][0] in ids and actor[b.id][1] >= 0.5]
        if cands:
            ref = max(cands, key=lambda b: b.area)
            tracks[role] = {"ref": ref, "last": ref, "status": "visible"}
    rows = []
    for c in caps:
        hm = hm0 if c is first else heightmap.build(c, boxes)
        bl = b0 if c is first else blobs_mod.extract(hm)
        acts = actor if c is first else labels.blob_actors(ep, c, bl)
        for role, tr in tracks.items():
            near = [b for b in bl if math.dist(b.centroid, tr["last"].centroid) <= GATE]
            truth = labels.object_state(ep, c, role)
            row = {"key": ep.key, "task": ep.task, "role": role, "capture": c.idx, "kind": c.kind, "turn": c.turn,
                   "true_tilt": round(truth["tilt"], 1), "true_dz": round(truth["dz"], 1),
                   "off_table": truth["z"] < c.table_z - 5}
            if near:
                b = min(near, key=lambda x: cost(x, tr["last"]))
                tr["last"], tr["status"] = b, "visible"
                ref = tr["ref"]
                row.update(status="visible", c0=round(score_c0(b, ref), 3),
                           h_ratio=round(b.h95 / max(ref.h95, 0.5), 3), l_ratio=round(b.length / ref.length, 3),
                           w_ratio=round(b.width / ref.width, 3), a_ratio=round(b.area / ref.area, 3),
                           hollow_ref=round(ref.hollow, 2), hollow_now=round(b.hollow, 2),
                           tilt_ref=ref.top_tilt and round(ref.top_tilt, 1), tilt_now=b.top_tilt and round(b.top_tilt, 1),
                           moved=round(math.dist(b.centroid, ref.centroid), 1), robot_gap=round(b.robot_gap, 1),
                           unknown_edge=round(b.unknown_edge, 2), ref_h95=round(ref.h95, 1), h95=round(b.h95, 1),
                           ref_lw=[round(ref.length, 1), round(ref.width, 1)], lw=[round(b.length, 1), round(b.width, 1)],
                           track_correct=acts[b.id][0] in roles[role])
            else:
                # hidden under the robot, or gone
                X, Y = hm.world_xy()
                near_cells = (X - tr["last"].centroid[0]) ** 2 + (Y - tr["last"].centroid[1]) ** 2 <= GATE ** 2
                hidden = bool((~np.isfinite(hm.height[near_cells])).mean() > 0.2 or hm.robot[near_cells].any())
                row.update(status="hidden" if hidden else "missing")
            rows.append(row)
    return rows


def label(r: dict) -> str:
    if r["off_table"] or r["true_tilt"] > 60:
        return "lost"
    return "ok" if r["true_tilt"] < 15 else "mid"


def report(path: Path) -> None:
    import collections
    rows = [json.loads(line) for line in open(path)]
    vis = [r for r in rows if r["status"] == "visible" and r["ref_h95"] >= 1.5 and r["true_dz"] < 2]
    for r in vis:
        r["label"] = label(r)
    print(f"{len(rows)} rows, {len({r['key'] for r in rows})} episodes; status {dict(collections.Counter(r['status'] for r in rows))}")
    print(f"monitored (visible, not flat, not lifted): {len(vis)}; labels {dict(collections.Counter(r['label'] for r in vis))}; "
          f"roles {dict(collections.Counter(r['role'] for r in vis))}")
    print(f"tracking correct: {sum(r['track_correct'] for r in vis)}/{len(vis)}")
    for name, sub in (("all", vis), ("arm >= 2 cm away", [r for r in vis if r["robot_gap"] >= 2.0])):
        pos = [r["c0"] for r in sub if r["label"] == "lost"]
        neg = sorted((r["c0"] for r in sub if r["label"] == "ok"), reverse=True)
        if not pos or not neg:
            continue
        auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
        cells = []
        for k in (0, 1, 2):
            thr = neg[k] if k < len(neg) else float("-inf")
            cells.append(f"{k} false alarms: detects {sum(p > thr for p in pos) / len(pos):.0%} (thr {thr:.2f})")
        print(f"  {name:18} lost {len(pos)}, ok {len(neg)}, AUC {auc:.3f}; " + "; ".join(cells))
    mid = [r for r in vis if r["label"] == "mid"]
    if mid:
        print(f"  15-60 deg tilts: {len(mid)}, code score > 2.34 in {sum(r['c0'] > 2.34 for r in mid)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    b = sub.add_parser("build")
    b.add_argument("--roots", type=Path, nargs="+", default=ROOTS)
    sub.add_parser("report")
    parser.add_argument("--out", type=Path, default=PROJECT/"outputs/analysis/j1_rows.jsonl")
    args = parser.parse_args()
    if args.step == "report":
        report(args.out)
        return
    boxes = robot_model.load()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n, seen = 0, set()
    with open(args.out, "w") as f:
        for root in args.roots:
            if not root.is_dir():
                continue
            for ep in episodes(root):
                if ep.key in seen:
                    continue
                seen.add(ep.key)
                rows = run_episode(ep, boxes)
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                n += len(rows)
                print(f"{ep.key}: {len(rows)} rows", flush=True)
    print(f"{n} rows from {len(seen)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
