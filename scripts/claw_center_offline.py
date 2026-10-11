#!/usr/bin/env python3
"""Claw-machine retest, broad version: from RGB alone, which 2 cm step brings the fingertips over the middle of a named
object? A companion to scripts/claw_retest_offline.py (same crop, question, scoring, interfaces and stay rules), with
many more items, to measure how well Clef reads a direction at each distance and degree of occlusion.

Items, from the replayed episodes of the 8 new tasks: every turn-start capture, every arm with its gripper open, every
tracked object whose model appears once in the scene (so the name is unambiguous) with its middle within 15 cm (x/y)
of the fingertips. The middle is the centroid of the object's visible top-camera pixels, back-projected with depth;
visibility is those pixels against the most the object shows in the episode (both from segmentation and depth:
labels only, never shown to a model). At most 3 items per (episode, arm, object) and distance bin, chosen by hash.
Rates weight every (episode, arm, object) equally; dev/test halves by episode as in the companion script.

    PYTHONPATH=src:scripts python scripts/claw_center_offline.py items
    PYTHONPATH=src:scripts python scripts/claw_center_offline.py clef [--interfaces img_tip blind]
    PYTHONPATH=src:scripts python scripts/claw_center_offline.py report
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claw_retest_offline as cr                         # noqa: E402

OUT = cr.OUT
PER_GROUP = 3


def build_items() -> list[dict]:
    from branchlab.rtscene import labels
    from branchlab.rtscene.capture import Episode
    from branchlab.rtscene.geometry import backproject
    items = []
    for ep_path in sorted((cr.REPLAY/"eval_baseline").glob("*/episode_*")):
        ep = Episode(ep_path)
        meta = ep.meta
        objs0 = meta["captures"][0]["state"].get("objects", {})
        models = collections.Counter(o.get("model") for r, o in objs0.items() if r not in ("wall", "table"))
        ids = {r: v for r, v in labels.object_ids(ep).items() if v and models[objs0[r].get("model")] == 1}
        seg_all = labels.segmentation(ep, "top_camera")
        caps = [c for c in ep.captures() if c.kind == "turn"]
        centre, seen = {}, collections.defaultdict(dict)
        for cap in caps:
            P, valid = backproject(cap.depth("top_camera"), cap.K("top_camera"), cap.E("top_camera"))
            seg = seg_all[cap.idx]
            for role, sids in ids.items():
                m = valid & np.isin(seg, sids) & (P[..., 2] > cap.table_z + 0.3)
                seen[role][cap.idx] = int(m.sum())
                if m.sum() >= 15:
                    centre[(role, cap.idx)] = P[m][:, :2].mean(0)
        trace = cr.trace_for(meta["task"], meta["episode"])
        for cap in caps:
            for arm in ("left", "right"):
                st = cap.arms[arm]
                if st.get("gripper", 1.0) < 0.5:
                    continue
                tip = list(meta["captures"][cap.idx]["state"][arm]["position_cm"])
                for role in ids:
                    c = centre.get((role, cap.idx))
                    if c is None:
                        continue
                    dist = math.hypot(c[0] - tip[0], c[1] - tip[1])
                    if dist > cr.REACH:
                        continue
                    most = max(seen[role].values()) or 1
                    items.append({"id": f"{meta['key']}/{arm}/c{cap.idx}/{role}", "key": meta["key"], "task": meta["task"],
                                  "episode": meta["episode"], "split": cr.split_of(meta["key"]), "arm": arm, "cap": cap.idx,
                                  "kind": "turn", "turn": cap.turn, "tip": tip, "grasp": [round(float(c[0]), 2), round(float(c[1]), 2), tip[2]],
                                  "dist": round(dist, 2), "approach": f"{meta['key']}/{arm}/{role}", "role": role,
                                  "target": cr.name_of(objs0[role].get("model")).replace("the ", "the middle of the ", 1),
                                  "visible": round(seen[role][cap.idx] / most, 2), "stratum": "centre",
                                  "gemini_next": cr.gemini_next(trace.get(cap.turn) or {}, arm)})
    # at most PER_GROUP items per (episode, arm, object) and distance bin
    groups = collections.defaultdict(list)
    for i in items:
        groups[(i["approach"], cr.bin_of(i["dist"]))].append(i)
    return [i for g in groups.values() for i in sorted(g, key=lambda i: cr.h(i["id"]))[:PER_GROUP]]


def cmd_items(args):
    items = build_items()
    (OUT/"centre_items.json").write_text(json.dumps(items))
    print(f"{len(items)} items, {len({i['approach'] for i in items})} (episode, arm, object) groups; by bin "
          f"{dict(collections.Counter(cr.bin_of(i['dist']) for i in items))}; visible<0.5: {sum(i['visible'] < 0.5 for i in items)}")


def cmd_clef(args):
    items = json.loads((OUT/"centre_items.json").read_text())
    out = OUT/"centre_clef.jsonl"
    done = {(r["id"], r["interface"]) for r in cr.read_jsonl(out)}
    jobs = [(i, f) for i in items for f in args.interfaces if (i["id"], f) not in done]
    print(f"{len(jobs)} Clef calls ({len(done)} done)")
    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as fh:
        for k, r in enumerate(pool.map(lambda j: cr.ask_clef(*j), jobs), 1):
            fh.write(json.dumps(r) + "\n")
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)


def cmd_report(args):
    items = json.loads((OUT/"centre_items.json").read_text())
    ans = collections.defaultdict(dict)
    for r in cr.read_jsonl(OUT/"centre_clef.jsonl"):
        ans[r["id"]][r["interface"]] = r
    L = ["# Claw-machine retest, object middles", "",
         f"{len(items)} items, {len({i['approach'] for i in items})} groups. Rates weight every (episode, arm, object) equally; "
         "brackets: 95% intervals over groups.", ""]
    conds = [(f"Clef {f} / {r}", lambda i, f=f, r=r: (lambda c: cr.score_step(i, c) if c else None)(cr.apply_rule(ans[i["id"]].get(f), r)))
             for f in ("img_tip", "blind") for r in cr.RULES if (f, r) != ("blind", "choice")]
    conds.append(("always up", lambda i: cr.score_step(i, "up")))
    for split in ("dev", "test"):
        for vis_name, pred in (("all", lambda i: True), ("visible >= 50%", lambda i: i["visible"] >= 0.5),
                               ("visible < 50%", lambda i: i["visible"] < 0.5)):
            sub = [i for i in items if i["split"] == split and pred(i)]
            L += ["", f"## {split} half, {vis_name}", "", "| condition | " + " | ".join(b + " cm" for _, _, b in cr.BINS) + " |",
                  "|---|" + "---|" * len(cr.BINS)]
            for name, fn in conds:
                cells = []
                for _, _, b in cr.BINS:
                    s2 = [i for i in sub if cr.bin_of(i["dist"]) == b]
                    rate, n = cr.weighted(s2, fn)
                    if not n:
                        cells.append("-")
                    elif b == "<1.5":
                        cells.append(f"stay_ok {rate.get('stay_ok', 0):.0%} ({n})")
                    else:
                        lo, hi = cr.bootstrap(s2, fn, "good", reps=300)
                        cells.append(f"good {rate.get('good', 0):.0%} [{lo:.0%},{hi:.0%}] / bad {rate.get('bad', 0):.0%} / "
                                     f"stall {rate.get('stall', 0):.0%} ({n})")
                L.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(L)
    (OUT/"centre_report.md").write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("items")
    s = sub.add_parser("clef")
    s.add_argument("--interfaces", nargs="+", default=["img_tip", "blind"])
    s.add_argument("--workers", type=int, default=8)
    sub.add_parser("report")
    args = parser.parse_args()
    {"items": cmd_items, "clef": cmd_clef, "report": cmd_report}[args.step](args)


if __name__ == "__main__":
    main()
