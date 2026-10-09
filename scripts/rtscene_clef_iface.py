#!/usr/bin/env python3
"""Interface tuning for Clef (Workers AI, where it sees images) on the judgments it would make in the harness.

Items come from the replayed episodes; each is rendered in several interface arms (cameras, scale, drawn marks,
code numbers in the text) so the arms are compared on the same items.

  reid   a task object at a later turn: the code's candidate blobs near where it was last seen, as numbered
         tiles at one scale, against the object's crops from the first turn. Which tile is it, or none?
         The last position is the true one (one tracking step); labels from segmentation. Code baseline: the
         J1 tracker's pick (lowest cost within 12 cm).
  topple a task object at the first turn and now, crops at the same scale: fallen over or off the table?
         Visible, not flat, tracked correctly, no gripper within 8 cm; tilts 15-60 degrees left out.
         Code baseline: the J1 shape-change score.
  grip   a close about to run, drawn on the turn's start view (the gripper does not hide the object there):
         two bars where the finger pads will be, a line along which they close. Will they close on an object?
         Label: the close's own result. Code baseline: the J3 memory geometry.

    PYTHONPATH=src python scripts/rtscene_clef_iface.py build [--n 60]
    PYTHONPATH=src python scripts/rtscene_clef_iface.py ask [--probes reid topple grip] [--arms ...] [--model clef]
    PYTHONPATH=src python scripts/rtscene_clef_iface.py augment-reid    # the same reid items, with outlined tiles
    PYTHONPATH=src python scripts/rtscene_clef_iface.py report
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT/"outputs/analysis/clef_iface"
ROOTS = [PROJECT/"outputs/replay_local/replay", *sorted((PROJECT/"outputs/robodawn_colab").glob("*/replay"))]
SKIP_ROLES = {"wall", "table", "cluttered_obj"}
SKIP_TASKS = {"open_laptop", "adjust_bottle"}
GATE = 12.0          # cm: the J1 tracker's search radius
CAND_RADIUS = 30.0   # cm: candidates shown to Clef
MAX_CANDS = 6
TILE = 224
PAD, SLAB = 4.0, 1.5
GAP_PER_OPENING = 9.0

ARMS = {
    "reid": ["top", "agent", "both", "both+num", "both+num+task", "o:top", "o:both", "o:both+num+task"],
    "topple": ["agent", "both", "agent+num", "pair"],
    "grip": ["top", "both", "both+now", "both+num"],
}


def episodes_all():
    from branchlab.rtscene.capture import episodes
    seen = {}
    for root in ROOTS:
        if root.is_dir():
            for ep in episodes(root):
                seen.setdefault(ep.key, ep)
    return list(seen.values())


def slug(key: str) -> str:
    return key.replace("/", "_")


class Parsed:
    """Blobs and their true actors per capture of one episode, built on demand."""

    def __init__(self, ep, boxes):
        self.ep, self.boxes, self._cache = ep, boxes, {}

    def __call__(self, cap):
        from branchlab.rtscene import blobs as blobs_mod, heightmap, labels
        if cap.idx not in self._cache:
            bl = [b for b in blobs_mod.extract(heightmap.build(cap, self.boxes)) if b.area >= 6.0 and b.robot_gap >= 0.5]
            self._cache[cap.idx] = (bl, labels.blob_actors(self.ep, cap, bl))
        return self._cache[cap.idx]


def true_blob(bl, act, ids):
    own = [b for b in bl if act[b.id][0] in ids and act[b.id][1] >= 0.5]
    return max(own, key=lambda b: b.area) if own else None


def half_for(*blobs) -> float:
    return float(np.clip(max(max(b.length, b.width) for b in blobs) / 2 + 4, 6, 16))


def save(img, name: str) -> str:
    img.save(OUT/"images"/name, quality=90)
    return name


def gripped(cap, xyz) -> bool:
    return any(math.dist(st["position_cm"], xyz) < 8.0 for st in cap.arms.values())


# ------------------------------------------------------------------------------------------------ build

def outlined(cap, cam, blob, colour=(0, 255, 255)):
    """The view with the blob's footprint, lifted to its top height, outlined."""
    from PIL import Image, ImageDraw
    from branchlab.rtscene.geometry import project
    from branchlab.rtscene.heightmap import CELL, X_RANGE, Y_RANGE
    img = Image.fromarray(cap.rgb(cam))
    K = cap.K(cam).copy()
    K[:2] *= 2 if cam in ("top_camera", "agent_camera") else 1
    xs = X_RANGE[0] + (blob.ix + 0.5) * CELL
    ys = Y_RANGE[0] + (blob.iy + 0.5) * CELL
    pts = np.concatenate([np.stack([xs + dx, ys + dy, np.full_like(xs, cap.table_z + max(blob.h95, 0.5))], axis=1)
                          for dx in (-CELL / 2, CELL / 2) for dy in (-CELL / 2, CELL / 2)])
    uv, _ = project(pts, K, cap.E(cam))
    hull = convex_hull(uv)
    if len(hull) >= 3:
        ImageDraw.Draw(img).line([tuple(p) for p in hull] + [tuple(hull[0])], fill=colour, width=4)
    return img


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set(map(tuple, np.round(points, 1))))
    if len(pts) < 3:
        return np.array(pts)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out
    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


def build_reid(eps, boxes, n, rng, only=None):
    """Items for the reid probe. With `only` (item ids), rebuild exactly those items (same candidates)."""
    from branchlab.rtscene import labels, render
    from rtscene_j1 import cost
    items, want = [], {"right": n, "wrong": n}
    if only is not None:
        keys = {i.rsplit("_", 2)[0] for i in only}
        eps = [e for e in eps if slug(e.key) in keys or any(k.startswith(slug(e.key) + "_") for k in keys)]
    for ep in eps:
        if only is None and all(v <= 0 for v in want.values()):
            break
        if ep.task in SKIP_TASKS:
            continue
        parsed = Parsed(ep, boxes)
        caps = [c for c in ep.captures() if c.kind in ("turn", "final")]
        first = caps[0]
        b0, a0 = parsed(first)
        roles = {r: ids for r, ids in labels.object_ids(ep).items() if r not in SKIP_ROLES and ids}
        per_ep = collections.Counter()
        for role, ids in roles.items():
            ref = true_blob(b0, a0, ids)
            if ref is None or ref.h95 < 1.5:
                continue
            last = ref
            for c in caps[1:]:
                bl, act = parsed(c)
                truth = true_blob(bl, act, ids)
                state = labels.object_state(ep, c, role)
                if state["dz"] >= 2 or truth is None:          # held, or hidden: nothing to re-identify
                    last = truth or last
                    continue
                cands = sorted((b for b in bl if math.dist(b.centroid, last.centroid) <= CAND_RADIUS),
                               key=lambda b: math.dist(b.centroid, last.centroid))[:MAX_CANDS]
                near = [b for b in cands if math.dist(b.centroid, last.centroid) <= GATE]
                code = min(near, key=lambda b: cost(b, last)) if near else None
                kind = "right" if code is truth else "wrong"
                prev, last = last, truth
                name = f"{slug(ep.key)}_{role}_{c.idx}"
                if only is not None:
                    if name not in only:
                        continue
                elif len(cands) < 2 or want[kind] <= 0 or per_ep[kind] >= (3 if kind == "right" else 6):
                    continue
                elif kind == "right" and rng.random() > 0.3:    # spread the easy cases over more episodes
                    continue
                half = half_for(ref, *cands)
                height = max(ref.h95, 4.0)
                refs = {cam: save(render.crop(first, cam, ref.centroid, half, height, size=TILE, label="REF"),
                                  f"{slug(ep.key)}_{role}_ref_{cam}.jpg") for cam in ("top_camera", "agent_camera")}
                sheets = {}
                for cam in ("top_camera", "agent_camera"):
                    tiles = [render.crop(c, cam, b.centroid, half, max(height, b.h95), size=TILE, label=str(k + 1))
                             for k, b in enumerate(cands)]
                    sheets[cam] = save(render.tile_sheet(tiles, cols=3), f"{name}_cands_{cam}.jpg")
                # the same views with the thing each tile stands for outlined
                refs_o = {cam: save(render.crop(first, cam, ref.centroid, half, height, size=TILE, label="REF",
                                                image=outlined(first, cam, ref)),
                                    f"{slug(ep.key)}_{role}_ref_o_{cam}.jpg") for cam in ("top_camera", "agent_camera")}
                sheets_o = {}
                for cam in ("top_camera", "agent_camera"):
                    tiles = [render.crop(c, cam, b.centroid, half, max(height, b.h95), size=TILE, label=str(k + 1),
                                         image=outlined(c, cam, b)) for k, b in enumerate(cands)]
                    sheets_o[cam] = save(render.tile_sheet(tiles, cols=3), f"{name}_cands_o_{cam}.jpg")
                truth_k = next((k + 1 for k, b in enumerate(cands) if b is truth), None)
                items.append({
                    "id": name, "task": ep.task, "instruction": ep.instruction, "kind": kind,
                    "truth": f"T{truth_k}" if truth_k else "none",
                    # every tile that is a piece of the object (the arm can split one object into two blobs)
                    "truths": [f"T{k + 1}" for k, b in enumerate(cands) if act[b.id][0] in ids and act[b.id][1] >= 0.5],
                    "code": f"T{cands.index(code) + 1}" if code is not None else "none",
                    "refs": refs, "sheets": sheets, "refs_o": refs_o, "sheets_o": sheets_o,
                    "ref_num": {"h95": round(ref.h95, 1), "lw": [round(ref.length), round(ref.width)]},
                    "cands": [{"dist": round(math.dist(b.centroid, prev.centroid)), "h95": round(b.h95, 1),
                               "lw": [round(b.length), round(b.width)]} for b in cands],
                    "moved": round(math.dist(truth.centroid, prev.centroid), 1)})
                want[kind] -= 1
                per_ep[kind] += 1
    if only is not None:
        order = {i: k for k, i in enumerate(only)}
        items.sort(key=lambda it: order[it["id"]])
    return items


def build_topple(eps, boxes, n, rng):
    from branchlab.rtscene import labels, render
    j1 = [json.loads(line) for line in open(PROJECT/"outputs/analysis/j1_rows.jsonl")]
    ep_of = {e.key: e for e in eps}

    def lab(r):
        return "lost" if (r["off_table"] or r["true_tilt"] > 60) else ("ok" if r["true_tilt"] < 15 else "mid")
    rows = [r for r in j1 if r["status"] == "visible" and r["ref_h95"] >= 1.5 and r["track_correct"]
            and lab(r) != "mid" and r["capture"] > 0 and r["key"] in ep_of]
    rng.shuffle(rows)
    items, want, per_ep = [], {"lost": n, "ok": n}, collections.Counter()
    parsed_of = {}
    for r in rows:
        k = lab(r)
        if want[k] <= 0 or per_ep[(r["key"], k)] >= 6:
            continue
        ep = ep_of[r["key"]]
        caps = {c.idx: c for c in ep.captures()}
        c = caps[r["capture"]]
        ids = labels.object_ids(ep)[r["role"]]
        obj = labels.objects(ep, c)[r["role"]]["position_cm"]
        if gripped(c, obj):
            continue
        parsed = parsed_of.setdefault(ep.key, Parsed(ep, boxes))
        ref = true_blob(*parsed(caps[0]), ids)
        now = true_blob(*parsed(c), ids)
        if ref is None or now is None:
            continue
        half = half_for(ref, now)
        height = max(ref.h95, now.h95, 4.0)
        name = f"{slug(ep.key)}_{r['role']}_{c.idx}"
        imgs = {}
        for cam in ("top_camera", "agent_camera"):
            imgs[f"ref_{cam}"] = save(render.crop(caps[0], cam, ref.centroid, half, height, size=TILE, label="START"),
                                      f"{name}_ref_{cam}.jpg")
            imgs[f"now_{cam}"] = save(render.crop(c, cam, now.centroid, half, height, size=TILE, label="NOW"),
                                      f"{name}_now_{cam}.jpg")
        from PIL import Image
        pair = render.tile_sheet([Image.open(OUT/"images"/imgs["ref_agent_camera"]),
                                  Image.open(OUT/"images"/imgs["now_agent_camera"])], cols=2)
        imgs["pair"] = save(pair, f"{name}_pair.jpg")
        items.append({"id": name, "task": ep.task, "label": k == "lost", "tilt": r["true_tilt"], "c0": r["c0"],
                      "imgs": imgs, "num": {"h_ref": round(ref.h95, 1), "h_now": round(now.h95, 1),
                                            "lw_ref": [round(ref.length), round(ref.width)],
                                            "lw_now": [round(now.length), round(now.width)]}})
        want[k] -= 1
        per_ep[(r["key"], k)] += 1
    return items


def draw_grip(cap, cam, st):
    """The view with the closing finger pads drawn: magenta pads, a yellow line between them."""
    from PIL import Image, ImageDraw
    from branchlab.rtscene.geometry import project
    img = Image.fromarray(cap.rgb(cam)).convert("RGBA")
    K = cap.K(cam).copy()
    K[:2] *= 2 if cam in ("top_camera", "agent_camera") else 1
    a = np.asarray(st["approach"], float); a /= np.linalg.norm(a)
    f = np.asarray(st["finger_axis"], float); f /= np.linalg.norm(f)
    nrm = np.cross(a, f)
    tcp = np.asarray(st["position_cm"], float)
    gap = GAP_PER_OPENING * float(st["gripper_real"])
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    centres = []
    for s in (-1, 1):
        c = tcp + s * f * gap / 2
        centres.append(c)
        corners = np.array([c + a * u + nrm * v for u, v in ((-PAD, -SLAB), (0.5, -SLAB), (0.5, SLAB), (-PAD, SLAB))])
        uv, _ = project(corners, K, cap.E(cam))
        d.polygon([tuple(p) for p in uv], fill=(255, 0, 255, 110), outline=(255, 0, 255, 255), width=3)
    uv, _ = project(np.array(centres), K, cap.E(cam))
    d.line([tuple(p) for p in uv], fill=(255, 255, 0, 255), width=3)
    return Image.alpha_composite(img, over).convert("RGB"), tcp


def clear_view(caps, cap, tcp, radius: float = 12.0):
    """The latest capture before `cap` with both arms' fingertips and wrists more than `radius` cm (in x/y) from
    the closing point, so the object there is in view; the first capture if none."""
    def clear(c):
        return all(math.hypot(st[k][0] - tcp[0], st[k][1] - tcp[1]) > radius
                   for st in c.arms.values() for k in ("position_cm", "wrist_cm") if k in st)
    before = [c for c in caps if c.idx < cap.idx and c.kind in ("turn", "preopen", "final")]
    return max((c for c in before if clear(c)), key=lambda c: c.idx, default=caps[0])


def build_grip(eps, boxes, n, rng):
    from branchlab.rtscene import render
    from rtscene_j3 import memory_features
    cache = {}
    j3 = [json.loads(line) for line in open(PROJECT/"outputs/analysis/j3_rows.jsonl")]
    ep_of = {e.key: e for e in eps}
    rows = [r for r in j3 if r["label"] in ("grasp", "empty") and r["key"] in ep_of and "mem_dist" in r]
    rng.shuffle(rows)
    items, want, per_ep = [], {"grasp": n, "empty": n}, collections.Counter()
    for r in rows:
        if want[r["label"]] <= 0 or per_ep[(r["key"], r["label"])] >= 4:
            continue
        ep = ep_of[r["key"]]
        caps = ep.captures()
        cap = next(c for c in caps if c.idx == r["capture"])
        st = cap.arms[r["arm"]]
        start = clear_view(caps, cap, st["position_cm"])
        name = f"{slug(r['key'])}_{cap.idx}_{r['arm']}"
        imgs = {}
        for when, c in (("start", start), ("now", cap)):
            for cam in ("top_camera", "agent_camera"):
                drawn, tcp = draw_grip(c, cam, st)
                h = max(tcp[2] - c.table_z + 3, 4.0)
                imgs[f"{when}_{cam}"] = save(render.crop(c, cam, tcp[:2], 8.0, h, size=320, image=drawn),
                                             f"{name}_{when}_{cam}.jpg")
        mem = memory_features(ep, cap, r["arm"], boxes, cache.setdefault(ep.key, {}), start=start)
        items.append({"id": name, "task": ep.task, "label": r["label"] == "grasp", "imgs": imgs,
                      "start": start.idx, "start_turn_gap": cap.turn - start.turn,
                      "code": {k: mem.get(k) for k in ("mem_dist", "mem_z_rel", "mem_extent_f", "mem_centre_f", "gap",
                                                      "mem_fits", "mem_h95")},
                      "code_turnstart": {k: r.get(k) for k in ("mem_dist", "mem_z_rel")}})
        want[r["label"]] -= 1
        per_ep[(r["key"], r["label"])] += 1
    return items


def build(n: int, probes, seed: int = 0):
    from branchlab.rtscene import robot_model
    rng = random.Random(seed)
    boxes = robot_model.load()
    (OUT/"images").mkdir(parents=True, exist_ok=True)
    eps = episodes_all()
    rng.shuffle(eps)
    path = OUT/"items.json"
    items = json.loads(path.read_text()) if path.exists() else {}
    for probe in probes:
        items[probe] = {"reid": build_reid, "topple": build_topple, "grip": build_grip}[probe](eps, boxes, n, rng)
        path.write_text(json.dumps(items))
        print(probe, len(items[probe]), collections.Counter(
            str(i.get("kind", i.get("label"))) for i in items[probe]), flush=True)


# ------------------------------------------------------------------------------------------------ ask

def uri(name: str) -> str:
    import base64
    return "data:image/jpeg;base64," + base64.b64encode((OUT/"images"/name).read_bytes()).decode()


def request(probe: str, arm: str, it: dict):
    """(state, questions, image file names) for one item in one interface arm."""
    T, A = "top_camera", "agent_camera"
    if probe == "reid":
        k = len(it["cands"])
        crit = {f"T{i}": f"tile {i}" for i in range(1, k + 1)}
        crit["none"] = "none of the tiles"
        outline = arm.startswith("o:")
        base = arm.removeprefix("o:")
        cams = {"top": [T], "agent": [A]}.get(base, [T, A])
        imgs = []
        for cam in cams:
            imgs += [it["refs_o" if outline else "refs"][cam], it["sheets_o" if outline else "sheets"][cam]]
        view = {T: "from straight above", A: "from the front, at an angle"}
        if outline:
            lines = [f"A robot works at a table. The REF image shows one object at the start, outlined in cyan, "
                     f"{view[cams[0]]}. The next image shows {k} numbered tiles of the same view now; in each tile one "
                     "thing found near where the object was last seen is outlined in cyan. All tiles are at the same "
                     "scale as REF."]
        else:
            lines = [f"A robot works at a table. The REF image shows one object at the start, {view[cams[0]]}. The next "
                     f"image shows {k} numbered tiles, the same view now, one tile per thing found near where the object "
                     "was last seen. All tiles are at the same scale as REF."]
        if len(cams) == 2:
            lines.append("Images 3 and 4 repeat this from the front, at an angle.")
        if "num" in base:
            rn = it["ref_num"]
            lines.append(f"Measured by depth: REF top {rn['h95']} cm above the table, footprint {rn['lw'][0]} x {rn['lw'][1]} cm.")
            for i, c in enumerate(it["cands"], 1):
                lines.append(f"Tile {i}: {c['dist']} cm from where the object was last seen, top {c['h95']} cm above the "
                             f"table, footprint {c['lw'][0]} x {c['lw'][1]} cm.")
        if "task" in base:
            lines.append(f"The robot's task: {it['instruction']}")
        what = "Which tile's outlined thing is" if outline else "Which tile shows"
        q = {"which": {"type": "choice", "criteria": crit,
                       "instructions": f"{what} the same physical object as REF? It may have been moved, turned or "
                                       "knocked over since the start, so match its colour, markings and shape."}}
        return " ".join(lines), q, imgs
    if probe == "topple":
        im = it["imgs"]
        if arm == "pair":
            imgs = [im["pair"]]
            state = "The image shows one object on a robot's table: START (left) at the start, NOW (right) now, same scale."
        elif arm == "both":
            imgs = [im["ref_top_camera"], im["now_top_camera"], im["ref_agent_camera"], im["now_agent_camera"]]
            state = ("One object on a robot's table, at the start (START) and now (NOW), at the same scale. Images 1-2: "
                     "from straight above. Images 3-4: from the front, at an angle.")
        else:
            imgs = [im["ref_agent_camera"], im["now_agent_camera"]]
            state = "One object on a robot's table, at the start (image 1, START) and now (image 2, NOW), at the same scale."
        if "num" in arm:
            m = it["num"]
            state += (f" Measured by depth: its top is {m['h_ref']} cm above the table at the start and {m['h_now']} cm now; "
                      f"its footprint is {m['lw_ref'][0]} x {m['lw_ref'][1]} cm at the start and "
                      f"{m['lw_now'][0]} x {m['lw_now'][1]} cm now.")
        q = {"fallen": {"type": "noul", "instructions": "Has the object fallen over or tipped onto its side since the start?"}}
        return state, q, imgs
    im = it["imgs"]
    imgs = {"top": [im["start_top_camera"]], "both": [im["start_top_camera"], im["start_agent_camera"]],
            "both+now": [im["start_top_camera"], im["start_agent_camera"], im["now_top_camera"], im["now_agent_camera"]],
            "both+num": [im["start_top_camera"], im["start_agent_camera"]]}[arm]
    state = ("A robot gripper is about to close. Two magenta bars mark where its two finger pads will be; they move "
             "toward each other along the yellow line. Image 1 is from straight above")
    state += {"top": ", taken before the arm moved there.", "both": " and image 2 from the front, both taken before the "
              "arm moved there.", "both+num": " and image 2 from the front, both taken before the arm moved there.",
              "both+now": " and image 2 from the front, both taken before the arm moved there; images 3-4 are the same "
              "views now, with the gripper in place."}[arm]
    if arm == "both+num":
        c = it["code"]
        state += (f" Measured by depth before the arm moved: the fingertips are {c['mem_dist']} cm from the nearest object's "
                  f"outline (0 means above it) and {c['mem_z_rel']} cm above its top (negative means below the top); "
                  f"across the closing line the object spans {c['mem_extent_f']} cm; the opening is {c['gap']} cm.")
    q = {"grip": {"type": "noul", "instructions": "Will the two finger pads close on an object and squeeze it between them?"}}
    return state, q, imgs


def cache_key(probe, arm, it, state, qs, imgs, model) -> str:
    h = hashlib.sha1(json.dumps([state, qs, model]).encode())
    for name in imgs:
        h.update((OUT/"images"/name).read_bytes())
    return f"{model}|{probe}|{arm}|{it['id']}|{h.hexdigest()[:10]}"


def ask(probes, arms, model, workers):
    from branchlab.clef import Clef
    items = json.loads((OUT/"items.json").read_text())
    clef = Clef(cache_path=OUT/"responses.jsonl", backend="workers")
    jobs = []
    for probe in probes:
        for arm in arms or ARMS[probe]:
            if arm not in ARMS[probe]:
                continue
            for it in items.get(probe, []):
                state, qs, imgs = request(probe, arm, it)
                jobs.append((cache_key(probe, arm, it, state, qs, imgs, model), state, qs, imgs))

    def run(job):
        key, state, qs, imgs = job
        try:
            return key, clef.ask(state, qs, [uri(i) for i in imgs], model=model, key=key)
        except Exception as exc:  # noqa: BLE001
            return key, {"error": str(exc)[:200]}
    with ThreadPoolExecutor(workers) as pool:
        errors = [(k, r["error"]) for k, r in pool.map(run, jobs) if "error" in r]
    print(f"{len(jobs)} asked, {len(errors)} errors", errors[:2])


# ------------------------------------------------------------------------------------------------ report

def auc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def block_at_keep(scores, labels, keep=0.9):
    """Share of negatives blocked when the threshold keeps `keep` of the positives."""
    pos = sorted((s for s, y in zip(scores, labels) if y), reverse=True)
    if not pos:
        return float("nan")
    thr = pos[min(len(pos) - 1, math.ceil(keep * len(pos)) - 1)]
    neg = [s for s, y in zip(scores, labels) if not y]
    return sum(s < thr for s in neg) / max(len(neg), 1)


def ranks(x):
    x = np.asarray(x, float)
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


def report(model):
    items = json.loads((OUT/"items.json").read_text())
    res = {}
    if (OUT/"responses.jsonl").exists():
        for line in open(OUT/"responses.jsonl"):
            row = json.loads(line)
            res[row["key"]] = row["result"]

    def answer(probe, arm, it):
        state, qs, imgs = request(probe, arm, it)
        return res.get(cache_key(probe, arm, it, state, qs, imgs, model))

    its = items.get("reid", [])
    if its:
        def right(pick, it):                 # any piece of the object counts; none is right when no tile is
            truths = it.get("truths", [it["truth"]] if it["truth"] != "none" else [])
            return pick in truths if truths else pick == "none"
        print(f"== reid ({len(its)} items: code right {sum(i['kind'] == 'right' for i in its)}, "
              f"wrong {sum(i['kind'] == 'wrong' for i in its)}; split into pieces "
              f"{sum(len(i.get('truths', [])) > 1 for i in its)}); accuracy counts any piece of the object")
        print(f"  {'code tracker':22} all {np.mean([right(i['code'], i) for i in its]):.0%}")
        print(f"  {'nearest blob':22} all {np.mean([right('T1', i) for i in its]):.0%}")
        for arm in ARMS["reid"]:
            got = [(i, answer("reid", arm, i)) for i in its]
            got = [(i, r["answers"]["which"]) for i, r in got if r]
            if not got:
                continue
            ok = {k: [right(a["choice"], i) for i, a in got if i["kind"] == k] for k in ("right", "wrong")}
            # fused with the distance prior: argmax of log p - dist / D
            fused = {}
            for D in (5, 10, 20):
                hits = []
                for i, a in got:
                    pr = a["probabilities"]
                    score = {f"T{k + 1}": math.log(max(pr.get(f"T{k + 1}", 0), 1e-6)) - c["dist"] / D
                             for k, c in enumerate(i["cands"])}
                    score["none"] = math.log(max(pr.get("none", 0), 1e-6)) - CAND_RADIUS / D
                    hits.append(right(max(score, key=score.get), i))
                fused[D] = np.mean(hits)
            print(f"  {arm:22} all {np.mean(ok['right'] + ok['wrong']):.0%} | code-right {np.mean(ok['right']):.0%} "
                  f"code-wrong {np.mean(ok['wrong']) if ok['wrong'] else float('nan'):.0%} | with distance prior "
                  f"D=5/10/20 cm: " + " / ".join(f"{v:.0%}" for v in fused.values()) + f"  (n {len(got)})")
    its = items.get("topple", [])
    if its:
        print(f"== topple ({len(its)} items, fallen {sum(i['label'] for i in its)})")
        print(f"  {'code shape score':22} AUC {auc([i['c0'] for i in its if i['label']], [i['c0'] for i in its if not i['label']]):.3f}")
        for arm in ARMS["topple"]:
            got = [(i, answer("topple", arm, i)) for i in its]
            got = [(i, r["answers"]["fallen"]["noul"]) for i, r in got if r]
            if not got:
                continue
            pos = [p for i, p in got if i["label"]]
            neg = [p for i, p in got if not i["label"]]
            acc = np.mean([(p >= 0.5) == i["label"] for i, p in got])
            print(f"  {arm:22} AUC {auc(pos, neg):.3f} acc@0.5 {acc:.0%} (n {len(got)})")
    its = items.get("grip", [])
    if its:
        print(f"== grip ({len(its)} items, grasps {sum(i['label'] for i in its)}); "
              "'blocks' = share of empty closes stopped while keeping 90% of grasps")
        y = [i["label"] for i in its]
        def geo(c):
            return -(c.get("mem_dist") or 0) - max(c.get("mem_z_rel") or 0, 0)
        old = [geo(i["code_turnstart"]) for i in its]
        print(f"  {'code, turn-start view':22} AUC {auc([s for s, l in zip(old, y) if l], [s for s, l in zip(old, y) if not l]):.3f} "
              f"blocks {block_at_keep(old, y):.0%}")
        code = [geo(i["code"]) for i in its]
        print(f"  {'code, clear view':22} AUC {auc([s for s, l in zip(code, y) if l], [s for s, l in zip(code, y) if not l]):.3f} "
              f"blocks {block_at_keep(code, y):.0%}")
        for arm in ARMS["grip"]:
            got = [answer("grip", arm, i) for i in its]
            if not all(got):
                continue
            p = [r["answers"]["grip"]["noul"] for r in got]
            both = list(ranks(p) + ranks(code))
            print(f"  {arm:22} AUC {auc([s for s, l in zip(p, y) if l], [s for s, l in zip(p, y) if not l]):.3f} "
                  f"blocks {block_at_keep(p, y):.0%} | with code: AUC "
                  f"{auc([s for s, l in zip(both, y) if l], [s for s, l in zip(both, y) if not l]):.3f} "
                  f"blocks {block_at_keep(both, y):.0%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=60)
    b.add_argument("--probes", nargs="+", default=["reid", "topple", "grip"])
    a = sub.add_parser("ask")
    a.add_argument("--probes", nargs="+", default=["reid", "topple", "grip"])
    a.add_argument("--arms", nargs="+")
    a.add_argument("--model", default="clef")
    a.add_argument("--workers", type=int, default=8)
    sub.add_parser("augment-reid")
    r = sub.add_parser("report")
    r.add_argument("--model", default="clef")
    args = parser.parse_args()
    if args.step == "build":
        build(args.n, args.probes)
    elif args.step == "augment-reid":
        from branchlab.rtscene import robot_model
        path = OUT/"items.json"
        items = json.loads(path.read_text())
        ids = [i["id"] for i in items["reid"]]
        new = build_reid(episodes_all(), robot_model.load(), 0, random.Random(0), only=ids)
        missing = set(ids) - {i["id"] for i in new}
        if missing:
            print(f"{len(missing)} items not rebuilt, kept as they were:", sorted(missing)[:3])
        by_id = {i["id"]: i for i in new}
        items["reid"] = [by_id.get(i["id"], i) for i in items["reid"]]
        path.write_text(json.dumps(items))
        print(f"rebuilt {len(new)} reid items")
    elif args.step == "ask":
        ask(args.probes, args.arms, args.model, args.workers)
    else:
        report(args.model)


if __name__ == "__main__":
    main()
