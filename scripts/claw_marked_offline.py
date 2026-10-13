#!/usr/bin/env python3
"""Claw-machine test with the grasp spot drawn on the crop (2026-10-13): when the planner's chosen grasp spot is in the
picture, can Clef (and Gemini) read where the red cross stands relative to it?

The claw retest (scripts/claw_retest_offline.py, 10-11) showed Clef a straight-down crop (+-16 cm, 320 px, about 1 mm
per pixel) of the raw top camera about the fingertips, a red cross on the table point under them (interface img_table)
and the target's name, and asked for one 2 cm step. The grasp point it was scored against (item["grasp"], the
fingertip position at the later successful close, carried rigidly to the object's pose in that capture) was never
shown. Clef could not tell aligned from 3-8 cm away (AUC of P(stay) about 0.5-0.6), and at 3-8 cm its direction matched
an image-free rule. With the target spot absent from the image, that cannot separate a perception limit from missing
information. Here the spot is drawn.

Items: outputs/analysis/claw_retest/items_v2.json, every item whose target is tracked (stratum != unverifiable; 242
items, both halves: nothing is tuned here, so the development half is used too; the test half alone is also reported
for continuity with the retest).

Marked crop: claw_retest_offline.crop_for(item, "img_table") unchanged (the very image Clef saw before), plus a blue
ring (RGB 0,90,255, 3 px wide) of radius 1.5 cm on the table plane centred on the table point under the grasp point
(item["grasp"] x/y at cap.table_z) with a filled blue centre dot (radius 3 px), and the red cross (same size, width and
place as crop_for) drawn again on top so it stays visible. The ring's pixel radius is the projected distance of a table
point 1.5 cm from its centre (crop_for's projection: K doubled, cap.E; the top camera looks straight down, so it is the
same in every direction). Both marks lie on the table plane, so their offset in the image is the x/y offset with no
parallax.

Questions (Clef on Workers AI, two separate calls per item; cache outputs/analysis/claw_marked/clef_responses.jsonl):
  dir  choice  the retest's options, with stay = "stay: the red cross is already inside the blue circle"; "Which
               single step brings the red cross closer to the centre of the blue circle? Steps are about 2 cm."
  in   noul    "Is the centre of the red cross inside the blue circle?"
Gemini 3.8 Flash (the retest's client: standard tier, default reasoning, max_tokens 8000) on the marked crop, only for
the 73 items Gemini already answered unmarked (outputs/analysis/claw_retest/gemini_v2.jsonl), so the two pair up; one
word, parsed as in the retest.

Scoring is the retest's: a 2 cm step that brings the fingertips over 0.5 cm closer (x/y) to the grasp point is good,
over 0.5 cm farther bad, else neutral; stay is stay_ok within 1.5 cm and a stall beyond. Rates weight every approach
equally; intervals resample approaches. Stay rules: forced (the likeliest of the four directions) and p0.5 (stay when
P(stay) >= 0.5). The unmarked condition is the retest's Clef img_table answer (clef_v2.jsonl) on the same items.
Alignment is the AUC (unweighted, Mann-Whitney; interval resampling approaches) of P(stay) (dir) or P(yes) (in) with
dist <= 1.5 cm as positives against 3-8 cm, and against 1.5-3 cm.

Pre-registered reading (fixed 2026-10-13, before any answer), Clef, all 242 items, marked condition:
  A  AUC of P(stay), <= 1.5 vs 3-8 cm, >= 0.85
  B  3-8 cm, forced rule: good >= 80% and bad <= 10%
  C  AUC of P(stay), <= 1.5 vs 3-8 cm, <= 0.70
  D  3-8 cm, forced rule: good is not at least 10 pp above the unmarked good on the same items
  reads the mark          A and B, and neither C nor D
  does not read the mark  C or D, and not (A and B)
  partial                 otherwise (also when A, B and D all hold: alignment read but the step no better than before;
                          the report names the conditions that held)
Reported, not part of the reading: P(yes) of the in question, the p0.5 rule, Gemini marked against unmarked, the test
half alone, and occlusion strata (item["occluded"], the share of a 2 cm disc about the grasp point the robot covers
in the top camera; the marks are drawn over the robot, so occlusion hides the object, not the mark).

    PYTHONPATH=src:scripts python scripts/claw_marked_offline.py render [--check-all]   # samples, no model calls;
        # --check-all reads the marks back from the drawn pixels of every item (geometry.json, used by the report)
    PYTHONPATH=src:scripts python scripts/claw_marked_offline.py clef [--workers 6]
    PYTHONPATH=src:scripts python scripts/claw_marked_offline.py gemini [--workers 6]
    PYTHONPATH=src:scripts python scripts/claw_marked_offline.py report
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
for p in (PROJECT/"src", PROJECT/"scripts", PROJECT/"third_party/robodawn"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import claw_retest_offline as cr      # noqa: E402  (the retest: items, crop, scoring; imported, never changed)

OUT = PROJECT/"outputs/analysis/claw_marked"
RETEST = cr.OUT
KEY = cr.KEY                          # the Vertex service account key (path only)

RING_CM = 1.5
RING_RGB = (0, 90, 255)
RING_W = 3
DOT_R = 3
CROSS_R = cr.SIZE // 22               # crop_for's cross: half length and width
CROSS_W = 3
CROSS_RGB = (255, 0, 0)
IFACE = "img_table"
STATE = ("A robot gripper is about to grasp something on a table. The picture looks straight down at the table; the "
         "robot stands at the bottom edge of the picture. The red cross marks the point on the table directly below "
         "the gripper's fingertips. The blue circle marks the spot the planner chose to grasp on {target}.")
OPTIONS = {**{k: v for k, v in cr.OPTIONS.items() if k != "stay"},
           "stay": "stay: the red cross is already inside the blue circle"}
QUESTIONS = {
    "dir": {"step": {"type": "choice", "criteria": OPTIONS,
                     "instructions": "Which single step brings the red cross closer to the centre of the blue circle? "
                                     "Steps are about 2 cm."}},
    "in": {"inside": {"type": "noul", "instructions": "Is the centre of the red cross inside the blue circle?"}},
}
GEMINI_Q = ("\n\nWhich single step (about 2 cm) brings the red cross closer to the centre of the blue circle? Answer with "
            "exactly one word: " + ", ".join(f"{k} ({v})" for k, v in OPTIONS.items()) + ".")
SAMPLES_PER_BIN = 3
OUTCOMES = cr.OUTCOMES


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def tracked_items() -> list[dict]:
    return [i for i in json.loads((RETEST/"items_v2.json").read_text()) if i["stratum"] != "unverifiable"]


# ----------------------------------------------------------------------------- images
def geometry(item: dict) -> dict:
    """crop_for's crop box and scale for img_table (the same arithmetic), and the cross and ring in crop pixels."""
    from branchlab.rtscene import render
    from branchlab.rtscene.geometry import project
    cap = cr.capture_of(item)
    cam = "top_camera"
    x, y, _ = item["tip"]
    u0, v0, u1, v1 = render.object_box(cap, cam, (x, y), cr.HALF, cap.table_z, cap.table_z + 2)
    side = max(u1 - u0, v1 - v0, 24)
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    box = (int(cu - side / 2), int(cv - side / 2), int(cu + side / 2), int(cv + side / 2))
    K = cap.K(cam).copy()
    K[:2] *= 2
    s = cr.SIZE / (box[2] - box[0])
    gx, gy = item["grasp"][0], item["grasp"][1]
    z = cap.table_z
    pts = np.array([[x, y, z], [gx, gy, z]] + [[gx + RING_CM * math.cos(a), gy + RING_CM * math.sin(a), z]
                                               for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)])
    uv, _ = project(pts, K, cap.E(cam))
    uv = (uv - np.array(box[:2])) * s
    radii = [float(np.linalg.norm(uv[k] - uv[1])) for k in range(2, 6)]
    return {"cross": (float(uv[0][0]), float(uv[0][1])), "ring": (float(uv[1][0]), float(uv[1][1])),
            "ring_r": radii[0], "ring_r_spread": max(radii) - min(radii), "box": box, "scale": s}


def draw_cross(img, px: float, py: float):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.line([px - CROSS_R, py, px + CROSS_R, py], fill=CROSS_RGB, width=CROSS_W)
    d.line([px, py - CROSS_R, px, py + CROSS_R], fill=CROSS_RGB, width=CROSS_W)


def marked_crop(item: dict, geo: dict | None = None):
    """(unmarked crop exactly as the retest made it, marked crop, geometry). The ring and dot go onto crop_for's
    image and the cross is drawn again on top; drawing is opaque, so this equals ring, dot, then cross on the raw crop.
    Raises if redrawing the cross at this module's position changes crop_for's image (the geometry would differ)."""
    from PIL import ImageDraw
    geo = geo or geometry(item)
    base = cr.crop_for(item, IFACE)
    check = base.copy()
    draw_cross(check, *geo["cross"])
    if not np.array_equal(np.asarray(check), np.asarray(base)):
        raise RuntimeError(f"{item['id']}: the cross position differs from crop_for's")
    out = base.copy()
    d = ImageDraw.Draw(out)
    (cx, cy), r = geo["ring"], geo["ring_r"]
    # PIL strokes inward from the box and its box is inclusive of the last pixel index, so pixel centres of a w px
    # stroke span rr - (w - 1) .. rr; rr = r + (w - 1) / 2 centres the 3 px stroke on the 1.5 cm radius (checked by a
    # circle fit to the drawn pixels in render --check-all; r + w / 2 put it about 0.5 px outside)
    rr = r + (RING_W - 1) / 2
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=RING_RGB, width=RING_W)
    d.ellipse([cx - DOT_R, cy - DOT_R, cx + DOT_R, cy + DOT_R], fill=RING_RGB)
    draw_cross(out, *geo["cross"])
    return base, out, geo


def est_dist(geo: dict) -> float:
    """The cross-to-ring-centre distance in cm read back from the image (pixels / ring radius * 1.5)."""
    return math.dist(geo["cross"], geo["ring"]) / geo["ring_r"] * RING_CM


def to_table(cap, geo: dict, px: float, py: float) -> np.ndarray:
    """A crop pixel back to the table plane (world cm): the inverse of crop_for's resize and projection (ray-plane
    intersection), independent of the forward projection used to draw."""
    cam = "top_camera"
    K = cap.K(cam).copy()
    K[:2] *= 2
    E = cap.E(cam)
    R, t = E[:3, :3], E[:3, 3]
    box = geo["box"]
    u = px / (cr.SIZE / (box[2] - box[0])) + box[0]
    v = py / (cr.SIZE / (box[3] - box[1])) + box[1]
    ray = R.T @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    origin = R.T @ (-t)
    lam = (cap.table_z / 100.0 - origin[2]) / ray[2]
    return (origin + lam * ray) * 100.0


def fit_circle(pts: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle (centre x, centre y, radius) through points."""
    x, y = pts[:, 0], pts[:, 1]
    (a, b, c), *_ = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones_like(x)], x ** 2 + y ** 2, rcond=None)
    return float(a), float(b), float(math.sqrt(c + a * a + b * b))


def drawn_marks(base, marked) -> dict:
    """The marks as found in the pixels (PIL pixel index convention): the cross centre as the mean of crop_for's pure
    red pixels, and the ring as a circle fitted to the ring-coloured pixels the marking added (the dot dropped)."""
    B, M = np.asarray(base).astype(int), np.asarray(marked).astype(int)
    ry, rx = np.nonzero(np.all(B == CROSS_RGB, axis=-1))
    by, bx = np.nonzero(np.all(M == RING_RGB, axis=-1) & np.any(M != B, axis=-1))
    pts = np.c_[bx, by].astype(float)
    cx, cy, r = fit_circle(pts)
    for _ in range(2):
        cx, cy, r = fit_circle(pts[np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) > r / 2])
    return {"cross": (float(rx.mean()), float(ry.mean())), "ring": (cx, cy), "ring_r": r}


def check_item(item: dict) -> dict:
    """Geometry of one item, checked three ways: crop_for's cross redrawn (raises if it moved), the drawn pixels
    back-projected to the table and compared with item["tip"], item["grasp"] and item["dist"], and the ring's reach
    beyond the crop edge."""
    geo = geometry(item)
    base, marked, _ = marked_crop(item, geo)
    cap = cr.capture_of(item)
    px = drawn_marks(base, marked)
    wc = to_table(cap, geo, *px["cross"])
    wr = to_table(cap, geo, *px["ring"])
    r_cm = np.mean([math.hypot(*(to_table(cap, geo, px["ring"][0] + px["ring_r"] * math.cos(a),
                                          px["ring"][1] + px["ring_r"] * math.sin(a)) - wr)[:2])
                    for a in (0, math.pi / 2, math.pi, 1.5 * math.pi)])
    gx, gy = geo["ring"]
    outer = geo["ring_r"] + RING_W / 2
    ang = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    ux, uy = gx + geo["ring_r"] * np.cos(ang), gy + geo["ring_r"] * np.sin(ang)
    return {"id": item["id"], "dist": item["dist"], "bin": cr.bin_of(item["dist"]), "split": item["split"],
            "dist_drawn": round(math.hypot(wr[0] - wc[0], wr[1] - wc[1]), 3),
            "offset_err": round(math.hypot((wr[0] - wc[0]) - (item["grasp"][0] - item["tip"][0]),
                                           (wr[1] - wc[1]) - (item["grasp"][1] - item["tip"][1])), 3),
            "cross_vs_tip": round(math.hypot(wc[0] - item["tip"][0], wc[1] - item["tip"][1]), 3),
            "ring_vs_grasp": round(math.hypot(wr[0] - item["grasp"][0], wr[1] - item["grasp"][1]), 3),
            "ring_r_px": round(geo["ring_r"], 2), "ring_r_drawn_px": round(px["ring_r"], 2),
            "ring_r_drawn_cm": round(float(r_cm), 3), "ring_r_spread_px": round(geo["ring_r_spread"], 3),
            "inside_drawn": math.dist(px["cross"], px["ring"]) <= px["ring_r"],
            "clipped": bool(gx - outer < 0 or gy - outer < 0 or gx + outer > cr.SIZE or gy + outer > cr.SIZE),
            "centre_in_crop": bool(0 <= gx < cr.SIZE and 0 <= gy < cr.SIZE),
            "ring_arc_in_crop": round(float(np.mean((ux >= 0) & (ux < cr.SIZE) & (uy >= 0) & (uy < cr.SIZE))), 3)}


# ----------------------------------------------------------------------------- render (no model calls)
def cmd_render(args):
    items = tracked_items()
    sd = OUT/"samples"
    sd.mkdir(parents=True, exist_ok=True)
    picks = []
    for lo, hi, b in cr.BINS:
        sub = sorted((i for i in items if lo <= i["dist"] < hi), key=lambda i: h(i["id"]))
        seen = set()
        for i in sub:                       # one per approach
            if i["approach"] not in seen and len([p for p in picks if p[0] == b]) < SAMPLES_PER_BIN:
                picks.append((b, i))
                seen.add(i["approach"])
    # plus every item whose ring reaches past the crop edge (only items near 15 cm can)
    picks += [("edge", i) for i in items if i["dist"] >= 14.0 and check_item(i)["clipped"]]
    index = []
    print(f"{'sample':14s} {'dist':>6s} {'px':>6s} {'r_px':>6s} {'est':>6s} {'err':>6s}  inside  id")
    for k, (b, i) in enumerate(picks):
        base, marked, geo = marked_crop(i)
        name = f"{k:02d}_{b.replace('<', 'lt').replace('.', 'p')}"
        base.save(sd/f"{name}_unmarked.png")
        marked.save(sd/f"{name}_marked.png")
        e = est_dist(geo)
        px = math.dist(geo["cross"], geo["ring"])
        index.append({"file": name, "id": i["id"], "bin": b, "dist": i["dist"], "target": i["target"],
                      "occluded": i["occluded"], "px": round(px, 1), "ring_r_px": round(geo["ring_r"], 2),
                      "est_cm": round(e, 2), "err_cm": round(e - i["dist"], 2)})
        print(f"{name:14s} {i['dist']:6.2f} {px:6.1f} {geo['ring_r']:6.2f} {e:6.2f} {e - i['dist']:+6.2f}  "
              f"{'yes' if px <= geo['ring_r'] else 'no ':6s}  {i['id']} ({i['target']})")
    (sd/"index.json").write_text(json.dumps(index, indent=1))
    bad = [x for x in index if abs(x["err_cm"]) > 0.3]
    print(f"{len(index)} samples in {sd}; |est - dist| > 0.3 cm: {len(bad)}")
    if args.check_all:
        rows = [check_item(i) for i in items]
        (OUT/"geometry.json").write_text(json.dumps(rows, indent=0))
        err = np.array([abs(r["dist_drawn"] - r["dist"]) for r in rows])
        print(f"all {len(rows)} items, from the drawn pixels back-projected to the table (independent of the drawing "
              f"projection): |drawn distance - dist| mean {err.mean():.3f} cm, max {err.max():.3f} cm; offset vector "
              f"error max {max(r['offset_err'] for r in rows):.3f} cm; cross vs tip max "
              f"{max(r['cross_vs_tip'] for r in rows):.3f} cm; ring centre vs grasp max "
              f"{max(r['ring_vs_grasp'] for r in rows):.3f} cm")
        print(f"ring radius: drawing {min(r['ring_r_px'] for r in rows):.2f}-{max(r['ring_r_px'] for r in rows):.2f} px "
              f"(spread over directions max {max(r['ring_r_spread_px'] for r in rows):.3f} px); fitted to the drawn "
              f"pixels minus drawing {min(r['ring_r_drawn_px'] - r['ring_r_px'] for r in rows):+.2f} to "
              f"{max(r['ring_r_drawn_px'] - r['ring_r_px'] for r in rows):+.2f} px; drawn radius on the table "
              f"{min(r['ring_r_drawn_cm'] for r in rows):.3f}-{max(r['ring_r_drawn_cm'] for r in rows):.3f} cm")
        wrong = [r for r in rows if r["inside_drawn"] != (r["dist"] <= RING_CM)]
        print(f"cross centre inside the drawn ring disagrees with dist <= 1.5 cm on {len(wrong)} items "
              + " ".join(f"{r['id']} ({r['dist']} cm)" for r in wrong))
        clipped = [r for r in rows if r["clipped"]]
        print(f"rings reaching past the crop edge: {len(clipped)} {dict(collections.Counter(r['bin'] for r in clipped))}; "
              f"ring centre outside the crop: {sum(not r['centre_in_crop'] for r in rows)}")
        for r in clipped:
            print(f"  {r['id']} dist {r['dist']} cm ({r['split']}): {r['ring_arc_in_crop']:.0%} of the ring in the crop")
        print(f"wrote {OUT/'geometry.json'}")


# ----------------------------------------------------------------------------- Clef
def read_jsonl(path: Path) -> list[dict]:
    return cr.read_jsonl(path)


def clef_item(clef, item: dict, todo: set) -> list[dict]:
    from branchlab.rtscene import render
    _, img, _ = marked_crop(item)
    uri = render.jpeg_uri(img)
    state = STATE.format(target=item["target"])
    rows = []
    for q in ("dir", "in"):
        if q not in todo:
            continue
        key = h(json.dumps(["claw_marked", q, state, QUESTIONS[q]]) + uri)
        ans = clef.ask(state, QUESTIONS[q], [uri], key=key)["answers"]
        if q == "dir":
            a = ans["step"]
            rows.append({"id": item["id"], "question": q, "choice": a.get("choice"),
                         "p": {o: round(a["probabilities"].get(o, 0.0), 4) for o in OPTIONS}})
        else:
            rows.append({"id": item["id"], "question": q, "p_yes": round(float(ans["inside"]["noul"]), 4)})
    return rows


def cmd_clef(args):
    from branchlab.clef import Clef
    OUT.mkdir(parents=True, exist_ok=True)
    items = tracked_items()
    out = OUT/"clef.jsonl"
    done = {(r["id"], r["question"]) for r in read_jsonl(out)}
    jobs = [(i, {q for q in ("dir", "in") if (i["id"], q) not in done}) for i in items]
    jobs = [j for j in jobs if j[1]]
    print(f"{sum(len(t) for _, t in jobs)} Clef calls on {len(jobs)} items ({len(done)} done)")
    clef = Clef(cache_path=OUT/"clef_responses.jsonl", backend="workers")
    if jobs:
        clef.check_images()                 # once, before the workers start (the client's own probe can race)
    with ThreadPoolExecutor(min(args.workers, 6)) as pool, out.open("a") as f:
        for k, rows in enumerate(pool.map(lambda j: clef_item(clef, *j), jobs), 1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)} items", flush=True)


# ----------------------------------------------------------------------------- Gemini
def gemini_ids() -> list[str]:
    return sorted({r["id"] for r in read_jsonl(RETEST/"gemini_v2.jsonl") if r["interface"] == IFACE})


def cmd_gemini(args):
    """Gemini 3.8 Flash on the marked crop: the retest's client and parsing (default reasoning; a one-word answer)."""
    import base64
    import io
    import run_robodawn_baseline as rb
    from harness.agent.llm_client import ChatClient
    OUT.mkdir(parents=True, exist_ok=True)
    by_id = {i["id"]: i for i in tracked_items()}
    ids = gemini_ids()
    missing = [x for x in ids if x not in by_id]
    if missing:
        print(f"warning: {len(missing)} Gemini ids are not tracked items: {missing[:3]}")
    out = OUT/"gemini.jsonl"
    done = {r["id"] for r in read_jsonl(out) if not r.get("error")}       # a failed call is retried on the next run
    pick = [by_id[x] for x in ids if x in by_id and x not in done]
    cls = rb.vertex_client(ChatClient, KEY, "standard")

    def run(item):
        client = cls(model=rb.MODEL, base_url=rb.api_base(rb.credentials_project(KEY)), max_tokens=8000,
                     temperature=None, timeout_s=300, log_path=OUT/"gemini_calls"/"llm_calls.jsonl")
        buf = io.BytesIO()
        marked_crop(item)[1].save(buf, format="PNG")
        text = STATE.format(target=item["target"]) + GEMINI_Q
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
                                             {"type": "text", "text": text}]}]
        r = client.chat(msgs, tag=item["id"])
        words = [w for w in re.findall(r"[a-z]+", (r.text or "").lower()) if w in OPTIONS]
        return {"id": item["id"], "interface": "marked", "choice": words[-1] if words else None,
                "text": (r.text or "")[-200:], "error": r.error, "latency_s": round(r.latency_s, 2)}
    print(f"{len(pick)} Gemini calls ({len(done)} done)")
    failed = 0
    with ThreadPoolExecutor(min(args.workers, 6)) as pool, out.open("a") as fh:
        for r in pool.map(run, pick):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            failed += bool(r["error"])
    print(f"done; {failed} calls failed (run again to retry them)")


# ----------------------------------------------------------------------------- report
def auc(pairs: list[tuple[float, bool]]) -> float:
    """Mann-Whitney AUC (ties count half) of score against label."""
    pos = np.array([s for s, y in pairs if y], dtype=float)
    neg = np.array([s for s, y in pairs if not y], dtype=float)
    if not len(pos) or not len(neg):
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins) / (len(pos) * len(neg))


def auc_boot(items: list[dict], score, label, reps: int = 1000, seed: int = 0) -> tuple[float, float, float, int, int]:
    """(AUC, 95% interval resampling approaches, positives, negatives) over items with a score and a label."""
    rows = [(i["approach"], s, y) for i in items if (s := score(i)) is not None and (y := label(i)) is not None]
    pairs = [(s, y) for _, s, y in rows]
    a = auc(pairs)
    by = collections.defaultdict(list)
    for ap, s, y in rows:
        by[ap].append((s, y))
    groups = list(by.values())
    rng = random.Random(seed)
    vals = []
    for _ in range(reps if groups else 0):
        v = auc([p for _ in groups for p in rng.choice(groups)])
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    lo, hi = (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (float("nan"),) * 2
    return a, lo, hi, sum(y for _, y in pairs), sum(not y for _, y in pairs)


def near_vs(lo: float, hi: float):
    """Label: True for dist <= 1.5 cm, False for lo <= dist < hi (1.5-3 excludes 1.5 itself), else None."""
    def f(i):
        d = i["dist"]
        if d <= cr.STAY:
            return True
        return False if lo <= d < hi and d > cr.STAY else None
    return f


def fmt_auc(r) -> str:
    a, lo, hi, npos, nneg = r
    return "-" if math.isnan(a) else f"{a:.2f} [{lo:.2f}, {hi:.2f}] ({npos}/{nneg})"


def rate_cell(sub: list[dict], fn, boot_good: bool = False) -> str:
    rate, n = cr.weighted(sub, fn)
    if not n:
        return "-"
    s = " / ".join(f"{o} {rate.get(o, 0):.0%}" for o in OUTCOMES if o in ("good", "bad") or rate.get(o, 0) > 0)
    if boot_good:
        lo, hi = cr.bootstrap(sub, fn, "good")
        s += f"; good [{lo:.0%}, {hi:.0%}]"
    return f"{s} ({n} ap, {sum(fn(i) is not None for i in sub)} it)"


def cmd_report(args):
    items = tracked_items()
    marked = collections.defaultdict(dict)
    for r in read_jsonl(OUT/"clef.jsonl"):
        marked[r["id"]][r["question"]] = r
    unmarked = {r["id"]: r for r in read_jsonl(RETEST/"clef_v2.jsonl") if r["interface"] == IFACE}
    gem_m = {}
    for r in read_jsonl(OUT/"gemini.jsonl"):          # a successful retry replaces a failed call
        if r["id"] not in gem_m or not r.get("error"):
            gem_m[r["id"]] = r
    gem_u = {r["id"]: r for r in read_jsonl(RETEST/"gemini_v2.jsonl") if r["interface"] == IFACE}

    def step_fn(source, rule):
        def f(i):
            a = source(i)
            c = cr.apply_rule(a, rule)
            return cr.score_step(i, c) if c else None
        return f
    src_m = lambda i: marked[i["id"]].get("dir")
    src_u = lambda i: unmarked.get(i["id"])
    p_stay_m = lambda i: (r := src_m(i)) and r["p"]["stay"]
    p_stay_u = lambda i: (r := src_u(i)) and r["p"]["stay"]
    p_in_m = lambda i: (r := marked[i["id"]].get("in")) and r["p_yes"]
    both = [i for i in items if src_m(i) and src_u(i)]          # the same items in both conditions

    L = ["# Claw-machine test with the grasp spot drawn (marked)", "",
         "Pre-registered reading (fixed 2026-10-13 before any answer; Clef, all tracked items, marked condition):", "",
         "- A: AUC of P(stay), <= 1.5 vs 3-8 cm, >= 0.85",
         "- B: 3-8 cm, forced rule: good >= 80% and bad <= 10%",
         "- C: AUC of P(stay), <= 1.5 vs 3-8 cm, <= 0.70",
         "- D: 3-8 cm, forced rule: good not at least 10 pp above the unmarked good on the same items",
         "- reads the mark: A and B, and neither C nor D; does not read the mark: C or D, and not (A and B); "
         "partial: otherwise (including A, B and D together).", "",
         "Reported, not part of the reading: P(yes) of the in question, the p0.5 rule, Gemini, the test half alone, "
         "occlusion strata.", "",
         f"Items {len(items)} (tracked targets, both halves), approaches {len({i['approach'] for i in items})}; by bin "
         f"{dict(collections.Counter(cr.bin_of(i['dist']) for i in items))}. Clef marked answers: dir "
         f"{sum('dir' in marked[i['id']] for i in items)}, in {sum('in' in marked[i['id']] for i in items)}; unmarked "
         f"(retest img_table) {sum(i['id'] in unmarked for i in items)}; both conditions {len(both)}. Rates weight "
         "every approach equally (ap = approaches, it = items); brackets are 95% intervals over approaches."]
    geo_path = OUT/"geometry.json"
    if geo_path.exists():
        geo = [g for g in json.loads(geo_path.read_text()) if g["id"] in {i["id"] for i in items}]
        clipped = [g for g in geo if g["clipped"]]
        L += ["", f"Marks (render --check-all, {len(geo)} items; drawn pixels back-projected to the table): |drawn "
              f"cross-to-ring distance - dist| max {max(abs(g['dist_drawn'] - g['dist']) for g in geo):.2f} cm; drawn "
              f"ring radius {min(g['ring_r_drawn_cm'] for g in geo):.2f}-{max(g['ring_r_drawn_cm'] for g in geo):.2f} cm; "
              f"cross centre inside the drawn ring disagrees with dist <= 1.5 cm on "
              f"{sum(g['inside_drawn'] != (g['dist'] <= RING_CM) for g in geo)} items. Rings reaching past the crop "
              f"edge: {len(clipped)}" + (" (" + "; ".join(f"{g['id']}, {g['dist']} cm, {g['ring_arc_in_crop']:.0%} of "
                                                        f"the ring in the crop" for g in clipped) + ")" if clipped else "")
              + f"; ring centre outside the crop: {sum(not g['centre_in_crop'] for g in geo)}."]
    else:
        L += ["", "Marks: geometry.json missing (run render --check-all)."]

    # --- verdict
    dec = [i for i in both if cr.bin_of(i["dist"]) == "3-8"]
    a_r = auc_boot(both, p_stay_m, near_vs(3, 8))
    mr, n_ap = cr.weighted(dec, step_fn(src_m, "forced"))
    ur, _ = cr.weighted(dec, step_fn(src_u, "forced"))
    L += ["", "## Reading (all items with both conditions)", ""]
    if mr and not math.isnan(a_r[0]):
        A, C = a_r[0] >= 0.85, a_r[0] <= 0.70
        B = mr["good"] >= 0.80 and mr["bad"] <= 0.10
        D = mr["good"] - ur.get("good", 0) < 0.10
        glo, ghi = cr.bootstrap(dec, step_fn(src_m, "forced"), "good")
        blo, bhi = cr.bootstrap(dec, step_fn(src_m, "forced"), "bad")
        verdict = ("reads the mark" if A and B and not (C or D) else
                   "does not read the mark" if (C or D) and not (A and B) else "partial")
        L += [f"- AUC of P(stay), <= 1.5 vs 3-8 cm: marked {fmt_auc(a_r)}, unmarked "
              f"{fmt_auc(auc_boot(both, p_stay_u, near_vs(3, 8)))}",
              f"- 3-8 cm forced: marked good {mr['good']:.1%} [{glo:.0%}, {ghi:.0%}], bad {mr['bad']:.1%} "
              f"[{blo:.0%}, {bhi:.0%}]; unmarked good {ur.get('good', 0):.1%}, bad {ur.get('bad', 0):.1%} "
              f"({n_ap} approaches, {len(dec)} items)",
              f"- A {'holds' if A else 'no'}; B {'holds' if B else 'no'}; C {'holds' if C else 'no'}; D {'holds' if D else 'no'}",
              f"- reading: **{verdict}**"]
    else:
        L.append("- no marked answers yet")

    # --- alignment
    for title, sub in (("all items", both), ("test half", [i for i in both if i["split"] == "test"])):
        L += ["", f"## Alignment AUC, {title}", "", "| score | <= 1.5 vs 3-8 cm | <= 1.5 vs 1.5-3 cm |", "|---|---|---|"]
        for name, sc in (("Clef unmarked P(stay)", p_stay_u), ("Clef marked P(stay) (dir)", p_stay_m),
                         ("Clef marked P(yes) (in)", p_in_m)):
            L.append(f"| {name} | {fmt_auc(auc_boot(sub, sc, near_vs(3, 8)))} | {fmt_auc(auc_boot(sub, sc, near_vs(1.5, 3)))} |")
        if title == "all items":
            L += ["", "Mean P(stay) / P(yes) by bin (unweighted):", "", "| score | " + " | ".join(b for *_, b in cr.BINS) + " |",
                  "|---|" + "---|" * len(cr.BINS)]
            for name, sc in (("unmarked P(stay)", p_stay_u), ("marked P(stay)", p_stay_m), ("marked P(yes)", p_in_m)):
                cells = []
                for *_, b in cr.BINS:
                    v = [s for i in sub if cr.bin_of(i["dist"]) == b and (s := sc(i)) is not None]
                    cells.append(f"{np.mean(v):.2f} ({len(v)})" if v else "-")
                L.append(f"| {name} | " + " | ".join(cells) + " |")

    # --- direction
    for title, sub in (("all items", both), ("test half", [i for i in both if i["split"] == "test"])):
        L += ["", f"## Direction by bin, {title}", "", "| condition | " + " | ".join(b + " cm" for *_, b in cr.BINS) + " |",
              "|---|" + "---|" * len(cr.BINS)]
        for rule in ("forced", "p0.5"):
            for name, src in (("unmarked", src_u), ("marked", src_m)):
                fn = step_fn(src, rule)
                L.append(f"| Clef {name} / {rule} | " + " | ".join(
                    rate_cell([i for i in sub if cr.bin_of(i["dist"]) == b], fn, boot_good=(b == "3-8"))
                    for *_, b in cr.BINS) + " |")

    # --- Gemini
    gsub = [i for i in items if i["id"] in gem_u and i["id"] in gem_m]
    L += ["", "## Gemini 3.8 Flash, marked against unmarked (paired items)", "",
          f"{len(gsub)} items answered in both (of {len(gem_u)} unmarked); no parsable word (scored as stay, as in the "
          f"retest): unmarked {sum(gem_u[i['id']]['choice'] is None for i in gsub)}, marked "
          f"{sum(gem_m[i['id']]['choice'] is None for i in gsub)}; marked errors {sum(bool(gem_m[i['id']].get('error')) for i in gsub)}.",
          "", "| condition | " + " | ".join(b + " cm" for *_, b in cr.BINS) + " |", "|---|" + "---|" * len(cr.BINS)]
    for name, src in (("Gemini unmarked", gem_u), ("Gemini marked", gem_m)):
        fn = lambda i, src=src: (lambda r: cr.score_step(i, r["choice"] or "stay") if r else None)(src.get(i["id"]))
        L.append(f"| {name} | " + " | ".join(rate_cell([i for i in gsub if cr.bin_of(i["dist"]) == b], fn) for *_, b in cr.BINS) + " |")
    for name, src in (("Clef unmarked / forced", src_u), ("Clef marked / forced", src_m)):
        fn = step_fn(src, "forced")
        L.append(f"| {name} (same items) | " + " | ".join(rate_cell([i for i in gsub if cr.bin_of(i["dist"]) == b], fn) for *_, b in cr.BINS) + " |")

    # --- occlusion
    L += ["", "## Occlusion strata (all items)", ""]
    for occ, pred in (("clear (occluded < 0.5)", lambda i: i["occluded"] < 0.5), ("covered (occluded >= 0.5)", lambda i: i["occluded"] >= 0.5)):
        sub = [i for i in both if pred(i)]
        L += [f"### {occ}: {len(sub)} items, {len({i['approach'] for i in sub})} approaches", "",
              f"- AUC P(stay) <= 1.5 vs 3-8: marked {fmt_auc(auc_boot(sub, p_stay_m, near_vs(3, 8)))}, unmarked "
              f"{fmt_auc(auc_boot(sub, p_stay_u, near_vs(3, 8)))}; marked P(yes) {fmt_auc(auc_boot(sub, p_in_m, near_vs(3, 8)))}", "",
              "| condition | " + " | ".join(b + " cm" for *_, b in cr.BINS) + " |", "|---|" + "---|" * len(cr.BINS)]
        for name, src in (("unmarked", src_u), ("marked", src_m)):
            fn = step_fn(src, "forced")
            L.append(f"| Clef {name} / forced | " + " | ".join(rate_cell([i for i in sub if cr.bin_of(i["dist"]) == b], fn) for *_, b in cr.BINS) + " |")
        L.append("")
    text = "\n".join(L)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"report.md").write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    s = sub.add_parser("render")
    s.add_argument("--check-all", action="store_true", help="also check the geometry of every item (no model calls)")
    s = sub.add_parser("clef")
    s.add_argument("--workers", type=int, default=6)
    s = sub.add_parser("gemini")
    s.add_argument("--workers", type=int, default=6)
    sub.add_parser("report")
    args = parser.parse_args()
    {"render": cmd_render, "clef": cmd_clef, "gemini": cmd_gemini, "report": cmd_report}[args.step](args)


if __name__ == "__main__":
    main()
