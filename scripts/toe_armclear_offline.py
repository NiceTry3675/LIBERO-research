#!/usr/bin/env python3
"""Shoe toe with the arm clear (2026-10-13): does Clef read which way a shoe's toe points when the whole shoe is in view,
with crops placed from run-time depth, not from the true pose?

Why. The toe tests so far asked at moments when the arm and gripper covered the shoe (release moment, P0: 14 of 26 toes
hidden, Clef read the visible heel as the toe; carried L1; right after the release R1/R3), and the one arm-clear number
(the design audit's pre-grasp read rotated by the gripper, AUC 0.951) came from crops placed on the true pose
(reports/앞코_판정_시점_재시험.md, reports/실험_설계_감사.md). The claw retest showed a judgment failing only because the
needed information was missing from the image; here the toe judgment is measured where the information is present.

Design (fixed 2026-10-13, before any answer).
1. Items (arm-clear). Turn-start captures of place_shoe / place_dual_shoes in outputs/robodawn_colab/rt8d/replay/
   eval_baseline that are replay-valid (turn <= the replay's divergence turn from g1_candidates/units.json, default 999:
   toe_code.py's rule; outcomes first differ during that turn, so its start state still matches the run), where a shoe
   is upright (its up axis within 30 deg of vertical: up_z >= cos 30), on the table (z >= table - 5 cm), not held (no
   holding arm within 12 cm, and the capture not inside a hold of that shoe), and both fingertips are >= 12 cm (xy) from
   the shoe centre. This is d_clef_share/toe_code.py's rule without its 'within 3 cm of the start height' condition, so
   placed shoes (on the mat or in the box) count. Robot-cover rule (added 2026-10-13 after the first QA sheets, before
   any answer, because the fingertip rule let the forearm lie over the toe of a kept item, place_shoe/9 c7): at most 2%
   of the shoe blob's cell points project onto robot pixels in the top camera and in the agent camera (depth
   back-projected and classified by the robot body model, rtscene.robot_model; run-time), and no robot-only height-map
   cell within 2 cm of the blob. Removed captures are listed in controls.json (cover_excluded). Phase: 'pre' = before that shoe's first grasp (the preclose capture of
   the first proprioceptive hold whose fingertips are within 12 cm of it), 'post' = after the end of a hold of that shoe.
   Event unit: per (episode, shoe), in capture order, keep the first capture and any later capture whose true toe yaw
   differs by >= 20 deg from every kept one; weight = 1 / (kept items of that (episode, shoe)). Kept captures are the
   items; all candidate captures are in items.json (keep flag) and carry the depth controls.
2. Truth (labels and scoring only): bucket() of the true toe direction (clef_semantic_offline.TOE_LOCAL) in world xy =
   the top picture's directions (left = -x, up = +y); margin to the nearest bucket edge. Clear = margin >= 10 deg.
3. Crops from run-time information. Height map from depth with the robot masked (rtscene.heightmap, noise-free), blobs as
   in toe_code.py (g1_candidates/scene.blobs). The shoe's blob is the one containing (else the one nearest, within 3 cm)
   the true shoe xy: the true position stands in for the planner's choice of target object; nothing else of the true
   pose is used. Crop = release_gate.crop_box of the 3-D box centred on the blob centroid, x/y half width
   max(12 cm, 0.65 x the blob's longest extent), z from the table to table + max(12 cm, blob h95 + 2 cm); the raw top
   camera, no overlay, 320 px (TOP); and the agent (oblique) camera's crop of the same box (for TOP+AGENT).
4. Clef (Workers AI, forced4 interface of clef_vantage_offline: options release_gate.TOE_OPTIONS, about 'the shoe in the
   centre', the question names the first, top, picture): TOP (top crop) and TOP+AGENT (top, then agent). One call per
   item per variant; the 320 px PNG crops are sent as JPEG quality 90 (rtscene.render.jpeg_uri, as the release gate and
   the earlier Clef studies sent them). Clef(cache_path=OUT/clef_responses.jsonl, backend='workers'), check_images() once
   before the pool, at most 6 workers.
5. Controls on the same items. (a) toe_code.py's depth rules H (heel = the end whose outer 30% has the higher 90th
   percentile height), W (toe = the narrower end), M (heel = the end nearer the highest cell) on the noise-free height
   map and the blob's PCA axis; rule H again with Gaussian noise added to the blob cells' heights (sigma 0.5 and 1.0 cm,
   3 seeds each, blob cells and axis fixed; mean reported). Scored 4-way (bucket of the predicted toe vector) and by
   sign along the axis (toe_code.py's metric). (b) The image-free majority bucket (in sample; overall and per task).
   (c) Gemini 3.8 Flash (claw_retest_offline.cmd_gemini's client: rb.vertex_client(ChatClient, KEY, 'standard'),
   default reasoning, max_tokens 8000, PNG, one-word answer) on the TOP crop, at most 80 clear-margin items by a fixed
   hash, stratified pre/post (up to 40 each, the rest filled from the other phase).
6. Gate use (secondary, exploratory). For every replay release event in clef_vantage items with vantage P0 (a held
   shoe), the reading capture is the last arm-clear capture (item rule above) of that shoe before the grasp of that
   hold and after the shoe's previous release. Its Clef reading is the answer on the kept item of its event unit (the
   capture itself when kept, else the kept item it duplicates: toe yaw within 20 deg), so no extra calls; depth rule H
   is computed on the reading capture itself. The spec set is the events whose reading is 'pre' (the shoe's first
   hold); later holds with a 'post' reading are reported apart; events with no reading are listed. The toe at release is the reading rotated by
   the gripper's rotation from the grasp to the release: R = R_grip(preopen) R_grip(grasp preclose)^T, from the arm's
   quat_wxyz in the capture state (proprioception; its columns are the approach and finger axes, checked against the
   stored approach / finger_axis vectors). Not the rotation from the reading capture: the gripper does not hold the shoe
   before the grasp, so a push between the reading and the grasp is an error of the method (flagged from the truth).
   The design's literal rotation (reading capture -> release) is kept only as a check (literal_* in gate_events.json).
   Clef: the four direction probabilities of TOP+AGENT (else TOP) are carried as picture/world vectors, rotated and
   re-bucketed; P(not left) = 1 - the mass landing on left (for the AUC and the 0.5 cut); the 4-way pick is the
   reading's argmax direction rotated and re-bucketed (the design's 'rotate the argmax'). Depth: rule H's toe vector
   rotated.
   Truth propagation (the true toe at the reading, rotated) checks the method. Scored against the truth at the release
   capture ('left' = correct-facing) and compared with the existing P0 release-moment forced4 answers on the same events
   (outputs/analysis/clef_vantage/answers_release-views.jsonl): 4-way accuracy, left/not-left accuracy at 0.5, and the
   AUC of P(not left) with wrong-facing as positives. Small n: exploratory.
7. Cost cap: Clef 2 calls per kept item; Gemini at most 80 calls.

Pre-registered reading (fixed 2026-10-13, before any answer). Primary = Clef 4-way accuracy, TOP+AGENT (TOP reported
beside it), on kept arm-clear items with margin >= 10 deg, event-weighted, pre and post pooled: 'reads the toe when
visible' if >= 0.85, 'does not' if <= 0.70, else 'partial'. Also per pre/post and per task, with 95% intervals by
resampling (episode, shoe) units.

    PYTHONPATH=src:scripts python scripts/toe_armclear_offline.py items     # items, crops, depth controls, QA (no calls)
    PYTHONPATH=src:scripts python scripts/toe_armclear_offline.py clef [--workers 6]
    PYTHONPATH=src:scripts python scripts/toe_armclear_offline.py gemini [--workers 6]
    PYTHONPATH=src:scripts python scripts/toe_armclear_offline.py report
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

import clef_vantage_offline as cv                                   # noqa: E402  (imported, never changed)
from clef_semantic_offline import bucket                             # noqa: E402
from branchlab.rtscene.release_gate import TOE_OPTIONS, crop_box, projector, DROP_CM  # noqa: E402

OUT = PROJECT/"outputs/analysis/toe_armclear"
EPISODES = PROJECT/"outputs/analysis/failure_ceiling/episodes.json"
SHARE = PROJECT/"outputs/analysis/next_exp_2026-10-12/next_exp"
G1 = SHARE/"g1_candidates"
TOE_CODE_ROWS = SHARE/"d_clef_share/toe_code_rows.json"
VANTAGE = PROJECT/"outputs/analysis/clef_vantage"
TASKS = ("place_shoe", "place_dual_shoes")

CLEAR_XY = 12.0                 # cm: both fingertips this far (xy) from the shoe centre
HOLD_CM = 12.0                  # a holding arm's fingertips this near the shoe hold it (labels only)
UPRIGHT_DEG = 30.0
ON_TABLE_CM = 5.0               # the shoe at most this far below the table top (else it fell off)
TURN_DEG = 20.0                 # event unit: a later capture counts when the toe yaw turned this far
MARGIN = 10.0
BLOB_REACH_CM = 3.0             # the shoe's blob: contains the true xy cell, else nearest cell within this
COVER_MAX = 0.02                # robot-cover rule (added after the QA sheets, before any answer): at most this share
GAP_MIN = 2.0                   # of the blob's points on robot pixels per camera, and no robot-only cell this near (cm)
HALF_MIN, HALF_K = 12.0, 0.65
RISE_MIN, RISE_PAD = 12.0, 2.0
SIZE = 320
NOISE_CM = (0.5, 1.0)
SEEDS = (0, 1, 2)
GEMINI_MAX = 80
TOE_CM = 9.0                    # QA arrows
VARIANTS = {"TOP": ("top",), "TOP+AGENT": ("top", "agent")}
DIRV = {"left": (-1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0), "up": (0.0, 1.0, 0.0), "down": (0.0, -1.0, 0.0)}
OPTS = tuple(TOE_OPTIONS)


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def request(variant: str) -> tuple[str, dict]:
    """The forced4 interface of clef_vantage_offline about the shoe in the centre (first picture = top)."""
    it = cv.interface(VARIANTS[variant], "centre", "forced4")
    assert it["questions"]["toe"]["criteria"] == TOE_OPTIONS
    return it["state"], it["questions"]


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def quat_R(q) -> np.ndarray:
    from branchlab.rtscene.geometry import quat_to_matrix
    return quat_to_matrix(q)


# ----------------------------------------------------------------------------- depth: blob and rules (toe_code.py)
def shoe_blob(hm, B, p) -> tuple[dict | None, float]:
    """The blob containing the cell under the true shoe xy, else the one with the nearest cell within BLOB_REACH_CM."""
    from branchlab.rtscene.heightmap import CELL, X_RANGE, Y_RANGE
    ix = int(math.floor((p[0] - X_RANGE[0]) / CELL))
    iy = int(math.floor((p[1] - Y_RANGE[0]) / CELL))
    best = (None, math.inf)
    for b in B:
        d = float(np.min(np.hypot(b["ix"].astype(float) - ix, b["iy"].astype(float) - iy))) * CELL
        if d < best[1]:
            best = (b, d)
    return best if best[1] <= BLOB_REACH_CM else (None, best[1])


def blob_axis(hm, b):
    """Cell world xy, PCA long axis and perpendicular projections of a blob (toe_code.py's arithmetic)."""
    X, Y = hm.world_xy()
    xs, ys = X[b["iy"], b["ix"]], Y[b["iy"], b["ix"]]
    P = np.stack([xs - xs.mean(), ys - ys.mean()], 1)
    ev, evec = np.linalg.eigh(np.cov(P.T))
    ax = evec[:, 1]
    return P @ ax, P @ evec[:, 0], ax


def rules(s, t, hcells) -> dict:
    """True = the +axis end is the toe. H: heel = the end whose outer 30% has the higher 90th percentile height;
    W: toe = the narrower end; M: heel = the end nearer the highest cell."""
    lo, hi = np.percentile(s, 30), np.percentile(s, 70)
    endA, endB = s >= hi, s <= lo
    hA, hB = np.percentile(hcells[endA], 90), np.percentile(hcells[endB], 90)
    wA, wB = np.ptp(t[endA]), np.ptp(t[endB])
    smax = s[np.argmax(hcells)]
    return {"H": bool(hA < hB), "W": bool(wA < wB), "M": bool(smax < 0), "hA": float(hA), "hB": float(hB)}


def robot_cover(cap, hm, b, boxes) -> dict:
    """Run-time check that the robot does not cover the shoe in a crop: the share of the blob's cell points (x, y,
    table + cell height) that project onto robot pixels (depth back-projected and classified by the robot body model)
    in the top and the agent camera, and the blob's distance to the nearest robot-only height-map cell."""
    from branchlab.rtscene import robot_model
    from branchlab.rtscene.geometry import backproject, project
    X, Y = hm.world_xy()
    hc = np.nan_to_num(hm.height[b["iy"], b["ix"]], nan=0.0)
    pts = np.stack([X[b["iy"], b["ix"]], Y[b["iy"], b["ix"]], hm.table_z + hc], 1)
    out = {"robot_gap": round(float(b["robot_gap"]), 1)}
    for cam, tag in (("top_camera", "top"), ("agent_camera", "agent")):
        P, valid = backproject(cap.depth(cam), cap.K(cam), cap.E(cam))
        rob = robot_model.mask(P, cap, boxes, 1.0) & valid
        uv, _ = project(pts, cap.K(cam), cap.E(cam))
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, rob.shape[1] - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, rob.shape[0] - 1)
        out[tag] = round(float(rob[v, u].mean()), 4)
    return out


def covered(c: dict) -> bool:
    return c["top"] > COVER_MAX or c["agent"] > COVER_MAX or c["robot_gap"] < GAP_MIN


def score_vec(vec, toe, label) -> dict:
    name, margin = bucket(vec[0], vec[1])
    return {"pred": name, "ok4": name == label}


# ----------------------------------------------------------------------------- episodes, holds, phases
def episode_list():
    div = {(u["task"], u["ep"]): u["div"] for u in json.loads((G1/"units.json").read_text())}
    for e in sorted(json.loads(EPISODES.read_text()), key=lambda e: (e["task"], e["episode"])):
        if e["task"] in TASKS and e["replay"]:
            yield e["task"], e["episode"], PROJECT/e["replay"], div.get((e["task"], e["episode"]), 999)


def hold_table(vep) -> list[dict]:
    """Proprioceptive holds (clef_vantage Episode.holds), each with its grasp (the arm's last preclose capture before
    the hold's first capture), its last holding capture, and the shoe within HOLD_CM of the fingertips (truth label)."""
    by = collections.defaultdict(list)
    for (idx, arm), n in vep.holds.items():
        by[(arm, n)].append(idx)
    out = []
    for (arm, n), idxs in sorted(by.items(), key=lambda kv: min(kv[1])):
        first, last = min(idxs), max(idxs)
        pre = [c["idx"] for c in vep.caps if c["kind"] == "preclose" and c["idx"] < first
               and c["command"].split()[0] == arm]
        grasp = max(pre) if pre else first
        st = vep.caps[vep.pos[first]]["state"]
        name, o, dist = cv.nearest_shoe(st["objects"], np.asarray(st[arm]["position_cm"], float))
        out.append(dict(arm=arm, hold=n, first=first, last=last, grasp=grasp,
                        grasp_from_preclose=bool(pre), shoe=name if dist <= HOLD_CM else None))
    return out


def phase_of(idx: int, shoe: str, holds: list[dict]) -> str:
    mine = [hd for hd in holds if hd["shoe"] == shoe]
    if any(hd["grasp"] <= idx <= hd["last"] for hd in mine):
        return "held"
    if not mine or idx < min(hd["grasp"] for hd in mine):
        return "pre"
    return "post"


# ----------------------------------------------------------------------------- crops
def crop_geometry(vep, idx, cam, centre_xy, half, table_z, top_z):
    params = vep.params(cam, idx)
    proj = projector(params)
    centre = np.array([centre_xy[0], centre_xy[1], table_z + DROP_CM])
    box = crop_box(proj, centre, half, top_z - centre[2])
    return proj, box


def save_crop(vep, idx, cam, box, path: Path):
    img = vep.image(idx, cam)
    img.crop(box).resize((SIZE, SIZE), _bicubic()).save(path)


def _bicubic():
    from PIL import Image
    return Image.BICUBIC


def uv_in_crop(proj, box, pts) -> list:
    k = SIZE / (box[2] - box[0])
    return ((proj(pts) - np.array(box[:2], float)) * k).round(1).tolist()


def seg_in_crop(vep, idx, cam, shoe, box) -> dict | None:
    """QA only (segmentation, never a filter): the share of the shoe's segmentation pixels inside the crop box and
    their smallest distance to the crop's edge, in crop pixels (0 = the shoe touches the edge)."""
    sid = vep.seg_ids.get(shoe)
    if sid is None or f"{cam}_seg" not in vep.npz:
        return None
    seg = vep.npz[f"{cam}_seg"][vep.pos[idx]]
    vs, us = np.nonzero(seg == sid)
    if not len(us):
        return {"share": None, "edge_px": None}
    img = vep.image(idx, cam)
    sc = img.size[0] / seg.shape[1]
    us, vs = (us + 0.5) * sc, (vs + 0.5) * sc
    inside = (us >= box[0]) & (us < box[2]) & (vs >= box[1]) & (vs < box[3])
    edge = np.minimum.reduce([us - box[0], box[2] - us, vs - box[1], box[3] - vs]).min() * SIZE / (box[2] - box[0])
    return {"share": round(float(inside.mean()), 4), "edge_px": round(float(max(edge, 0.0)), 1)}


# ----------------------------------------------------------------------------- items
def cmd_items(args):
    sys.path.insert(0, str(G1))
    from scene import blobs as g1_blobs, BOXES               # toe_code.py's blobs and robot boxes
    from branchlab.rtscene import capture, heightmap
    (OUT/"img").mkdir(parents=True, exist_ok=True)
    for old in (OUT/"img").glob("*.png"):
        old.unlink()
    rows, skips, ctrl, cover_out = [], collections.Counter(), {}, []
    hold_info = {}
    cos_up = math.cos(math.radians(UPRIGHT_DEG))
    for task, n, rdir, div in episode_list():
        vep = cv.Episode(rdir/"meta.json")
        rep = capture.Episode(rdir)
        caps = rep.captures()
        assert all(c.idx == k for k, c in enumerate(caps)) and all(c["idx"] == k for k, c in enumerate(vep.caps))
        holds = hold_table(vep)
        hold_info[vep.key] = holds
        for cap in caps:
            if cap.kind != "turn":
                continue
            c = vep.caps[cap.idx]
            st, tz = c["state"], cap.table_z
            hm = B = None
            for shoe, o in vep.shoes(st).items():
                p = np.asarray(o["position_cm"], float)
                toe = cv.toe_dir(o["quat_wxyz"])
                u = cv.up_z(o["quat_wxyz"])
                arm_xy = min(math.dist(st[a]["position_cm"][:2], p[:2]) for a in ("left", "right"))
                held_now = any(cv.holding(st[a]) and math.dist(st[a]["position_cm"], p) < HOLD_CM
                               for a in ("left", "right"))
                phase = phase_of(cap.idx, shoe, holds)
                why = ("after the divergence turn" if cap.turn > div else
                       "off the table" if p[2] < tz - ON_TABLE_CM else
                       "not upright" if u < cos_up else
                       "held" if held_now or phase == "held" else
                       "arm within 12 cm" if arm_xy < CLEAR_XY else None)
                if why:
                    skips[why] += 1
                    continue
                if hm is None:
                    hm = heightmap.build(cap, BOXES, with_rgb=False)
                    B = g1_blobs(hm)
                b, bd = shoe_blob(hm, B, p)
                if b is None:
                    skips["no blob within 3 cm of the shoe"] += 1
                    continue
                iid = f"{vep.key}|c{cap.idx:03d}|{shoe}"
                label, margin = bucket(toe[0], toe[1])
                # ---- depth rules (run time: height map, blob, axis); truth only for scoring
                s, t, ax = blob_axis(hm, b)
                hc = np.nan_to_num(hm.height[b["iy"], b["ix"]], nan=0.0)
                r = rules(s, t, hc)
                a_is_toe = float(np.dot(toe[:2], ax)) > 0
                dep = {"axis_cos": round(abs(float(np.dot(toe[:2] / (np.linalg.norm(toe[:2]) + 1e-9), ax))), 4),
                       "hA": round(r["hA"], 2), "hB": round(r["hB"], 2)}
                for rule in "HWM":
                    vec = ax if r[rule] else -ax
                    sc = score_vec(vec, toe, label)
                    dep[rule] = {"sign_ok": r[rule] == a_is_toe, "pred": sc["pred"], "ok4": sc["ok4"],
                                 "vec": [round(float(vec[0]), 4), round(float(vec[1]), 4)]}
                noisy = {}
                for sigma in NOISE_CM:
                    sg, ok4 = [], []
                    for seed in SEEDS:
                        rng = np.random.default_rng([seed, int(h(iid)[:8], 16), int(sigma * 10)])
                        rn = rules(s, t, hc + rng.normal(0.0, sigma, size=hc.shape))
                        vec = ax if rn["H"] else -ax
                        sg.append(rn["H"] == a_is_toe)
                        ok4.append(bucket(vec[0], vec[1])[0] == label)
                    noisy[str(sigma)] = {"sign_ok": float(np.mean(sg)), "ok4": float(np.mean(ok4))}
                dep["H_noise"] = noisy
                ctrl[iid] = dep
                cov = robot_cover(cap, hm, b, BOXES)
                if covered(cov):
                    skips["robot over the shoe in a crop (run-time cover rule)"] += 1
                    dep["cover_excluded"] = True
                    cover_out.append(dict(id=iid, task=task, episode=n, turn=cap.turn, shoe=shoe, phase=phase,
                                          arm_xy_cm=round(arm_xy, 1), robot_cover=cov,
                                          vis_frac_scaled=vep.seg_facts(c, shoe).get("vis_frac_scaled")))
                    continue
                # ---- crops (blob centroid, blob extent)
                half = max(HALF_MIN, HALF_K * b["length"])
                top_z = tz + max(RISE_MIN, b["h95"] + RISE_PAD)
                files, uv, pic, in_crop = {}, {}, None, {}
                for cam, tag in (("top_camera", "top"), ("agent_camera", "agent")):
                    proj, box = crop_geometry(vep, cap.idx, cam, (b["cx"], b["cy"]), half, tz, top_z)
                    name = f"{h(iid)[:12]}_{tag}.png"
                    save_crop(vep, cap.idx, cam, box, OUT/"img"/name)
                    files[tag] = name
                    in_crop[tag] = seg_in_crop(vep, cap.idx, cam, shoe, box)
                    uv[tag] = {"centre": uv_in_crop(proj, box, [p])[0], "toe": uv_in_crop(proj, box, [p + TOE_CM * toe])[0],
                               "blob": uv_in_crop(proj, box, [[b["cx"], b["cy"], tz + b["h95"] / 2]])[0],
                               "side_px": box[2] - box[0]}
                    if tag == "top":
                        pd = cv.pic_dir(proj, p, toe)
                        pic = pd[0] if pd else None
                others = [k for k, oo in st["objects"].items() if k not in (shoe, "wall", "table")
                          and math.hypot(oo["position_cm"][0] - b["cx"], oo["position_cm"][1] - b["cy"]) < b["length"] / 2
                          and oo["position_cm"][2] > tz - ON_TABLE_CM]
                rows.append(dict(
                    id=iid, task=task, episode=n, ep=vep.key, unit=f"{vep.key}|{shoe}", idx=cap.idx, turn=cap.turn,
                    div=div, shoe=shoe, phase=phase, label=label, margin=round(margin, 1), clear=margin >= MARGIN,
                    toe_yaw=round(cv.yaw(toe), 1), toe_z=round(float(toe[2]), 3), up_z=round(u, 3),
                    tilt_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, u)))), 1),
                    z_above_table=round(float(p[2] - tz), 1), arm_xy_cm=round(arm_xy, 1),
                    arm_xy_blob_cm=round(min(math.dist(st[a]["position_cm"][:2], (b["cx"], b["cy"]))
                                             for a in ("left", "right")), 1),
                    blob={"cx": round(b["cx"], 2), "cy": round(b["cy"], 2), "length": round(b["length"], 1),
                          "width": round(b["width"], 1), "h95": round(b["h95"], 1), "cells": int(len(b["ix"])),
                          "reach_cm": round(bd, 1), "robot_gap": round(b["robot_gap"], 1),
                          "unknown_edge": round(b["unknown_edge"], 3)},
                    blob_off_cm=round(math.hypot(b["cx"] - p[0], b["cy"] - p[1]), 1), blob_others=others,
                    robot_cover=cov, half_cm=round(half, 1), top_z_cm=round(top_z - tz, 1), images=files, truth_uv=uv,
                    pic=pic, shoe_in_crop=in_crop,
                    **{k: v for k, v in vep.seg_facts(c, shoe).items() if k in ("vis_frac_scaled", "toe_visible",
                                                                                  "heel_visible", "seg_px")}))
    # ---- event units
    units = collections.defaultdict(list)
    for r in rows:
        units[r["unit"]].append(r)
    for g in units.values():
        g.sort(key=lambda r: r["idx"])
        kept = []
        for r in g:
            r["keep"] = all(abs(wrap(r["toe_yaw"] - k["toe_yaw"])) >= TURN_DEG for k in kept)
            if r["keep"]:
                kept.append(r)
                r["event_id"] = r["id"]
            else:                               # the kept item it duplicates (nearest toe yaw): same event
                r["event_id"] = min(kept, key=lambda k: abs(wrap(r["toe_yaw"] - k["toe_yaw"])))["id"]
        for r in g:
            r["weight"] = round(1.0 / len(kept), 4) if r["keep"] else 0.0
            r["unit_kept"] = len(kept)
    rows.sort(key=lambda r: (r["task"], r["episode"], r["idx"], r["shoe"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"items.json").write_text(json.dumps(rows, indent=0))
    # ---- depth controls: agreement with toe_code.py on its own items
    tc = {(r["task"], r["ep"], r["turn"], r["role"]): r for r in json.loads(TOE_CODE_ROWS.read_text()) if r["found"]}
    mine = {(r["task"], r["episode"], r["turn"], r["shoe"]): r for r in rows + cover_out}
    both = sorted(set(tc) & set(mine))
    agree = {rule: sum(ctrl[mine[k]["id"]][rule]["sign_ok"] == tc[k][rule] for k in both) for rule in "HWM"}
    only_tc = sorted(set(tc) - set(mine))
    only_me = sorted(set(mine) - set(tc))
    comparison = {"matched": len(both), "toe_code_rows": len(tc), "agree": agree,
                  "disagree": {rule: [list(k) for k in both if ctrl[mine[k]["id"]][rule]["sign_ok"] != tc[k][rule]]
                               for rule in "HWM"},
                  "toe_code_sign_ok_on_matched": {rule: sum(tc[k][rule] for k in both) for rule in "HWM"},
                  "mine_sign_ok_on_matched": {rule: sum(ctrl[mine[k]["id"]][rule]["sign_ok"] for k in both)
                                              for rule in "HWM"},
                  "only_toe_code": [list(k) for k in only_tc], "only_here": [list(k) for k in only_me]}
    (OUT/"controls.json").write_text(json.dumps({"per_capture": ctrl, "toe_code_comparison": comparison,
                                                 "skips": dict(skips), "cover_excluded": cover_out}, indent=0))
    # ---- gate events (geometry only; Clef readings are joined in the report)
    gate = gate_events(rows, ctrl, hold_info)
    (OUT/"gate_events.json").write_text(json.dumps(gate, indent=0))
    made = qa_sheets(rows, ctrl)
    print_counts(rows, skips, comparison, gate)
    print("  removed by the run-time cover rule:", [(x["id"], x["phase"], x["robot_cover"], x["vis_frac_scaled"])
                                                    for x in cover_out])
    print(f"QA sheets: {made}")


def crop_flags(rows) -> list:
    out = []
    for r in rows:
        for tag, q in (r.get("shoe_in_crop") or {}).items():
            if q and q["share"] is not None and (q["share"] < 1.0 or q["edge_px"] < 5.0):
                out.append((r["id"], tag, q["share"], q["edge_px"]))
    return out


def print_counts(rows, skips, comparison, gate):
    kept = [r for r in rows if r["keep"]]
    print(f"candidate captures {len(rows)}; kept items {len(kept)}; (episode, shoe) units {len({r['unit'] for r in rows})}"
          f"; episodes {len({r['ep'] for r in rows})}")
    print("skipped (shoe-captures at turn starts):", dict(skips))
    for name, sub in (("all captures", rows), ("kept items", kept), ("kept, clear margin", [r for r in kept if r["clear"]])):
        c = collections.Counter((r["task"], r["phase"]) for r in sub)
        print(f"  {name:20s} n={len(sub):3d} units={len({r['unit'] for r in sub}):3d} "
              + " ".join(f"{t}/{ph}={c[(t, ph)]}" for t in TASKS for ph in ("pre", "post")))
    print("  kept labels:", dict(collections.Counter(r["label"] for r in kept)),
          " clear labels:", dict(collections.Counter(r["label"] for r in kept if r["clear"])))
    print("  top-crop picture direction of the truth differs from the world bucket on",
          sum(r["pic"] != r["label"] for r in rows), "captures",
          [(r["id"], r["label"], r["pic"], r["margin"]) for r in rows if r["pic"] != r["label"]])
    print("  blob: length", f"{min(r['blob']['length'] for r in rows):.1f}-{max(r['blob']['length'] for r in rows):.1f} cm;",
          "centroid off the true centre max", f"{max(r['blob_off_cm'] for r in rows):.1f} cm;",
          "other objects within half a blob length of the centroid:",
          [(r["id"], r["blob_others"]) for r in rows if r["blob_others"]])
    print("  shoe in its crops (segmentation, QA only): kept items with < 100% of the shoe's pixels inside, or touching "
          "the edge (< 5 crop px):", crop_flags(kept))
    vis = [r.get("vis_frac_scaled") for r in rows if r.get("vis_frac_scaled") is not None]
    print("  segmentation (analysis only): vis_frac_scaled min", f"{min(vis):.2f}" if vis else "-",
          "toe visible", sum(bool(r.get("toe_visible")) for r in rows), "/", len(rows),
          "heel visible", sum(bool(r.get("heel_visible")) for r in rows), "/", len(rows))
    print("  toe_code.py comparison: matched", comparison["matched"], "of", comparison["toe_code_rows"],
          "rows; agreement", comparison["agree"], "; toe_code sign-ok", comparison["toe_code_sign_ok_on_matched"],
          "here", comparison["mine_sign_ok_on_matched"], "; only toe_code", comparison["only_toe_code"],
          "; only here", comparison["only_here"])
    ev = gate["events"]
    print(f"  gate events: P0 with a shoe {len(ev) + len(gate['no_read'])}; with a reading {len(ev)} "
          f"(pre {sum(e['read_phase'] == 'pre' for e in ev)}, post {sum(e['read_phase'] == 'post' for e in ev)}); "
          f"no reading {len(gate['no_read'])}")


# ----------------------------------------------------------------------------- gate events (section 6)
def gate_events(rows, ctrl, hold_info) -> dict:
    p0 = [i for i in map(json.loads, open(VANTAGE/"items.jsonl")) if i["vantage"] == "P0" and i["obj"]]
    clear_caps = collections.defaultdict(list)
    for r in rows:
        clear_caps[(r["ep"], r["shoe"])].append(r)
    eps = {}
    events, no_read = [], []
    for e in p0:
        if e["ep"] not in eps:
            task, n = e["ep"].split("/")[1], int(e["ep"].split("/")[2])
            eps[e["ep"]] = cv.Episode(PROJECT/"outputs/robodawn_colab/rt8d/replay"/e["ep"].split("/")[0]/task
                                      / f"episode_{n:03d}"/"meta.json")
        vep = eps[e["ep"]]
        holds = hold_info.get(e["ep"]) or hold_table(vep)
        hd = next((x for x in holds if x["arm"] == e["arm"] and x["hold"] == e["hold"]), None)
        if hd is None:
            no_read.append({"event": e["id"], "why": "no proprioceptive hold at the preopen"})
            continue
        prev = [x["last"] for x in holds if x["shoe"] == e["obj"] and x["last"] < hd["grasp"]]
        after = max(prev) if prev else -1
        reads = [r for r in clear_caps[(e["ep"], e["obj"])] if after < r["idx"] < hd["grasp"]]
        if not reads:
            no_read.append({"event": e["id"], "why": "no arm-clear capture of the shoe between its previous release "
                                                    "(or the start) and this grasp", "label": e["label"],
                            "first_hold_of_shoe": not prev})
            continue
        rd = max(reads, key=lambda r: r["idx"])         # the last arm-clear capture before the grasp
        cg, cr_ = vep.caps[vep.pos[hd["grasp"]]], vep.caps[vep.pos[e["idx"]]]
        arm = e["arm"]
        Rg, Rr = quat_R(cg["state"][arm]["quat_wxyz"]), quat_R(cr_["state"][arm]["quat_wxyz"])
        R = Rr @ Rg.T
        # the quaternion's columns against the stored approach / finger axes (proprioception, rounded to 0.01)
        ax_err = max(math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(M[:, k], v) / np.linalg.norm(v))))))
                     for M, s in ((Rg, cg["state"][arm]), (Rr, cr_["state"][arm]))
                     for k, v in ((0, np.asarray(s["approach"], float)), (1, np.asarray(s["finger_axis"], float))))
        cread = vep.caps[vep.pos[rd["idx"]]]
        toe_read = cv.toe_dir(cread["state"]["objects"][e["obj"]]["quat_wxyz"])
        toe_grasp = cv.toe_dir(cg["state"]["objects"][e["obj"]]["quat_wxyz"])
        toe_rel = cv.toe_dir(cr_["state"]["objects"][e["obj"]]["quat_wxyz"])
        prop = R @ toe_read
        hv = np.array(ctrl[rd["id"]]["H"]["vec"] + [0.0])
        hprop = R @ hv
        # the design's literal rotation (reading capture -> release), kept only as a check of the choice above: it
        # also counts the gripper's turns before it holds the shoe
        Rl = Rr @ quat_R(cread["state"][arm]["quat_wxyz"]).T
        prop_l = Rl @ toe_read
        events.append(dict(
            event=e["id"], task=e["task"], ep=e["ep"], arm=arm, obj=e["obj"], hold=e["hold"], release_idx=e["idx"],
            grasp_idx=hd["grasp"], grasp_from_preclose=hd["grasp_from_preclose"], read_capture=rd["id"],
            read_id=rd["event_id"], read_idx=rd["idx"], read_phase=rd["phase"], read_label=rd["label"],
            read_margin=rd["margin"],
            label=e["label"], margin=e["margin"], wrong_facing=e["label"] != "left",
            R=np.round(R, 5).tolist(), grip_yaw_change_deg=round(wrap(math.degrees(math.atan2(R[1, 0], R[0, 0]))), 1),
            axis_check_deg=round(ax_err, 2),
            truth_prop=bucket(prop[0], prop[1])[0],
            truth_prop_err_deg=round(cv._angle(prop, toe_rel), 1),
            pushed_before_grasp_deg=round(cv._angle(toe_read, toe_grasp), 1),
            turned_in_hold_vs_gripper_deg=round(cv._angle(R @ toe_grasp, toe_rel), 1),
            depth_H_prop=bucket(hprop[0], hprop[1])[0],
            literal_grip_yaw_change_deg=round(wrap(math.degrees(math.atan2(Rl[1, 0], Rl[0, 0]))), 1),
            literal_truth_prop=bucket(prop_l[0], prop_l[1])[0],
            literal_truth_prop_err_deg=round(cv._angle(prop_l, toe_rel), 1)))
    return {"events": events, "no_read": no_read}


def rotate_probs(probs: dict, R) -> dict:
    out = {k: 0.0 for k in OPTS}
    for k, v in probs.items():
        w = np.asarray(R, float) @ np.array(DIRV[k])
        out[bucket(w[0], w[1])[0]] += v
    return out


# ----------------------------------------------------------------------------- QA sheets (truth shown; no answers exist)
def qa_sheets(rows, ctrl) -> list[str]:
    from PIL import Image, ImageDraw, ImageFont
    sd = OUT/"samples"
    sd.mkdir(parents=True, exist_ok=True)
    for old in sd.glob("*.png"):
        old.unlink()
    kept = sorted((r for r in rows if r["keep"]), key=lambda r: h(r["id"]))
    post = [r for r in kept if r["phase"] == "post"][:8]
    pre = [r for r in kept if r["phase"] == "pre"][:16 - len(post)]
    picks = [("sample", r) for r in pre + post]
    flagged = [r for r in rows if r["blob_others"] or r["blob_off_cm"] > 2.0 or (r.get("vis_frac_scaled") or 1) < 0.9
               or not r.get("toe_visible", True) or not r.get("heel_visible", True)]
    picks += [("flag", r) for r in flagged if r not in pre + post]
    font = ImageFont.load_default(size=14)
    T = 240
    made = []
    per = 8
    for s in range(math.ceil(len(picks) / per)):
        chunk = picks[s * per:(s + 1) * per]
        cols = 2
        W = cols * (2 * T + 18) + 6
        Hrow = T + 44
        sheet = Image.new("RGB", (W, math.ceil(len(chunk) / cols) * Hrow + 6), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for k, (kind, r) in enumerate(chunk):
            x0, y0 = 6 + (k % cols) * (2 * T + 18), 6 + (k // cols) * Hrow
            for j, tag in enumerate(("top", "agent")):
                img = Image.open(OUT/"img"/r["images"][tag]).convert("RGB").resize((T, T), Image.BICUBIC)
                dd = ImageDraw.Draw(img)
                kk = T / SIZE
                uv = r["truth_uv"][tag]
                (cu, cv_), (tu, tv) = [(a * kk, b * kk) for a, b in (uv["centre"], uv["toe"])]
                dd.line([cu, cv_, tu, tv], fill=(255, 0, 255), width=3)
                dd.ellipse([tu - 6, tv - 6, tu + 6, tv + 6], outline=(255, 0, 255), width=3)
                bu, bv = uv["blob"][0] * kk, uv["blob"][1] * kk
                dd.line([bu - 6, bv, bu + 6, bv], fill=(0, 200, 255), width=2)
                dd.line([bu, bv - 6, bu, bv + 6], fill=(0, 200, 255), width=2)
                sheet.paste(img, (x0 + j * (T + 4), y0 + 40))
            c = ctrl[r["id"]]
            d.text((x0, y0 + 2), f"{kind} {r['ep'].split('/', 1)[1]} c{r['idx']} t{r['turn']} {r['shoe']} {r['phase']} "
                                 f"w={r['weight']}", fill=(0, 0, 0), font=font)
            d.text((x0, y0 + 20), f"truth={r['label']} m={r['margin']} H={c['H']['pred']} arm={r['arm_xy_cm']} "
                                  f"vis={r.get('vis_frac_scaled')} blob={r['blob']['length']} off={r['blob_off_cm']}"
                                  f"{' +' + ','.join(r['blob_others']) if r['blob_others'] else ''}",
                   fill=(0, 120, 0) if r["label"] == "left" else (200, 0, 0), font=font)
        path = sd/f"qa_{s + 1}.png"
        sheet.save(path)
        made.append(str(path.relative_to(PROJECT)))
    return made


# ----------------------------------------------------------------------------- Clef
def items_kept() -> list[dict]:
    return [r for r in json.loads((OUT/"items.json").read_text()) if r["keep"]]


def clef_job(clef, item: dict, variant: str) -> dict:
    from PIL import Image
    from branchlab.rtscene.render import jpeg_uri         # JPEG q90, as the release gate sends its crop
    state, q = request(variant)
    uris = [jpeg_uri(Image.open(OUT/"img"/item["images"][v]).convert("RGB")) for v in VARIANTS[variant]]
    key = h(json.dumps(["toe_armclear", variant, state, q]) + "".join(uris))
    a = clef.ask(state, q, uris, key=key)["answers"]["toe"]
    probs = {o: round(float(a["probabilities"].get(o, 0.0)), 4) for o in OPTS}
    return {"id": item["id"], "variant": variant, "choice": a.get("choice"), "probs": probs,
            "argmax": max(probs, key=probs.get)}


def cmd_clef(args):
    from branchlab.clef import Clef
    items = items_kept()
    out = OUT/"clef.jsonl"
    done = {(r["id"], r["variant"]) for r in read_jsonl(out)}
    jobs = [(i, v) for i in items for v in VARIANTS if (i["id"], v) not in done]
    print(f"{len(jobs)} Clef calls on {len(items)} items ({len(done)} done; cap {2 * len(items)})")
    if not jobs:
        return
    assert len(jobs) + len(done) <= 2 * len(items)
    clef = Clef(cache_path=OUT/"clef_responses.jsonl", backend="workers")
    clef.check_images()                         # once, before the workers start
    with ThreadPoolExecutor(min(args.workers, 6)) as pool, out.open("a") as f:
        for k, r in enumerate(pool.map(lambda j: clef_job(clef, *j), jobs), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)


# ----------------------------------------------------------------------------- Gemini
GEMINI_Q = ("\n\nWhich way does the toe of the shoe in the centre point, in the picture? Answer with exactly one word: "
            + ", ".join(f"{k} ({v})" for k, v in TOE_OPTIONS.items()) + ".")


def gemini_pick(items: list[dict]) -> list[dict]:
    clear = sorted((i for i in items if i["clear"]), key=lambda i: h("gemini|" + i["id"]))
    pre = [i for i in clear if i["phase"] == "pre"]
    post = [i for i in clear if i["phase"] == "post"]
    half = GEMINI_MAX // 2
    a, b = pre[:half], post[:half]
    a += pre[half:half + max(0, half - len(b))]
    b += post[half:half + max(0, half - len(a))]
    return (a + b)[:GEMINI_MAX]


def cmd_gemini(args):
    """Gemini 3.8 Flash on the TOP crop: claw_retest_offline's client (standard tier, default reasoning)."""
    import base64
    import claw_retest_offline as cr
    import run_robodawn_baseline as rb
    from harness.agent.llm_client import ChatClient
    pick = gemini_pick(items_kept())
    out = OUT/"gemini.jsonl"
    done = {r["id"] for r in read_jsonl(out) if not r.get("error")}
    todo = [i for i in pick if i["id"] not in done]
    print(f"{len(todo)} Gemini calls ({len(done)} done; {len(pick)} picked, cap {GEMINI_MAX})")
    assert len(pick) <= GEMINI_MAX
    if not todo:
        return
    KEY = cr.KEY                                # the Vertex service account key (path only)
    cls = rb.vertex_client(ChatClient, KEY, "standard")
    state, _ = request("TOP")

    def run(item):
        client = cls(model=rb.MODEL, base_url=rb.api_base(rb.credentials_project(KEY)), max_tokens=8000,
                     temperature=None, timeout_s=300, log_path=OUT/"gemini_calls"/"llm_calls.jsonl")
        data = (OUT/"img"/item["images"]["top"]).read_bytes()
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}},
            {"type": "text", "text": state + GEMINI_Q}]}]
        r = client.chat(msgs, tag=item["id"])
        words = [w for w in re.findall(r"[a-z]+", (r.text or "").lower()) if w in TOE_OPTIONS]
        return {"id": item["id"], "choice": words[-1] if words else None, "text": (r.text or "")[-200:],
                "error": r.error, "latency_s": round(r.latency_s, 2)}
    failed = 0
    with ThreadPoolExecutor(min(args.workers, 6)) as pool, out.open("a") as fh:
        for r in pool.map(run, todo):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            failed += bool(r["error"])
    print(f"done; {failed} calls failed (run again to retry them)")


# ----------------------------------------------------------------------------- report
def wacc(rows, fn) -> tuple[float, float, int]:
    num = den = 0.0
    n = 0
    for i in rows:
        v = fn(i)
        if v is None:
            continue
        num += i["weight"] * float(v)
        den += i["weight"]
        n += 1
    return (num / den if den else float("nan")), den, n


def boot(rows, fn, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    units = collections.defaultdict(list)
    for i in rows:
        units[i["unit"]].append(i)
    keys = sorted(units)
    if not keys:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        a, d, _ = wacc([i for _ in keys for i in units[rng.choice(keys)]], fn)
        if d:
            vals.append(a)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def cell(rows, fn) -> str:
    a, d, n = wacc(rows, fn)
    if not n:
        return "-"
    lo, hi = boot(rows, fn)
    return f"{a:.3f} [{lo:.2f}, {hi:.2f}] (n={n}, units={len({i['unit'] for i in rows if fn(i) is not None})})"


def auc(pairs) -> float:
    pos = np.array([s for s, y in pairs if y], float)
    neg = np.array([s for s, y in pairs if not y], float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def cmd_report(args):
    rows = json.loads((OUT/"items.json").read_text())
    kept = [r for r in rows if r["keep"]]
    ctrl = json.loads((OUT/"controls.json").read_text())
    dep, comp = ctrl["per_capture"], ctrl["toe_code_comparison"]
    gate = json.loads((OUT/"gate_events.json").read_text())
    ans = collections.defaultdict(dict)
    for r in read_jsonl(OUT/"clef.jsonl"):
        ans[r["id"]][r["variant"]] = r
    gem = {}
    for r in read_jsonl(OUT/"gemini.jsonl"):
        if r["id"] not in gem or not r.get("error"):
            gem[r["id"]] = r
    clear = [r for r in kept if r["clear"]]
    clef_ok = lambda v: (lambda i: (ans[i["id"]][v]["argmax"] == i["label"]) if v in ans[i["id"]] else None)
    dep_ok = lambda rule: (lambda i: dep[i["id"]][rule]["ok4"])
    noise_ok = lambda s: (lambda i: dep[i["id"]]["H_noise"][s]["ok4"])
    maj_all = collections.Counter()
    for r in kept:
        maj_all[r["label"]] += r["weight"]
    maj = max(maj_all, key=maj_all.get)
    maj_task = {}
    for t in TASKS:
        c = collections.Counter()
        for r in kept:
            if r["task"] == t:
                c[r["label"]] += r["weight"]
        maj_task[t] = max(c, key=c.get) if c else None
    conds = [("Clef TOP+AGENT", clef_ok("TOP+AGENT")), ("Clef TOP", clef_ok("TOP")),
             ("depth H (noise-free)", dep_ok("H")), ("depth W", dep_ok("W")), ("depth M", dep_ok("M")),
             ("depth H, noise 0.5 cm (mean of 3)", noise_ok("0.5")), ("depth H, noise 1.0 cm (mean of 3)", noise_ok("1.0")),
             (f"majority bucket ({maj}, in sample)", lambda i: i["label"] == maj),
             ("per-task majority (in sample)", lambda i: i["label"] == maj_task[i["task"]])]

    L = ["# Shoe toe with the arm clear (toe_armclear)", "",
         "Pre-registered reading (fixed 2026-10-13, before any answer): primary = Clef 4-way accuracy, TOP+AGENT (TOP "
         "beside it), on kept arm-clear items with margin >= 10 deg, event-weighted (1 / kept items of the (episode, "
         "shoe)), pre and post pooled. **reads the toe when visible** if >= 0.85, **does not** if <= 0.70, else "
         "**partial**. Also per pre/post and per task, with 95% intervals by resampling (episode, shoe) units.", "",
         "Crops are placed from run-time information only: the shoe's depth blob (robot-masked noise-free height map) "
         "gives the centre and size. Which blob is the shoe was chosen with the true shoe position, standing in for "
         "the planner's choice of target object; nothing else of the true pose places a crop.", "",
         "Item rule beyond the fixed design: a run-time robot-cover rule (at most 2% of the shoe blob's points on robot "
         "pixels in the top and the agent camera, no robot-only height-map cell within 2 cm of the blob) was added after "
         "the first QA sheets and before any answer, because the fingertip rule let the forearm lie over the toe of one "
         f"kept item. Removed captures: {[(x['id'], x['phase'], x['robot_cover']) for x in ctrl.get('cover_excluded', [])]}.",
         "",
         "Other notes on the design as run:",
         f"- Replay validity is toe_code.py's rule, turn <= the divergence turn (the start of the divergence turn comes "
         f"before its first differing outcome); kept items at the divergence turn itself: "
         f"{sum(r['turn'] == r['div'] for r in kept)}.",
         f"- One more item condition, the shoe on the table (centre >= table - {ON_TABLE_CM:g} cm): it removed "
         f"{ctrl['skips'].get('off the table', 0)} shoe-capture(s).",
         f"- Crop QA (segmentation, analysis only, never a filter): kept crops with less than all of the shoe's pixels "
         f"inside or the shoe within 5 crop px of the edge: {crop_flags(kept) or 'none'}.",
         "- Clef gets the 320 px crops as JPEG quality 90 (the release gate's transport); Gemini gets the PNG.",
         ""]
    n_clef = sum(len(v) for v in ans.values())
    L += ["## Items", "",
          f"- candidate captures {len(rows)}, kept items {len(kept)} ({len({r['unit'] for r in kept})} (episode, shoe) "
          f"units, {len({r['ep'] for r in kept})} episodes); clear margin {len(clear)} items.",
          "- kept by task/phase: " + ", ".join(f"{t} {ph} {sum(r['task'] == t and r['phase'] == ph for r in kept)}"
                                              f" (clear {sum(r['task'] == t and r['phase'] == ph for r in clear)})"
                                              for t in TASKS for ph in ("pre", "post")),
          f"- kept truth buckets {dict(collections.Counter(r['label'] for r in kept))}; skipped shoe-captures "
          f"{ctrl['skips']}",
          f"- Clef answers {n_clef} (cap {2 * len(kept)}); Gemini answers {len(gem)} "
          f"(errors {sum(bool(r.get('error')) for r in gem.values())}).", ""]
    # ---- verdict
    a_ta, _, n_ta = wacc(clear, clef_ok("TOP+AGENT"))
    a_t, _, n_t = wacc(clear, clef_ok("TOP"))
    L += ["## Reading", ""]
    if n_ta:
        verdict = "reads the toe when visible" if a_ta >= 0.85 else "does not" if a_ta <= 0.70 else "partial"
        L += [f"- primary, Clef TOP+AGENT, clear margin, pooled: {cell(clear, clef_ok('TOP+AGENT'))}",
              f"- Clef TOP, same items: {cell(clear, clef_ok('TOP'))}"
              + (f" (by the same thresholds: {'reads' if a_t >= 0.85 else 'does not' if a_t <= 0.70 else 'partial'})"
                 if n_t else ""),
              f"- reading: **{verdict}**", ""]
    else:
        L += ["- no Clef answers yet", ""]
    # ---- table: all conditions by stratum
    strata = [("clear, pooled", clear), ("clear, pre", [r for r in clear if r["phase"] == "pre"]),
              ("clear, post", [r for r in clear if r["phase"] == "post"])]
    strata += [(f"clear, {t}", [r for r in clear if r["task"] == t]) for t in TASKS]
    strata += [("all margins, pooled", kept), ("all margins, pre", [r for r in kept if r["phase"] == "pre"]),
               ("all margins, post", [r for r in kept if r["phase"] == "post"])]
    L += ["## 4-way accuracy (event-weighted; [95% interval over (episode, shoe) units])", "",
          "| condition | " + " | ".join(s for s, _ in strata) + " |", "|---|" + "---|" * len(strata)]
    for name, fn in conds:
        L.append(f"| {name} | " + " | ".join(cell(sub, fn) for _, sub in strata) + " |")
    L += ["", "Depth rules by sign along the blob's axis (toe_code.py's metric), clear items: " + ", ".join(
        f"{rule} {wacc(clear, lambda i, rule=rule: dep[i['id']][rule]['sign_ok'])[0]:.3f}" for rule in "HWM")
        + "; H with noise " + ", ".join(f"{s} cm {wacc(clear, lambda i, s=s: dep[i['id']]['H_noise'][s]['sign_ok'])[0]:.3f}"
                                         for s in ("0.5", "1.0")), ""]
    for v in VARIANTS:
        sub = [r for r in clear if v in ans[r["id"]]]
        if sub:
            conf = collections.Counter((r["label"], ans[r["id"]][v]["argmax"]) for r in sub)
            flips = sum(n for (a, b), n in conf.items() if {a, b} in ({"left", "right"}, {"up", "down"}))
            L.append(f"- Clef {v} confusion (truth, answer), clear items: {dict(sorted(conf.items()))}; 180-degree "
                     f"flips {flips}/{len(sub)}")
    # ---- Gemini
    gsub = [r for r in clear if r["id"] in gem]
    if gsub:
        gok = lambda i: (gem[i["id"]]["choice"] == i["label"]) if i["id"] in gem else None
        L += ["", "## Gemini 3.8 Flash on the TOP crop (its items only)", "",
              f"- {len(gsub)} items (pre {sum(r['phase'] == 'pre' for r in gsub)}, post "
              f"{sum(r['phase'] == 'post' for r in gsub)}); no parsable word {sum(gem[r['id']]['choice'] is None for r in gsub)}",
              f"- Gemini {cell(gsub, gok)}; Clef TOP on the same items {cell(gsub, clef_ok('TOP'))}; Clef TOP+AGENT "
              f"{cell(gsub, clef_ok('TOP+AGENT'))}; depth H {cell(gsub, dep_ok('H'))}"]
    # ---- toe_code comparison
    L += ["", "## Depth control check against toe_code.py", "",
          f"- matched {comp['matched']} of {comp['toe_code_rows']} toe_code rows by (task, episode, turn, shoe); sign "
          f"agreement H {comp['agree']['H']}, W {comp['agree']['W']}, M {comp['agree']['M']} of {comp['matched']}; "
          f"toe_code sign-ok {comp['toe_code_sign_ok_on_matched']}, here {comp['mine_sign_ok_on_matched']}",
          f"- disagreements {comp['disagree']}; only in toe_code {comp['only_toe_code']}; only here {comp['only_here']}"]
    # ---- gate use
    p0 = {r["id"]: r for r in read_jsonl(VANTAGE/"answers_release-views.jsonl")
          if r["vantage"] == "P0" and r["interface"] == "forced4"}
    allev = gate["events"]
    L += ["", "## Gate use (secondary, exploratory): read with the arm clear, carry by the gripper's rotation", "",
          "- Reading: the last arm-clear capture of the shoe between its previous release (or the start) and this "
          "grasp. Its Clef reading is the answer on the kept item of its event unit (the capture itself when kept, "
          "else the earlier kept item whose toe yaw is within 20 deg), so no extra calls; depth H is computed on the "
          "reading capture itself.",
          "- Rotation (proprioception only): R = R_grip(release preopen) R_grip(grasp preclose)^T, from the arm's "
          "quat_wxyz in the capture states, checked against the stored approach / finger_axis vectors. The design's "
          "literal rotation (reading capture -> release) also counts the gripper's turns before it holds the shoe; "
          "carrying the true toe with it gives a median error of "
          + (f"{np.median([e['literal_truth_prop_err_deg'] for e in allev]):.1f} deg and the wrong bucket on "
             f"{sum(e['literal_truth_prop'] != e['label'] for e in allev)}/{len(allev)} events, against "
             f"{np.median([e['truth_prop_err_deg'] for e in allev]):.1f} deg and "
             f"{sum(e['truth_prop'] != e['label'] for e in allev)}/{len(allev)} for the grasp -> release rotation used "
             "here." if allev else "- (no events)."),
          "- The design's event set is every P0 event whose shoe has a 'pre' item earlier in the episode. Events of a "
          "later hold are carried from the reading after the shoe's previous release ('post'), not across a release "
          "and re-grasp; they are reported apart, and the events with no reading are listed at the end.", ""]
    for title, sel in (("spec set: first hold of the shoe, 'pre' reading", "pre"),
                       ("later holds, 'post' reading after the previous release", "post")):
        ev = [e for e in gate["events"] if e["read_phase"] == sel]
        if not ev:
            L += [f"### {title}: no events", ""]
            continue

        def clef_rot(e):
            """(the four probabilities carried by R, the carried argmax) of the reading's Clef answer
            (TOP+AGENT, else TOP), or None."""
            a = ans[e["read_id"]]
            v = "TOP+AGENT" if "TOP+AGENT" in a else "TOP" if "TOP" in a else None
            if v is None:
                return None
            w = np.asarray(e["R"], float) @ np.array(DIRV[a[v]["argmax"]])
            return rotate_probs(a[v]["probs"], e["R"]), bucket(w[0], w[1])[0]
        lines = []
        for name, pnot, pick in (
                ("truth propagated (oracle reading)", lambda e: float(e["truth_prop"] != "left"), lambda e: e["truth_prop"]),
                ("Clef read + gripper rotation", lambda e: (1 - c[0]["left"]) if (c := clef_rot(e)) else None,
                 lambda e: c[1] if (c := clef_rot(e)) else None),
                ("depth H read + gripper rotation", lambda e: float(e["depth_H_prop"] != "left"), lambda e: e["depth_H_prop"]),
                ("P0 release-moment Clef (existing)", lambda e: p0[e["event"]]["p_not_left"] if e["event"] in p0 else None,
                 lambda e: (lambda pr: max(pr, key=pr.get))(p0[e["event"]]["answer"]["probabilities"])
                 if e["event"] in p0 else None)):
            sub = [e for e in ev if pnot(e) is not None]
            if not sub:
                lines.append(f"| {name} | - | - | - | - |")
                continue
            acc4 = np.mean([pick(e) == e["label"] for e in sub])
            acc2 = np.mean([(pnot(e) >= 0.5) == e["wrong_facing"] for e in sub])
            a = auc([(pnot(e), e["wrong_facing"]) for e in sub])
            lines.append(f"| {name} | {len(sub)} | {acc4:.3f} | {acc2:.3f} | {a:.3f} |")
        L += [f"### {title}: {len(ev)} events ({sum(e['wrong_facing'] for e in ev)} wrong-facing at release, "
              f"{len({e['ep'] for e in ev})} episodes)", "",
              "| reading | n | 4-way acc | left/not-left acc (0.5) | AUC P(not left) |", "|---|---|---|---|---|", *lines, "",
              "| event | truth at release (margin) | read capture (phase) | grip yaw change | Clef carried argmax, "
              "P(not left) | depth H carried | truth carried | P0 P(not left) |", "|---|---|---|---|---|---|---|---|",
              *[f"| {e['event'].split('|', 1)[1].replace('|', ' ')} | {e['label']} ({e['margin']}) | c{e['read_idx']} ({e['read_phase']}) | "
                f"{e['grip_yaw_change_deg']} | "
                + ((lambda c: f"{c[1]}, {1 - c[0]['left']:.2f}")(clef_rot(e)) if clef_rot(e) else "-")
                + f" | {e['depth_H_prop']} | {e['truth_prop']} | "
                + (f"{p0[e['event']]['p_not_left']:.2f}" if e["event"] in p0 else "-") + " |" for e in ev], "",
              f"- truth-propagation error median {np.median([e['truth_prop_err_deg'] for e in ev]):.1f} deg; pushed "
              f">= 20 deg between the reading and the grasp: {sum(e['pushed_before_grasp_deg'] >= 20 for e in ev)}; "
              f"axis check (quaternion columns vs stored approach / finger axes) max "
              f"{max(e['axis_check_deg'] for e in ev):.2f} deg", ""]
    L += [f"- events without a reading: {len(gate['no_read'])}: "
          + "; ".join(f"{x['event'].split('|', 1)[1].replace('|', ' ')} (truth {x.get('label')}, "
                      f"{'first hold' if x.get('first_hold_of_shoe') else 'later hold'}: {x['why']})"
                      for x in gate["no_read"]), ""]
    text = "\n".join(L)
    (OUT/"report.md").write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("items")
    for name in ("clef", "gemini"):
        s = sub.add_parser(name)
        s.add_argument("--workers", type=int, default=6)
    sub.add_parser("report")
    args = parser.parse_args()
    {"items": cmd_items, "clef": cmd_clef, "gemini": cmd_gemini, "report": cmd_report}[args.step](args)


if __name__ == "__main__":
    main()
