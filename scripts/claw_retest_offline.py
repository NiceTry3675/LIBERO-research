#!/usr/bin/env python3
"""Claw-machine alignment retest (2026-10-11): can Clef, from RGB alone, pick the step that brings a gripper over the
spot it is about to grasp?

The first tests of this idea (scripts/clef_servo_offline.py, clef_servo_topview.py, 2026-10-07/08) went through
OpenRouter, which never passed Clef the images, so they say nothing about Clef's vision. This retest uses Clef on
Workers AI and only what the RoboDawn baseline gives the robot at run time: camera images, camera calibration and the
fingertip positions (proprioception). No depth; the depth parser appears only as a privileged reference.

Items, from the replayed episodes of the 8 new tasks (outputs/robodawn_colab/rt8d/replay):
  grasp      a close that caught something in the replay AND in the original run ("fingers stopped" in both), after
             which the arm held it (moved 3 cm with the fingers still apart, or kept them apart to the end); the
             grasp point is the fingertip position at that close. Other closes end a segment but give no items.
  target     the object that moved with the arm after the close; else the nearest object within 12 cm that ever
             moves in the episode (never a fixture such as a pad or box); else the task's untracked object (bread,
             bottle) by name; else the close gives no items.
  items      every turn-start or pre-close capture of that arm since its previous grasp, gripper open, fingertips
             within 15 cm (x/y) of the grasp point; a pre-close capture at the same fingertip position as a turn-start
             capture is dropped. The approach often pushes the target before the close, so for a tracked target the
             grasp point is carried rigidly to where the object stands in the item's capture (simulator poses, labels
             only); untracked targets (bread, dustbin bottles) cannot be carried and form the unverifiable stratum.
             Items also record whether the target moved at all (stationary: at most 1 cm and 10 degrees).
Clef sees a straight-down crop of the raw top camera (+-16 cm) about the fingertips, a red cross on them, and the
target's name, and picks one step: left, right, up (away from the robot), down (toward it) or stay. A 2 cm step that
brings the fingertips over 0.5 cm closer (x/y) to the grasp point is good, over 0.5 cm farther bad, else neutral;
stay is right within 1.5 cm (stay_ok) and a stall beyond.

  interfaces  img_table  the cross on the table point under the fingertips
              img_tip    the cross at the fingertips' own height (less parallax against a raised grasp point)
              blind      img_tip's text with a blank grey image (what the name and Clef's priors give)
  stay rules  choice     Clef's own pick
              p0.5       stay only when P(stay) >= 0.5, else the likeliest step
              forced     never stay
References: Gemini's actual next x/y move at that turn (the baseline's behaviour), Gemini on the same crop (a subset),
the depth parser's blob nearest the grasp point (privileged: depth and the true grasp point), always-stay.
Rates are weighted so that every approach (episode, arm, grasp) counts once per distance bin; confidence intervals
resample approaches.

Decision rule. The first version (fixed before any result) was scored on the development half with a labelling bug
and on wrong grasp points, as an adversarial review found; this version was fixed on 2026-10-11 after that review and
the development half, before any Clef answer on the test half existed. The interface and stay rule are the
development half's best (highest weighted good at 3-8 cm with bad < 5%). On the test half, tracked targets at 3-8 cm
(the first version of this rule used stationary targets only; there were 12 such test items from 6 approaches, too few,
so it was widened to all tracked targets with the carried grasp point before any test answer existed):
  go    good >= 60%, bad < 5%, good >= Gemini's actual good - 5 pp, and good - blind good (same stay rule) >= 15 pp
  stop  otherwise
Reported, not part of the rule: stay_ok within 1.5 cm (a closed loop could stop on oscillation instead), the moved and
unverifiable strata, and occlusion (the share of a 2 cm disc about the grasp point covered by the robot in the top
camera, from segmentation; labels only).

    PYTHONPATH=src:scripts python scripts/claw_retest_offline.py items
    PYTHONPATH=src:scripts python scripts/claw_retest_offline.py clef --split dev|test [--interfaces img_table img_tip blind]
    PYTHONPATH=src:scripts python scripts/claw_retest_offline.py depth
    PYTHONPATH=src:scripts python scripts/claw_retest_offline.py choose        # the development half's best policy
    PYTHONPATH=src:scripts python scripts/claw_retest_offline.py gemini [--n 80]
    PYTHONPATH=src:scripts python scripts/claw_retest_offline.py report
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
REPLAY = PROJECT/"outputs/robodawn_colab/rt8d/replay"
COLAB = PROJECT/"outputs/robodawn_colab"
OUT = PROJECT/"outputs/analysis/claw_retest"
KEY = PROJECT/"gemini0614-499409-42e14e861064.json"          # the Vertex service account key (path only)
for p in (PROJECT/"src", PROJECT/"scripts", PROJECT/"third_party/robodawn"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

REACH = 15.0               # cm: items within this x/y distance of the grasp point
STEP, MARGIN, STAY = 2.0, 0.5, 1.5
HALF = 16.0                # cm: the crop's half width about the fingertips
SIZE = 320
HELD_MOVE, HELD_OPEN = 3.0, 0.02
CARRY_XY = 3.0
STILL_CM, STILL_DEG = 1.0, 10.0
BINS = ((0, 1.5, "<1.5"), (1.5, 3, "1.5-3"), (3, 8, "3-8"), (8, 15.01, "8-15"))
DIRS = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}
INTERFACES = ("img_table", "img_tip", "blind")
RULES = ("choice", "p0.5", "forced")
TASK_TARGET = {"place_bread_basket": "the bread", "put_bottles_dustbin": "the bottle", "pick_diverse_bottles": "the bottle"}
STATE = ("A robot gripper is about to grasp something on a table. The picture looks straight down at the table; the "
         "robot stands at the bottom edge of the picture. The red cross marks {where}. The planner wants the "
         "fingertips directly above the spot to grasp on {target}.")
WHERE = {"img_table": "the point on the table directly below the gripper's fingertips",
         "img_tip": "the gripper's fingertips", "blind": "the gripper's fingertips"}
OPTIONS = {"left": "one step toward the left edge of the picture",
           "right": "one step toward the right edge of the picture",
           "up": "one step toward the top edge of the picture (away from the robot)",
           "down": "one step toward the bottom edge of the picture (toward the robot)",
           "stay": "stay: the red cross is already on the spot to grasp (within about 1.5 cm)"}
QUESTION = {"step": {"type": "choice", "criteria": OPTIONS,
                     "instructions": "Which single step brings the red cross closer to the spot the planner wants to "
                                     "grasp? Steps are about 2 cm."}}
OUTCOMES = ("good", "bad", "neutral", "stay_ok", "stall")


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def split_of(key: str) -> str:
    return "dev" if int(h(key)[:2], 16) % 2 == 0 else "test"


def bin_of(d: float) -> str:
    return next(b for lo, hi, b in BINS if lo <= d < hi)


def trace_for(task: str, episode: int) -> dict:
    for tier in ("standard", "flex"):
        found = sorted(COLAB.glob(f"*/robodawn/gemini_flash_{tier}/{task}/shard_*/episode_{episode:03d}/trace.json"))
        if found:
            return {r["turn"]: r for r in json.loads(found[0].read_text()) if "turn" in r}
    return {}


def name_of(model: str) -> str:
    return "the " + (re.sub(r"^\d+_", "", str(model)).replace("-", " ").replace("_", " ") or "object")


def quat_matrix(q) -> np.ndarray:
    from branchlab.rtscene.geometry import quat_to_matrix
    return quat_to_matrix(q)


def quat_angle(q1, q2) -> float:
    d = abs(sum(a * b for a, b in zip(q1, q2)))
    return math.degrees(2 * math.acos(min(1.0, d)))


def gripper_value(cmd: str):
    p = cmd.split()
    if len(p) != 3 or p[1] != "gripper":
        return None
    return 0.0 if p[2] in ("close", "closed") else 1.0 if p[2] == "open" else 0.5 if p[2] == "half" else float(p[2])


# ----------------------------------------------------------------------------- items
def original_caught(trace: dict, turn: int, cmd_index: int, arm: str) -> bool:
    res = (trace.get(turn) or {}).get("results") or []
    if cmd_index >= len(res):
        return False
    r = res[cmd_index]
    return str(r.get("command", "")).startswith(f"{arm} gripper") and str(r.get("note", "")).startswith("fingers stopped")


def original_empty(trace: dict, turn: int, cmd_index: int, arm: str) -> bool:
    res = (trace.get(turn) or {}).get("results") or []
    if cmd_index >= len(res):
        return False
    r = res[cmd_index]
    return str(r.get("command", "")).startswith(f"{arm} gripper") and str(r.get("note", "")).startswith("fingers closed fully")


def hold_window(log: list, k: int, arm: str) -> tuple[bool, list]:
    """(held, the log entries after close k until this arm's next gripper command)."""
    g = log[k]["arms"][arm]["position_cm"]
    window, held = [], False
    for c in log[k + 1:]:
        if str(c.get("command", "")).startswith(f"{arm} gripper"):
            break
        window.append(c)
        st = c["arms"][arm]
        if st.get("gripper_real", 0.0) <= HELD_OPEN:
            return False, window
        if math.dist(st["position_cm"], g) >= HELD_MOVE:
            held = True
    else:
        held = held or bool(window) and window[-1]["arms"][arm].get("gripper_real", 0.0) > HELD_OPEN
    return held, window


def carried_role(log: list, k: int, arm: str, window: list) -> str | None:
    """The tracked object that moved with the arm while it held (its x/y displacement follows the arm's, and it rises
    if the arm rises); when several do (the other arm can lift another object at the same time), the one nearest the
    grasp point at the close."""
    if not window:
        return None
    a0 = np.array(log[k]["arms"][arm]["position_cm"])
    at_close = log[k].get("objects") or {}
    matches = set()
    for c in window:
        a = np.array(c["arms"][arm]["position_cm"]) - a0
        if np.linalg.norm(a) < HELD_MOVE:
            continue
        for role, o in (c.get("objects") or {}).items():
            o0 = at_close.get(role)
            if role in ("wall", "table") or not o0 or "position_cm" not in o:
                continue
            d = np.array(o["position_cm"]) - np.array(o0["position_cm"])
            if float(np.linalg.norm(d[:2] - a[:2])) < CARRY_XY and (a[2] < 2 or d[2] > 0.5 * a[2]):
                matches.add(role)
    if not matches:
        return None
    return min(matches, key=lambda r: math.dist(at_close[r]["position_cm"][:2], a0[:2]))


def build_items() -> tuple[list[dict], list[dict]]:
    items, closes = [], []
    for meta_path in sorted(REPLAY.glob("eval_baseline/*/episode_*/meta.json")):
        meta = json.loads(meta_path.read_text())
        task, ep, key = meta["task"], meta["episode"], meta["key"]
        trace = trace_for(task, ep)
        caps = meta["captures"]
        log = meta["command_log"]
        models = {r: o.get("model") for r, o in caps[0]["state"].get("objects", {}).items()}
        span = collections.defaultdict(float)          # how far each tracked object ever moves in the episode
        for cap in caps:
            for r, o in (cap["state"].get("objects") or {}).items():
                o0 = caps[0]["state"]["objects"].get(r)
                if o0 and "position_cm" in o:
                    span[r] = max(span[r], math.dist(o["position_cm"], o0["position_cm"]))
        for arm in ("left", "right"):
            start_turn = 0
            for k, c in enumerate(log):
                value = gripper_value(str(c.get("command") or ""))
                if not str(c.get("command", "")).startswith(arm) or value is None or value >= 0.5 or not c.get("ok"):
                    continue
                if not str(c.get("note", "")).startswith("fingers stopped"):
                    continue
                held, window = hold_window(log, k, arm)
                orig = original_caught(trace, c["turn"], c["cmd_index"], arm)
                g = c["arms"][arm]["position_cm"]
                role = carried_role(log, k, arm, window) if held else None
                if role is None and held:
                    near = sorted((math.dist(o["position_cm"][:2], g[:2]), r) for r, o in (c.get("objects") or {}).items()
                                  if r not in ("wall", "table") and "position_cm" in o and span.get(r, 0) > 0.5)
                    role = near[0][1] if near and near[0][0] < 12 else None
                target = name_of(models.get(role)) if role else TASK_TARGET.get(task)
                closes.append({"key": key, "arm": arm, "turn": c["turn"], "held": held, "original_caught": orig,
                               "role": role, "target": target})
                if not (held and orig and target):
                    start_turn = c["turn"]
                    continue
                pre = (log[k - 1].get("objects") or {}).get(role) if role and k > 0 else None
                seen = set()
                for cap in sorted(caps, key=lambda x: (x["kind"] != "turn", x["idx"])):
                    if cap["kind"] not in ("turn", "preclose") or not (start_turn < cap["turn"] <= c["turn"]):
                        continue
                    if cap["kind"] == "preclose" and ((cap["turn"], cap["cmd_index"]) > (c["turn"], c["cmd_index"])
                                                      or not str(cap.get("command") or "").startswith(arm)):
                        continue
                    st = cap["state"][arm]
                    if st.get("gripper", 1.0) < 0.5:
                        continue
                    tip = st["position_cm"]
                    pos = tuple(round(v, 1) for v in tip)
                    if pos in seen:
                        continue
                    seen.add(pos)
                    stratum, moved, rot, gi = "unverifiable", None, None, list(g)
                    if role and pre:
                        now = cap["state"]["objects"].get(role)
                        if now:
                            moved = math.dist(now["position_cm"], pre["position_cm"])
                            rot = quat_angle(now["quat_wxyz"], pre["quat_wxyz"])
                            stratum = "stationary" if moved <= STILL_CM and rot <= STILL_DEG else "moved"
                            # the grasp point on the object as it stands in this capture (rigid; labels only)
                            gi = list(np.array(now["position_cm"]) + quat_matrix(now["quat_wxyz"]) @ quat_matrix(pre["quat_wxyz"]).T
                                      @ (np.array(g) - np.array(pre["position_cm"])))
                    dist = math.hypot(gi[0] - tip[0], gi[1] - tip[1])
                    if dist > REACH:
                        continue
                    rec = trace.get(cap["turn"]) or {}
                    items.append({"id": f"{key}/{arm}/c{cap['idx']}", "key": key, "task": task, "episode": ep,
                                  "split": split_of(key), "arm": arm, "cap": cap["idx"], "kind": cap["kind"],
                                  "turn": cap["turn"], "tip": tip, "grasp": [round(v, 2) for v in gi], "grasp_close": g,
                                  "dist": round(dist, 2),
                                  "approach": f"{key}/{arm}/t{c['turn']}", "role": role, "target": target,
                                  "stratum": stratum, "moved_cm": moved and round(moved, 2), "rot_deg": rot and round(rot, 1),
                                  "gemini_next": gemini_next(rec, arm) if cap["kind"] == "turn" else "close"})
                start_turn = c["turn"]
    return items, closes


def nearest_middle(ep, cap, tip, radius: float = 12.0):
    """(name, middle x/y) of the object whose visible top-camera middle is nearest the fingertips (segmentation and
    depth: labels only), or None."""
    from branchlab.rtscene import labels
    from branchlab.rtscene.geometry import backproject
    seg = labels.segmentation(ep, "top_camera")[cap.idx]
    names = labels.entity_names(ep)
    robot = set(labels.robot_ids(ep))
    P, valid = backproject(cap.depth("top_camera"), cap.K("top_camera"), cap.E("top_camera"))
    up = valid & (P[..., 2] > cap.table_z + 0.3)
    best = None
    for sid in np.unique(seg[up]):
        sid = int(sid)
        if sid in robot or names.get(sid, "") in labels.SCENE or names.get(sid, "").endswith(("wall", "table", "ground")):
            continue
        m = up & (seg == sid)
        if m.sum() < 15:
            continue
        mid = P[m][:, :2].mean(0)
        d = math.hypot(mid[0] - tip[0], mid[1] - tip[1])
        if d <= radius and (best is None or d < best[0]):
            best = (d, names.get(sid, "object"), mid)
    return (name_of(best[1]), [round(float(best[2][0]), 2), round(float(best[2][1]), 2), tip[2]]) if best else None


def build_empty_items() -> list[dict]:
    """The last test (fixed 2026-10-11 before any answer). At the pre-close capture of an empty close (empty in the
    replay and in the original run), does Clef say stay (agreeing with the bad close) or step toward the target?
      empty_grasp  the same arm later, in the same segment, made a held grasp of a tracked object (the build_items
                   grasp): the target is that grasp point, carried to the object's pose at the capture
      empty_mid    otherwise: the middle of the object nearest the fingertips (segmentation and depth, labels only)
      held         controls: the pre-close capture of every held grasp that was also caught in the original run
                   (the fingertips are on the spot; stay is right)
    Decision, on the empty items whose fingertips are over 1.5 cm (x/y) off their target (misses an x/y step could
    fix; the rest missed in height or the object slipped, reported apart), approach-weighted:
      go  AUC of P(stay), held vs those empty items, >= 0.75 for img_table or img_tip; and for that interface, with
          the step forced, good >= 60% and bad <= 15% on those empty items"""
    from branchlab.rtscene.capture import Episode
    items = []
    for meta_path in sorted(REPLAY.glob("eval_baseline/*/episode_*/meta.json")):
        meta = json.loads(meta_path.read_text())
        task, ep_i, key = meta["task"], meta["episode"], meta["key"]
        ep = Episode(meta_path.parent)
        cap_obj = {c.idx: c for c in ep.captures()}
        trace = trace_for(task, ep_i)
        caps = meta["captures"]
        log = meta["command_log"]
        models = {r: o.get("model") for r, o in caps[0]["state"].get("objects", {}).items()}
        pre_caps = {(c["turn"], c["cmd_index"]): c for c in caps if c["kind"] == "preclose"}

        def add(kind, close, arm, target, gi, approach):
            cap = pre_caps.get((close["turn"], close["cmd_index"]))
            if cap is None or cap["state"][arm].get("gripper", 1.0) < 0.5:
                return
            tip = cap["state"][arm]["position_cm"]
            if gi is None:
                nm = nearest_middle(ep, cap_obj[cap["idx"]], tip)
                if nm is None:
                    return
                target, gi = nm[0].replace("the ", "the middle of the ", 1), nm[1]
            dist = math.hypot(gi[0] - tip[0], gi[1] - tip[1])
            if dist > REACH:
                return
            nxt = trace.get(close["turn"] + 1) or {}
            items.append({"id": f"{key}/{arm}/c{cap['idx']}/{kind}", "key": key, "task": task, "episode": ep_i,
                          "split": "all", "arm": arm, "cap": cap["idx"], "kind": kind, "turn": cap["turn"], "tip": tip,
                          "grasp": [round(float(v), 2) for v in gi], "dist": round(dist, 2), "approach": approach,
                          "target": target, "stratum": "tracked", "occluded": None,
                          "gemini_next": gemini_next(nxt, arm) if kind != "held" else "close"})

        for arm in ("left", "right"):
            start_turn, empties = 0, []
            for k, c in enumerate(log):
                value = gripper_value(str(c.get("command") or ""))
                if not str(c.get("command", "")).startswith(arm) or value is None or value >= 0.5 or not c.get("ok"):
                    continue
                note = str(c.get("note", ""))
                if note.startswith("fingers closed fully"):
                    if original_empty(trace, c["turn"], c["cmd_index"], arm):
                        empties.append(c)
                    continue
                if not note.startswith("fingers stopped"):
                    continue
                held, window = hold_window(log, k, arm)
                orig = original_caught(trace, c["turn"], c["cmd_index"], arm)
                role = carried_role(log, k, arm, window) if held else None
                pre = (log[k - 1].get("objects") or {}).get(role) if role and k > 0 else None
                approach = f"{key}/{arm}/t{c['turn']}"
                if held and orig:
                    add("held", c, arm, name_of(models.get(role)) if role else TASK_TARGET.get(task, "the object"),
                        c["arms"][arm]["position_cm"], approach)
                if held and orig and role and pre:
                    g = np.array(c["arms"][arm]["position_cm"])
                    for e in empties:
                        cap = pre_caps.get((e["turn"], e["cmd_index"]))
                        now = cap and cap["state"]["objects"].get(role)
                        if now:
                            gi = list(np.array(now["position_cm"]) + quat_matrix(now["quat_wxyz"]) @ quat_matrix(pre["quat_wxyz"]).T
                                      @ (g - np.array(pre["position_cm"])))
                            add("empty_grasp", e, arm, name_of(models.get(role)), gi, approach)
                else:
                    for e in empties:
                        add("empty_mid", e, arm, None, None, f"{key}/{arm}/e{e['turn']}")
                start_turn, empties = c["turn"], []
            for e in empties:                       # empty closes after the arm's last grasp
                add("empty_mid", e, arm, None, None, f"{key}/{arm}/e{e['turn']}")
    return items


def gemini_next(rec: dict, arm: str):
    """Gemini's actual next x/y move of this arm at that turn (net, until its first gripper command), or 'close'."""
    dx = dy = 0.0
    for cmd in rec.get("commands") or []:
        p = cmd.lower().split()
        if len(p) < 3 or p[0] != arm:
            continue
        if p[1] == "gripper":
            break
        if p[1] == "move" and p[2] in ("x", "y") and len(p) > 3:
            dx, dy = (dx + float(p[3]), dy) if p[2] == "x" else (dx, dy + float(p[3]))
    return [round(dx, 2), round(dy, 2)]


# ----------------------------------------------------------------------------- scoring
def score_step(item: dict, choice: str) -> str:
    gx, gy = item["grasp"][0] - item["tip"][0], item["grasp"][1] - item["tip"][1]
    dist = math.hypot(gx, gy)
    if choice == "stay":
        return "stay_ok" if dist <= STAY else "stall"
    ux, uy = DIRS[choice]
    new = math.hypot(gx - STEP * ux, gy - STEP * uy)
    return "good" if new < dist - MARGIN else "bad" if new > dist + MARGIN else "neutral"


def score_move(item: dict, move) -> str:
    """Gemini's own move, whatever its size."""
    gx, gy = item["grasp"][0] - item["tip"][0], item["grasp"][1] - item["tip"][1]
    dist = math.hypot(gx, gy)
    if move == "close" or math.hypot(*move) < 0.5:
        return "stay_ok" if dist <= STAY else "stall"
    new = math.hypot(gx - move[0], gy - move[1])
    return "good" if new < dist - MARGIN else "bad" if new > dist + MARGIN else "neutral"


def apply_rule(answer: dict | None, rule: str):
    if not answer:
        return None
    p = answer["p"]
    best = max(DIRS, key=lambda k: p.get(k, 0.0))
    if rule == "choice":
        return answer["choice"]
    if rule == "p0.5":
        return "stay" if p.get("stay", 0.0) >= 0.5 else best
    return best


def weighted(items: list[dict], outcome_of) -> tuple[dict, float]:
    """Outcome shares with every approach weighted equally; and the number of approaches."""
    by = collections.defaultdict(list)
    for i in items:
        o = outcome_of(i)
        if o is not None:
            by[i["approach"]].append(o)
    tot = collections.Counter()
    for outs in by.values():
        for o in outs:
            tot[o] += 1 / len(outs)
    n = sum(tot.values())
    return ({o: tot[o] / n for o in OUTCOMES} if n else {}), len(by)


def bootstrap(items: list[dict], outcome_of, key: str, reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of a weighted rate, resampling approaches (a repeated approach counts as a new one)."""
    by = collections.defaultdict(list)
    for i in items:
        by[i["approach"]].append(i)
    groups = list(by.values())
    if not groups:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        sample = []
        for n in range(len(groups)):
            sample += [{**i, "approach": f"{i['approach']}#{n}"} for i in rng.choice(groups)]
        rate, _ = weighted(sample, outcome_of)
        vals.append(rate.get(key, 0.0))
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


# ----------------------------------------------------------------------------- images
_ep_cache = {}


def episode(path: Path):
    from branchlab.rtscene.capture import Episode
    k = str(path)
    if k not in _ep_cache:
        _ep_cache[k] = Episode(path)
    return _ep_cache[k]


def ep_dir(item: dict) -> Path:
    return REPLAY/"eval_baseline"/item["task"]/f"episode_{item['episode']:03d}"


def capture_of(item: dict):
    return {c.idx: c for c in episode(ep_dir(item)).captures()}[item["cap"]]


def crop_for(item: dict, interface: str):
    """The raw top camera about the fingertips, with a red cross on the table point under them (img_table) or on the
    fingertips themselves (img_tip)."""
    from PIL import Image, ImageDraw
    from branchlab.rtscene import render
    from branchlab.rtscene.geometry import project
    cap = capture_of(item)
    cam = "top_camera"
    img = Image.fromarray(cap.rgb(cam))
    x, y, z = item["tip"]
    u0, v0, u1, v1 = render.object_box(cap, cam, (x, y), HALF, cap.table_z, cap.table_z + 2)
    side = max(u1 - u0, v1 - v0, 24)
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    box = (int(cu - side / 2), int(cv - side / 2), int(cu + side / 2), int(cv + side / 2))
    K = cap.K(cam).copy()
    K[:2] *= 2
    mark_z = cap.table_z if interface == "img_table" else z
    (pu, pv), = project(np.array([[x, y, mark_z]]), K, cap.E(cam))[0]
    out = img.crop(box).resize((SIZE, SIZE), Image.LANCZOS)
    s = SIZE / (box[2] - box[0])
    px, py = (pu - box[0]) * s, (pv - box[1]) * s
    d = ImageDraw.Draw(out)
    r = SIZE // 22
    d.line([px - r, py, px + r, py], fill=(255, 0, 0), width=3)
    d.line([px, py - r, px, py + r], fill=(255, 0, 0), width=3)
    return out


def blank():
    from PIL import Image
    return Image.new("RGB", (SIZE, SIZE), (128, 128, 128))


# ----------------------------------------------------------------------------- Clef
_clef = None


def ask_clef(item: dict, interface: str) -> dict:
    global _clef
    from branchlab.rtscene import render
    if _clef is None:
        from branchlab.clef import Clef
        _clef = Clef(cache_path=OUT/"clef_responses.jsonl", backend="workers")
    img = blank() if interface == "blind" else crop_for(item, interface)
    uri = render.jpeg_uri(img)
    state = STATE.format(where=WHERE[interface], target=item["target"])
    k = hashlib.sha1(json.dumps([state, QUESTION]).encode() + uri.encode()).hexdigest()
    ans = _clef.ask(state, QUESTION, [uri], key=k)["answers"]["step"]
    return {"id": item["id"], "interface": interface, "choice": ans.get("choice"),
            "p": {o: round(ans["probabilities"].get(o, 0.0), 4) for o in OPTIONS}}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def load_answers() -> dict:
    out = collections.defaultdict(dict)
    for r in read_jsonl(OUT/"clef_v2.jsonl"):
        out[r["id"]][r["interface"]] = r
    return out


# ----------------------------------------------------------------------------- steps
def occlusion(item: dict) -> float:
    """Share of a 2 cm disc about the grasp point (at its height) that the robot covers in the top camera (labels)."""
    from branchlab.rtscene import labels
    from branchlab.rtscene.geometry import project
    ep = episode(ep_dir(item))
    cap = capture_of(item)
    seg = labels.segmentation(ep, "top_camera")[cap.idx]
    robot = set(labels.robot_ids(ep))
    K = cap.K("top_camera").copy()          # segmentation is stored at the halved resolution of the depth
    gx, gy, gz = item["grasp"]
    pts = np.array([[gx + 2 * math.cos(a) * r, gy + 2 * math.sin(a) * r, gz] for r in (0, 0.5, 1) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)])
    uv, _ = project(pts, K, cap.E("top_camera"))
    hh, ww = seg.shape[:2]
    hits = [seg[int(v), int(u)] in robot for u, v in uv if 0 <= int(u) < ww and 0 <= int(v) < hh]
    return round(sum(hits) / len(hits), 2) if hits else 1.0


def cmd_items(args):
    items, closes = build_items()
    for i in items:
        i["occluded"] = occlusion(i)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"items_v2.json").write_text(json.dumps(items))
    (OUT/"closes_v2.json").write_text(json.dumps(closes))
    c = collections.Counter((x["held"], x["original_caught"], bool(x["target"])) for x in closes)
    print(f"caught closes {len(closes)}: (held, caught in the original, target) {dict(c)}")
    print(f"{len(items)} items, approaches {len({i['approach'] for i in items})}; strata {dict(collections.Counter(i['stratum'] for i in items))}")
    for s in ("dev", "test"):
        sub = [i for i in items if i["split"] == s and i["stratum"] == "stationary"]
        print(f"  {s} stationary by bin: {dict(collections.Counter(bin_of(i['dist']) for i in sub))}, approaches in 3-8: "
              f"{len({i['approach'] for i in sub if bin_of(i['dist']) == '3-8'})}")


def cmd_clef(args):
    items = [i for i in json.loads((OUT/"items_v2.json").read_text()) if args.split in ("all", i["split"])]
    out = OUT/"clef_v2.jsonl"
    done = {(r["id"], r["interface"]) for r in read_jsonl(out)}
    jobs = [(i, f) for i in items for f in args.interfaces if (i["id"], f) not in done]
    print(f"{len(jobs)} Clef calls ({len(done)} done)")
    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as f:
        for k, r in enumerate(pool.map(lambda j: ask_clef(*j), jobs), 1):
            f.write(json.dumps(r) + "\n")
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)


def cmd_depth(args):
    """Privileged reference: the step toward the depth blob nearest the true grasp point (depth + true identity)."""
    from branchlab.rtscene import monitor as mon, robot_model
    boxes = robot_model.load()
    items = json.loads((OUT/"items_v2.json").read_text())
    out = {}
    for i in items:
        bl = [b for b in mon.parse(capture_of(i), boxes) if b.h95 >= 1.0]
        if not bl:
            out[i["id"]] = "stay"
            continue
        b = min(bl, key=lambda b: math.dist(b.centroid, i["grasp"][:2]))
        gx, gy = b.centroid[0] - i["tip"][0], b.centroid[1] - i["tip"][1]
        out[i["id"]] = "stay" if math.hypot(gx, gy) <= STAY else max(DIRS, key=lambda k: gx * DIRS[k][0] + gy * DIRS[k][1])
    (OUT/"depth_v2.json").write_text(json.dumps(out))
    print(f"{len(out)} depth references")


def policies():
    return [(f, r) for f in ("img_table", "img_tip") for r in RULES]


def cmd_choose(args):
    items = [i for i in json.loads((OUT/"items_v2.json").read_text())
             if i["split"] == "dev" and i["stratum"] != "unverifiable" and bin_of(i["dist"]) == "3-8"]
    ans = load_answers()
    rows = []
    for f, r in policies():
        rate, n = weighted(items, lambda i: score_step(i, c) if (c := apply_rule(ans[i["id"]].get(f), r)) else None)
        rows.append((f, r, rate.get("good", 0), rate.get("bad", 0), rate.get("stall", 0), n))
        print(f"{f:10s} {r:7s} good {rate.get('good', 0):.0%} bad {rate.get('bad', 0):.0%} stall {rate.get('stall', 0):.0%} (approaches {n})")
    ok = [x for x in rows if x[3] < 0.05] or rows
    best = max(ok, key=lambda x: (x[2], -RULES.index(x[1])))
    (OUT/"chosen.json").write_text(json.dumps({"interface": best[0], "rule": best[1]}))
    print(f"chosen on dev: {best[0]} / {best[1]}")


def cmd_gemini(args):
    """Gemini 3.8 Flash on the chosen interface's crop and text (default reasoning; a one-word answer)."""
    import base64
    import io
    import run_robodawn_baseline as rb
    from harness.agent.llm_client import ChatClient
    chosen = json.loads((OUT/"chosen.json").read_text())
    f = chosen["interface"]
    items = [i for i in json.loads((OUT/"items_v2.json").read_text()) if i["split"] == "test" and i["stratum"] != "unverifiable"]
    pick = []
    for lo, hi, b in BINS:
        sub = sorted((i for i in items if lo <= i["dist"] < hi), key=lambda i: h(i["id"]))
        per, seen = [], collections.Counter()
        for i in sub:                       # at most two items per approach
            if seen[i["approach"]] < 2:
                per.append(i)
                seen[i["approach"]] += 1
        pick += per[:args.n // len(BINS)]
    out = OUT/"gemini_v2.jsonl"
    done = {(r["id"], r["interface"]) for r in read_jsonl(out)}
    pick = [i for i in pick if (i["id"], f) not in done]
    cls = rb.vertex_client(ChatClient, KEY, "standard")

    def run(item):
        client = cls(model=rb.MODEL, base_url=rb.api_base(rb.credentials_project(KEY)), max_tokens=8000,
                     temperature=None, timeout_s=300, log_path=OUT/"gemini_calls"/"llm_calls.jsonl")
        buf = io.BytesIO()
        crop_for(item, f).save(buf, format="PNG")
        text = (STATE.format(where=WHERE[f], target=item["target"]) + "\n\nWhich single step (about 2 cm) brings the "
                "red cross closer to the spot to grasp? Answer with exactly one word: "
                + ", ".join(f"{k} ({v})" for k, v in OPTIONS.items()) + ".")
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
                                             {"type": "text", "text": text}]}]
        r = client.chat(msgs, tag=item["id"])
        words = [w for w in re.findall(r"[a-z]+", (r.text or "").lower()) if w in OPTIONS]
        return {"id": item["id"], "interface": f, "choice": words[-1] if words else None, "text": (r.text or "")[-200:],
                "error": r.error, "latency_s": round(r.latency_s, 2)}
    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as fh:
        for r in pool.map(run, pick):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print(f"{len(pick)} Gemini calls on {f}")


def cmd_report(args):
    items = json.loads((OUT/"items_v2.json").read_text())
    ans = load_answers()
    depth = json.loads((OUT/"depth_v2.json").read_text()) if (OUT/"depth_v2.json").exists() else {}
    chosen = json.loads((OUT/"chosen.json").read_text()) if (OUT/"chosen.json").exists() else None
    gem = {}
    for r in read_jsonl(OUT/"gemini_v2.jsonl"):
        gem[(r["id"], r["interface"])] = r
    L = ["# Claw-machine retest (v2)", "",
         f"items {len(items)}, approaches {len({i['approach'] for i in items})}; strata "
         f"{dict(collections.Counter(i['stratum'] for i in items))}. Rates weight every approach equally; "
         "brackets are 95% intervals over approaches.", ""]
    if chosen:
        L.append(f"Chosen on the development half: interface {chosen['interface']}, stay rule {chosen['rule']}.")
    conds = [(f"Clef {f} / {r}", lambda i, f=f, r=r: (lambda c: score_step(i, c) if c else None)(apply_rule(ans[i["id"]].get(f), r)))
             for f in INTERFACES for r in RULES]
    conds += [("Gemini actual next move", lambda i: score_move(i, i["gemini_next"])),
              ("depth blob (privileged)", lambda i: score_step(i, depth[i["id"]]) if i["id"] in depth else None),
              ("always stay", lambda i: score_step(i, "stay"))]
    if chosen:
        f = chosen["interface"]
        conds.append((f"Gemini on the same crop ({f})", lambda i: (lambda r: score_step(i, r["choice"] or "stay") if r else None)(gem.get((i["id"], f)))))

    def cell(sub, fn, b):
        rate, n = weighted(sub, fn)
        if not n:
            return "-"
        if b == "<1.5":
            return f"stay_ok {rate.get('stay_ok', 0):.0%} ({n})"
        return f"good {rate.get('good', 0):.0%} / bad {rate.get('bad', 0):.0%} / stall {rate.get('stall', 0):.0%} ({n})"
    for split in ("dev", "test"):
        for stratum, pred in (("tracked (decision stratum)", lambda i: i["stratum"] != "unverifiable"),
                              ("stationary only", lambda i: i["stratum"] == "stationary"),
                              ("unverifiable", lambda i: i["stratum"] == "unverifiable")):
            sub = [i for i in items if i["split"] == split and pred(i)]
            if not sub:
                continue
            L += ["", f"## {split} half, {stratum} targets", "", "| condition | " + " | ".join(b + " cm" for _, _, b in BINS) + " |",
                  "|---|" + "---|" * len(BINS)]
            for name, fn in conds:
                L.append(f"| {name} | " + " | ".join(cell([i for i in sub if bin_of(i["dist"]) == b], fn, b) for _, _, b in BINS) + " |")
    if chosen:
        f, r = chosen["interface"], chosen["rule"]
        dec = [i for i in items if i["split"] == "test" and i["stratum"] != "unverifiable" and bin_of(i["dist"]) == "3-8"]
        clef_fn = lambda i: (lambda c: score_step(i, c) if c else None)(apply_rule(ans[i["id"]].get(f), r))
        blind_fn = lambda i: (lambda c: score_step(i, c) if c else None)(apply_rule(ans[i["id"]].get("blind"), r))
        gem_fn = lambda i: score_move(i, i["gemini_next"])
        cr, n = weighted(dec, clef_fn)
        br, _ = weighted(dec, blind_fn)
        gr, _ = weighted(dec, gem_fn)
        if cr:
            lo, hi = bootstrap(dec, clef_fn, "good")
            blo, bhi = bootstrap(dec, clef_fn, "bad")
            checks = [("good >= 60%", cr["good"] >= 0.60), ("bad < 5%", cr["bad"] < 0.05),
                      ("good >= Gemini actual - 5 pp", cr["good"] >= gr.get("good", 0) - 0.05),
                      ("good - blind >= 15 pp", cr["good"] - br.get("good", 0) >= 0.15)]
            L += ["", "## Decision (test half, tracked targets, 3-8 cm)", "",
                  f"Clef {f}/{r}: good {cr['good']:.1%} [{lo:.0%}, {hi:.0%}], bad {cr['bad']:.1%} [{blo:.0%}, {bhi:.0%}], "
                  f"stall {cr['stall']:.1%}; Gemini actual good {gr.get('good', 0):.1%}, blind good {br.get('good', 0):.1%}; "
                  f"{n} approaches, {len(dec)} items"]
            L += [f"- {name}: {'pass' if ok else 'FAIL'}" for name, ok in checks]
            L.append(f"- verdict: {'GO' if all(ok for _, ok in checks) else 'STOP'}")
            for occ_name, pred in (("clear (<50% covered)", lambda i: i["occluded"] < 0.5), ("covered (>=50%)", lambda i: i["occluded"] >= 0.5)):
                s2 = [i for i in dec if pred(i)]
                rr, nn = weighted(s2, clef_fn)
                if rr:
                    L.append(f"- {occ_name}: good {rr['good']:.0%}, bad {rr['bad']:.0%} ({nn} approaches)")
    text = "\n".join(L)
    (OUT/"report_v2.md").write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("items")
    s = sub.add_parser("clef")
    s.add_argument("--split", choices=["dev", "test", "all"], required=True)
    s.add_argument("--interfaces", nargs="+", default=list(INTERFACES))
    s.add_argument("--workers", type=int, default=8)
    sub.add_parser("depth")
    sub.add_parser("choose")
    s = sub.add_parser("gemini")
    s.add_argument("--n", type=int, default=80)
    s.add_argument("--workers", type=int, default=6)
    sub.add_parser("report")
    args = parser.parse_args()
    {"items": cmd_items, "clef": cmd_clef, "depth": cmd_depth, "choose": cmd_choose, "gemini": cmd_gemini,
     "report": cmd_report}[args.step](args)


if __name__ == "__main__":
    main()
