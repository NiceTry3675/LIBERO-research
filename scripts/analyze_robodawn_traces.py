#!/usr/bin/env python3
"""Offline analysis of the RoboDawn reproduction traces (Gemini 3.8 Flash, RoboTwin 2.0).

Reads outputs/robodawn_colab/{robotwin,robotwin2}/robodawn/gemini_flash_flex/<task>/shard_<k>/
  results.json, transport.jsonl, episode_<nnn>/trace.json
and prints tables for:
  A. turn anatomy (commands, kinds, move sizes, time split, turns per outcome)
  B. phases (heuristic per-turn labels, time per phase, where failed episodes end)
  C. judgment persistence (segments between gripper events, same-phase runs, plan-text similarity)
  D. spatial estimates (object coordinates parsed from the `scene` field vs ground truth,
     fingertip-to-object distance at gripper closes)
  E. corrections (axis reversals between turns, fine-correction turns before a close)

stdlib + numpy only; no model calls. Ground-truth object poses come from state['objects'], which
the controller never saw. Usage:
  python scripts/analyze_robodawn_traces.py [--root DIR] [--dump turns.jsonl]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

ROOT_DEFAULT = "outputs/robodawn_colab"
FINE_CM = 3.0            # a move below this magnitude counts as a fine correction
DT = 1.0 / 250.0         # RoboTwin physics step (harness/robotwin/env.py:691)
PRESS_TASKS = {"click_bell"}   # tasks where 'close' is used to make a pressing tool, not to grasp
SKIP_OBJECTS = {"wall", "table"}


# ----------------------------------------------------------------------------- loading
def load_episodes(root: str) -> list[dict]:
    """One dict per episode: meta from results.json, trace turns, per-turn transport usage."""
    episodes = []
    pattern = os.path.join(root, "*", "robodawn", "gemini_flash_flex", "*", "shard_*")
    for shard in sorted(glob.glob(pattern)):
        task = shard.rstrip("/").split("/")[-2]
        res = json.load(open(os.path.join(shard, "results.json")))
        # transport.jsonl holds one line per model call in run order; tags restart at turn1 per episode
        runs, cur = [], []
        tpath = os.path.join(shard, "transport.jsonl")
        if os.path.exists(tpath):
            for line in open(tpath):
                rec = json.loads(line)
                if rec.get("tag") == "turn1" and cur:
                    runs.append(cur)
                    cur = []
                cur.append(rec)
            if cur:
                runs.append(cur)
        for i, meta in enumerate(res["episodes"]):
            ep_dir = os.path.join(shard, f"episode_{meta['episode_index']:03d}")
            trace = json.load(open(os.path.join(ep_dir, "trace.json")))
            usage = runs[i] if i < len(runs) and len(runs[i]) == len(trace) else [None] * len(trace)
            for turn, u in zip(trace, usage):
                turn["_usage"] = (u or {}).get("usage") or {}
            episodes.append({"task": task, "index": meta["episode_index"], "success": bool(meta["success"]),
                             "finished_reason": meta["finished_reason"], "turns": trace,
                             "instruction": meta.get("instruction", "")})
    return episodes


# ----------------------------------------------------------------------------- helpers
def pct(a, b) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "-"


def q(values, p):
    return float(np.percentile(values, p)) if len(values) else float("nan")


def table(title: str, header: list[str], rows: list[list], note: str = "") -> None:
    print(f"\n### {title}")
    if note:
        print(note)
    cells = [[str(h) for h in header]] + [[f"{c:.2f}" if isinstance(c, float) else str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    for j, r in enumerate(cells):
        print("  ".join(c.rjust(w) if k else c.ljust(w) for k, (c, w) in enumerate(zip(r, widths))))
        if j == 0:
            print("  ".join("-" * w for w in widths))


def objects_of(state: dict) -> dict:
    return {k: v for k, v in (state.get("objects") or {}).items() if k not in SKIP_OBJECTS}


def ref_points(obj: dict) -> list[list[float]]:
    """Object origin plus its functional points (both are candidates for 'where the object is')."""
    return [obj["position_cm"]] + list(obj.get("functional_points_cm") or [])


def holding(state: dict, arm: str) -> bool:
    a = state.get(arm) or {}
    return a.get("gripper", 1.0) < 0.5 and a.get("gripper_real", 0.0) > 0.08


def robot_seconds(results: list[dict]) -> float:
    return sum(float(r.get("robot_seconds") or 0.0) + DT * float(r.get("settle_steps") or 0) for r in results)


_CMD_RE = re.compile(r"^(left|right) (move|rotate|gripper|home|point)\s*([xyz]|roll|pitch|yaw|down45|down|forward)?\s*([-+]?\d+(?:\.\d+)?)?")


def parse_cmd(text: str) -> dict:
    """Parse the normalised command text written by Command.text() (harness/core/commands.py:50)."""
    m = _CMD_RE.match(text)
    if not m:
        return {"kind": text, "arm": None}
    arm, kind, axis, val = m.groups()
    return {"kind": kind, "arm": arm, "axis": axis, "value": float(val) if val is not None else None}


_PARTIAL_RE = re.compile(r"fingertips moved \(([-+]?\d+\.\d+), ([-+]?\d+\.\d+), ([-+]?\d+\.\d+)\)")


def executed_delta(cmd: dict, res: dict):
    """Fingertip displacement (cm) caused by one command, or None when unknown (home)."""
    if cmd["kind"] == "move":
        d = np.zeros(3)
        if res.get("ok"):
            d["xyz".index(cmd["axis"])] = cmd["value"]
            return d
        m = _PARTIAL_RE.search(res.get("note") or "")
        return np.array([float(v) for v in m.groups()]) if m else np.zeros(3)
    if cmd["kind"] == "home":
        return None
    return np.zeros(3)   # rotate/point keep the tool point (env.py:634-650), gripper/wait do not move it


# ----------------------------------------------------------------------------- per-turn features
def turn_features(ep: dict) -> list[dict]:
    """Derive per-turn features: arms, moves, phase label, gripper events, segment ids."""
    feats = []
    turns = ep["turns"]
    closes_so_far = 0
    for i, t in enumerate(turns):
        st = t["state"]
        cmds = [parse_cmd(c) for c in t.get("commands") or []]
        results = t.get("results") or []
        arms = sorted({c["arm"] for c in cmds if c.get("arm")})
        moves = [c for c in cmds if c["kind"] == "move"]
        net = {a: np.zeros(3) for a in ("left", "right")}
        for c, r in zip(cmds, results):
            if c["kind"] == "move":
                d = executed_delta(c, r)
                if d is not None:
                    net[c["arm"]] += d
        hold0 = {a: holding(st, a) for a in ("left", "right")}
        closes = [(c, r) for c, r in zip(cmds, results) if c["kind"] == "gripper" and c["value"] < 0.5]
        opens = [(c, r) for c, r in zip(cmds, results) if c["kind"] == "gripper" and c["value"] >= 0.5]
        kinds = [c["kind"] for c in cmds]
        failed = [r for r in results if r.get("kind") in ("move", "rotate", "point", "home") and not r.get("ok")]

        # phase label (first matching rule; see report section B)
        if not cmds:
            phase = "no_command"            # parse failure / empty reply
        elif all(k == "done" for k in kinds) or kinds == ["wait"]:
            phase = "done_or_wait"
        elif any(not hold0[c["arm"]] for c, _ in closes):
            phase = "grasp"
        elif any(hold0[c["arm"]] for c, _ in opens):
            phase = "release"
        elif "home" in kinds or any((not hold0[c["arm"]]) and st[c["arm"]]["gripper"] < 0.5 for c, _ in opens):
            phase = "recover"
        elif any(hold0[a] for a in arms):
            phase = "transport"
        elif all(k in ("rotate", "point", "wait") for k in kinds):
            phase = "orient"
        elif moves and all(abs(c["value"]) < FINE_CM for c in moves):
            phase = "align_fine"
        else:
            a = arms[0] if arms else "left"
            n = net[a]
            phase = "descend" if n[2] <= -FINE_CM and abs(n[2]) >= np.linalg.norm(n[:2]) else "approach"

        closes_so_far += len(closes)
        feats.append({
            "turn": t["turn"], "phase": phase, "arms": arms, "n_cmds": len(cmds), "kinds": kinds,
            "move_sizes": [abs(c["value"]) for c in moves], "net": {a: net[a].tolist() for a in net},
            "hold0": hold0, "closes": closes, "opens": opens, "n_failed": len(failed),
            "latency": float(t.get("latency_s") or 0.0), "robot_s": robot_seconds(results),
            "sim_wall_s": sum(float(r.get("seconds") or 0.0) for r in results),
            "reasoning_tokens": int(((t["_usage"].get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0),
            "prompt_tokens": int(t["_usage"].get("prompt_tokens") or 0),
            "completion_tokens": int(t["_usage"].get("completion_tokens") or 0),
            "plan": t.get("plan") or "", "scene": t.get("scene") or "", "progress": t.get("progress") or "",
            "memory": t.get("memory") or "", "closes_before": closes_so_far - len(closes),
            "empty_close": any("closed fully" in (r.get("note") or "") for _, r in closes),
            "done_cmd": "done" in kinds, "home": "home" in kinds,
        })
    # segments: a new segment starts at turn 1 and after every turn with a gripper event or a home command
    seg = 0
    for i, f in enumerate(feats):
        if i > 0:
            p = feats[i - 1]
            if p["closes"] or p["opens"] or p["home"]:
                seg += 1
        f["segment"] = seg
    # sub-goals: a new sub-goal starts at turn 1, when the holding status of an arm changed since the previous
    # turn (successful grasp, release, drop) or when the commanded arm set changes (arm switch). Empty closes and
    # planner failures do NOT start a new sub-goal: the goal is the same, only the attempt failed.
    sub, last_arms = 0, None
    for i, f in enumerate(feats):
        if i > 0:
            p = feats[i - 1]
            if f["hold0"] != p["hold0"] or (f["arms"] and last_arms is not None and f["arms"] != last_arms):
                sub += 1
        if f["arms"]:
            last_arms = f["arms"]
        f["subgoal"] = sub
        # turn category (section C): what the model had to decide on this turn
        if i == 0 or (i > 0 and f["subgoal"] != feats[i - 1]["subgoal"]):
            f["category"] = "new_subgoal"
        else:
            p = feats[i - 1]
            if p["empty_close"] or p["n_failed"] or p["done_cmd"] or p["phase"] == "no_command" or p["home"] \
                    or (p["opens"] and not any(p["hold0"][c["arm"]] for c, _ in p["opens"])):
                f["category"] = "retry_after_failure"
            else:
                f["category"] = "continuation"
    return feats


# ----------------------------------------------------------------------------- coordinate parsing
NUM = r"([-+−]?\d+(?:\.\d+)?)"
APPROX = r"(?:~|≈|approx\.?|about|around)?\s*"
TUPLE_RE = re.compile(
    rf"\(\s*(?:x\s*[=:]\s*)?{APPROX}{NUM}\s*(?:cm)?\s*,\s*(?:y\s*[=:]\s*)?{APPROX}{NUM}\s*(?:cm)?"
    rf"(?:\s*,\s*(?:z\s*[=:]\s*)?{APPROX}{NUM}\s*(?:cm)?)?\s*\)", re.I)
SEP = r"\s*(?:=|≈|~|:|is|of)\s*(?:~|≈)?\s*"
NAMED_RE = re.compile(
    rf"\bx{SEP}{NUM}\s*(?:cm)?\s*,?\s*(?:and\s+)?y{SEP}{NUM}(?:\s*(?:cm)?\s*,?\s*(?:and\s+)?z{SEP}{NUM})?", re.I)
GRIPPER_WORDS = re.compile(r"gripper|finger|tcp|wrist|\barms?\b|\bhand\b|\bL@|\bR@|tool|\bmove\b|\btarget\b|\bto\b\s*$", re.I)
ARM_ONLY = re.compile(r"^\W*(?:and\s+)?(?:the\s+)?(?:left|right)\b\s*(?:one\s*)?(?:is\s*)?(?:at|@|:)?\s*$", re.I)
PART_WORDS = re.compile(r"handle|corner|edge|rim|\btip\b|\bends?\b|\bhead\b|spout|hinge|\blid\b|neck|\bcap\b|"
                        r"\bside\b|opening|mouth|\bleg\b|\bbase\b", re.I)   # object parts, not the object centre
SYNONYMS = {"block": "box", "cube": "box", "mat": "pad", "stand": "displaystand",
            "screen": "laptop", "button": "bell", "pan": "pot", "kettle": "pot"}
STOP = {"the", "and", "with", "near", "its", "top", "side", "center", "centre", "centered", "centred", "located",
        "lying", "at", "is", "are", "now", "about", "approximately", "around", "body", "base", "left", "right",
        "object", "table", "cm", "which", "still", "edge", "front", "behind", "between", "position", "rests",
        "resting", "sits", "sitting", "standing", "upright", "horizontally", "on", "of", "in", "a", "an"}


def _num(s: str) -> float:
    return float(s.replace("−", "-"))


def name_tokens(key: str, obj: dict) -> set[str]:
    toks = set(re.split(r"[_\-\d\s]+", f"{key}_{obj.get('model', '')}".lower()))
    return {t for t in toks if len(t) >= 3 and t not in {"obj", "object", "target", "cluttered"}}


def parse_scene_coords(text: str) -> list[dict]:
    """All (x, y[, z]) mentions with the subject text that precedes each one."""
    hits = []
    for rx in (TUPLE_RE, NAMED_RE):
        for m in rx.finditer(text):
            hits.append((m.start(), m.end(), [_num(g) if g is not None else None for g in m.groups()]))
    hits.sort()
    out, last_end = [], 0
    for s, e, vals in hits:
        if s < last_end:          # overlapping match (a tuple that also matched the named form)
            continue
        subj = text[last_end:s]
        cut = max(subj.rfind(". "), subj.rfind("; "))
        subj = subj[cut + 1:] if cut >= 0 else subj
        out.append({"subject": subj.strip(), "x": vals[0], "y": vals[1], "z": vals[2]})
        last_end = e
    return out


def match_object(subject: str, xy, objs: dict):
    """Name rule first (latest object word in the subject wins), else nearest object within 15 cm (xy)."""
    words = [w for w in re.findall(r"[a-z]+", subject.lower()) if len(w) >= 3 and w not in STOP]
    words = [SYNONYMS.get(w, w) for w in words] + words
    toks = {k: name_tokens(k, o) for k, o in objs.items()}
    for w in reversed(words):
        cands = [k for k, ts in toks.items() if any(w == n or (len(w) >= 3 and (w in n or n in w)) for n in ts)]
        if cands:
            best = min(cands, key=lambda k: min(np.hypot(p[0] - xy[0], p[1] - xy[1]) for p in ref_points(objs[k])))
            return best, "name"
    best, dist = None, 1e9
    for k, o in objs.items():
        d = min(np.hypot(p[0] - xy[0], p[1] - xy[1]) for p in ref_points(o))
        if d < dist:
            best, dist = k, d
    return (best, "nearest") if dist <= 15.0 else (None, "unmatched")


def spatial_estimates(ep: dict) -> list[dict]:
    rows = []
    for t in ep["turns"]:
        st = t["state"]
        objs = objects_of(st)
        tcp = {a: st[a]["position_cm"] for a in ("left", "right")}
        for c in parse_scene_coords(t.get("scene") or ""):
            xy = (c["x"], c["y"])
            near_tcp = min(np.hypot(tcp[a][0] - xy[0], tcp[a][1] - xy[1]) for a in tcp)
            if GRIPPER_WORDS.search(c["subject"]) or ARM_ONLY.match(c["subject"]) or not c["subject"]:
                rows.append({"kind": "gripper_subject"})
                continue
            if near_tcp < 1.0:
                rows.append({"kind": "equals_tcp"})
                continue
            if PART_WORDS.search(c["subject"][-40:]):
                rows.append({"kind": "object_part"})
                continue
            key, rule = match_object(c["subject"], xy, objs)
            row = {"kind": rule, "task": ep["task"], "turn": t["turn"], "subject": c["subject"],
                   "est": [c["x"], c["y"], c["z"]], "obj": key}
            if key:
                o = objs[key]
                p0 = o["position_cm"]
                best = min(ref_points(o), key=lambda p: np.hypot(p[0] - xy[0], p[1] - xy[1]))
                row.update(err_origin=[xy[0] - p0[0], xy[1] - p0[1], (c["z"] - p0[2]) if c["z"] is not None else None],
                           err_best=[xy[0] - best[0], xy[1] - best[1]],
                           gt=p0, gripper_dist=min(math.dist(tcp[a][:2], p0[:2]) for a in tcp),
                           nearest_is_same=(key == min(objs, key=lambda k: min(np.hypot(p[0] - xy[0], p[1] - xy[1])
                                                                                 for p in ref_points(objs[k])))))
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------- grasp geometry
def tcp_at_commands(t: dict, next_state) -> list:
    """Fingertip position (cm) just before each command of a turn, rebuilt from the turn's start state
    and the executed moves; after a 'home' the position is unknown (None)."""
    st = t["state"]
    pos = {a: np.array(st[a]["position_cm"], dtype=float) for a in ("left", "right")}
    out = []
    for ctext, res in zip(t.get("commands") or [], t.get("results") or []):
        c = parse_cmd(ctext)
        out.append({a: (None if pos[a] is None else pos[a].copy()) for a in pos})
        if c.get("arm"):
            d = executed_delta(c, res)
            if d is None:
                pos[c["arm"]] = None
            elif pos[c["arm"]] is not None:
                pos[c["arm"]] = pos[c["arm"]] + d
    return out


def grasp_events(ep: dict) -> list[dict]:
    """Every executed 'close' on an arm that was not holding: fingertip offset to the nearest object and outcome."""
    if ep["task"] in PRESS_TASKS:
        return []
    out = []
    turns = ep["turns"]
    for i, t in enumerate(turns):
        nxt = turns[i + 1]["state"] if i + 1 < len(turns) else None
        pre = tcp_at_commands(t, nxt)
        for j, (ctext, res) in enumerate(zip(t.get("commands") or [], t.get("results") or [])):
            c = parse_cmd(ctext)
            if c["kind"] != "gripper" or c["value"] >= 0.5 or holding(t["state"], c["arm"]) or not res.get("ok"):
                continue
            p = pre[j][c["arm"]]
            if p is None:
                continue
            objs = objects_of(t["state"])
            best_k, best_d, best_p = None, 1e9, None
            for k, o in objs.items():
                for rp in ref_points(o):
                    d = math.hypot(p[0] - rp[0], p[1] - rp[1])
                    if d < best_d:
                        best_k, best_d, best_p = k, d, rp
            # outcome
            if "closed fully" in (res.get("note") or ""):
                outcome = "empty"
            else:
                z0 = objs[best_k]["position_cm"][2]
                later = [turns[k]["state"] for k in range(i + 1, min(i + 5, len(turns)))]
                rose = max((objects_of(s).get(best_k, {}).get("position_cm", [0, 0, z0])[2] - z0 for s in later),
                           default=0.0)
                if rose >= 3.0:
                    outcome = "lifted"
                elif ep["success"] and i == len(turns) - 1:
                    outcome = "success_end"
                else:
                    outcome = "held_not_lifted"
            out.append({"task": ep["task"], "turn": t["turn"], "outcome": outcome, "obj": best_k,
                        "obj_pos": objs[best_k]["position_cm"],
                        "dxy": best_d, "dz_table": p[2] - t["state"]["table_z_cm"],
                        "dz_obj": p[2] - (best_p[2] if best_p is not None else 0), "tcp": p.tolist(),
                        "ep_success": ep["success"], "closes_before": sum(1 for f in turns[:i] for cc in (f.get("commands") or [])
                                                                          if re.search(r"gripper 0\.[0-4]", cc))})
    return out


# ----------------------------------------------------------------------------- text similarity
WORD = re.compile(r"[a-z]{3,}")
STOPWORDS = {"the", "and", "then", "to", "of", "a", "an", "it", "its", "with", "for", "at", "on", "in", "by",
             "from", "into", "onto", "is", "are", "be", "this", "that", "cm", "now", "next", "slightly"}


def words(s: str) -> set[str]:
    return {w for w in WORD.findall(s.lower()) if w not in STOPWORDS}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


ACTION_KEYS = {
    "orient": r"point|rotate|yaw|roll|pitch|orient",
    "descend": r"descend|lower|down to|go down|move down",
    "lift": r"\blift|raise|move up|ascend",
    "close": r"close|grasp|grip\b|pinch",
    "open": r"\bopen|release|let go|drop",
    "translate": r"move|translate|shift|align|position|transport|carry|center",
    "home": r"\bhome\b",
}


def plan_actions(text: str) -> set[str]:
    return {k for k, rx in ACTION_KEYS.items() if re.search(rx, text.lower())}


def command_actions(f: dict) -> set[str]:
    acts = set()
    for k in f["kinds"]:
        if k in ("rotate", "point"):
            acts.add("orient")
        if k == "home":
            acts.add("home")
    for c, _ in f["closes"]:
        acts.add("close")
    for c, _ in f["opens"]:
        acts.add("open")
    for a in f["arms"]:
        n = np.array(f["net"][a])
        if n[2] <= -1:
            acts.add("descend")
        if n[2] >= 1:
            acts.add("lift")
        if np.linalg.norm(n[:2]) >= 0.5:
            acts.add("translate")
    return acts


# ----------------------------------------------------------------------------- analyses
def section_a(eps, feats):
    print("\n## A. Turn anatomy")
    allf = [f for fs in feats for f in fs]
    n = len(allf)
    ncmd = Counter(f["n_cmds"] for f in allf)
    table("commands per turn", ["n_cmds", "turns", "share"], [[k, v, pct(v, n)] for k, v in sorted(ncmd.items())])
    kinds = Counter(k for f in allf for k in f["kinds"])
    tot = sum(kinds.values())
    table("command kinds", ["kind", "count", "share"], [[k, v, pct(v, tot)] for k, v in kinds.most_common()])
    sizes = [s for f in allf for s in f["move_sizes"]]
    bins = [0, 1, 2, 3, 5, 10, 15, 20.01]
    hist = np.histogram(sizes, bins=bins)[0]
    table("move magnitude |cm|", ["bin", "count", "share"],
          [[f"[{bins[i]}, {bins[i + 1]:.0f})", int(c), pct(c, len(sizes))] for i, c in enumerate(hist)],
          note=f"moves: {len(sizes)}, median {np.median(sizes):.1f} cm, share < {FINE_CM:.0f} cm: "
               f"{pct(sum(s < FINE_CM for s in sizes), len(sizes))}")
    lat = np.array([f["latency"] for f in allf])
    rob = np.array([f["robot_s"] for f in allf])
    sim = np.array([f["sim_wall_s"] for f in allf])
    table("time per turn (s)", ["quantity", "sum", "mean", "median", "p90"],
          [["model latency", lat.sum(), lat.mean(), np.median(lat), q(lat, 90)],
           ["robot motion (simulated)", rob.sum(), rob.mean(), np.median(rob), q(rob, 90)],
           ["simulator wall time", sim.sum(), sim.mean(), np.median(sim), q(sim, 90)]],
          note=f"model latency share of (latency + robot motion): {pct(lat.sum(), lat.sum() + rob.sum())}")
    rt = np.array([f["reasoning_tokens"] for f in allf])
    ct = np.array([f["completion_tokens"] for f in allf])
    pt = np.array([f["prompt_tokens"] for f in allf])
    table("tokens per turn", ["quantity", "mean", "median", "p90"],
          [["prompt", pt.mean(), np.median(pt), q(pt, 90)], ["completion", ct.mean(), np.median(ct), q(ct, 90)],
           ["reasoning", rt.mean(), np.median(rt), q(rt, 90)]],
          note=f"latency vs reasoning tokens: Pearson r = {np.corrcoef(lat, rt)[0, 1]:.2f}")
    rows = []
    for name, sel in (("success", lambda e: e["success"]), ("failure", lambda e: not e["success"])):
        tt = [len(e["turns"]) for e in eps if sel(e)]
        ms = [sum(f["latency"] for f in fs) / 60 for e, fs in zip(eps, feats) if sel(e)]
        rows.append([name, len(tt), np.mean(tt), np.median(tt), np.mean(ms)])
    table("turns per episode by outcome", ["outcome", "episodes", "mean turns", "median", "mean model min"], rows)
    fr = Counter(e["finished_reason"] for e in eps)
    table("finished_reason", ["reason", "episodes"], [[k, v] for k, v in fr.most_common()])
    total_lat = lat.sum()
    succ_lat = sum(f["latency"] for e, fs in zip(eps, feats) if e["success"] for f in fs)
    print(f"model time in failed episodes: {pct(total_lat - succ_lat, total_lat)} of all model time; "
          f"turns in failed episodes: {pct(sum(len(e['turns']) for e in eps if not e['success']), n)}")


PHASES = ["approach", "orient", "descend", "align_fine", "grasp", "transport", "release", "recover",
          "done_or_wait", "no_command"]


def section_b(eps, feats):
    print("\n## B. Phases")
    print("labels (first rule wins): no_command (unparsed reply) | done_or_wait | grasp (close on an empty arm) | "
          "release (open on a holding arm) | recover (home, or re-open after a failed close) | transport (a "
          f"commanded arm holds an object) | orient (only point/rotate) | align_fine (all moves < {FINE_CM:.0f} cm) | "
          f"descend (net z <= -{FINE_CM:.0f} cm, mostly vertical) | approach (other moves)")
    allf = [(e, f) for e, fs in zip(eps, feats) for f in fs]
    n = len(allf)
    tot_lat = sum(f["latency"] for _, f in allf)
    rows = []
    for ph in PHASES:
        sel = [(e, f) for e, f in allf if f["phase"] == ph]
        if not sel:
            continue
        s = [f for e, f in sel if e["success"]]
        fl = [f for e, f in sel if not e["success"]]
        rows.append([ph, len(sel), pct(len(sel), n), pct(sum(f["latency"] for _, f in sel), tot_lat),
                     len(s), len(fl), np.mean([f["latency"] for _, f in sel]),
                     np.mean([f["robot_s"] for _, f in sel])])
    table("turns and model time per phase", ["phase", "turns", "share", "model time", "in succ eps", "in fail eps",
                                             "mean latency s", "mean robot s"], rows)
    rows = []
    for task in sorted({e["task"] for e in eps}):
        tf = [f for e, f in allf if e["task"] == task]
        c = Counter(f["phase"] for f in tf)
        rows.append([task, len(tf)] + [pct(c[p], len(tf)) for p in PHASES[:8]])
    table("phase mix per task (share of turns)", ["task", "turns"] + PHASES[:8], rows)
    # failed episodes: where they end
    rows = []
    last_phase, held_ever, fails = Counter(), Counter(), [ (e, fs) for e, fs in zip(eps, feats) if not e["success"]]
    for e, fs in fails:
        lp = Counter(f["phase"] for f in fs[-5:]).most_common(1)[0][0]
        ever = any(f["hold0"]["left"] or f["hold0"]["right"] for f in fs)
        n_close = sum(len(f["closes"]) for f in fs)
        n_empty = sum(1 for f in fs if f["empty_close"])
        rows.append([e["task"], e["index"], e["finished_reason"], len(fs), lp, "yes" if ever else "no", n_close, n_empty])
        last_phase[lp] += 1
        held_ever["held" if ever else "never held"] += 1
    rows.sort()
    table("failed episodes (last-5-turn dominant phase)", ["task", "ep", "reason", "turns", "last phase", "ever held",
                                                            "closes", "empty closes"], rows)
    print("failed episodes by last phase:", dict(last_phase), "| held an object at some point:", dict(held_ever))
    # success episodes: turns until first grasp attempt
    first = [next((i for i, f in enumerate(fs) if f["closes"]), None) for e, fs in zip(eps, feats)
             if e["success"] and e["task"] not in PRESS_TASKS]
    first = [x + 1 for x in first if x is not None]
    print(f"successful episodes (grasp tasks): turn of first close median {np.median(first):.0f}, mean {np.mean(first):.1f}")


def section_c(eps, feats):
    print("\n## C. Judgment persistence")
    allf = [f for fs in feats for f in fs]
    n = len(allf)
    tot_lat = sum(f["latency"] for f in allf)
    # (0) turn taxonomy: new sub-goal / retry after a failed attempt / plain continuation
    cats = ("new_subgoal", "retry_after_failure", "continuation")
    rows = []
    for cat in cats:
        sel = [(e, f) for e, fs in zip(eps, feats) for f in fs if f["category"] == cat]
        fsel = [f for _, f in sel]
        ph = Counter(f["phase"] for f in fsel).most_common(3)
        rows.append([cat, len(sel), pct(len(sel), n), pct(sum(f["latency"] for f in fsel), tot_lat),
                     pct(sum(1 for e, _ in sel if not e["success"]), len(sel)),
                     np.mean([f["latency"] for f in fsel]), np.mean([f["reasoning_tokens"] for f in fsel]),
                     ", ".join(f"{p} {pct(c, len(fsel))}" for p, c in ph)])
    subs = [len({f["subgoal"] for f in fs}) for fs in feats]
    sub_len = [c for fs in feats for c in Counter(f["subgoal"] for f in fs).values()]
    table("turn taxonomy", ["category", "turns", "share", "model time", "in failed eps", "mean latency s",
                            "mean reasoning tok", "top phases"], rows,
          note="new_subgoal: turn 1, holding status changed, or arm switch | retry_after_failure: previous turn had "
               "an empty close, a failed motion, a home, a re-open, a rejected done or no command | continuation: rest. "
               f"Sub-goals per episode: median {np.median(subs):.0f}; turns per sub-goal median {np.median(sub_len):.0f}, "
               f"mean {np.mean(sub_len):.1f}, p90 {q(sub_len, 90):.0f}")
    rows = []
    for name, sel in (("success", True), ("failure", False)):
        fsel = [f for e, fs in zip(eps, feats) if e["success"] == sel for f in fs]
        c = Counter(f["category"] for f in fsel)
        rows.append([name, len(fsel)] + [pct(c[k], len(fsel)) for k in cats])
    table("taxonomy by episode outcome", ["episodes", "turns"] + list(cats), rows)
    # (1) segments between gripper events / home
    first_in_seg = 0
    seg_lens = []
    for fs in feats:
        segs = Counter(f["segment"] for f in fs)
        seg_lens += list(segs.values())
        first_in_seg += len(segs)
    cont = [f for fs in feats for i, f in enumerate(fs) if i > 0 and f["segment"] == fs[i - 1]["segment"]]
    # (2) same arm and same phase as previous turn
    same_phase = [f for fs in feats for i, f in enumerate(fs)
                  if i > 0 and f["phase"] == fs[i - 1]["phase"] and f["arms"] == fs[i - 1]["arms"]]
    # (3) strict: same segment, same arms, every move fine, no gripper event, no orientation command
    strict = [f for fs in feats for i, f in enumerate(fs)
              if i > 0 and f["segment"] == fs[i - 1]["segment"] and f["arms"] == fs[i - 1]["arms"]
              and f["move_sizes"] and all(s < FINE_CM for s in f["move_sizes"]) and not f["closes"] and not f["opens"]
              and not any(k in ("rotate", "point", "home") for k in f["kinds"])]
    # (4) "trigger" turns: obviously new information (first turn, after a gripper event, after a planner
    # failure, after an empty close, after a rejected done, after an unparsed reply)
    trig = []
    for fs in feats:
        for i, f in enumerate(fs):
            if i == 0:
                trig.append(f)
                continue
            p = fs[i - 1]
            if p["closes"] or p["opens"] or p["home"] or p["n_failed"] or p["done_cmd"] or p["phase"] == "no_command":
                trig.append(f)
    rows = [["continuation within a gripper-event segment", len(cont), pct(len(cont), n),
             pct(sum(f["latency"] for f in cont), tot_lat)],
            ["same arm + same phase as previous turn", len(same_phase), pct(len(same_phase), n),
             pct(sum(f["latency"] for f in same_phase), tot_lat)],
            ["strict: same segment, only fine moves (< 3 cm)", len(strict), pct(len(strict), n),
             pct(sum(f["latency"] for f in strict), tot_lat)],
            ["turns following an event (new information)", len(trig), pct(len(trig), n),
             pct(sum(f["latency"] for f in trig), tot_lat)]]
    table("continuation shares", ["definition", "turns", "share of turns", "share of model time"], rows,
          note=f"segments: {len(seg_lens)}, turns per segment median {np.median(seg_lens):.0f}, mean "
               f"{np.mean(seg_lens):.1f}, p90 {q(seg_lens, 90):.0f}")
    # continuation by outcome
    rows = []
    for name, sel in (("success", True), ("failure", False)):
        fsel = [fs for e, fs in zip(eps, feats) if e["success"] == sel]
        nn = sum(len(fs) for fs in fsel)
        c1 = sum(1 for fs in fsel for i, f in enumerate(fs) if i > 0 and f["segment"] == fs[i - 1]["segment"])
        rows.append([name, nn, pct(c1, nn)])
    table("segment continuation by outcome", ["episodes", "turns", "continuation share"], rows)
    # text similarity of plan/scene/progress between consecutive turns
    rows = []
    for field in ("plan", "progress", "scene", "memory"):
        within, across = [], []
        for fs in feats:
            for i in range(1, len(fs)):
                if not fs[i][field] or not fs[i - 1][field]:
                    continue
                j = jaccard(words(fs[i][field]), words(fs[i - 1][field]))
                (within if fs[i]["segment"] == fs[i - 1]["segment"] else across).append(j)
        rows.append([field, np.median(within), np.median(across), pct(sum(x >= 0.5 for x in within), len(within))])
    table("word-set Jaccard between consecutive turns (numbers ignored)",
          ["field", "median within segment", "median across event", "within >= 0.5"], rows)
    # deferred plan steps: the previous plan named an action it did not execute and this turn executes it
    deferred = carried = 0
    for fs in feats:
        for i in range(1, len(fs)):
            prev_plan = plan_actions(fs[i - 1]["plan"]) - {"translate"}
            left_over = prev_plan - command_actions(fs[i - 1])
            if left_over:
                deferred += 1
                if left_over & command_actions(fs[i]):
                    carried += 1
    print(f"turns whose previous plan named a not-yet-executed action (orient/descend/lift/close/open/home): "
          f"{deferred} ({pct(deferred, n)}); of these the next turn executed it: {carried} ({pct(carried, deferred)})")
    mem_coords = sum(1 for f in allf if parse_scene_coords(f["memory"]))
    print(f"memory field contains coordinates in {pct(mem_coords, n)} of turns")
    # does the model reuse the object coordinate it stored in its notes, or re-estimate it?
    dd = []
    for e in eps:
        T = e["turns"]
        for i in range(1, len(T)):
            objs = objects_of(T[i]["state"])
            stored = {}
            for c in parse_scene_coords(T[i - 1].get("memory") or ""):
                if c["subject"] and not GRIPPER_WORDS.search(c["subject"]):
                    k, rule = match_object(c["subject"], (c["x"], c["y"]), objs)
                    if k and rule == "name":
                        stored.setdefault(k, (c["x"], c["y"]))
            for c in parse_scene_coords(T[i].get("scene") or ""):
                if c["subject"] and not GRIPPER_WORDS.search(c["subject"]):
                    k, rule = match_object(c["subject"], (c["x"], c["y"]), objs)
                    if rule == "name" and k in stored:
                        dd.append(math.dist(stored.pop(k), (c["x"], c["y"])))
    dd = np.array(dd)
    print(f"object coordinate in notes (turn t-1) vs this turn's scene (same object, name-matched): n={len(dd)}, "
          f"reused exactly {pct((dd < 0.1).sum(), len(dd))}, within 1 cm {pct((dd < 1).sum(), len(dd))}, "
          f"median change {np.median(dd):.1f} cm")
    has = sum(1 for e in eps for t in e["turns"]
              if any(c["subject"] and not GRIPPER_WORDS.search(c["subject"]) for c in parse_scene_coords(t.get("scene") or "")))
    print(f"turns whose scene states an object coordinate: {pct(has, n)}")


def section_d(eps, feats):
    print("\n## D. Spatial estimates")
    rows = [r for e in eps for r in spatial_estimates(e)]
    kinds = Counter(r["kind"] for r in rows)
    tot = len(rows)
    table("coordinate mentions in `scene`", ["class", "count", "share"], [[k, v, pct(v, tot)] for k, v in kinds.most_common()],
          note="gripper_subject: subject names a gripper/arm/target; equals_tcp: within 1 cm of a fingertip; "
               "object_part: subject names a part (handle, edge, corner, lid, ...), excluded from the error stats; "
               "name: matched by object word vs key/model name; nearest: no word match, nearest object within 15 cm")
    named = [r for r in rows if r["kind"] == "name"]
    print(f"name-matched estimates whose nearest object is a different one: "
          f"{pct(sum(not r['nearest_is_same'] for r in named), len(named))}")
    obj_rows = [r for r in rows if r["kind"] in ("name", "nearest")]

    def stats(label, errs):
        a = np.abs(np.array(errs))
        return [label, len(errs), float(np.mean(errs)), float(np.median(a)), q(a, 90), pct(sum(a <= 2), len(a)),
                pct(sum(a <= 5), len(a))]

    ex = [r["err_origin"][0] for r in obj_rows]
    ey = [r["err_origin"][1] for r in obj_rows]
    ez = [r["err_origin"][2] for r in obj_rows if r["err_origin"][2] is not None]
    exy = [math.hypot(r["err_origin"][0], r["err_origin"][1]) for r in obj_rows]
    bxy = [math.hypot(*r["err_best"]) for r in obj_rows]
    table("estimate error vs ground truth (cm; matched mentions)",
          ["axis", "n", "mean signed", "median |e|", "p90 |e|", "<= 2 cm", "<= 5 cm"],
          [stats("x (origin)", ex), stats("y (origin)", ey), stats("z (origin; origin height varies)", ez),
           stats("xy norm (origin)", exy), stats("xy norm (nearest of origin/functional pts)", bxy)])
    nm = [math.hypot(*r["err_best"]) for r in named]
    print(f"name-matched only: xy (best ref) median {np.median(nm):.1f} cm, p90 {q(nm, 90):.1f} cm (n={len(nm)})")
    rows_t = []
    for task in sorted({r["task"] for r in obj_rows}):
        v = [math.hypot(*r["err_best"]) for r in obj_rows if r["task"] == task]
        rows_t.append([task, len(v), np.median(v), q(v, 90)])
    table("xy error by task (best ref)", ["task", "n", "median", "p90"], rows_t)
    # by turn number and by gripper distance
    rows_b = []
    for lo, hi in ((1, 1), (2, 3), (4, 6), (7, 12), (13, 45)):
        v = [math.hypot(*r["err_best"]) for r in obj_rows if lo <= r["turn"] <= hi]
        if v:
            rows_b.append([f"turn {lo}-{hi}", len(v), np.median(v), q(v, 90)])
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 200)):
        v = [math.hypot(*r["err_best"]) for r in obj_rows if lo <= r["gripper_dist"] < hi]
        if v:
            rows_b.append([f"nearest fingertip {lo}-{hi} cm (xy)", len(v), np.median(v), q(v, 90)])
    table("xy error vs turn and vs fingertip distance", ["bucket", "n", "median", "p90"], rows_b)
    # jitter: consecutive re-estimates of the same object while the object did not move
    jit, drift_toward = [], []
    by_ep = defaultdict(dict)
    for e in eps:
        est = defaultdict(dict)
        for r in spatial_estimates(e):
            if r.get("obj") and r["kind"] in ("name", "nearest") and r["obj"] not in est[r["turn"]]:
                est[r["turn"]][r["obj"]] = r
        turns = sorted(est)
        for a, b in zip(turns, turns[1:]):
            if b != a + 1:
                continue
            for k in est[a].keys() & est[b].keys():
                ra, rb = est[a][k], est[b][k]
                if math.dist(ra["gt"], rb["gt"]) < 0.5:
                    jit.append(math.hypot(rb["est"][0] - ra["est"][0], rb["est"][1] - ra["est"][1]))
                    drift_toward.append(math.hypot(*rb["err_best"]) - math.hypot(*ra["err_best"]))
    jit = np.array(jit)
    dt = np.array(drift_toward)
    print(f"re-estimates of a static object on consecutive turns: n={len(jit)}, change median {np.median(jit):.1f} cm, "
          f"p90 {q(jit, 90):.1f} cm, changed >= 2 cm: {pct((jit >= 2).sum(), len(jit))}; "
          f"error got smaller: {pct((dt < -0.5).sum(), len(dt))}, larger: {pct((dt > 0.5).sum(), len(dt))}")
    # frozen vs fresh: keep the first estimate of a static object vs re-estimating it every turn
    fro, fre = [], []
    for e in eps:
        est = defaultdict(dict)
        for r in spatial_estimates(e):
            if r.get("obj") and r["kind"] == "name" and r["obj"] not in est[r["turn"]]:
                est[r["turn"]][r["obj"]] = r
        first = {}
        for t in sorted(est):
            for k, r in est[t].items():
                f = first.get(k)
                if f is None or math.dist(f["gt"], r["gt"]) > 0.5:   # first mention, or the object moved
                    first[k] = r
                    continue
                fro.append(math.hypot(f["est"][0] - r["gt"][0], f["est"][1] - r["gt"][1]))
                fre.append(math.hypot(r["est"][0] - r["gt"][0], r["est"][1] - r["gt"][1]))
    fro, fre = np.array(fro), np.array(fre)
    print(f"frozen first estimate vs fresh re-estimate of a static object (n={len(fro)}): xy error median "
          f"{np.median(fro):.1f} vs {np.median(fre):.1f} cm; fresh better by > 1 cm {pct((fre < fro - 1).sum(), len(fro))}, "
          f"worse by > 1 cm {pct((fre > fro + 1).sum(), len(fro))}")
    # grasp closes
    g = [x for e in eps for x in grasp_events(e)]
    rows_g = []
    for oc in ("lifted", "success_end", "held_not_lifted", "empty"):
        v = [x for x in g if x["outcome"] == oc]
        if v:
            d = [x["dxy"] for x in v]
            z = [x["dz_table"] for x in v]
            rows_g.append([oc, len(v), np.median(d), q(d, 90), pct(sum(x <= 2 for x in d), len(d)), np.median(z)])
    table("fingertip offset at 'close' (grasp tasks; xy to nearest object origin/functional point)",
          ["outcome", "closes", "median dxy", "p90 dxy", "dxy <= 2 cm", "median z above table"], rows_g,
          note=f"excluded: {sorted(PRESS_TASKS)} (close = pressing tool). total closes: {len(g)}")
    rows_t = []
    for task in sorted({x["task"] for x in g}):
        v = [x for x in g if x["task"] == task]
        c = Counter(x["outcome"] for x in v)
        med = lambda oc, key: (np.median([x[key] for x in v if x["outcome"] == oc])
                               if any(x["outcome"] == oc for x in v) else "-")
        rows_t.append([task, len(v), c["lifted"] + c["success_end"], c["held_not_lifted"], c["empty"],
                       med("lifted", "dxy"), med("empty", "dxy"), med("lifted", "dz_table"), med("empty", "dz_table")])
    table("closes per task", ["task", "closes", "lifted", "held", "empty", "dxy lifted", "dxy empty",
                              "z lifted", "z empty"], rows_t,
          note="dxy is measured to the object origin/functional point; for pots (handles), laptops (lid edge) and "
               "fans the grasp point is far from both, so dxy is not a grasp-quality measure there")
    # retry chains after an empty close
    shifts, gaps, runs = [], [], []
    for e in eps:
        ge = grasp_events(e)
        r = 0
        for a, b in zip(ge, ge[1:]):
            if a["outcome"] == "empty":
                shifts.append(math.dist(a["tcp"][:2], b["tcp"][:2]))
                gaps.append(b["turn"] - a["turn"])
        for x in ge + [{"outcome": "end"}]:
            if x["outcome"] == "empty":
                r += 1
            elif r:
                runs.append(r)
                r = 0
    print(f"after an empty close: next close {np.median(gaps):.0f} turns later (median; mean {np.mean(gaps):.1f}), "
          f"fingertip xy shift median {np.median(shifts):.1f} cm (p90 {q(shifts, 90):.1f}); "
          f"runs of consecutive empty closes: n={len(runs)}, median {np.median(runs):.0f}, >= 5: "
          f"{sum(r >= 5 for r in runs)} runs holding {sum(r for r in runs if r >= 5)} closes")
    reach, succ, after = Counter(), Counter(), Counter()
    for e in eps:
        k = 0
        for x in grasp_events(e):
            if x["outcome"] == "empty":
                k += 1
                reach[k] += 1
                succ[k] += e["success"]
                after[k] += len(e["turns"]) - x["turn"]
    table("episode success after k empty closes (grasp tasks)", ["k", "episodes reaching k", "later succeeded",
                                                                  "rate", "turns spent after the k-th"],
          [[k, reach[k], succ[k], pct(succ[k], reach[k]), after[k]] for k in (1, 2, 3, 4, 5, 6, 8, 10) if reach[k]])
    # would an earlier estimate have been enough? compare it with the xy where a lifting close happened
    seg_first, ep_first = [], []
    for e, fs in zip(eps, feats):
        est = [r for r in spatial_estimates(e) if r.get("obj") and r["kind"] in ("name", "nearest")]
        for x in grasp_events(e):
            if x["outcome"] not in ("lifted", "success_end"):
                continue
            seg = next(f["segment"] for f in fs if f["turn"] == x["turn"])
            seg_start = min(f["turn"] for f in fs if f["segment"] == seg)
            # only estimates made while the object was where it was grasped (not pushed in between)
            cands = [r for r in est if r["obj"] == x["obj"] and math.dist(r["gt"], x["obj_pos"]) < 1.0]
            s = [r for r in cands if r["turn"] == seg_start]
            f0 = [r for r in cands if r["turn"] == 1]
            if s:
                seg_first.append(math.hypot(s[0]["est"][0] - x["tcp"][0], s[0]["est"][1] - x["tcp"][1]))
            if f0:
                ep_first.append(math.hypot(f0[0]["est"][0] - x["tcp"][0], f0[0]["est"][1] - x["tcp"][1]))
    for label, v in (("segment-start estimate", seg_first), ("turn-1 estimate", ep_first)):
        if v:
            print(f"{label} vs xy of a close that lifted the object: n={len(v)}, median {np.median(v):.1f} cm, "
                  f"p90 {q(v, 90):.1f} cm, within 2 cm {pct(sum(d <= 2 for d in v), len(v))}")


def section_e(eps, feats):
    print("\n## E. Corrections")
    allf = [f for fs in feats for f in fs]
    n = len(allf)
    rev_axes, rev_by_cat, ratios = Counter(), Counter(), []
    xy_rev = []
    for fs in feats:
        for i in range(1, len(fs)):
            p, f = fs[i - 1], fs[i]
            hit_xy = False
            for a in set(p["arms"]) & set(f["arms"]):
                if p["hold0"][a] != f["hold0"][a]:
                    continue
                for k, ax in enumerate("xyz"):
                    u, v = p["net"][a][k], f["net"][a][k]
                    if abs(u) >= 0.5 and abs(v) >= 0.5 and u * v < 0:
                        rev_axes[ax] += 1
                        if ax != "z":
                            hit_xy = True
                            ratios.append(abs(v) / abs(u))
            if hit_xy:
                xy_rev.append(f)
                rev_by_cat[f["category"]] += 1
    cat_n = Counter(f["category"] for f in allf)
    ratios = np.array(ratios)
    print(f"axis reversals vs the previous turn (same arm, same holding status, both |net| >= 0.5 cm): by axis "
          f"{dict(rev_axes)}")
    print(f"turns with an x/y reversal: {len(xy_rev)} ({pct(len(xy_rev), n)}), model time "
          f"{pct(sum(f['latency'] for f in xy_rev), sum(f['latency'] for f in allf))}; share within category: "
          + ", ".join(f"{k} {pct(rev_by_cat[k], cat_n[k])}" for k in ("new_subgoal", "retry_after_failure", "continuation"))
          + f"; reversal size / previous move: median {np.median(ratios):.2f}, >= 0.8 (near-undo) "
          f"{pct((ratios >= 0.8).sum(), len(ratios))}")
    # fine corrections before each close
    rows, fine_counts, fine_lat = [], [], []
    for e, fs in zip(eps, feats):
        if e["task"] in PRESS_TASKS:
            continue
        for i, f in enumerate(fs):
            if not f["closes"] or any(f["hold0"][c["arm"]] for c, _ in f["closes"]):
                continue
            k, cnt, lat = i - 1, 0, 0.0
            while k >= 0 and fs[k]["segment"] == f["segment"] and fs[k]["phase"] == "align_fine":
                cnt += 1
                lat += fs[k]["latency"]
                k -= 1
            fine_counts.append(cnt)
            fine_lat.append(lat)
    fc = Counter(fine_counts)
    table("fine-correction turns immediately before a close (grasp tasks)", ["fine turns", "closes", "share"],
          [[k, v, pct(v, len(fine_counts))] for k, v in sorted(fc.items())],
          note=f"model time in these turns: {sum(fine_lat) / 60:.1f} min")
    # blind closes: the close follows a move in the same turn, so no image of the pre-close pose was seen
    blind = Counter()
    for e in eps:
        if e["task"] in PRESS_TASKS:
            continue
        by_turn = {x["turn"]: x["outcome"] for x in grasp_events(e)}
        for t in e["turns"]:
            if t["turn"] not in by_turn:
                continue
            cmds = [parse_cmd(c) for c in t["commands"]]
            idx = next(i for i, c in enumerate(cmds) if c["kind"] == "gripper" and c["value"] < 0.5)
            moved = any(c["kind"] in ("move", "rotate", "point", "home") for c in cmds[:idx])
            blind[("after_move" if moved else "close_first", by_turn[t["turn"]])] += 1
    nb = sum(blind.values())
    rows = []
    for k in ("after_move", "close_first"):
        tot_k = sum(v for (a, _), v in blind.items() if a == k)
        rows.append([k, tot_k, pct(tot_k, nb), pct(blind[(k, "lifted")] + blind[(k, "success_end")], tot_k),
                     pct(blind[(k, "empty")], tot_k)])
    table("closes issued after a move in the same turn (pre-close pose never observed)",
          ["close position in turn", "closes", "share", "lifted", "empty"], rows)
    gt = [f for f in allf if f["phase"] == "grasp"]
    print(f"grasp-attempt turns with a move >= {FINE_CM:.0f} cm: {pct(sum(1 for f in gt if any(s >= FINE_CM for s in f['move_sizes'])), len(gt))}; "
          f"turns with a failed motion: {sum(1 for f in allf if f['n_failed'])} ({pct(sum(1 for f in allf if f['n_failed']), n)})")
    # retries: closes per object-acquisition
    ex = [f for f in allf if f["empty_close"]]
    print(f"empty closes (fingers closed fully), all tasks: {len(ex)}")
    small_turn = sum(1 for f in allf if f["move_sizes"] and all(s < FINE_CM for s in f["move_sizes"]))
    print(f"turns whose moves are all < {FINE_CM:.0f} cm: {small_turn} ({pct(small_turn, n)}), model time "
          f"{pct(sum(f['latency'] for f in allf if f['move_sizes'] and all(s < FINE_CM for s in f['move_sizes'])), sum(f['latency'] for f in allf))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--dump", help="write per-turn features (jsonl) for later offline tests")
    args = ap.parse_args()
    eps = load_episodes(args.root)
    feats = [turn_features(e) for e in eps]
    n_turns = sum(len(e["turns"]) for e in eps)
    print(f"# RoboDawn trace analysis: {len(eps)} episodes, {n_turns} turns, "
          f"success {sum(e['success'] for e in eps)}/{len(eps)}")
    section_a(eps, feats)
    section_b(eps, feats)
    section_c(eps, feats)
    section_d(eps, feats)
    section_e(eps, feats)
    if args.dump:
        with open(args.dump, "w") as fh:
            for e, fs in zip(eps, feats):
                for f in fs:
                    rec = {k: v for k, v in f.items() if k not in ("closes", "opens")}
                    rec.update(task=e["task"], episode=e["index"], ep_success=e["success"],
                               n_close=len(f["closes"]), n_open=len(f["opens"]))
                    fh.write(json.dumps(rec) + "\n")
        print(f"\nper-turn features written to {args.dump}")


if __name__ == "__main__":
    main()
