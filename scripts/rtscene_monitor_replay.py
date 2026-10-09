#!/usr/bin/env python3
"""The harness's object monitor (branchlab.rtscene.monitor), run offline on the replayed episodes exactly as the
harness runs it, to see whether it would end hopeless episodes early and never one that went on to succeed.

Per episode, in the harness's order:
  turn 1    the monitor binds the coordinates in Gemini's own first reply (`scene`, from the recorded trace)
            to raised blobs of the turn-1 capture;
  turn t>1  it checks the bound objects at the turn-start capture (the harness runs this beside the model call);
            then every gripper command of turn t is reported as the harness reports it (a close that caught
            something, an open), from the replay's command log.
The monitor's ObjectMonitor, parser and Clef interfaces are the harness's own; Clef answers are cached.
Segmentation and true poses are read only for the labels: which task object each bound blob is, whether the
picked blob is that object, and the object's true tilt.

    PYTHONPATH=src:scripts python scripts/rtscene_monitor_replay.py run [--runs eval_baseline ...] [--procs 4]
    PYTHONPATH=src:scripts python scripts/rtscene_monitor_replay.py report
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT/"outputs/analysis/monitor_replay"
ROOTS = [PROJECT/"outputs/replay_local/replay", *sorted((PROJECT/"outputs/robodawn_colab").glob("*/replay"))]
COLAB = PROJECT/"outputs/robodawn_colab"
SKIP_TASKS = {"open_laptop", "adjust_bottle"}      # the harness's MONITOR_SKIP_TASKS
EMPTY_CLOSE_OPENING = 0.08


def trace_path(meta: dict) -> Path | None:
    """The recorded trace of the replayed episode (its turn-1 reply holds Gemini's object coordinates)."""
    k = meta["episode"]
    if meta["run"] == "eval_baseline":
        pats = [f"*/robodawn/gemini_flash_flex/{meta['task']}/shard_*/episode_{k:03d}/trace.json"]
    else:
        pats = [f"*/robodawn_dev/{meta['run']}/{meta['task']}/shard_*/episode_{k:03d}/trace.json"]
    for pat in pats:
        found = sorted(COLAB.glob(pat))
        if found:
            return found[0]
    return None


class CachedAsk:
    """Clef on Workers AI with answers cached by content (state, questions, images)."""

    def __init__(self):
        from branchlab.clef import Clef
        self.clef = Clef(cache_path=OUT/"responses.jsonl", backend="workers")

    def __call__(self, state, questions, images):
        h = hashlib.sha1(json.dumps([state, questions]).encode())
        for uri in images:
            h.update(uri.encode())
        return self.clef.ask(state, questions, images, key=h.hexdigest())["answers"]


NO_CLEF = False          # set by `run --no-clef`: the code-only ablation (nearest blob, shape score, height veto)


def run_episode(ep_dir: str) -> list[dict]:
    from branchlab.rtscene import labels, monitor as mon, robot_model
    from branchlab.rtscene.capture import Episode
    ep = Episode(Path(ep_dir))
    if ep.task in SKIP_TASKS:
        return []
    tp = trace_path(ep.meta)
    if tp is None:
        return [{"key": ep.key, "run": ep.meta["run"], "task": ep.task, "episode_end": True, "error": "no trace"}]
    trace = json.loads(tp.read_text())
    trace = trace if isinstance(trace, list) else trace["trace"]
    scenes = {}
    for rec in trace:                                  # the model's own replies (relay steps have none)
        if rec.get("source", "model") == "model" and rec["turn"] not in scenes:
            scenes[rec["turn"]] = str(rec.get("scene") or "")

    # the parser's blobs per capture, cached on disk (the parse is the slow part), and their true actors
    cache_file = OUT/"parse"/f"{ep.key.replace('/', '_')}.pkl"
    cache = pickle.loads(cache_file.read_bytes()) if cache_file.exists() else {}
    real_parse = mon.parse
    parsed_now = {}

    def parse(cap, boxes):
        if cap.idx not in cache:
            cache[cap.idx] = real_parse(cap, boxes)
        parsed_now[cap.idx] = cache[cap.idx]
        return cache[cap.idx]
    mon.parse = parse

    boxes = robot_model.load()
    monitor = (mon.ObjectMonitor(None, boxes, policy="code", need=2) if NO_CLEF
               else mon.ObjectMonitor(CachedAsk(), boxes, policy="both", need=2))
    caps = {c.idx: c for c in ep.captures()}
    turn_caps = {c.turn: c for c in caps.values() if c.kind == "turn"}
    ids = {r: v for r, v in labels.object_ids(ep).items() if v}
    role_of = {i: r for r, v in ids.items() for i in v}
    bound, roles, bound_turn = [], [], None
    log = {(c["turn"], c["cmd_index"]): c for c in ep.meta["command_log"]}
    rows = []
    for t in sorted(turn_caps):
        cap = turn_caps[t]
        if not monitor.objects:
            # as the harness: bind on this turn's start view once this turn's reply gives coordinates
            bound = monitor.bind(cap, mon.coordinates(scenes.get(t, "")))
            if monitor.objects:
                bound_turn = t
                act0 = labels.blob_actors(ep, cap, parsed_now[cap.idx])
                roles = [role_of.get(act0[o.ref.id][0]) if act0[o.ref.id][1] >= 0.5 else None for o in monitor.objects]
        elif monitor.objects:
            out = monitor.check(cap)
            acts = labels.blob_actors(ep, cap, parsed_now[cap.idx]) if cap.idx in parsed_now else {}
            for obj, role, row in zip(monitor.objects, roles, out):
                truth = labels.object_state(ep, cap, role) if role else None
                picked = obj.picked if row.get("status") == "seen" and row.get("pick") != "none" else None
                rows.append({"key": ep.key, "run": ep.meta["run"], "task": ep.task, "turn": t, "capture": cap.idx,
                             "role": role, **{k: v for k, v in row.items() if k != "at"}, "at": list(obj.at),
                             "pick_right": (picked is not None and role is not None and acts.get(picked.id, (None, 0))[0]
                                            in ids.get(role, []) and acts[picked.id][1] >= 0.5),
                             **({"true_tilt": round(truth["tilt"], 1), "true_dz": round(truth["dz"], 1),
                                 "off_table": truth["z"] < cap.table_z - 5} if truth else {})})
        # the turn's gripper commands, as the harness reports them to the monitor
        k = 0
        while (t, k) in log:
            c = log[(t, k)]
            parts = str(c.get("command") or "").split()
            if len(parts) == 3 and parts[1] == "gripper" and c.get("ok"):
                arm, value = parts[0], float(parts[2])
                after = (c.get("arms") or {}).get(arm) or {}
                if after.get("position_cm"):
                    if value < 0.5:
                        monitor.note_close(arm, after["position_cm"], after.get("gripper_real", 0.0) > EMPTY_CLOSE_OPENING)
                    else:
                        monitor.note_open(arm, after["position_cm"])
            k += 1
    mon.parse = real_parse
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(pickle.dumps(cache))
    rows.append({"key": ep.key, "run": ep.meta["run"], "task": ep.task, "episode_end": True,
                 "success": ep.meta["success"], "replay_success": ep.meta["replay_success"],
                 "turns": max(turn_caps), "bound": bound, "bound_roles": roles, "bound_turn": bound_turn,
                 "clef_calls": monitor.calls, "points": len(mon.coordinates(scenes.get(1, "")))})
    return rows


def run(runs, procs, rows_name="rows.jsonl"):
    OUT.mkdir(parents=True, exist_ok=True)
    dirs = {}
    for root in ROOTS:
        for meta in sorted(Path(root).glob("*/*/episode_*/meta.json")) if root.is_dir() else []:
            m = json.loads(meta.read_text())
            if m["run"] in runs:
                dirs.setdefault(m["key"], str(meta.parent))
    out = OUT/rows_name
    done = {json.loads(line)["key"] for line in open(out) if '"episode_end"' in line} if out.exists() else set()
    todo = [d for k, d in sorted(dirs.items()) if k not in done]
    print(f"{len(dirs)} episodes, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(procs) as pool, open(out, "a") as fh:
        for rows in pool.map(run_episode, todo):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            if rows:
                print(rows[-1]["key"], rows[-1].get("error", ""), flush=True)


def report(code_thr: float, rows_name: str = "rows.jsonl"):
    rows = [json.loads(line) for line in open(OUT/rows_name)]
    ends = {r["key"]: r for r in rows if r.get("episode_end") and "error" not in r}
    checks = [r for r in rows if not r.get("episode_end")]
    seen = [r for r in checks if r.get("status") == "seen" and r.get("pick") != "none" and "code" in r]
    n_roles = collections.Counter(bool(x) for e in ends.values() for x in e["bound_roles"])
    print(f"{len(ends)} episodes; bound objects {sum(len(e['bound']) for e in ends.values())} "
          f"(a task object {n_roles[True]}, other {n_roles[False]}); episodes with none bound "
          f"{sum(not e['bound'] for e in ends.values())}")
    print(f"checks {len(checks)}: {dict(collections.Counter(r['status'] for r in checks))}; "
          f"pick right {np.mean([r['pick_right'] for r in seen]):.0%} of {len(seen)}; "
          f"Clef calls {sum(e['clef_calls'] for e in ends.values())}")
    def agree(r):                       # Clef's own choice is the picked tile (the monitor's gate)
        return r.get("clef_choice") == r["pick"]
    policies = {
        "code": lambda r: agree(r) and r["code"] > code_thr,
        "clef": lambda r: agree(r) and (r.get("clef") or 0) >= 0.5,
        "code & clef (monitor)": lambda r: agree(r) and r["code"] > code_thr and (r.get("clef") or 0) >= 0.5 and not r.get("upright"),
        "code & clef, no height veto": lambda r: agree(r) and r["code"] > code_thr and (r.get("clef") or 0) >= 0.5,
        "code, height veto": lambda r: agree(r) and r["code"] > code_thr and not r.get("upright"),
        "clef, height veto": lambda r: agree(r) and (r.get("clef") or 0) >= 0.5 and not r.get("upright"),
        "code & clef, any pick": lambda r: r["code"] > code_thr and (r.get("clef") or 0) >= 0.5,
        "code & clef, Clef sure": lambda r: agree(r) and r["code"] > code_thr and (r.get("clef") or 0) >= 0.5 and r.get("clef_p", 0) >= 0.8,
    }
    by_ep = collections.defaultdict(list)
    for r in seen:
        by_ep[r["key"]].append(r)
    groups = {"all": list(ends.values())}
    for e in ends.values():
        groups.setdefault(e["run"], []).append(e)
    for group, eps in groups.items():
        total = sum(e["turns"] for e in eps)
        print(f"== {group}: {len(eps)} episodes, replay successes {sum(e['replay_success'] for e in eps)}, turns {total}")
        for name, flag in policies.items():
            for need in (1, 2, 3):
                false_stop, stops, saved, true_fall = 0, 0, 0, 0
                for e in eps:
                    streak = collections.Counter()
                    stop = None
                    for r in sorted(by_ep.get(e["key"], []), key=lambda r: r["turn"]):
                        key = tuple(r["at"])
                        streak[key] = streak[key] + 1 if flag(r) else 0
                        if streak[key] >= need:
                            stop = r
                            break
                    if stop is None:
                        continue
                    stops += 1
                    true_fall += bool(stop.get("off_table") or (stop.get("true_tilt", 0) > 60))
                    if e["replay_success"]:
                        false_stop += 1
                    else:
                        saved += e["turns"] - stop["turn"]      # the stop turn's call is made (beside the check)
                print(f"  {name:24} x{need}: stops {stops:3} (object really down {true_fall:2}), "
                      f"of successful episodes {false_stop:2}, turns saved {saved:4} ({saved / max(total, 1):.0%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs", nargs="+", default=["eval_baseline", "relay_baseline_m16000_standard",
                                                 "relay_open_m16000_standard", "relay_checked_lite_m16000_standard",
                                                 "relay_checked_clef_m16000_standard"])
    r.add_argument("--procs", type=int, default=4)
    r.add_argument("--no-clef", action="store_true", help="the code-only ablation (rows_noclef.jsonl)")
    p = sub.add_parser("report")
    p.add_argument("--code-thr", type=float, default=2.34)
    p.add_argument("--no-clef", action="store_true")
    args = parser.parse_args()
    name = "rows_noclef.jsonl" if args.no_clef else "rows.jsonl"
    if args.step == "run":
        global NO_CLEF
        NO_CLEF = args.no_clef
        run(set(args.runs), args.procs, name)
    else:
        report(args.code_thr, name)


if __name__ == "__main__":
    main()
