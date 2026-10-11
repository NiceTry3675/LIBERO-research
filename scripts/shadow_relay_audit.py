#!/usr/bin/env python3
"""Shadow audit of a one-packet relay (Oracle's proposal of 2026-10-11): can Gemini leave a next-turn continuation
that code and Clef can safely authorize, and does Clef widen the safe coverage over code?

Offline, on the replayed episodes of the 8 new tasks (outputs/robodawn_colab/rt8d/replay). A window is a source turn t
whose replay matches the original run at t and t+1, so the replay's views are what Gemini saw and the recorded turn t+1
is the true successor of the recorded commands. A window is clean when, in addition, t+1 comes before the episode's
first divergence: the first turn whose state differs (fingertips over 1 cm, commanded or real gripper opening over 0.1,
a recorded object over 1.5 cm or 15 degrees) or whose outcome differs (an ok flag, or a close that caught something in
one run and nothing in the other); objects the state does not list (breads, bottles) are covered by this prefix rule.
The statistics are given on the clean windows.

Per window, Gemini (unchanged settings: Vertex standard, default reasoning, 8000 reply tokens) is asked:
  P0   the unchanged prompt on the source observation: the run's system prompt and demonstration (rebuilt with the
       harness's own functions), the replay's turn-t views and the recorded turn-t text
  P1   the same with one format delta, an optional "continuation" packet (next-turn commands plus conditions), to be
       left out when the next step depends on what Gemini will see
  P1b  the same packet, asked for whenever Gemini can already say what it would send next (how often it can)
  N1   the unchanged prompt at t+1: a second sample of Gemini's next decision (the recorded reply is the first)

A packet is judged at t+1 only when its reply's own commands equal the recorded ones within 1 cm / 5 degrees
(resolved; otherwise the recorded successor is not theirs, and the window stays in the denominator as unresolved). The
gate requires every command of turn t to have run ok plus the packet's conditions. Code checks them; object_unmoved,
which needs to know which depth blob is the object, is decided three ways:
  code   the nearest tall-enough blob, abstaining when another is nearly as close
  clef   the object monitor's re-identification (Clef picks among numbered top-camera tiles, fused with distance)
  truth  the simulator pose of the recorded object nearest the packet's coordinate (offline upper bound; unknown
         when no recorded object is there)
code and clef abstain while a fingertip is within 10 cm of the object (the top view is covered), as the monitor does.

The packet is compared with Gemini's next decision (recorded and N1), arm by arm and command by command:
  full      it equals the whole next decision: a skipped call
  partial   for its arms it equals the next decision up to some command: safe, but Gemini is still needed for the rest
  critical  at the best-aligned prefix an event differs in kind, arm or gripper action, or a motion by over 3 cm /
            15 degrees, or the packet has a further gripper event or large motion
  conflict  any other difference
An accepted packet that is full against either reference is a safe skip; an accepted packet that is not a safe skip
and is critical against either is a critical authorization. Gemini against itself (N1 vs the recorded next decision)
is the reference for these rates.
A one-packet relay cannot skip two calls in a row, so a per-call skip rate p removes p / (1 + p) of the calls.

    PYTHONPATH=src:scripts python scripts/shadow_relay_audit.py windows [--n 60]
    PYTHONPATH=src:scripts python scripts/shadow_relay_audit.py query [--variants P0 P1 P1b N1] [--workers 6]
    PYTHONPATH=src:scripts python scripts/shadow_relay_audit.py gate
    PYTHONPATH=src:scripts python scripts/shadow_relay_audit.py report
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import io
import json
import math
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT/"third_party/robodawn"
REPLAY = PROJECT/"outputs/robodawn_colab/rt8d/replay"
COLAB = PROJECT/"outputs/robodawn_colab"
OUT = PROJECT/"outputs/analysis/shadow_relay"
KEY = PROJECT/"gemini0614-499409-42e14e861064.json"          # the Vertex service account key (path only)
VIEWS = ("agent_camera", "top_camera", "left_camera", "right_camera")   # the order of the harness's views
TASKS = ("pick_diverse_bottles", "place_object_scale", "rotate_qrcode", "scan_object", "place_bread_basket",
         "place_dual_shoes", "place_shoe", "put_bottles_dustbin")
PRESS_TASKS = {"click_bell", "click_alarmclock", "press_stapler", "turn_switch"}
FINGER_TOL, OBJECT_TOL, OBJECT_ROT, GRIP_TOL = 1.0, 1.5, 15.0, 0.1   # replay matches the original run
MOVE_TOL, ROT_TOL = 1.5, 10.0              # cm, deg: two motions are equal (per arm; capped at the critical level)
CRIT_MOVE, CRIT_ROT = 3.0, 15.0            # a difference beyond these (or in a gripper action) is critical
STRICT_MOVE, STRICT_ROT = 1.0, 5.0         # a reply's commands equal the recorded ones (resolved)
UNMOVED_XY, UNMOVED_H, UNMOVED_ROT = 2.0, 1.5, 10.0   # object_unmoved (cm, cm, deg)
AMBIGUOUS = 2.0                            # cm: code abstains when the second candidate is this much farther or less
CONDITIONS = ("object_unmoved", "object_held", "gripper_open")
PACKET_VARIANTS = ("P1", "P1b")

CONTINUATION_FIELD = ('  "continuation": {"object": "<object>", "if": ["<condition>", ...], "commands": ["<command>", ...]}'
                      '   // optional, see CONTINUATION below\n')
CONTINUATION_NOTE = (
    "CONTINUATION (optional): if you already know the commands you would send on the next turn, assuming every "
    "command above runs ok and the scene then looks as you expect, give them in \"continuation\": \"object\" is the "
    "object those next commands act on, with its (x, y) in cm; \"if\" lists the conditions that must hold; "
    "\"commands\" are 1 to 4 commands in the same grammar. The harness may then run them without asking you, but only "
    "when every command of this turn ran ok and every listed condition holds after this turn's commands; otherwise you "
    "are asked as usual. You see their results on your next turn either way. Conditions to choose from: "
    "\"object_unmoved\" (that object is still where it was at the start of this turn, within 2 cm), \"object_held\" "
    "(the arm of the next commands is closed on something and still holding it), \"gripper_open\" (that arm's gripper "
    "is open and empty). Leave \"continuation\" out when the next step depends on what you will see, for example a "
    "fine adjustment before closing.\n")
_P1_END = ("Leave \"continuation\" out when the next step depends on what you will see, for example a "
           "fine adjustment before closing.\n")
_P1B_END = "Give a continuation whenever you can already say what you would send next; leave it out only when you cannot.\n"
CONTINUATION_NOTE_B = CONTINUATION_NOTE.replace(_P1_END, _P1B_END)
assert CONTINUATION_NOTE_B != CONTINUATION_NOTE


def setup_paths():
    for p in (PROJECT/"src", PROJECT/"scripts", REPO):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


# ----------------------------------------------------------------------------- the recorded runs
def trace_dir(task: str, episode: int) -> Path | None:
    for tier in ("standard", "flex"):
        found = sorted(COLAB.glob(f"*/robodawn/gemini_flash_{tier}/{task}/shard_*/episode_{episode:03d}/trace.json"))
        if found:
            return found[0].parent
    return None


def load_trace(tdir: Path) -> dict:
    return {r["turn"]: r for r in json.loads((tdir/"trace.json").read_text()) if "turn" in r}


def call_type(rec: dict, task: str) -> str:
    """The kind of the call that follows turn `rec`: retry after a failure, a new sub-goal after a gripper event,
    else a continuation (as in the harness analysis)."""
    cmds = [c.lower() for c in rec.get("commands") or []]
    res = rec.get("results") or []
    empty = any(r.get("kind") == "gripper" and str(r.get("note", "")).startswith("fingers closed fully") for r in res)
    if not cmds or rec.get("error") or rec.get("parse_error") \
            or any(r.get("ok") is False and r.get("kind") != "done" for r in res) \
            or (empty and task not in PRESS_TASKS) or any(" home" in c for c in cmds) or rec.get("command_errors"):
        return "retry"
    if any(" gripper " in c for c in cmds):
        return "new"
    return "continuation"


def quat_angle(q1, q2) -> float:
    d = abs(sum(a * b for a, b in zip(q1, q2)))
    return math.degrees(2 * math.acos(min(1.0, d)))


def state_match(rec_state: dict, cap_state: dict) -> bool:
    for arm in ("left", "right"):
        r, c = rec_state[arm], cap_state[arm]
        if math.dist(r["position_cm"], c["position_cm"]) > FINGER_TOL:
            return False
        if abs(r["gripper"] - c["gripper"]) > GRIP_TOL or abs(r.get("gripper_real", 0) - c.get("gripper_real", 0)) > GRIP_TOL:
            return False
    ro, co = rec_state.get("objects") or {}, cap_state.get("objects") or {}
    for name, o in ro.items():
        if name in ("wall", "table") or name not in co or "position_cm" not in o:
            continue
        if math.dist(o["position_cm"], co[name]["position_cm"]) > OBJECT_TOL:
            return False
        if "quat_wxyz" in o and "quat_wxyz" in co[name] and quat_angle(o["quat_wxyz"], co[name]["quat_wxyz"]) > OBJECT_ROT:
            return False
    return True


def _outcome(r: dict) -> tuple:
    note = str(r.get("note") or "")
    grasp = "empty" if note.startswith("fingers closed fully") else "caught" if note.startswith("fingers stopped") else None
    return bool(r.get("ok")), grasp


def first_divergence(meta: dict, trace: dict) -> tuple[float, float]:
    """(first turn whose state differs, first turn whose outcome differs) between the original run and its replay."""
    caps = {c["turn"]: c for c in meta["captures"] if c["kind"] == "turn"}
    log = collections.defaultdict(dict)
    for c in meta["command_log"]:
        log[c["turn"]][c["cmd_index"]] = c
    d_state = d_outcome = math.inf
    for t in sorted(trace):
        if d_state == math.inf and (t not in caps or not state_match(trace[t]["state"], caps[t]["state"])):
            d_state = t
        if d_outcome == math.inf:
            executed = [r for r in trace[t].get("results") or [] if r.get("kind") not in ("invalid", "done")]
            for k, r in enumerate(executed):
                rep = log[t].get(k)
                if rep is None or _outcome(r) != _outcome(rep):
                    d_outcome = t
                    break
    return d_state, d_outcome


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


# ----------------------------------------------------------------------------- windows
def candidates() -> list[dict]:
    """Source turns whose state matches at t and t+1 (the first, looser selection rule), each with its clean flag."""
    out = []
    for meta_path in sorted(REPLAY.glob("eval_baseline/*/episode_*/meta.json")):
        meta = json.loads(meta_path.read_text())
        tdir = trace_dir(meta["task"], meta["episode"])
        if tdir is None:
            continue
        trace = load_trace(tdir)
        caps = {c["turn"]: c for c in meta["captures"] if c["kind"] == "turn"}
        d_state, d_outcome = first_divergence(meta, trace)
        for t, rec in sorted(trace.items()):
            nxt = trace.get(t + 1)
            if nxt is None or t not in caps or t + 1 not in caps:
                continue
            if rec.get("error") or rec.get("parse_error") or nxt.get("error") or nxt.get("parse_error"):
                continue
            if not rec.get("commands") or not nxt.get("commands"):
                continue
            if not (loose_match(rec["state"], caps[t]["state"]) and loose_match(nxt["state"], caps[t + 1]["state"])):
                continue
            wid = f"{meta['task']}/{meta['episode']}/t{t}"
            out.append({"id": wid, "task": meta["task"], "episode": meta["episode"], "turn": t,
                        "replay": str(meta_path.parent.relative_to(PROJECT)), "trace": str(tdir.relative_to(PROJECT)),
                        "cap": caps[t]["idx"], "cap_next": caps[t + 1]["idx"], "call_type": call_type(rec, meta["task"]),
                        "success": bool(meta["success"]), "clean": t + 1 < d_state and t < d_outcome})
    return out


def loose_match(rec_state: dict, cap_state: dict) -> bool:
    """The selection rule the windows were picked with: fingertips, commanded grippers, recorded object positions."""
    for arm in ("left", "right"):
        if math.dist(rec_state[arm]["position_cm"], cap_state[arm]["position_cm"]) > FINGER_TOL:
            return False
        if abs(rec_state[arm]["gripper"] - cap_state[arm]["gripper"]) > GRIP_TOL:
            return False
    ro, co = rec_state.get("objects") or {}, cap_state.get("objects") or {}
    return all(math.dist(o["position_cm"], co[n]["position_cm"]) <= OBJECT_TOL for n, o in ro.items()
               if n not in ("wall", "table") and n in co and "position_cm" in o)


def pick_windows(cands: list[dict], n: int) -> list[dict]:
    """Per task: continuation 4, new 2, retry 1 (then more continuations), alternating successful and failed
    episodes, at most two windows per episode; order within a group by hash, so the pick is reproducible."""
    quota = {"continuation": 4, "new": 2, "retry": 1}
    by = collections.defaultdict(list)
    for c in cands:
        by[(c["task"], c["call_type"])].append(c)
    chosen, per_ep = [], collections.Counter()

    def take(task, kind, k):
        pool = sorted(by[(task, kind)], key=lambda c: h(c["id"]))
        got = 0
        for want_success in [True, False] * len(pool):
            if got >= k:
                break
            nxt = next((c for c in pool if c["success"] == want_success and c not in chosen
                        and per_ep[(c["task"], c["episode"])] < 2), None)
            nxt = nxt or next((c for c in pool if c not in chosen and per_ep[(c["task"], c["episode"])] < 2), None)
            if nxt is None:
                break
            chosen.append(nxt)
            per_ep[(nxt["task"], nxt["episode"])] += 1
            got += 1
        return got

    tasks = sorted({c["task"] for c in cands})
    for task in tasks:
        for kind, k in quota.items():
            take(task, kind, k)
    for task in sorted(tasks, key=lambda t: -len(by[(t, "continuation")])):
        if len(chosen) >= n:
            break
        take(task, "continuation", 1)
    return chosen[:n]


def population() -> collections.Counter:
    """Every Gemini call of the 8 tasks' original evaluation runs, by kind: the episode's first call, else the kind
    the previous turn makes it (call_type)."""
    pop = collections.Counter()
    for task in TASKS:
        for ep in range(10):
            tdir = trace_dir(task, ep)
            if tdir is None:
                continue
            trace = load_trace(tdir)
            turns = sorted(trace)
            pop["first"] += 1
            for a, b in zip(turns, turns[1:]):
                pop[call_type(trace[a], task)] += 1
    return pop


# ----------------------------------------------------------------------------- requests
_demo_cache = {}
_lock = threading.Lock()


def demo_msgs(entry: str) -> list[dict]:
    """The primer plus the bank entry shown in the episode, as the harness renders them (demos.demo_messages)."""
    setup_paths()
    from harness.agent.demos import build_demo, demo_messages, load_saved_demo
    with _lock:
        if entry not in _demo_cache:
            primer = build_demo(REPO/"demos/robotwin2/primer")
            _demo_cache[entry] = demo_messages([primer, load_saved_demo(REPO/"demos/robotwin2/expert"/entry)])
        return _demo_cache[entry]


def packet_system(system: str, variant: str) -> str:
    """The format delta: the continuation field in the response format and its note before the length rule. P1b
    also adds the comma after the commands line that P1's template lacked."""
    if variant == "P1b":
        line = '["<command>", ...]   // 1 to'
        assert system.count(line) == 1
        system = system.replace(line, '["<command>", ...],   // 1 to')
    field_at = system.index("}\nKeep every text field short")
    out = system[:field_at] + CONTINUATION_FIELD + system[field_at:]
    rule = out.index("Keep every text field short")
    return out[:rule] + (CONTINUATION_NOTE if variant == "P1" else CONTINUATION_NOTE_B) + out[rule:]


def png_part(path: Path) -> dict:
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def request(w: dict, variant: str) -> list[dict]:
    tdir = PROJECT/w["trace"]
    trace = load_trace(tdir)
    entry = Path(trace[min(trace)]["demo"][0]).name
    system = (tdir/"system_prompt.txt").read_text()
    if variant in PACKET_VARIANTS:
        system = packet_system(system, variant)
    turn, cap = (w["turn"], w["cap"]) if variant != "N1" else (w["turn"] + 1, w["cap_next"])
    images = [png_part(PROJECT/w["replay"]/"images"/f"c{cap:03d}_{v}.jpg") for v in VIEWS]
    return ([{"role": "system", "content": system}] + demo_msgs(entry)
            + [{"role": "user", "content": images + [{"type": "text", "text": trace[turn]["prompt"]}]}])


# ----------------------------------------------------------------------------- Gemini
_local = threading.local()


def client():
    if not hasattr(_local, "client"):
        setup_paths()
        import run_robodawn_baseline as rb
        from harness.agent.llm_client import ChatClient
        cls = rb.vertex_client(ChatClient, KEY, "standard")
        _local.client = cls(model=rb.MODEL, base_url=rb.api_base(rb.credentials_project(KEY)), max_tokens=8000,
                            temperature=None, timeout_s=300, log_path=OUT/"gemini"/"llm_calls.jsonl")
    return _local.client


def ask_gemini(job):
    w, variant = job
    r = client().chat(request(w, variant), tag=f"{w['id']}|{variant}")
    return {"key": f"{w['id']}|{variant}", "id": w["id"], "variant": variant, "text": r.text, "error": r.error,
            "latency_s": round(r.latency_s, 2), "usage": r.usage, "attempts": r.attempts}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


# ----------------------------------------------------------------------------- parsing and comparison
def parse_reply(text: str) -> dict:
    """{ok, commands, errors, packet: {raw, valid, object, conditions, commands, problems}}"""
    setup_paths()
    from harness.agent.prompts import parse_agent_json
    from harness.core.commands import parse_command_list
    try:
        js = parse_agent_json(text or "")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200], "commands": [], "errors": [], "packet": None}
    raw = js.get("commands") or []
    raw = [raw] if isinstance(raw, str) else raw
    cmds, errs = parse_command_list(raw)
    out = {"ok": True, "commands": [c.text() for c in cmds[:4]], "errors": errs, "packet": None}
    cont = js.get("continuation")
    if cont:
        if not isinstance(cont, dict):
            out["packet"] = {"raw": cont, "valid": False, "problems": ["not an object"]}
            return out
        problems = []
        pc_raw = cont.get("commands") or []
        pc_raw = [pc_raw] if isinstance(pc_raw, str) else pc_raw
        pcmds, perrs = parse_command_list(pc_raw)
        conds = cont.get("if") or []
        conds = [conds] if isinstance(conds, str) else conds
        if perrs:
            problems.append(f"commands: {perrs}")
        if not pcmds:
            problems.append("no commands")
        if len(pcmds) > 4:
            problems.append("more than 4 commands")
        if any(c.kind in ("wait", "done") for c in pcmds):
            problems.append("wait/done")
        unknown = [c for c in conds if c not in CONDITIONS]
        if unknown:
            problems.append(f"unknown conditions: {unknown}")
        out["packet"] = {"raw": cont, "object": str(cont.get("object") or ""), "conditions": [c for c in conds if c in CONDITIONS],
                         "commands": [c.text() for c in pcmds[:4]], "valid": not problems, "problems": problems}
    return out


def malformed(rep: dict | None) -> bool | None:
    """A reply that failed, did not parse, held no command or a command error."""
    if rep is None:
        return None
    if rep.get("error"):
        return True
    pr = parse_reply(rep["text"])
    return not pr["ok"] or not pr["commands"] or bool(pr["errors"])


def events(cmds: list[str]) -> list[tuple]:
    """Net effect: runs of moves and rotations summed per arm, then each gripper/point/home/wait/done event."""
    setup_paths()
    from harness.core.commands import parse_command_list
    parsed, _ = parse_command_list(cmds)
    out, acc = [], {}

    def flush():
        if acc:
            out.append(("motion", {a: tuple(round(v, 2) for v in vec) for a, vec in sorted(acc.items())}))
            acc.clear()
    for c in parsed:
        if c.kind in ("move", "rotate"):
            vec = list(acc.get(c.arm, (0.0,) * 6))
            vec[{"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}[c.axis]] += c.value
            acc[c.arm] = tuple(vec)
        else:
            flush()
            if c.kind == "gripper":
                out.append(("gripper", c.arm, "close" if c.value < 0.5 else "open"))
            else:
                out.append((c.kind, c.arm, c.axis))
    flush()
    return out


def motion_diff(a: dict, b: dict) -> tuple[float, float]:
    """Largest translation (cm) and rotation (deg) difference over both arms."""
    dm = dr = 0.0
    for arm in set(a) | set(b):
        va, vb = a.get(arm, (0.0,) * 6), b.get(arm, (0.0,) * 6)
        dm = max(dm, math.dist(va[:3], vb[:3]))
        dr = max(dr, max(abs(x - y) for x, y in zip(va[3:], vb[3:])))
    return dm, dr


def event_equal(e: tuple, f: tuple, strict: bool = False) -> bool:
    """Equal kind and discrete content; motions per arm within 1.5 cm or a quarter of their size, capped at 3 cm, and
    10 degrees (strict: 1 cm and 5 degrees)."""
    if e[0] != f[0]:
        return False
    if e[0] != "motion":
        return e == f
    if set(e[1]) != set(f[1]):
        return False
    for arm in e[1]:
        va, vb = e[1][arm], f[1][arm]
        if strict:
            tol, rtol = STRICT_MOVE, STRICT_ROT
        else:
            tol = min(max(MOVE_TOL, 0.25 * max(math.hypot(*va[:3]), math.hypot(*vb[:3]))), CRIT_MOVE)
            rtol = ROT_TOL
        if math.dist(va[:3], vb[:3]) > tol or max(abs(x - y) for x, y in zip(va[3:], vb[3:])) > rtol:
            return False
    return True


def events_equal(p: list, n: list, strict: bool = False) -> bool:
    return len(p) == len(n) and all(event_equal(a, b, strict) for a, b in zip(p, n))


def same_commands(a: list[str], b: list[str], strict: bool = False) -> bool:
    return events_equal(events(a), events(b), strict)


def arm_of(cmd: str):
    w = cmd.split()[0].lower() if cmd.split() else ""
    return w if w in ("left", "right") else None


def _pair_cost(a: tuple, b: tuple | None) -> float:
    if b is None:
        return 50.0
    if a[0] != b[0] or (a[0] != "motion" and a != b) or (a[0] == "motion" and set(a[1]) != set(b[1])):
        return 100.0
    if a[0] == "motion":
        dm, dr = motion_diff(a[1], b[1])
        return dm + dr / 5
    return 0.0


def _critical_event(a: tuple, b: tuple | None) -> bool:
    if b is None:                                   # a packet event Gemini's next decision has nothing for
        return a[0] != "motion" or any(math.hypot(*v[:3]) > CRIT_MOVE or max(map(abs, v[3:])) > CRIT_ROT
                                       for v in a[1].values())
    if a[0] != b[0] or (a[0] != "motion" and a != b) or (a[0] == "motion" and set(a[1]) != set(b[1])):
        return True
    if a[0] == "motion":
        dm, dr = motion_diff(a[1], b[1])
        return dm > CRIT_MOVE or dr > CRIT_ROT
    return False


def compare(packet: list[str], nxt: list[str]) -> str:
    """full | partial | critical | conflict (module docstring)."""
    p = events(packet)
    if not p or not nxt:
        return "conflict"
    if events_equal(p, events(nxt)):
        return "full"
    arms = {arm_of(c) for c in packet} - {None}
    nf = [c for c in nxt if arm_of(c) in arms]
    best = None
    for j in range(len(nf) + 1):
        pre = events(nf[:j])
        if events_equal(p, pre):
            return "partial"
        cost = sum(_pair_cost(a, pre[k] if k < len(pre) else None) for k, a in enumerate(p)) + 50.0 * max(0, len(pre) - len(p))
        if best is None or cost < best[0]:
            best = (cost, pre)
    pre = best[1]
    if any(_critical_event(a, pre[k] if k < len(pre) else None) for k, a in enumerate(p)):
        return "critical"
    return "conflict"


# ----------------------------------------------------------------------------- association
class Association:
    """object_unmoved for a packet's object: bind at the source capture, find it again at the successor."""

    def __init__(self):
        setup_paths()
        from branchlab.rtscene import robot_model
        self.boxes = robot_model.load()
        self._clef = None
        self._parsed = {}

    def clef(self):
        if self._clef is None:
            from branchlab.clef import Clef
            clef = Clef(cache_path=OUT/"clef_responses.jsonl", backend="workers")

            def ask(state, questions, images):
                hh = hashlib.sha1(json.dumps([state, questions]).encode())
                for uri in images:
                    hh.update(uri.encode())
                return clef.ask(state, questions, images, key=hh.hexdigest())["answers"]
            self._clef = ask
        return self._clef

    def parse(self, ep, cap):
        from branchlab.rtscene import monitor as mon
        k = (ep.key, cap.idx)
        if k not in self._parsed:
            self._parsed[k] = mon.parse(cap, self.boxes)
        return self._parsed[k]

    def evaluate(self, w: dict, obj_text: str) -> dict:
        """{modes: {code|clef|truth: True/False/None}} (None: abstain, or truth unknown), with details."""
        from branchlab.rtscene import labels, monitor as mon, render
        from branchlab.rtscene.capture import Episode
        ep = Episode(PROJECT/w["replay"])
        caps = {c.idx: c for c in ep.captures()}
        src, nxt = caps[w["cap"]], caps[w["cap_next"]]
        pts = mon.coordinates(obj_text)
        row = {"coords": pts[:1]}
        modes = {"code": None, "clef": None, "truth": None}
        if not pts:
            return {**row, "bound": False, "modes": modes, "why": "no coordinate"}
        p = pts[0]
        # truth: the recorded object nearest the packet's coordinate (each instance has its own pose)
        o0, o1 = labels.objects(ep, src), labels.objects(ep, nxt)
        roles = sorted(((math.dist(o["position_cm"][:2], p), r) for r, o in o0.items()
                        if r not in ("wall", "table") and "position_cm" in o and r in o1), key=lambda x: x[0])
        if roles and roles[0][0] <= mon.BIND_RADIUS:
            role = roles[0][1]
            moved = math.dist(o0[role]["position_cm"][:2], o1[role]["position_cm"][:2])
            dz = abs(o1[role]["position_cm"][2] - o0[role]["position_cm"][2])
            rot = quat_angle(o0[role]["quat_wxyz"], o1[role]["quat_wxyz"])
            modes["truth"] = moved < UNMOVED_XY and dz < UNMOVED_H and rot < UNMOVED_ROT
            row.update(role=role, true_moved=round(moved, 2), true_dz=round(dz, 2), true_rot=round(rot, 1))
        bl0 = [b for b in self.parse(ep, src) if b.h95 >= 1.0]
        near = [b for b in bl0 if math.dist(b.centroid, p) <= mon.BIND_RADIUS]
        tips0 = [st["position_cm"][:2] for st in src.arms.values()]
        if not near:
            why = "gripper point" if any(math.dist(p, t) <= mon.GRIPPER_POINT for t in tips0) else "no blob"
            return {**row, "bound": False, "modes": modes, "why": why}
        ref = min(near, key=lambda b: math.dist(b.centroid, p))
        row.update(bound=True, ref_centroid=[round(v, 1) for v in ref.centroid], ref_h=round(ref.h95, 1))
        if mon.arm_near(nxt, ref):
            return {**row, "arm_near": True, "modes": modes, "why": "arm near"}
        floor = 0.5 * min(ref.length, ref.width, ref.h95)
        cands = sorted((b for b in self.parse(ep, nxt) if math.dist(b.centroid, ref.centroid) <= mon.CAND_RADIUS
                        and b.h95 >= floor), key=lambda b: math.dist(b.centroid, ref.centroid))[:mon.MAX_CANDS]
        if not cands:
            modes.update(code=False, clef=False)
            return {**row, "modes": modes, "why": "nothing there"}
        dists = [math.dist(b.centroid, ref.centroid) for b in cands]

        def unmoved(b):
            return math.dist(b.centroid, ref.centroid) < UNMOVED_XY and abs(b.h95 - ref.h95) < UNMOVED_H
        modes["code"] = None if len(cands) > 1 and dists[1] - dists[0] < AMBIGUOUS else unmoved(cands[0])
        half = mon.half_for(ref)
        height = max(ref.h95, 4.0)
        ref_img = render.crop(src, mon.T, ref.centroid, half, height, size=mon.TILE, label="REF")
        tiles = [render.crop(nxt, mon.T, b.centroid, half, max(height, b.h95), size=mon.TILE, label=str(k + 1))
                 for k, b in enumerate(cands)]
        state, q = mon.reid_request(len(cands))
        ans = self.clef()(state, q, [render.jpeg_uri(ref_img), render.jpeg_uri(render.tile_sheet(tiles, cols=3))])["which"]
        pick, p_pick = mon.fuse(ans["probabilities"], [round(d) for d in dists])
        if pick != "none" and p_pick >= mon.SURE and ans.get("choice") == pick:
            modes["clef"] = unmoved(cands[int(pick[1:]) - 1])
        elif pick == "none" and p_pick >= mon.SURE and ans.get("choice") == "none":
            modes["clef"] = False
        row.update(n_cands=len(cands), d1=round(dists[0], 2), clef_pick=pick, clef_p=round(p_pick, 3),
                   clef_choice=ans.get("choice"))
        return {**row, "modes": modes}


# ----------------------------------------------------------------------------- steps
def cmd_windows(args):
    cands = candidates()
    chosen = pick_windows(cands, args.n)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"candidates.json").write_text(json.dumps(cands))
    (OUT/"windows.json").write_text(json.dumps(chosen, indent=1))
    print(f"candidates {len(cands)} {dict(collections.Counter(c['call_type'] for c in cands))}, clean "
          f"{sum(c['clean'] for c in cands)}; chosen {len(chosen)} {dict(collections.Counter(c['call_type'] for c in chosen))}, "
          f"clean {sum(c['clean'] for c in chosen)}")


def cmd_query(args):
    windows = json.loads((OUT/"windows.json").read_text())
    if args.limit:
        windows = windows[:args.limit]
    out = OUT/"replies.jsonl"
    done = {r["key"] for r in read_jsonl(out) if not r.get("error")}
    jobs = [(w, v) for w in windows for v in args.variants if f"{w['id']}|{v}" not in done]
    print(f"{len(jobs)} Gemini calls to make ({len(done)} cached)")
    if args.dry_run:
        if not jobs:
            return
        w, v = jobs[0]
        msgs = request(w, v)
        n_img = sum(1 for m in msgs if isinstance(m["content"], list) for p in m["content"] if p.get("type") == "image_url")
        print(json.dumps({"window": w["id"], "variant": v, "messages": len(msgs), "images": n_img,
                          "system_chars": len(msgs[0]["content"])}, indent=1))
        if v in PACKET_VARIANTS:
            print(msgs[0]["content"][-2700:])
        return
    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as f:
        for k, r in enumerate(pool.map(ask_gemini, jobs), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            u = r.get("usage") or {}
            print(f"[{k}/{len(jobs)}] {r['key']} {r['latency_s']}s prompt {u.get('prompt_tokens')} "
                  f"out {u.get('completion_tokens')}{' ERROR ' + str(r['error'])[:80] if r['error'] else ''}", flush=True)


def logged_usage(w: dict) -> dict | None:
    """The original run's usage for turn t of this episode (llm_calls.jsonl; episodes split at turn1 tags)."""
    shard = (PROJECT/w["trace"]).parent
    order = [e["episode_index"] for e in json.loads((shard/"results.json").read_text())["episodes"]]
    groups, cur = [], None
    for r in read_jsonl(shard/"llm_calls.jsonl"):
        if r.get("tag") == "turn1":
            cur = []
            groups.append(cur)
        if cur is not None:
            cur.append(r)
    if len(groups) != len(order) or w["episode"] not in order:
        return None
    calls = [r for r in groups[order.index(w["episode"])] if r.get("tag") == f"turn{w['turn']}" and r.get("usage")]
    return calls[-1]["usage"] if calls else None


def window_clean(w: dict) -> bool:
    meta = json.loads((PROJECT/w["replay"]/"meta.json").read_text())
    d_state, d_outcome = first_divergence(meta, load_trace(PROJECT/w["trace"]))
    return w["turn"] + 1 < d_state and w["turn"] < d_outcome


def summarize(rep: dict | None) -> dict | None:
    if rep is None:
        return None
    pr = parse_reply(rep["text"]) if not rep.get("error") else {"ok": False, "error": rep["error"], "commands": [], "errors": [], "packet": None}
    u = rep.get("usage") or {}
    return {"ok": pr["ok"] and bool(pr["commands"]), "malformed": malformed(rep), "commands": pr["commands"],
            "packet": pr.get("packet"), "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "latency_s": rep.get("latency_s"), "error": pr.get("error")}


def judge_packet(w: dict, row: dict, variant: str, rec: dict, nxt: dict, assoc: Association) -> dict | None:
    rep = row.get(variant)
    pk = (rep or {}).get("packet")
    if not pk:
        return None
    out = {"valid": pk["valid"], "problems": pk.get("problems"), "conditions": pk.get("conditions"),
           "commands": pk.get("commands"), "object": pk.get("object")}
    if not (pk["valid"] and rep["ok"]):
        return out
    out["resolved"] = same_commands(rep["commands"], rec["commands"], strict=True)
    out["resolved_loose"] = same_commands(rep["commands"], rec["commands"])
    out["vs_logged"] = compare(pk["commands"], nxt["commands"])
    out["vs_n1"] = compare(pk["commands"], row["N1"]["commands"]) if row.get("N1") and row["N1"]["ok"] else None
    if not out["resolved"]:
        return out
    from branchlab.rtscene.capture import Episode
    capn = {c.idx: c for c in Episode(PROJECT/w["replay"]).captures()}[w["cap_next"]]
    res = rec.get("results") or []
    ok = bool(res) and all(r.get("ok") for r in res) and not rec.get("command_errors")   # a done counts as not ok
    arms = {arm_of(c) for c in pk["commands"]} - {None}
    held = bool(arms) and all(capn.arms[a].get("gripper", 1.0) < 0.5 and capn.arms[a].get("gripper_real", 0.0) > 0.02 for a in arms)
    opened = bool(arms) and all(capn.arms[a].get("gripper", 0.0) >= 0.9 and capn.arms[a].get("gripper_real", 0.0) >= 0.5 for a in arms)
    code_conds = {"commands_ok": ok, "object_held": held, "gripper_open": opened}
    needs_obj = "object_unmoved" in pk["conditions"]
    obj = assoc.evaluate(w, pk["object"]) if needs_obj else None
    gate, unknown = {}, {}
    for mode in ("code", "clef", "truth"):
        vals = [code_conds["commands_ok"]] + [code_conds[c] for c in pk["conditions"] if c != "object_unmoved"]
        if needs_obj:
            vals.append(obj["modes"][mode] is True)
        gate[mode] = all(vals)
        unknown[mode] = bool(needs_obj and obj["modes"][mode] is None)
    out.update(code_conditions=code_conds, object_check=obj, gate=gate, abstain=unknown)
    return out


def cmd_gate(args):
    windows = json.loads((OUT/"windows.json").read_text())
    replies = {}
    for r in read_jsonl(OUT/"replies.jsonl"):
        if not r.get("error") or r["key"] not in replies:
            replies[r["key"]] = r
    assoc = Association()
    rows = []
    for w in windows:
        trace = load_trace(PROJECT/w["trace"])
        rec, nxt = trace[w["turn"]], trace[w["turn"] + 1]
        row = {**{k: w[k] for k in ("id", "task", "episode", "turn", "call_type", "success")},
               "clean": window_clean(w), "logged": rec["commands"], "logged_next": nxt["commands"],
               "logged_prompt_tokens": (logged_usage(w) or {}).get("prompt_tokens")}
        for v in ("P0", "N1", *PACKET_VARIANTS):
            row[v] = summarize(replies.get(f"{w['id']}|{v}"))
        if row["P0"] and row["P0"]["ok"]:
            row["P0_same_as_logged"] = same_commands(row["P0"]["commands"], rec["commands"])
        if row["N1"] and row["N1"]["ok"]:
            row["N1_same_as_logged_next"] = same_commands(row["N1"]["commands"], nxt["commands"])
            row["N1_vs_logged_next"] = compare(row["N1"]["commands"], nxt["commands"])
        for v in PACKET_VARIANTS:
            if row[v] and row[v]["ok"] and row["P0"] and row["P0"]["ok"]:
                row[f"{v}_same_as_P0"] = same_commands(row[v]["commands"], row["P0"]["commands"])
            row[f"{v}_eval"] = judge_packet(w, row, v, rec, nxt, assoc)
        rows.append(row)
    (OUT/"rows.json").write_text(json.dumps(rows, indent=1, default=str))
    print(f"{len(rows)} windows judged -> {OUT/'rows.json'} (clean {sum(r['clean'] for r in rows)})")


def _safe(e: dict, mode: str) -> bool:
    return bool(e and e.get("gate", {}).get(mode) and "full" in (e.get("vs_logged"), e.get("vs_n1")))


def _critical(e: dict, mode: str) -> bool:
    return bool(e and e.get("gate", {}).get(mode) and not _safe(e, mode) and "critical" in (e.get("vs_logged"), e.get("vs_n1")))


def cmd_report(args):
    rows_all = json.loads((OUT/"rows.json").read_text())
    rows = [r for r in rows_all if r["clean"]]
    n = len(rows)
    pct = lambda a, b: f"{a}/{b} ({a / b:.0%})" if b else f"{a}/0"
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    pop = population()
    total = sum(pop.values())
    L = ["# Shadow relay audit", "",
         f"Windows {len(rows_all)}, clean {n} (statistics below are on the clean windows); call types of the clean windows "
         f"{dict(collections.Counter(r['call_type'] for r in rows))}.",
         f"Population: {total} Gemini calls of the 8 tasks' 80 original episodes: "
         + ", ".join(f"{k} {v} ({v / total:.0%})" for k, v in pop.most_common()), ""]
    p0ok = [r for r in rows if r["P0"] and r["P0"]["ok"]]
    fid = [r for r in p0ok if r["logged_prompt_tokens"]]
    L += ["## Fidelity and replies",
          f"- P0 prompt tokens equal to the original run's: {pct(sum(r['P0']['prompt_tokens'] == r['logged_prompt_tokens'] for r in fid), len(fid))}"]
    for v in ("P0", "P1", "P1b", "N1"):
        got = [r for r in rows if r.get(v)]
        if got:
            L.append(f"- {v}: malformed {pct(sum(bool(r[v]['malformed']) for r in got), len(got))}; median reply tokens "
                     f"{med([r[v]['completion_tokens'] for r in got if r[v]['completion_tokens']])}, reasoning "
                     f"{med([r[v]['reasoning_tokens'] for r in got if r[v]['reasoning_tokens']])}, latency "
                     f"{med([r[v]['latency_s'] for r in got if r[v]['latency_s']])} s")
    L += ["", "## Prompt effect on the current commands (net-effect equality)",
          f"- P0 vs the recorded commands (Gemini against itself): {pct(sum(bool(r.get('P0_same_as_logged')) for r in p0ok), len(p0ok))}"]
    for v in PACKET_VARIANTS:
        k = f"{v}_same_as_P0"
        got = [r for r in rows if k in r]
        if got:
            L.append(f"- {v} vs P0: {pct(sum(bool(r[k]) for r in got), len(got))}")
    n1 = [r for r in rows if "N1_vs_logged_next" in r]
    L += [f"- N1 vs the recorded next decision (Gemini against itself at t+1): "
          f"{dict(collections.Counter(r['N1_vs_logged_next'] for r in n1))} of {len(n1)}", ""]
    for v in PACKET_VARIANTS:
        usable = [r for r in rows if r.get(v) and r[v]["ok"]]
        if not usable:
            continue
        evs = [r[f"{v}_eval"] for r in usable if r.get(f"{v}_eval")]
        valid = [e for e in evs if e["valid"]]
        resolved = [e for e in valid if e.get("resolved")]
        L += [f"## {v} packets",
              f"- given {pct(len(evs), len(usable))}; valid {pct(len(valid), len(evs))}; resolved (own commands = recorded, "
              f"1 cm) {pct(len(resolved), len(valid))} (loose rule {sum(bool(e.get('resolved_loose')) for e in valid)})",
              f"- by call type: " + ", ".join(f"{k} {sum(1 for r in usable if r['call_type'] == k and r.get(f'{v}_eval'))}/"
                                              f"{sum(1 for r in usable if r['call_type'] == k)}" for k in ("continuation", "new", "retry")),
              f"- conditions {dict(collections.Counter(c for e in valid for c in e['conditions']))}",
              f"- every valid packet vs the recorded next decision {dict(collections.Counter(e['vs_logged'] for e in valid))}, "
              f"vs N1 {dict(collections.Counter(e['vs_n1'] for e in valid if e.get('vs_n1')))}",
              "", "| mode | accepted | safe skips (full vs recorded or N1) | partial | critical | abstained on object_unmoved |",
              "|---|---|---|---|---|---|"]
        for mode in ("code", "clef", "truth"):
            acc = [e for e in resolved if e["gate"][mode]]
            safe = [e for e in acc if _safe(e, mode)]
            part = [e for e in acc if not _safe(e, mode) and "partial" in (e["vs_logged"], e.get("vs_n1"))]
            crit = [e for e in acc if _critical(e, mode)]
            L.append(f"| {mode} | {len(acc)} | {len(safe)} | {len(part)} | {len(crit)} | {sum(e['abstain'][mode] for e in resolved)} |")
        L += ["", "Calls removed, weighted by call kind over the population (first calls cannot be skipped; "
              "p = per-call skip rate, removed = p / (1 + p)):"]
        for mode in ("code", "clef", "truth"):
            p_rate = 0.0
            for kind in ("continuation", "new", "retry"):
                sub = [r for r in rows if r["call_type"] == kind and r.get(v)]
                if sub:
                    p_rate += pop[kind] / total * sum(_safe(r.get(f"{v}_eval"), mode) for r in sub) / len(sub)
            boots = []
            rng = random.Random(0)
            strata = {k: [r for r in rows if r["call_type"] == k and r.get(v)] for k in ("continuation", "new", "retry")}
            for _ in range(2000):
                b = 0.0
                for kind, sub in strata.items():
                    if sub:
                        draw = [rng.choice(sub) for _ in sub]
                        b += pop[kind] / total * sum(_safe(r.get(f"{v}_eval"), mode) for r in draw) / len(draw)
                boots.append(b / (1 + b))
            boots.sort()
            L.append(f"- {mode}: p {p_rate:.1%}, removed {p_rate / (1 + p_rate):.1%} (bootstrap 95% {boots[50]:.1%}-{boots[1949]:.1%})")
        diff = sum(_safe(r.get(f"{v}_eval"), "clef") - _safe(r.get(f"{v}_eval"), "code") for r in rows if r.get(v))
        L += [f"- clef minus code safe skips over the windows: {diff:+d}", ""]
    text = "\n".join(L)
    (OUT/"report.md").write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    s = sub.add_parser("windows")
    s.add_argument("--n", type=int, default=60)
    s = sub.add_parser("query")
    s.add_argument("--variants", nargs="+", default=["P0", "P1", "P1b", "N1"])
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--limit", type=int)
    s.add_argument("--dry-run", action="store_true")
    sub.add_parser("gate")
    sub.add_parser("report")
    args = parser.parse_args()
    setup_paths()
    {"windows": cmd_windows, "query": cmd_query, "gate": cmd_gate, "report": cmd_report}[args.step](args)


if __name__ == "__main__":
    main()
