#!/usr/bin/env python3
"""Success ceiling per judgment type on our own 8-task baseline (80 episodes, rt8a-c).

The semantic tests (toe, QR, re-ID, claw) measured how often Clef answers correctly, but never how many of our
failures a correct answer could have turned into successes. This script gives the coders what they need to count
that, and then counts it.

    digests   one markdown digest per episode: instruction, the RoboTwin checker, ground-truth object poses per turn
              (trace state: the robot never saw these), task metrics against the checker, gripper feedback, Gemini's
              scene/progress/plan and commands, and the first turn where the replay capture stops matching the run.
    report    reads the coding workflow's output (coding.json: first and second coders, adjudication, skeptic verdicts)
              and writes per-judgment ceilings, agreement and the
              success-vs-failure persistence rates.

    python3 scripts/failure_ceiling.py digests
    python3 scripts/failure_ceiling.py report
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLAB = ROOT/"outputs/robodawn_colab"
REPLAY = COLAB/"rt8d/replay/eval_baseline"
OUT = ROOT/"outputs/analysis/failure_ceiling"

# RoboTwin envs/<task>.py at commit 96c1feab, as summarised in reports/RoboDawn_40과제_실패_분석.md §4
CHECKER = {
    "place_shoe": "shoe xy within ±5 cm (x) and ±2 cm (y) of (0, -8) cm; every quaternion component within 0.07 of the "
                  "target [0.5, 0.5, -0.5, -0.5] (about ±8°); both grippers open",
    "place_dual_shoes": "both shoes at their slots inside the shoe box; quaternion within 0.07 of the same target as "
                        "place_shoe; height near the box floor; both grippers open",
    "rotate_qrcode": "every quaternion component within 0.05 of [0.707, 0.707, 0, 0] (about ±6°); on the table; both "
                     "grippers open",
    "scan_object": "object within 2.5 cm of the scanner's beam axis (functional point), distance 0-7 cm; both grippers "
                   "closed",
    "put_bottles_dustbin": "every bottle inside the dustbin region (x -45±22 cm, y 0±32 cm), height 0.2-0.7 m",
    "place_bread_basket": "every bread within ±5 cm (xy) of the basket; on the table; both grippers open",
    "pick_diverse_bottles": "both bottles' functional points within 10 cm of (±6, -10.5) cm and higher than 0.89 m; if a "
                            "bottle ever drops below 0.78 m the episode is marked failed for good",
    "place_object_scale": "object within 3.5 cm of the scale's functional point; the used arm's gripper open",
}
NO_TRUTH = {"put_bottles_dustbin": "the bottles", "place_bread_basket": "the breads"}
TARGET_QUAT = {"place_shoe": [0.5, 0.5, -0.5, -0.5], "place_dual_shoes": [0.5, 0.5, -0.5, -0.5],
               "rotate_qrcode": [0.7071, 0.7071, 0.0, 0.0]}


def quat_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                     [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                     [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])


def rel_angles(q, q_ref) -> tuple[float, float, float]:
    """(total angle, tilt of the reference's up axis, yaw about world z) of q relative to q_ref, in degrees."""
    R = quat_matrix(q) @ quat_matrix(q_ref).T
    total = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return total, tilt, yaw


def quat_err(q, t) -> float:
    return min(max(abs(a - b) for a, b in zip(q, t)), max(abs(a + b) for a, b in zip(q, t)))


def episodes() -> list[dict]:
    """The 80 counted episodes: results from rt8b/rt8c, the trace from the same shard (rt8a holds some of them)."""
    out = []
    for res in sorted(glob.glob(str(COLAB/"rt8[bc]/robodawn/gemini_flash_*/*/shard_*/results.json"))):
        p = Path(res)
        task, shard = p.parts[-3], p.parts[-2]
        traces = sorted(p.parent.glob("episode_*/trace.json")) or \
            sorted(COLAB.glob(f"rt8a/robodawn/gemini_flash_*/{task}/{shard}/episode_*/trace.json"))
        r = json.loads(p.read_text())
        for e in r["episodes"]:
            n = e["episode_index"]
            tr = [t for t in traces if t.parent.name == f"episode_{n:03d}"]
            rep = REPLAY/task/f"episode_{n:03d}"
            out.append({"task": task, "episode": n, "success": bool(e["success"]), "finished": e["finished_reason"],
                        "turns": e["turns"], "steps": f"{e['steps_used']}/{e['step_limit']}", "instruction": e["instruction"],
                        "trace": str(tr[0].relative_to(ROOT)) if tr else None, "results": str(p.relative_to(ROOT)),
                        "replay": str(rep.relative_to(ROOT)) if (rep/"meta.json").is_file() else None})
    return sorted(out, key=lambda e: (e["task"], e["episode"]))


def gripper_outcomes(notes_by_turn: dict) -> dict:
    """turn -> tuple of close outcomes ('empty' or the opening to 0.05) and planner refusals, to compare run and replay."""
    out = {}
    for t, notes in notes_by_turn.items():
        sig = []
        for n in notes:
            if n.startswith("fingers closed fully"):
                sig.append("empty")
            elif n.startswith("fingers stopped at opening"):
                sig.append(round(round(float(n.split()[4].rstrip(":")) / 0.05) * 0.05, 2))
            elif n.startswith("motion planner could not reach"):
                sig.append("refused")
        out[t] = tuple(sig)
    return out


def replay_divergence(ep: dict, trace: list[dict]) -> str:
    if not ep["replay"]:
        return "no replay capture"
    meta = json.loads((ROOT/ep["replay"]/"meta.json").read_text())
    run = gripper_outcomes({r["turn"]: [x.get("note", "") for x in r.get("results") or []] for r in trace})
    rep_notes = collections.defaultdict(list)
    for c in meta["command_log"]:
        rep_notes[c["turn"]].append(c.get("note", ""))
    rep = gripper_outcomes(rep_notes)
    for t in sorted(run):
        a, b = run[t], rep.get(t)
        if b is None:
            return f"replay stops before turn {t} (replay success {meta['replay_success']})"
        if len(a) != len(b) or any((x != y) if isinstance(x, str) or isinstance(y, str) else abs(x - y) > 0.051
                                   for x, y in zip(a, b)):
            return (f"gripper/planner outcomes first differ at turn {t} (run {a} vs replay {b}); replay images "
                    f"(images/c<idx>_<camera>.jpg, turn-start captures) match the run only before that")
    return f"gripper/planner outcomes match on every turn (replay success {meta['replay_success']})"


def held_by(state: dict, role: str, rest_z: float) -> str | None:
    o = state["objects"][role]
    for arm in ("left", "right"):
        a = state[arm]
        if a.get("gripper_real", 1) < 0.9 and o["position_cm"][2] > rest_z + 2 and \
                math.dist(a["position_cm"], o["position_cm"]) < 15:
            return arm
    return None


def task_metrics(task: str, objs: dict) -> list[str]:
    L = []
    tq = TARGET_QUAT.get(task)
    for role, o in objs.items():
        if tq and ("shoe" in str(o.get("model")) or task == "rotate_qrcode") and "box" not in str(o.get("model")):
            tot, tilt, yaw = rel_angles(o["quat_wxyz"], tq)
            L.append(f"{role}: quaternion error {quat_err(o['quat_wxyz'], tq):.3f} (checker needs <= "
                     f"{0.05 if task == 'rotate_qrcode' else 0.07}); {tot:.0f}° from the target pose, of which tilt "
                     f"{tilt:.0f}° and yaw {yaw:+.0f}° about vertical (about ±180° = facing reversed)")
    if task == "place_shoe" and "shoe" in objs and "target_block" in objs:
        s, b = objs["shoe"]["position_cm"], objs["target_block"]["position_cm"]
        L.append(f"shoe vs mat centre: dx {s[0]-b[0]:+.1f} cm (needs ±5), dy {s[1]-b[1]:+.1f} cm (needs ±2)")
    if task == "place_dual_shoes" and "shoe_box" in objs:
        b = objs["shoe_box"]["position_cm"]
        for role in ("left_shoe", "right_shoe"):
            if role in objs:
                s = objs[role]["position_cm"]
                L.append(f"{role} vs box centre: ({s[0]-b[0]:+.1f}, {s[1]-b[1]:+.1f}, {s[2]-b[2]:+.1f}) cm")
    if task == "scan_object" and {"scanner", "object"} <= set(objs):
        fs, fo = objs["scanner"].get("functional_points_cm") or [], objs["object"].get("functional_points_cm") or []
        if fs and fo:
            L.append(f"scanner functional points {fs}; object functional points {fo}; nearest pair "
                     f"{min(math.dist(a, b) for a in fs for b in fo):.1f} cm apart")
    if task == "pick_diverse_bottles":
        for role in ("bottle1", "bottle2"):
            if role in objs:
                fp = (objs[role].get("functional_points_cm") or [objs[role]["position_cm"]])[0]
                d = min(math.dist(fp[:2], (sx, -10.5)) for sx in (-6, 6))
                L.append(f"{role} functional point {fp}: {d:.1f} cm from the nearer hold spot (needs <= 10), "
                         f"z {fp[2]:.1f} cm (needs > 89)")
    if task == "place_object_scale" and {"object", "scale"} <= set(objs):
        fs = (objs["scale"].get("functional_points_cm") or [objs["scale"]["position_cm"]])[0]
        L.append(f"object vs scale functional point {fs}: {math.dist(objs['object']['position_cm'], fs):.1f} cm "
                 "(needs <= 3.5)")
    return L


def short(text, n=240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n] + "…"


def digest(ep: dict) -> str:
    trace = [r for r in json.loads((ROOT/ep["trace"]).read_text()) if "turn" in r]
    s0 = trace[0]["state"]
    table = s0.get("table_z_cm", 0)
    roles = [r for r in s0["objects"] if r not in ("wall", "table")]
    o0 = {r: s0["objects"][r] for r in roles}
    L = [f"# {ep['task']} episode {ep['episode']}: {'SUCCESS' if ep['success'] else 'FAILURE'} "
         f"({ep['finished']}), {ep['turns']} turns, steps {ep['steps']}", "",
         f"- Instruction: {ep['instruction']}",
         f"- Checker: {CHECKER[ep['task']]}",
         f"- Trace (full record, prompts and replies): {ep['trace']}",
         f"- Replay capture: {ep['replay'] or 'none'}; {replay_divergence(ep, trace)}",
         f"- Table top z = {table} cm. Positions are cm in the robot frame (x right, y away from the robot, z up).", ""]
    if ep["task"] in NO_TRUTH:
        L += [f"- No ground truth for {NO_TRUTH[ep['task']]}: RoboTwin keeps them in a list the trace does not record. "
              "Use the gripper feedback (truthful), the replay images while the replay still matches, and Gemini's text "
              "(not truthful).", ""]
    L += ["## Ground-truth objects (the robot never saw these numbers)", ""]
    last = trace[-1]["state"]["objects"]
    for r in roles:
        a, b = o0[r], last.get(r, o0[r])
        zs = [t["state"]["objects"][r]["position_cm"][2] for t in trace if r in t["state"]["objects"]]
        tot, tilt, yaw = rel_angles(b["quat_wxyz"], a["quat_wxyz"])
        off = next((t["turn"] for t in trace if r in t["state"]["objects"] and
                    t["state"]["objects"][r]["position_cm"][2] < table - 10), None)
        L.append(f"- {r} ({a.get('model')}): start ({', '.join(f'{v:.1f}' for v in a['position_cm'])}), last recorded "
                 f"({', '.join(f'{v:.1f}' for v in b['position_cm'])}); z range {min(zs):.1f}..{max(zs):.1f}; at the end "
                 f"tilted {tilt:.0f}° and turned {yaw:+.0f}° about vertical relative to its start"
                 + (f"; BELOW THE TABLE TOP from turn {off}" if off else ""))
    metrics = task_metrics(ep["task"], last)
    if metrics:
        L += ["", "Task metrics at the last recorded state (start of the last turn):", ""] + [f"- {m}" for m in metrics]
    L += ["", "## Turns", "",
          "Each turn: arm fingertips (x, y, z) and real opening at the start of the turn, ground-truth object events since "
          "the previous turn, Gemini's commands with the harness feedback, then Gemini's scene / progress / plan text.", ""]
    prev = o0
    for rec in trace:
        st, t = rec["state"], rec["turn"]
        arms = "; ".join(f"{arm[0].upper()} ({', '.join(f'{v:.1f}' for v in st[arm]['position_cm'])}) open "
                         f"{st[arm].get('gripper_real', float('nan')):.2f}" for arm in ("left", "right"))
        ev = []
        for r in roles:
            if r not in st["objects"]:
                continue
            o, p = st["objects"][r], prev.get(r, o0[r])
            d = math.dist(o["position_cm"][:2], p["position_cm"][:2])
            _, dtilt, dyaw = rel_angles(o["quat_wxyz"], p["quat_wxyz"])
            _, tilt0, _ = rel_angles(o["quat_wxyz"], o0[r]["quat_wxyz"])
            h = held_by(st, r, o0[r]["position_cm"][2])
            bits = []
            if h:
                bits.append(f"held by {h}, z {o['position_cm'][2]:.1f}")
            if d > 1.5 or dtilt > 15 or abs(dyaw) > 20:
                bits.append(f"moved {d:.1f} cm to ({o['position_cm'][0]:.1f}, {o['position_cm'][1]:.1f}, "
                            f"{o['position_cm'][2]:.1f}), tilt {dtilt:+.0f}° (now {tilt0:.0f}° from start), yaw {dyaw:+.0f}°")
            if o["position_cm"][2] < table - 10 and p["position_cm"][2] >= table - 10:
                bits.append("FELL BELOW THE TABLE TOP")
            if bits:
                ev.append(f"{r}: " + ", ".join(bits))
        prev = {r: st["objects"].get(r, prev.get(r)) for r in roles}
        res = []
        for c, x in zip(rec.get("commands") or [], rec.get("results") or []):
            res.append(f"{c} -> {'ok' if x.get('ok') else 'FAILED'}{(': ' + short(x['note'], 90)) if x.get('note') else ''}")
        if rec.get("error"):
            res.append(f"ERROR {short(rec['error'], 120)}")
        if rec.get("command_errors"):
            res.append(f"command errors {short(json.dumps(rec['command_errors']), 160)}")
        L.append(f"**t{t:02d}** {arms}" + (f" | events: {'; '.join(ev)}" if ev else ""))
        L.append(f"  - commands: {' | '.join(res) or '(none)'}")
        L.append(f"  - scene: {short(rec.get('scene'))} | progress: {short(rec.get('progress'))} | plan: {short(rec.get('plan'))}")
    return "\n".join(L) + "\n"


def cmd_digests(args):
    eps = episodes()
    (OUT/"digests").mkdir(parents=True, exist_ok=True)
    for ep in eps:
        ep["digest"] = str((OUT/"digests"/f"{ep['task']}__e{ep['episode']:02d}.md").relative_to(ROOT))
        (ROOT/ep["digest"]).write_text(digest(ep))
    (OUT/"episodes.json").write_text(json.dumps(eps, indent=1))
    c = collections.Counter((e["task"], e["success"]) for e in eps)
    print(f"{len(eps)} episodes, {sum(e['success'] for e in eps)} successes; "
          + ", ".join(f"{t} {c[(t, True)]}/{c[(t, True)] + c[(t, False)]}" for t in sorted({e['task'] for e in eps})))


JTYPES = ("facing", "identity", "alignment", "topple_watch", "collision_risk", "placement", "held_state",
          "task_condition", "progress_stall")
JNAME = {"facing": "방향(앞코·QR 면)", "identity": "대상 정체", "alignment": "닫기 전 정렬", "topple_watch": "넘어짐·밀림 감지",
         "collision_risk": "충돌 예측", "placement": "놓기 위치", "held_state": "쥠 유지", "task_condition": "숨은 채점 조건",
         "progress_stall": "정체 감지"}


def ceiling(e: dict, t: str) -> dict | None:
    return next((j for j in e.get("judgments", []) if j["type"] == t), None)


def kappa(pairs: list[tuple]) -> tuple[float, float, int]:
    """Cohen's kappa, raw agreement and n over (a, b) label pairs."""
    n = len(pairs)
    if not n:
        return float("nan"), float("nan"), 0
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), po, n


def mde_episodes(p: float, n: int) -> float:
    """Smallest difference (episodes out of n) two independent n-episode arms detect with 80% power at two-sided 5%."""
    return (1.96 + 0.84) * math.sqrt(2 * p * (1 - p) / n) * n


def cmd_report(args):
    data = json.loads((OUT/"coding.json").read_text())
    eps = {f"{e['task']}#{e['episode']}": e for e in json.loads((OUT/"episodes.json").read_text())}
    stages = [("1차 코딩", data["first"]), ("판정 후", data["final"]), ("반박 검토 후", data["post_skeptic"])]
    post = data["post_skeptic"]
    fails = {k: e for k, e in post.items() if not eps[k]["success"]}
    succ = {k: e for k, e in post.items() if eps[k]["success"]}
    n_succ = sum(e["success"] for e in eps.values())
    L = ["# 판단 유형별 성공 상한 (8과제 기준선 80 에피소드)", "",
         f"실패 {len(fails)}개, 성공 {len(succ)}개를 정답 자세로 다시 코딩했다. 기준은 outputs/analysis/failure_ceiling/rubric.md, "
         "에피소드 요약은 digests/, 코딩 원자료는 coding.json에 있다.", "",
         "## 1. 판단 유형별 상한", "",
         "'유력'은 그 회차에 정답을 한 문장으로 알려 줬다면 성공했을 가능성이 더 큰 실패, '가능'은 20~50% 정도인 실패다.", "",
         "| 판단 | " + " | ".join(f"{name} 유력/유력+가능" for name, _ in stages) + " | 이미 프롬프트에 있던 정보 | 카메라에 보임(예/부분/아니오) |",
         "|---|" + "---|" * (len(stages) + 2)]
    for t in JTYPES:
        cells = []
        for _, codes in stages:
            js = [ceiling(e, t) for k, e in codes.items() if not eps[k]["success"]]
            lik = sum(1 for j in js if j and j["ceiling"] == "likely")
            pos = sum(1 for j in js if j and j["ceiling"] in ("likely", "possible"))
            cells.append(f"{lik} / {pos}")
        js = [ceiling(e, t) for e in fails.values()]
        hit = [j for j in js if j and j["ceiling"] in ("likely", "possible")]
        avail = sum(j["already_available"] for j in hit)
        vis = collections.Counter(j["visible"] for j in hit)
        L.append(f"| {JNAME[t]} (`{t}`) | " + " | ".join(cells) + f" | {avail}/{len(hit)} | "
                 f"{vis['yes']}/{vis['partial']}/{vis['no']}" + (f" (+{vis['unknown']} 모름)" if vis["unknown"] else "") + " |")
    # union and per-task
    def any_level(e, levels, new_only=False, visible_only=False):
        return any(j["ceiling"] in levels and (not new_only or not j["already_available"]) and
                   (not visible_only or j["visible"] in ("yes", "partial")) for j in e.get("judgments", []))
    rows = [("어떤 판단이든", {}), ("새 정보만(이미 프롬프트에 있던 것 제외)", {"new_only": True}),
            ("새 정보이면서 카메라에 보이는 것", {"new_only": True, "visible_only": True})]
    L += ["", "## 2. 합친 상한 (반박 검토 후)", "", "| 범위 | 유력 | 유력+가능 |", "|---|---|---|"]
    for name, kw in rows:
        L.append(f"| {name} | {sum(any_level(e, ('likely',), **kw) for e in fails.values())} | "
                 f"{sum(any_level(e, ('likely', 'possible'), **kw) for e in fails.values())} |")
    p0 = n_succ / len(eps)
    L += ["", f"- 기준선 성공 {n_succ}/{len(eps)}. 80 에피소드 두 팔 비교(독립, 검정력 80%, 양측 5%)가 감지하는 최소 차이는 약 "
          f"{mde_episodes(p0, len(eps)):.0f} 에피소드다.", ""]
    tasks = sorted({e["task"] for e in eps.values()})
    L += ["### 과제별 (반박 검토 후, 유력/유력+가능)", "", "| 과제 | 실패 | " + " | ".join(JNAME[t] for t in JTYPES) + " |",
          "|---|---|" + "---|" * len(JTYPES)]
    for task in tasks:
        fe = [e for k, e in fails.items() if eps[k]["task"] == task]
        cells = []
        for t in JTYPES:
            js = [ceiling(e, t) for e in fe]
            cells.append(f"{sum(1 for j in js if j and j['ceiling'] == 'likely')}/"
                         f"{sum(1 for j in js if j and j['ceiling'] in ('likely', 'possible'))}")
        L.append(f"| {task} | {len(fe)} | " + " | ".join(cells) + " |")
    L += ["", "## 3. 결정적 판단과 주 원인 (반박 검토 후)", ""]
    dj = collections.Counter(e["decisive_judgment"] for e in fails.values())
    L += ["- 결정적 판단: " + ", ".join(f"{k} {v}" for k, v in dj.most_common())]
    pc = collections.Counter(e["primary_cause"] for e in fails.values())
    L += ["- 주 원인: " + ", ".join(f"{k} {v}" for k, v in pc.most_common())]
    off = [k for k, e in fails.items() if e["end_state"].get("off_table")
           and not str(e["end_state"]["off_table"]).lower().startswith(("no", "none"))]
    L += [f"- 물체가 탁자를 벗어났다고 코딩된 실패: {len(off)}개 ({', '.join(sorted(off))}). 병·빵 과제의 일부는 정답 자세가 없어 "
          "Gemini의 말이나 하네스 멈춤에서 추정했다.", ""]
    # agreement
    first, second = data["first"], data["second"]
    both = [k for k in second if k in first]
    L += ["## 4. 코더 간 일치 (판정 전, 이중 코딩한 실패 " + str(len(both)) + "개)", "",
          "| 항목 | kappa | 단순 일치 | n |", "|---|---|---|---|"]
    lvl = lambda j: "yes" if j and j["ceiling"] in ("likely", "possible") else "no"
    items = [("주 원인", [(first[k]["primary_cause"], second[k]["primary_cause"]) for k in both]),
             ("결정적 판단", [(first[k]["decisive_judgment"], second[k]["decisive_judgment"]) for k in both])]
    pooled = []
    for t in JTYPES:
        pr = [(lvl(ceiling(first[k], t)), lvl(ceiling(second[k], t))) for k in both]
        pooled += pr
        items.append((f"{JNAME[t]}: 유력+가능 여부", pr))
    items.append(("아홉 판단 합침: 유력+가능 여부", pooled))
    for name, pr in items:
        kp, po, n = kappa(pr)
        L.append(f"| {name} | {kp:.2f} | {po:.0%} | {n} |")
    L += ["", f"- 불일치가 있던 에피소드 {len(data['disputes'])}개, 판정 {len(data['adjudication']['decisions'])}건. "
          f"반박 검토는 유력·가능 주장 {len(data['claims'])}건 중 "
          + ", ".join(f"{k} {v}" for k, v in collections.Counter(v["verdict"] for v in data["verdicts"]).most_common()) + ".", ""]
    # persistence
    L += ["## 5. 판단 유지: 성공과 실패의 비교", "", "| 항목 | 성공 | 실패 | 20회차 이상 성공 |", "|---|---|---|---|"]
    long_s = {k: e for k, e in succ.items() if eps[k]["turns"] >= 20}
    for f in ("dropped_own_judgment", "flip_flop", "retarget"):
        cell = lambda d: f"{sum(e['persistence'][f] for e in d.values())}/{len(d)}"
        L.append(f"| {f} | {cell(succ)} | {cell(fails)} | {cell(long_s)} |")
    rec = collections.Counter(r["type"] for e in post.values() for r in e.get("recovered_errors", []))
    L += ["", "- 회복된 판단 오류(성공·실패 전체): " + (", ".join(f"{JNAME[k]} {v}" for k, v in rec.most_common()) or "없음"), ""]
    # likely list
    L += ["## 6. 반박 검토 뒤에도 유력·가능으로 남은 주장", "", "| 에피소드 | 판단 | 등급 | 회차 | 이미 있던 정보 | 보임 | 근거 |",
          "|---|---|---|---|---|---|---|"]
    for k, e in sorted(fails.items()):
        for j in e.get("judgments", []):
            if j["ceiling"] in ("likely", "possible"):
                L.append(f"| {k} | {j['type']} | {j['ceiling']} | {j.get('turn')} | {'예' if j['already_available'] else '아니오'} | "
                         f"{j['visible']} | {short(j['why'], 300).replace('|', '/')} |")
    text = "\n".join(L) + "\n"
    (OUT/"report.md").write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("digests")
    sub.add_parser("report")
    args = parser.parse_args()
    {"digests": cmd_digests, "report": cmd_report}[args.step](args)


if __name__ == "__main__":
    main()
