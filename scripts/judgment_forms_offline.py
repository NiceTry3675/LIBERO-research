"""Offline tests of judgment forms a big model could leave for a fast model,
on our 100 RoboDawn reproduction episodes (outputs/robodawn_colab). See
reports/RoboDawn_하네스_분석.md for the forms and why.

  candidates  Code draws nine numbered candidate points (3 x 3, 3 cm apart) on
              the straight-down map around the true grasp point; the model
              picks the mark where the gripper should close. Scored by the
              distance from the picked mark to the true grasp point.
  checks      Yes/no questions the code cannot answer from the robot state:
              preclose  turns that begin with a close: "would closing now
                        grasp the target?" (label: fingers stopped on it)
              aligned   every step before a grasp: "are the fingertips within
                        about 2 cm of the grasp point horizontally?"
  retry       No model calls: what a cap of N empty grasps (fingers closed on
              nothing) per episode would save and lose.
  textual     The spatial facts as text, no images: the fingertips and an
              object table; per axis, which way to step (as in the servo
              test). Two tables: the simulator's object poses (an upper
              bound) and Gemini's own estimates from the first turn of the
              sub-goal (what a big model would leave behind).

Ground truth for a grasp point is the fingertip position at a close whose
fingers stopped on something, as in scripts/clef_servo_topview.py.

    uv run python scripts/judgment_forms_offline.py retry
    GOOGLE_APPLICATION_CREDENTIALS=<key> uv run --with google-auth --with requests \\
        python scripts/judgment_forms_offline.py candidates checks textual [--gemini]
"""

import argparse
import collections
import io
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from branchlab import PROJECT_ROOT
from branchlab.clef import Clef, image_uri
from branchlab.vlm import VLM
from clef_servo_topview import (AXIS_NAME, BINS, TOL, close_point, crop_image, episodes, label, load_items,
                                to_sign, WORDS, Z)

OUT = PROJECT_ROOT / "outputs/clef_offline"
DRAW = OUT / "judgment_images"
GEMINI = "google/gemini-3.8-flash"
COLOR = {"left": "cyan", "right": "orange"}
NOT_OBJECTS = {"wall", "table"}


# ----------------------------------------------------------------------------- top camera geometry
# DEFAULT_TOP_CAMERA in third_party/robodawn/harness/robotwin/env.py: 640 x 480, fovy 50 deg, at
# (0, -0.10, 1.65 + table_z_bias) m looking straight down, image right = +x, image up = +y.
def top_project(points_cm, table_z_cm):
    f = 240.0 / math.tan(math.radians(25.0))
    cam_z = 165.0 + (table_z_cm - 74.0)
    out = []
    for x, y, z in points_cm:
        depth = cam_z - z
        out.append((320.0 + f * x / depth, 240.0 - f * (y + 10.0) / depth))
    return out


def check_projection(items, n=60):
    """Mean pixel distance between projected fingertips and the harness's drawn markers."""
    from clef_servo_topview import marker_centre
    errs = []
    for item in items[:n]:
        img = Image.open(item["images"]["top_camera"]).convert("RGB")
        found = marker_centre(img, item["arm"])
        if found:
            (u, v), = top_project([item["state"][item["arm"]]["position_cm"]], item["state"]["table_z_cm"])
            errs.append(math.hypot(u - found[0], v - found[1]))
    return np.median(errs), np.percentile(errs, 90), len(errs)


# ----------------------------------------------------------------------------- shared
def jpeg_uri(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask_all(jobs, workers):
    """jobs: (key, model, state, questions, images); returns {key: result or None}."""
    clients = {"clef": Clef(cache_path=OUT / "responses_judgment.jsonl")}
    if any(j[1] == GEMINI for j in jobs):
        clients["gemini"] = VLM(cache_path=OUT / "responses_judgment_gemini.jsonl", backend="vertex")

    def one(job):
        key, model, state, questions, images = job
        try:
            client = clients["gemini" if model == GEMINI else "clef"]
            return key, client.ask(state, questions, images, model=model, key=key)
        except Exception as e:
            print(f"failed {key}: {str(e)[:120]}", flush=True)
            return key, None
    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, jobs))


def grasps():
    """Successful grasps: (task, episode dir, trace, grasp turn index, arm, grasp point, segment start)."""
    out = []
    for task, ep_dir, trace in episodes():
        last = -1
        for k, turn in enumerate(trace):
            for i, res in enumerate(turn["results"]):
                if res["kind"] == "gripper" and res.get("ok") and "probably grasped" in (res.get("note") or ""):
                    point = close_point(trace, k, i, res["arm"])
                    if point is not None:
                        out.append((task, ep_dir, trace, k, res["arm"], point, last + 1))
                    last = k
    return out


def target_text(trace, k):
    from clef_offline import abstract
    return abstract(trace[k].get("plan") or "")


# ----------------------------------------------------------------------------- candidates
def draw_candidates(path, out_path, points_cm, table_z_cm, crop_px=170, size=512):
    img = Image.open(path).convert("RGB")
    uv = top_project(points_cm, table_z_cm)
    cu, cv = np.mean(uv, axis=0)
    half = crop_px / 2
    box = (int(cu - half), int(cv - half), int(cu + half), int(cv + half))
    scale = size / crop_px
    img = img.crop(box).resize((size, size), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=22)
    for n, (u, v) in enumerate(uv, start=1):
        x, y = (u - box[0]) * scale, (v - box[1]) * scale
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 0, 255), outline=(0, 0, 0))
        draw.text((x + 7, y - 26), str(n), fill=(255, 255, 0), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def candidates_test(models, workers):
    rng = random.Random(0)
    items = []
    for task, ep_dir, trace, k, arm, point, start in grasps():
        # the latest turn of the segment with the fingertips still > 8 cm away (the target is not hidden under them)
        view = None
        for j in range(start, k + 1):
            tip = trace[j]["state"][arm]["position_cm"]
            if math.hypot(point[0] - tip[0], point[1] - tip[1]) > 8.0:
                view = j
        if view is None:
            continue
        img = ep_dir / f"turn{int(trace[view]['turn']):03d}_top_camera.png"
        if not img.exists():
            continue
        off = (rng.uniform(-3, 3), rng.uniform(-3, 3))
        grid = [(point[0] + off[0] + dx, point[1] + off[1] + dy, point[2])
                for dy in (3, 0, -3) for dx in (-3, 0, 3)]  # reading order: top row first
        out = DRAW / "candidates" / f"{task}__{ep_dir.parent.name}__{ep_dir.name}__t{k}.png"
        if not out.exists():
            draw_candidates(img, out, grid, trace[view]["state"]["table_z_cm"])
        items.append({"id": f"{task}/{ep_dir.parent.name}/{ep_dir.name}#{k}", "arm": arm, "grid": grid, "point": point,
                      "image": out, "target": target_text(trace, k), "task_text": trace[k]["prompt"].split("\n", 1)[0]})
    q = {"mark": {"type": "choice", "criteria": {str(n): f"mark {n}" for n in range(1, 10)},
                  "instructions": None}}
    jobs = []
    for model in models:
        for it in items:
            qq = json.loads(json.dumps(q))
            qq["mark"]["instructions"] = (
                f"The image is an enlarged view looking straight down at the table. Nine numbered magenta dots mark "
                f"candidate points. Which numbered dot lies where the {it['arm'].upper()} gripper should close its "
                f"fingers to grasp the target, i.e. at the centre of the part to grasp? Target: {it['target']}")
            jobs.append((f"C|{model}|{it['id']}", model,
                         {"task": it["task_text"], "target_from_planner": it["target"]}, qq, [image_uri(it["image"])]))
    res = ask_all(jobs, workers)
    print(f"\n=== candidates: {len(items)} successful grasps, 9 marks 3 cm apart around the true grasp point")
    dist = lambda it, n: math.hypot(it["grid"][n][0] - it["point"][0], it["grid"][n][1] - it["point"][1])
    rows = {"oracle (nearest mark)": [min(dist(it, n) for n in range(9)) for it in items],
            "random mark": [np.mean([dist(it, n) for n in range(9)]) for it in items],
            "centre mark (5)": [dist(it, 4) for it in items]}
    for model in models:
        picks = []
        for it in items:
            r = res.get(f"C|{model}|{it['id']}")
            choice = r and r["answers"]["mark"].get("choice")
            if choice:
                picks.append(dist(it, int(choice) - 1))
        rows[model] = picks
    print(f"{'picker':24} {'n':>4} {'median cm':>10} {'mean cm':>8} {'<=2 cm':>7} {'<=4 cm':>7}")
    for name, d in rows.items():
        d = np.array(d)
        print(f"{name:24} {len(d):4d} {np.median(d):10.2f} {d.mean():8.2f} {np.mean(d <= 2):7.0%} {np.mean(d <= 4):7.0%}")
    print("reference: Gemini's own xy estimate of objects, median error 3.9 cm (reports/RoboDawn_하네스_분석.md)")


# ----------------------------------------------------------------------------- checks
def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    return np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])


def report_binary(name, pairs):
    scores, labels = [p for p, _ in pairs], [l for _, l in pairs]
    yes = [p >= 0.5 for p in scores]
    tp = sum(y and l for y, l in zip(yes, labels))
    fp = sum(y and not l for y, l in zip(yes, labels))
    tn = sum((not y) and (not l) for y, l in zip(yes, labels))
    fn = sum((not y) and l for y, l in zip(yes, labels))
    print(f"{name:28} n {len(pairs):4d}  positives {sum(labels):4d}  AUC {auc(scores, labels):.2f}  "
          f"yes-rate {np.mean(yes):.0%}  precision {tp / max(tp + fp, 1):.0%}  recall {tp / max(tp + fn, 1):.0%}  "
          f"false-yes {fp / max(fp + tn, 1):.0%}")


def checks_test(models, workers):
    # preclose: turns whose first command closes an empty gripper
    pre = []
    for task, ep_dir, trace in episodes():
        for k, turn in enumerate(trace):
            res = turn["results"]
            if not res or res[0]["kind"] != "gripper" or not res[0].get("ok"):
                continue
            value = float(turn["commands"][0].split()[-1]) if turn["commands"][0].split()[-1][0].isdigit() else 0.0
            arm = res[0]["arm"]
            if value >= 0.5 or turn["state"][arm].get("gripper", 1) < 0.5:
                continue
            note = res[0].get("note") or ""
            wrist = ep_dir / f"turn{int(turn['turn']):03d}_{arm}_camera.png"
            if wrist.exists():
                pre.append({"id": f"{task}/{ep_dir.parent.name}/{ep_dir.name}#{k}", "arm": arm, "wrist": wrist,
                            "top": ep_dir / f"turn{int(turn['turn']):03d}_top_camera.png",
                            "state": turn["state"], "label": "probably grasped" in note,
                            "target": target_text(trace, k), "task_text": turn["prompt"].split("\n", 1)[0]})
    steps = load_items()[0]
    jobs = []
    for model in models:
        for it in pre:
            crop = crop_image({"id": "pre/" + it["id"], "arm": it["arm"], "images": {"top_camera": it["top"]}})
            images = [jpeg_uri(it["wrist"], (320, 240))] + ([jpeg_uri(crop, (320, 320))] if crop else [])
            q = {"grasp": {"type": "noul", "instructions":
                           f"The {it['arm'].upper()} gripper is open. If it closed its fingers right now, without moving, "
                           f"would it grasp the target between its fingers? Target: {it['target']}. Image 1 is the "
                           f"{it['arm']} wrist camera looking along the fingers; image 2 is an enlarged straight-down view "
                           f"centred on the {COLOR[it['arm']]} fingertip marker."}}
            jobs.append((f"P|{model}|{it['id']}", model, {"task": it["task_text"], "target_from_planner": it["target"]}, q, images))
    for it in steps:
        crop = crop_image(it)
        images = [jpeg_uri(it["images"][f"{it['arm']}_camera"], (320, 240))] + ([jpeg_uri(crop, (320, 320))] if crop else [])
        q = {"aligned": {"type": "noul", "instructions":
                         f"Are the {it['arm'].upper()} arm's fingertips within about 2 cm, horizontally, of the point "
                         f"where they should grasp the target (they may still be above it)? Target: {it['target']}. "
                         f"Image 1 is the {it['arm']} wrist camera looking along the fingers; image 2 is an enlarged "
                         f"straight-down view centred on the {COLOR[it['arm']]} fingertip marker."}}
        jobs.append((f"A|clef|{it['id']}", "clef", {"task": it["instruction"], "target_from_planner": it["target"]}, q, images))
    res = ask_all(jobs, workers)
    print(f"\n=== checks (P(yes) >= 0.5 counts as yes)")
    for model in models:
        pairs = [(res[f"P|{model}|{it['id']}"]["answers"]["grasp"]["noul"], it["label"])
                 for it in pre if res.get(f"P|{model}|{it['id']}")]
        report_binary(f"preclose / {model}", pairs)
    pairs = [(res[f"A|clef|{it['id']}"]["answers"]["aligned"]["noul"], math.hypot(*it["delta"][:2]) <= 2.0)
             for it in steps if res.get(f"A|clef|{it['id']}")]
    report_binary("aligned / clef", pairs)


# ----------------------------------------------------------------------------- retry
def retry_test():
    rows = []
    for task, ep_dir, trace in episodes():
        res_path = ep_dir.parent / "results.json"
        meta = {e["episode_index"]: e for e in json.loads(res_path.read_text())["episodes"]}
        success = bool(meta[int(ep_dir.name.split("_")[1])]["success"])
        empties = [k for k, turn in enumerate(trace) for r in turn["results"]
                   if r["kind"] == "gripper" and "closed fully" in (r.get("note") or "")]
        secs = [float(t["latency_s"] or 0) for t in trace]
        rows.append((task, success, empties, len(trace), secs))
    total_turns = sum(r[3] for r in rows)
    total_secs = sum(sum(r[4]) for r in rows)
    succ = sum(r[1] for r in rows)
    print(f"\n=== retry cap: stop (or hand back) at the N-th empty grasp; {len(rows)} episodes, {succ} successes, "
          f"{total_turns} turns")
    print(f"{'N':>3} {'episodes reaching N':>20} {'of which succeeded':>19} {'turns after':>12} {'model time after':>17}")
    for n in range(1, 7):
        hit = [r for r in rows if len(r[2]) >= n]
        lost = sum(r[1] for r in hit)
        after = sum(r[3] - (r[2][n - 1] + 1) for r in hit)
        after_s = sum(sum(r[4][r[2][n - 1] + 1:]) for r in hit)
        print(f"{n:3d} {len(hit):20d} {lost:19d} {after:6d} ({after / total_turns:4.0%}) {after_s / total_secs:16.0%}")


# ----------------------------------------------------------------------------- textual
def object_table(state):
    lines = []
    for key, o in state["objects"].items():
        if key in NOT_OBJECTS:
            continue
        p = o["position_cm"]
        pts = "; marked points " + ", ".join(f"({a:.1f}, {b:.1f}, {c:.1f})" for a, b, c in o.get("functional_points_cm", [])) \
            if o.get("functional_points_cm") else ""
        lines.append(f"{key} ({o.get('model', '')}): origin ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}){pts}")
    return "\n".join(lines)


def gemini_table(trace, start, j):
    """Gemini's coordinates from the first turn of the segment that has any (its judgment, kept fixed)."""
    from analyze_robodawn_traces import parse_scene_coords
    for t in trace[start: j + 1]:
        coords = parse_scene_coords(t.get("scene") or "")
        if coords:
            return "\n".join(f"{c['subject'][-80:]}: ({c['x']}, {c['y']}{', ' + str(c['z']) if c['z'] is not None else ''})"
                             for c in coords), t["turn"]
    return None, None


def textual_test(workers):
    steps, _ = load_items()
    traces = {}
    for task, ep_dir, trace in episodes():
        traces[f"{task}/{ep_dir.parent.name}/{ep_dir.name}"] = trace
    segments = {}
    for task, ep_dir, trace, k, arm, point, start in grasps():
        segments[f"{task}/{ep_dir.parent.name}/{ep_dir.name}#{k}"] = start
    jobs, meta = [], {}
    for it in steps:
        ep_id, turn_no = it["id"].split("#")
        trace = traces[ep_id]
        j = next(i for i, t in enumerate(trace) if str(t["turn"]) == turn_no)
        grasp_k = min(k for k in (int(s.split("#")[1]) for s in segments if s.startswith(ep_id + "#")) if k >= j)
        start = segments[f"{ep_id}#{grasp_k}"]
        tip = it["state"][it["arm"]]["position_cm"]
        common = {"conventions": "World frame in cm: +x right, +y forward (away from the robot), +z up. "
                                 f"Table top at z = {it['state']['table_z_cm']}.",
                  "task": it["instruction"], "target_from_planner": it["target"], "moving_arm": it["arm"].upper(),
                  "fingertips": f"{it['arm'].upper()} fingertip centre at ({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f})"}
        qs = {}
        for axis in "xyz":
            sides = Z if axis == "z" else WORDS["world"][axis]
            criteria = {key: text for key, text in sides.values()}
            criteria["stay"] = f"stay: already within about 2 cm of the grasp point {AXIS_NAME[axis]}"
            qs[axis] = {"type": "choice", "criteria": criteria, "instructions":
                        f"Using the coordinates, which way should the {it['arm'].upper()} fingertips take one small step "
                        f"{AXIS_NAME[axis]} to reach the grasp point on the target, or stay?"}
        jobs.append((f"X|gt|{it['id']}", "clef", {**common, "objects": object_table(it["state"])}, qs, []))
        table, from_turn = gemini_table(trace, start, j)
        if table:
            jobs.append((f"X|gemini|{it['id']}", "clef",
                         {**common, f"objects (estimated at turn {from_turn})": table}, qs, []))
        meta[it["id"]] = it
    res = ask_all(jobs, workers)
    print("\n=== textual: per-axis step direction from coordinates in text, no images (clef)")
    for table in ("gt", "gemini"):
        print(f"--- object table: {'simulator poses (upper bound)' if table == 'gt' else 'Gemini estimates, fixed at the sub-goal start'}")
        for a, axis in enumerate("xyz"):
            pairs = [(meta[i]["delta"][a], to_sign(axis, res[f"X|{table}|{i}"]["answers"][axis].get("choice"), "world"))
                     for i in meta if res.get(f"X|{table}|{i}")]
            cells = []
            for lo, hi in BINS:
                sel = [(d, p) for d, p in pairs if (abs(d) <= TOL if lo == 0 else lo < abs(d) <= hi)]
                if sel:
                    ok = np.mean([label(d) == p for d, p in sel])
                    wrong = np.mean([p not in (None, "stay") and p != label(d) for d, p in sel])
                    cells.append(f"{lo}-{hi if hi < 999 else ''}cm n{len(sel)} ok {ok:.2f} wrong {wrong:.2f}")
            acc = np.mean([label(d) == p for d, p in pairs]) if pairs else float("nan")
            print(f"  {axis}: acc {acc:.2f} (n {len(pairs)}) | " + " | ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tests", nargs="+", choices=["candidates", "checks", "retry", "textual", "projection"])
    parser.add_argument("--gemini", action="store_true", help="also ask Gemini 3.8 Flash (Vertex) in candidates/checks")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    models = ["clef"] + ([GEMINI] if args.gemini else [])
    for test in args.tests:
        if test == "projection":
            print("top camera projection vs drawn markers: median %.1f px, p90 %.1f px (n=%d)"
                  % check_projection(load_items()[0]))
        elif test == "retry":
            retry_test()
        elif test == "candidates":
            candidates_test(models, args.workers)
        elif test == "checks":
            checks_test(models, args.workers)
        elif test == "textual":
            textual_test(args.workers)


if __name__ == "__main__":
    main()
