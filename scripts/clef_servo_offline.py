"""Offline test of Clef as a claw-machine style servo: at every step it only
says, per axis, which way to step (or to stay), never how far.

Ground truth comes from RoboDawn's RoboTwin 2.0 demonstrations. Each turn in
which a gripper closes on an object (the effect note says "holding ...") gives
the fingertip position at the moment of closing, which is taken as the grasp
point. Every earlier turn since the previous grasp, for the same arm, then
gets a label per axis: step + or -, or stay if the fingertips are within
`TOL` cm of the grasp point along that axis.

The model sees the overview image (10 cm grid, fingertip markers), the
fingertip state, the task, and as the target the grasp turn's plan with
numbers removed (what the planner would hand over: "grasp the small yellow
block"). Those plans often say "lower" or "descend", so the z axis is given a
hint; x and y, the claw machine's lever, are the fair test.

Clef is asked three ways: with the image, without it, and with the image but
x/y options named only by image direction (toward the top edge, ...) instead
of world words (forward, back, ...).

    uv run python scripts/clef_servo_offline.py [--models clef google/gemini-3.8-flash] [--workers 6]
"""

import argparse
import collections
import glob
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from branchlab import PROJECT_ROOT
from branchlab.clef import MODELS, Clef, image_uri
from branchlab.vlm import VLM
from clef_offline import CONVENTIONS, abstract

DEMOS = PROJECT_ROOT / "third_party/robodawn/demos/robotwin2/expert"
OUT = PROJECT_ROOT / "outputs/clef_offline"
TOL = 2.0  # cm; within this along an axis the right answer is "stay"
BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 999)]

POS = re.compile(r"(LEFT|RIGHT) fingertips \(([-+\d.]+), ([-+\d.]+), ([-+\d.]+)\) opening ([\d.]+)")
HOLD = re.compile(r"holding the ([^;,)]+)")
CLOSE = re.compile(r"(left|right) gripper (close|0(\.[0-5]\d*)?|0\.60?)$")

AXES = {
    "x": ("left-right", {"-": ("left", "step LEFT in the image (-x)"), "+": ("right", "step RIGHT in the image (+x)")}),
    "y": ("forward-back", {"+": ("forward", "step forward, away from the robot, UP in the image (+y)"),
                           "-": ("back", "step back towards the robot, DOWN in the image (-y)")}),
    "z": ("up-down", {"-": ("down", "step down, towards the table (-z)"), "+": ("up", "step up, away from the table (+z)")}),
}


# Variant wording: name each option only by where it moves the marker in the
# overview image, with no "forward/back" or "up/down" world words.
IMAGE_WORDS = {
    "x": {"-": ("toward_left_edge", "move the marker toward the LEFT edge of the overview image"),
          "+": ("toward_right_edge", "move the marker toward the RIGHT edge of the overview image")},
    "y": {"+": ("toward_top_edge", "move the marker toward the TOP edge of the overview image"),
          "-": ("toward_bottom_edge", "move the marker toward the BOTTOM edge of the overview image")},
}


def tips(state):
    return {m.group(1).lower(): [float(m.group(i)) for i in (2, 3, 4)] for m in POS.finditer(state)}


def label(value):
    return "stay" if abs(value) <= TOL else ("+" if value > 0 else "-")


def load_items():
    items = []
    for path in sorted(glob.glob(str(DEMOS / "*" / "demo.json"))):
        demo = json.load(open(path))
        name, frames = path.split("/")[-2], demo["frames"]
        last = -1
        for k, frame in enumerate(frames):
            cmds = frame.get("commands") or []
            close = [i for i, c in enumerate(cmds) if CLOSE.match(c)]
            hold = HOLD.search(frame.get("effect") or "")
            if not close or not hold:
                continue
            arm = cmds[close[0]].split()[0]
            grasp = tips(frame["state"]).get(arm)
            if grasp is None:
                continue
            for c in cmds[: close[0]]:  # moves of that arm before the close, within the same turn
                t = c.split()
                if t[0] == arm and t[1] == "move":
                    grasp["xyz".index(t[2])] += float(t[3])
            target = abstract(frame.get("plan", ""))
            for j in range(last + 1, k + 1):
                f = frames[j]
                cur = tips(f.get("state", "")).get(arm)
                if cur is None or not f.get("images"):
                    continue
                table = re.search(r"table z = (\d+)", f["state"])
                items.append({
                    "id": f"{name}#{j}", "arm": arm, "task": demo["instruction"], "target": target,
                    "image": DEMOS / name / f["images"][0], "state": f["state"],
                    "table_z": int(table.group(1)) if table else 74,
                    "delta": [grasp[a] - cur[a] for a in range(3)],
                })
            last = k
    return items


def axis_sides(axis, wording):
    return IMAGE_WORDS[axis] if wording == "image" and axis in IMAGE_WORDS else AXES[axis][1]


def questions(arm, wording="world"):
    who = f"the {arm.upper()} arm's fingertips ({'cyan' if arm == 'left' else 'orange'} marker)"
    qs = {}
    for axis, (name, _) in AXES.items():
        sides = axis_sides(axis, wording)
        criteria = {key: text for key, text in sides.values()}
        criteria["stay"] = f"stay: {who} are already within about 2 cm of the grasp point {name}"
        qs[axis] = {"type": "choice", "criteria": criteria,
                    "instructions": f"To bring {who} to the grasp point on the target, which way should they "
                                    f"take one small step {name}, or stay?"}
    return qs


def state_for(item):
    return {"conventions": CONVENTIONS.format(table_z=item["table_z"]), "task": item["task"],
            "target_from_planner": item["target"], "moving_arm": item["arm"].upper(),
            "robot_state": item["state"]}


def to_sign(axis, answer, wording="world"):
    choice = answer.get("choice")
    if choice == "stay":
        return "stay"
    for sign, (key, _) in axis_sides(axis, wording).items():
        if key == choice:
            return sign
    return None


def run(clients, job):
    model, cond, item = job
    images = [image_uri(item["image"])] if cond.startswith("img") else []
    wording = "image" if cond == "img_imagewords" else "world"
    key = f"S|{cond}|{model}|{item['id']}"
    client = clients["clef"] if model in MODELS else clients["vlm"]
    try:
        return job, client.ask(state_for(item), questions(item["arm"], wording), images, model=model, key=key)
    except Exception as e:
        print(f"failed {key}: {str(e)[:120]}", flush=True)
        return job, None


def summarize(results):
    groups = collections.defaultdict(list)
    for (model, cond, item), res in results:
        if res is None:
            continue
        for a, axis in enumerate("xyz"):
            wording = "image" if cond == "img_imagewords" else "world"
            groups[(axis, model, cond)].append((item["delta"][a], to_sign(axis, res["answers"][axis], wording)))
    for axis in "xyz":
        print(f"\n=== axis {axis}" + ("  (target text often says 'lower': hinted)" if axis == "z" else ""))
        print(f"{'model / input':34} {'all':>5} | " + " | ".join(f"{lo}-{hi if hi < 999 else '':>2} cm".ljust(22) for lo, hi in BINS))
        print(f"{'':34} {'acc':>5} | " + " | ".join(f"{'n  ok  wrong  stall':22}" for _ in BINS))
        for (ax, model, cond), pairs in sorted(groups.items()):
            if ax != axis:
                continue
            acc = np.mean([label(d) == p for d, p in pairs])
            cells = []
            for lo, hi in BINS:
                sel = [(d, p) for d, p in pairs if (abs(d) <= TOL if lo == 0 else lo < abs(d) <= hi)]
                if not sel:
                    cells.append(" " * 22)
                    continue
                ok = np.mean([label(d) == p for d, p in sel])
                if lo == 0:
                    cells.append(f"{len(sel):3d} {ok:4.2f}  move {1 - ok:4.2f}".ljust(22))
                else:
                    wrong = np.mean([p not in (None, "stay") and p != label(d) for d, p in sel])
                    stall = np.mean([p == "stay" for d, p in sel])
                    cells.append(f"{len(sel):3d} {ok:4.2f} {wrong:5.2f} {stall:6.2f}".ljust(22))
            print(f"{model + ' / ' + cond:34} {acc:5.2f} | " + " | ".join(cells))
        maj = collections.Counter(label(d) for d, _ in next(p for k, p in groups.items() if k[0] == axis))
        print(f"{'labels':34} {dict(maj)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["clef", "google/gemini-3.8-flash"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    items = load_items()
    print(f"{len(items)} steps before {len({i['id'].split('#')[0] for i in items})} demos' grasps")
    clients = {"clef": Clef(cache_path=OUT / "responses_servo_clef.jsonl"),
               "vlm": VLM(cache_path=OUT / "responses_servo_vlm.jsonl")}
    jobs = []
    for model in args.models:
        for cond in (("img", "noimg", "img_imagewords") if model in MODELS else ("img",)):
            jobs += [(model, cond, item) for item in items]
    print(f"{len(jobs)} calls (cached ones are free)")
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda j: run(clients, j), jobs))
    failed = sum(1 for _, r in results if r is None)
    if failed:
        print(f"{failed} calls failed and are left out")
    summarize(results)
    for model in args.models:
        rs = [r for (m, _, _), r in results if m == model and r]
        print(f"latency {model}: median {np.median([r['latency'] for r in rs]):.2f}s over {len(rs)} calls")


if __name__ == "__main__":
    main()
