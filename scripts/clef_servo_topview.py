"""Offline test of the claw-machine servo on our RoboDawn reproduction runs:
does a straight-down view fix the forward/back confusion seen with the
oblique overview?

The earlier test (scripts/clef_servo_offline.py) used RoboDawn's clean-scene
expert demonstrations and only the oblique overview image; there Clef chose
the wrong forward/back direction 51% of the time. Here the steps come from the
100 domain-randomized evaluation episodes we ran with Gemini 3.8 Flash
(outputs/robodawn_colab), which also saved the harness's straight-down map
view (top_camera, same fingertip markers and table grid) and both wrist
cameras every turn.

Ground truth, as before: a gripper close whose result says the fingers
stopped on something ("probably grasped") gives the grasp point, the
fingertip position at the close. Every earlier turn of that arm since the
previous grasp gets, per axis, step +, step - or stay (within TOL cm).

Clef is asked (one call per step, three ternary questions):
  agent      the oblique overview, world words        (the earlier setup)
  top        the straight-down map, map-edge words
  top_wrist  the map plus the moving arm's wrist camera
  top_crop   a crop of the map about 30 cm across, centred on the moving arm's
             marker and enlarged, so target and fingertip take more pixels
and, as a reference for whether the image holds the answer at all,
  gemini_top Gemini 3.8 Flash on the full map (Vertex direct, default thinking),
             on a random subset of --gemini-limit steps

The script also reports how many of Gemini's turns were spent close to the
grasp point (the share of turns the align command could take over).

    uv run python scripts/clef_servo_topview.py [--conditions agent top top_wrist top_crop] [--workers 8] [--limit N]
    GOOGLE_APPLICATION_CREDENTIALS=<key> uv run --with google-auth --with requests \
        python scripts/clef_servo_topview.py --conditions top top_crop gemini_top
"""

import argparse
import collections
import glob
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from branchlab import PROJECT_ROOT
from branchlab.clef import Clef, image_uri
from branchlab.vlm import VLM
from clef_offline import abstract
from clef_servo_offline import BINS, TOL, label

RUNS = PROJECT_ROOT / "outputs/robodawn_colab"
OUT = PROJECT_ROOT / "outputs/clef_offline"
CROP_DIR = OUT / "topview_crops"
CROP_PX, CROP_OUT = 180, 512  # crop side on the 640x480 map (about 30 cm), then enlarged
MARKER_RGB = {"left": (0, 200, 255), "right": (255, 120, 0)}  # harness ring colours (robotwin/env.py)
NEAR_CM = 5.0  # a turn that starts within this horizontal distance of the coming grasp point counts as "near"

COLOR = {"left": "cyan", "right": "orange"}
VIEW = {
    "agent": ("the overview image is a perspective view from above and behind the robot; its grid lies on the table "
              "top, so for a raised fingertip the marker's position against the grid is shifted by perspective"),
    "top": ("the map image looks straight down at the table: +x is to the RIGHT, +y (away from the robot) is UP "
            "the image; the grid lies on the table top, every 10 cm"),
    "crop": ("the map image is an enlarged crop, about 30 cm across, of a camera looking straight down at the table, "
             "centred on the moving arm's fingertip marker: +x is to the RIGHT, +y (away from the robot) is UP the "
             "image; the grey grid lines lie on the table top, every 10 cm"),
}
WORDS = {  # option key -> description, per wording
    "world": {"x": {"-": ("left", "step LEFT (-x)"), "+": ("right", "step RIGHT (+x)")},
              "y": {"+": ("forward", "step forward, away from the robot (+y)"),
                    "-": ("back", "step back, towards the robot (-y)")}},
    "map": {"x": {"-": ("toward_left_edge", "move the marker toward the LEFT edge of the map image (-x)"),
                  "+": ("toward_right_edge", "move the marker toward the RIGHT edge of the map image (+x)")},
            "y": {"+": ("toward_top_edge", "move the marker toward the TOP edge of the map image (+y)"),
                  "-": ("toward_bottom_edge", "move the marker toward the BOTTOM edge of the map image (-y)")}},
}
Z = {"-": ("down", "step down, towards the table (-z)"), "+": ("up", "step up, away from the table (+z)")}
AXIS_NAME = {"x": "left-right", "y": "forward-back", "z": "up-down"}


def episodes():
    for trace in sorted(glob.glob(str(RUNS / "*/robodawn/gemini_flash_flex/*/shard_*/episode_*/trace.json"))):
        path = Path(trace)
        yield path.parts[-4], path.parent, json.loads(path.read_text())


def close_point(trace, k, res_index, arm):
    """Fingertip position when the fingers closed in turn k, or None if it cannot be told exactly."""
    turn = trace[k]
    results = turn["results"]
    if res_index == len(results) - 1 and k + 1 < len(trace):
        return list(trace[k + 1]["state"][arm]["position_cm"])   # nothing ran after the close in that turn
    pos = list(turn["state"][arm]["position_cm"])
    for res in results[:res_index]:
        if res.get("arm") != arm:
            continue
        if res["kind"] != "move" or not res.get("ok"):
            return None                                         # a rotation or failed move: position unknown
        _, _, axis, amount = res["command"].split()
        pos["xyz".index(axis)] += float(amount)
    return pos


def load_items():
    items, turn_stats = [], []
    for task, ep_dir, trace in episodes():
        events = []
        for k, turn in enumerate(trace):
            for i, res in enumerate(turn["results"]):
                if res["kind"] == "gripper" and res.get("ok") and "probably grasped" in (res.get("note") or ""):
                    events.append((k, i, res["arm"]))
        near = set()
        last = -1
        for k, i, arm in events:
            point = close_point(trace, k, i, arm)
            if point is not None:
                target = abstract(trace[k].get("plan") or "")
                for j in range(last + 1, k + 1):
                    turn = trace[j]
                    cur = turn["state"][arm]["position_cm"]
                    delta = [point[a] - cur[a] for a in range(3)]
                    if np.hypot(delta[0], delta[1]) <= NEAR_CM:
                        near.add(j)
                    images = {v: ep_dir / f"turn{int(turn['turn']):03d}_{v}.png"
                              for v in ("agent_camera", "top_camera", f"{arm}_camera")}
                    if not all(p.exists() for p in images.values()):
                        continue
                    items.append({"id": f"{task}/{ep_dir.parent.name}/{ep_dir.name}#{turn['turn']}", "task": task,
                                  "arm": arm, "target": target, "images": images, "state": turn["state"],
                                  "instruction": turn["prompt"].split("\n", 1)[0].removeprefix("TASK: "),
                                  "delta": delta})
            last = k
        turn_stats.append((len(trace), len(near), sum(float(t["latency_s"] or 0) for t in trace),
                           sum(float(t["latency_s"] or 0) for j, t in enumerate(trace) if j in near)))
    return items, turn_stats


def questions(arm, wording, wrist):
    who = f"the {arm.upper()} arm's fingertips ({COLOR[arm]} marker)"
    qs = {}
    for axis in "xyz":
        sides = Z if axis == "z" else WORDS[wording][axis]
        criteria = {key: text for key, text in sides.values()}
        criteria["stay"] = f"stay: {who} are already within about 2 cm of the grasp point {AXIS_NAME[axis]}"
        hint = (" Judge height from the wrist camera." if axis == "z" and wrist else "")
        qs[axis] = {"type": "choice", "criteria": criteria,
                    "instructions": f"To bring {who} to the grasp point on the target, which way should they take "
                                    f"one small step {AXIS_NAME[axis]}, or stay?{hint}"}
    return qs


def state_for(item, view, wrist):
    st = item["state"]
    lines = [f"{arm.upper()} fingertips at ({', '.join(f'{v:.1f}' for v in st[arm]['position_cm'])}) cm, "
             f"opening {st[arm].get('gripper_real', 0):.2f}" for arm in ("left", "right")]
    images = ["the " + ("map" if view == "top" else "overview") + " image"]
    if wrist:
        images.append(f"the {item['arm'].upper()} wrist camera, looking along its fingers")
    return {"conventions": f"Dual-arm robot at a table. World frame in cm: +x right, +y forward (away from the robot), "
                           f"+z up. Table top at z = {st['table_z_cm']}. LEFT fingertips: cyan circle, RIGHT: orange "
                           f"circle. Images in order: {'; '.join(images)}. {VIEW[view]}.",
            "task": item["instruction"], "target_from_planner": item["target"],
            "moving_arm": item["arm"].upper(), "robot_state": "\n".join(lines)}


CONDITIONS = {"agent": ("agent", "world", False), "top": ("top", "map", False), "top_wrist": ("top", "map", True),
              "top_crop": ("crop", "map", False), "gemini_top": ("top", "map", False)}
GEMINI = "google/gemini-3.8-flash"


def marker_centre(img, arm):
    """Pixel centre of the arm's fingertip ring: the point with the most marker-coloured pixels at radius 5."""
    mask = np.all(np.asarray(img)[..., :3] == MARKER_RGB[arm], axis=-1)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring = np.stack([np.round(5 * np.sin(angles)), np.round(5 * np.cos(angles))], axis=1).astype(int)
    best, best_score = None, -1
    for cy in range(max(ys.min() - 2, 5), min(ys.max() + 3, mask.shape[0] - 5)):
        for cx in range(max(xs.min() - 2, 5), min(xs.max() + 3, mask.shape[1] - 5)):
            score = mask[cy + ring[:, 0], cx + ring[:, 1]].sum()
            if score > best_score:
                best, best_score = (cx, cy), score
    return best if best_score >= 12 else None


def crop_image(item):
    """Enlarged crop of the map around the moving arm's marker (cached as PNG); None if no marker is found."""
    from PIL import Image
    path = CROP_DIR / (item["id"].replace("/", "__").replace("#", "_t") + f"_{item['arm']}.png")
    if not path.exists():
        img = Image.open(item["images"]["top_camera"]).convert("RGB")
        centre = marker_centre(img, item["arm"])
        if centre is None:
            return None
        half = CROP_PX // 2
        box = (centre[0] - half, centre[1] - half, centre[0] + half, centre[1] + half)
        CROP_DIR.mkdir(parents=True, exist_ok=True)
        img.crop(box).resize((CROP_OUT, CROP_OUT), Image.LANCZOS).save(path)  # outside the frame: black
    return path


def run(clients, job):
    cond, item = job
    view, wording, wrist = CONDITIONS[cond]
    if view == "crop":
        crop = crop_image(item)
        if crop is None:
            return job, None
        images = [image_uri(crop)]
    else:
        images = [image_uri(item["images"]["top_camera" if view == "top" else "agent_camera"])]
    if wrist:
        images.append(image_uri(item["images"][f"{item['arm']}_camera"]))
    gemini = cond.startswith("gemini")
    key = f"T|{cond}|{'gemini' if gemini else 'clef'}|{item['id']}"
    try:
        return job, clients["vlm" if gemini else "clef"].ask(
            state_for(item, view, wrist), questions(item["arm"], wording, wrist), images,
            model=GEMINI if gemini else "clef", key=key)
    except Exception as e:
        print(f"failed {key}: {str(e)[:120]}", flush=True)
        return job, None


def to_sign(axis, choice, wording):
    if choice == "stay":
        return "stay"
    for sign, (key, _) in (Z if axis == "z" else WORDS[wording][axis]).items():
        if key == choice:
            return sign
    return None


def summarize(results):
    groups = collections.defaultdict(list)
    for (cond, item), res in results:
        if res is not None:
            for a, axis in enumerate("xyz"):
                groups[(axis, cond)].append((item["delta"][a], to_sign(axis, res["answers"][axis].get("choice"),
                                                                       CONDITIONS[cond][1])))
    for axis in "xyz":
        print(f"\n=== axis {axis}" + ("  (target text often says 'lower': hinted)" if axis == "z" else ""))
        print(f"{'condition':12} {'acc':>5} | " + " | ".join(f"{lo}-{hi if hi < 999 else '':>2} cm".ljust(22) for lo, hi in BINS))
        print(f"{'':12} {'':>5} | " + " | ".join(f"{'n  ok  wrong  stall':22}" for _ in BINS))
        for (ax, cond), pairs in sorted(groups.items()):
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
            print(f"{cond:12} {acc:5.2f} | " + " | ".join(cells))
        first = next(p for k, p in groups.items() if k[0] == axis)
        print(f"{'labels':12} {dict(collections.Counter(label(d) for d, _ in first))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="random subset of steps, for a quick look")
    parser.add_argument("--gemini-limit", type=int, default=300, help="random subset of steps for gemini_top")
    args = parser.parse_args()
    items, turn_stats = load_items()
    turns, near, secs, near_secs = map(sum, zip(*turn_stats))
    print(f"{len(turn_stats)} episodes, {turns} Gemini turns; {near} turns ({near / turns:.0%}) and "
          f"{near_secs / secs:.0%} of model time started within {NEAR_CM:.0f} cm (horizontal) of the coming grasp point")
    print(f"{len(items)} steps before {len({i['id'].split('#')[0] for i in items})} episodes' grasps")
    if args.limit:
        items = random.Random(0).sample(items, min(args.limit, len(items)))
    clients = {"clef": Clef(cache_path=OUT / "responses_servo_topview.jsonl")}
    if any(c.startswith("gemini") for c in args.conditions):
        clients["vlm"] = VLM(cache_path=OUT / "responses_servo_topview_gemini.jsonl", backend="vertex")
    gemini_items = random.Random(1).sample(items, min(args.gemini_limit, len(items)))
    jobs = [(cond, item) for cond in args.conditions for item in (gemini_items if cond.startswith("gemini") else items)]
    print(f"{len(jobs)} calls (cached ones are free)")
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda j: run(clients, j), jobs))
    failed = sum(r is None for _, r in results)
    if failed:
        print(f"{failed} calls failed and are left out")
    summarize(results)
    lat = [r["latency"] for _, r in results if r and "latency" in r]
    if lat:
        print(f"\nClef latency: median {np.median(lat):.2f}s over {len(lat)} calls")


if __name__ == "__main__":
    main()
