#!/usr/bin/env python3
"""Clef on the semantic judgments found in the 40-task reading (reports/RoboDawn_40과제_실패_분석.md): which way an
object faces, and which object is the one picked at the first turn. Depth geometry gives an object's long axis or
plane but not which end is the toe or which face carries the QR code, and it does not match descriptions.

Items come from baseline runs: turn-start images, and the true poses in the trace's privileged state, used only to
label and to place the crops (at run time the code's depth blobs would place them; here that step is idealised).

  toe   place_shoe, place_dual_shoes: each shoe in view. Which way does its toe point (left/right/up/down of the
        picture)? Label from the pose; the checker's target pose has the toe at -x ("tips left", checked on images).
        Comparators: Gemini's own scene text when it states a toe direction for a shoe it names; depth's long axis,
        which leaves the sign to chance (the "axis" column restricts Clef to the two options along the true axis).
  qr    rotate_qrcode: which side of the sign faces the robot? Label from the yaw to the checker's target pose.
  reid  tasks with several objects: the target's crop at turn 1 (REF) against tiles of every object now, at one
        scale. Which tile is REF? Comparators: chance, and the tile nearest REF's turn-1 position.

    python scripts/clef_semantic_offline.py build                # items and crops from outputs/robodawn_colab/*
    python scripts/clef_semantic_offline.py ask [--probes toe qr reid] [--model clef]
    python scripts/clef_semantic_offline.py report

Needs numpy and Pillow; Clef calls go to Workers AI (CLOUDFLARE_* in .env).
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import io
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"src"))
from branchlab.rtscene.release_gate import TOE_OPTIONS, crop_box  # noqa: E402  (the release gate's crop and options)
OUT = PROJECT/"outputs/analysis/clef_semantic"
RUNS = PROJECT/"outputs/robodawn_colab"

TOE_TASKS = ("place_shoe", "place_dual_shoes")
QR_TASKS = ("rotate_qrcode",)
REID_TASKS = ("pick_diverse_bottles", "place_object_scale", "put_bottles_dustbin", "place_bread_basket")
SKIP_OBJECTS = {"wall", "table"}
TILE = 256

# the harness's own cameras (third_party/robodawn/harness/robotwin/env.py), raised with the table (table_z_bias)
CAMERAS = {"agent_camera": dict(fovy=60.0, pos=[0.0, -0.36, 1.50], fwd=[0.0, 0.42, -0.91], left=[-1.0, 0.0, 0.0]),
           "top_camera": dict(fovy=50.0, pos=[0.0, -0.10, 1.65], fwd=[0.0, 0.0, -1.0], left=[-1.0, 0.0, 0.0])}


def project(cam: str, pts_cm, table_z_cm: float, W: int = 640, H: int = 480) -> np.ndarray:
    c = CAMERAS[cam]
    f = np.array(c["fwd"], float); f /= np.linalg.norm(f)
    left = np.array(c["left"], float); left /= np.linalg.norm(left)
    up = np.cross(f, left)
    R = np.stack([-left, -up, f])                                # world -> OpenCV camera
    t = np.array(c["pos"], float) + [0.0, 0.0, (table_z_cm - 74.0) / 100.0]
    fy = (H / 2) / math.tan(math.radians(c["fovy"]) / 2)
    K = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])
    p = (np.asarray(pts_cm, float).reshape(-1, 3) / 100.0 - t) @ R.T
    return ((p / p[:, 2:3]) @ K.T)[:, :2]


def rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                     [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                     [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])


TOE_LOCAL = rot([0.5, 0.5, -0.5, -0.5]).T @ np.array([-1.0, 0.0, 0.0])   # place_shoe / place_dual_shoes target
QR_TARGET = rot([0.707, 0.707, 0.0, 0.0])                                  # rotate_qrcode target

DIRS = {"right": 0.0, "up": 90.0, "left": 180.0, "down": 270.0}           # picture directions (top view: up = +y)


def bucket(vx: float, vy: float) -> tuple[str, float]:
    """Picture direction of a world xy vector and its margin to the nearest bucket boundary (degrees)."""
    a = math.degrees(math.atan2(vy, vx)) % 360
    name = min(DIRS, key=lambda k: abs((a - DIRS[k] + 180) % 360 - 180))
    return name, 45 - abs((a - DIRS[name] + 180) % 360 - 180)


def crop(img: Image.Image, cam: str, centre_cm, half_cm: float, table_z: float, rise_cm: float = 12.0,
         marks: dict | None = None) -> Image.Image:
    """Square crop around a box of +-half_cm in x/y and [z-3, z+rise] cm, scaled to TILE; marks: label -> world cm,
    drawn after scaling so that they stay legible. The box is the release gate's (branchlab.rtscene.release_gate),
    so that the gate's live crops are these."""
    box = crop_box(lambda pts: project(cam, pts, table_z, *img.size), centre_cm, half_cm, rise_cm)
    out = img.crop(box).resize((TILE, TILE), Image.BICUBIC)
    if marks:
        from PIL import ImageFont
        draw, font, k = ImageDraw.Draw(out), ImageFont.load_default(size=26), TILE / (box[2] - box[0])
        for label, pt in marks.items():
            u, v = (project(cam, pt, table_z, *img.size)[0] - box[:2]) * k
            draw.ellipse([u - 15, v - 15, u + 15, v + 15], fill=(255, 0, 255), outline=(255, 255, 255), width=2)
            draw.text((u, v), label, fill=(255, 255, 255), font=font, anchor="mm")
    return out


def tile_sheet(tiles: list[Image.Image], cols: int = 3) -> Image.Image:
    from PIL import ImageDraw
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * (TILE + 8), rows * (TILE + 8)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, t in enumerate(tiles):
        x, y = (i % cols) * (TILE + 8), (i // cols) * (TILE + 8)
        sheet.paste(t, (x, y))
        draw.rectangle([x, y, x + 34, y + 26], fill=(0, 0, 0))
        draw.text((x + 5, y + 5), f"T{i + 1}", fill=(255, 255, 255))
    return sheet


def uri(img: Image.Image) -> str:
    buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def original_trace(task: str, episode: int):
    for path in RUNS.glob(f"*/robodawn/gemini_flash_*/{task}/shard_*/episode_{episode:03d}/trace.json"):
        return json.loads(path.read_text())
    return None


def frames(tasks, source: str):
    """(task, key, frame id, kind, turn, state, {camera: image path}, first frame, Gemini's scene text or None).

    trace: the run's own turn-start images (annotated). replay: scripts/replay_capture.py captures (raw images) at
    every turn start and right before every close and every open of a holding gripper; Gemini's text is attached
    to a turn capture only when the replay is within 2 cm of the original run there."""
    if source == "trace":
        for path in sorted(RUNS.glob("*/robodawn/gemini_flash_*/*/shard_*/episode_*/trace.json")):
            task, ep = path.parts[-4], path.parent
            if task not in tasks:
                continue
            key, trace = f"{path.parts[-7]}/{task}/{ep.name}", json.loads(path.read_text())
            out = []
            for t in trace:
                n = int(t["turn"])
                paths = {c: ep/f"turn{n:03d}_{c}.png" for c in CAMERAS}
                if all(p.exists() for p in paths.values()):
                    out.append((task, key, f"t{n}", "turn", n, t["state"], paths, t.get("scene")))
            for f in out:
                yield (*f[:7], out[0], f[7])
        return
    for path in sorted(RUNS.glob("*/replay/*/*/episode_*/meta.json")):
        meta = json.loads(path.read_text())
        task = meta["task"]
        if task not in tasks:
            continue
        key = f"{meta['run']}/{task}/{path.parent.name}"
        trace = original_trace(task, meta["episode"]) or []
        scene = {int(t["turn"]): t.get("scene") for t in trace}
        out = []
        for c in meta["captures"]:
            if c["kind"] not in ("turn", "preopen", "preclose"):
                continue
            paths = {cam: path.parent/"images"/f"c{c['idx']:03d}_{cam}_raw.jpg" for cam in CAMERAS}
            if not all(p.exists() for p in paths.values()):
                continue
            text = scene.get(c["turn"]) if c["kind"] == "turn" and c.get("divergence_cm", 99) < 2.0 else None
            out.append((task, key, f"c{c['idx']}", c["kind"], c["turn"], c["state"], paths, text))
        for f in out:
            yield (*f[:7], out[0], f[7])


def held_or_near(state, p) -> tuple[bool, bool]:
    """(held, near): a fingertip centre within 10 cm and the object 3 cm off the table; a fingertip within 8 cm in xy."""
    held = near = False
    for arm in ("left", "right"):
        f = np.asarray(state[arm]["position_cm"])
        near |= float(np.linalg.norm(f[:2] - p[:2])) < 8.0
        held |= float(np.linalg.norm(f - p)) < 10.0 and p[2] > state["table_z_cm"] + 3.0 and state[arm]["gripper"] < 0.9
    return bool(held), bool(near)


def in_view(img_size, uv, margin=10) -> bool:
    W, H = img_size
    return margin <= uv[0] <= W - margin and margin <= uv[1] <= H - margin


# ------------------------------------------------------------------ Gemini's own statements (toe comparator)
TOE_RE = re.compile(r"\b(?:tip|toe)s?\b[^.;]{0,45}?\b(left|right|forward|away|backward|back|toward(?:s)? the robot|"
                    r"\+x|-x|\+y|-y)", re.I)
WORD = {"left": "left", "-x": "left", "right": "right", "+x": "right", "forward": "up", "away": "up", "+y": "up",
        "backward": "down", "back": "down", "-y": "down"}


def gemini_toe_claims(task: str, text: str) -> dict[str, str]:
    """Shoe -> picture direction for sentences that name exactly one shoe and state its toe/tip direction."""
    out = {}
    for sentence in re.split(r"(?<=[.;])\s+", text or ""):
        names = set()
        if task == "place_shoe":
            names = {"shoe"} if re.search(r"\bshoe\b", sentence, re.I) else set()
        else:
            names = {f"{m.lower()}_shoe" for m in re.findall(r"\b(left|right)\s+(?:white\s+)?shoe\b", sentence, re.I)}
        m = TOE_RE.search(sentence)
        if len(names) == 1 and m:
            w = m.group(1).lower()
            out[names.pop()] = "down" if w.startswith("toward") else WORD.get(w, "")
    return {k: v for k, v in out.items() if v}


# ------------------------------------------------------------------ build
def build(source: str):
    (OUT/"img").mkdir(parents=True, exist_ok=True)
    items = []
    for task, key, fid, kind, n, st, paths, first_frame, scene in frames(TOE_TASKS + QR_TASKS + REID_TASKS, source):
        first, first_paths = first_frame[5], first_frame[6]
        if True:
            imgs = {c: Image.open(p).convert("RGB") for c, p in paths.items()}
            tz = st["table_z_cm"]
            objs = st["objects"]
            if task in TOE_TASKS:
                claims = gemini_toe_claims(task, scene or "")
                for name, o in objs.items():
                    if o["model"] != "041_shoe":
                        continue
                    p = np.asarray(o["position_cm"])
                    toe = rot(o["quat_wxyz"]) @ TOE_LOCAL
                    if p[2] < tz - 5 or abs(toe[2]) > 0.6 or not in_view((640, 480), project("top_camera", p, tz)[0]):
                        continue
                    label, margin = bucket(toe[0], toe[1])
                    held, near = held_or_near(st, p)
                    iid = f"toe|{key}|{fid}|{name}"
                    crop(imgs["top_camera"], "top_camera", p, 15, tz).save(OUT/"img"/f"{_h(iid)}_top.jpg")
                    crop(imgs["agent_camera"], "agent_camera", p, 15, tz).save(OUT/"img"/f"{_h(iid)}_agent.jpg")
                    # "ends": the code marks both ends of the long axis (here from the pose; depth would give it)
                    a_toe = int(_h(iid), 16) % 2 == 0
                    ends = {"A": p + 9 * toe if a_toe else p - 9 * toe, "B": p - 9 * toe if a_toe else p + 9 * toe}
                    for cam, sfx in (("top_camera", "topAB"), ("agent_camera", "agentAB")):
                        crop(imgs[cam], cam, p, 15, tz, marks=ends).save(OUT/"img"/f"{_h(iid)}_{sfx}.jpg")
                    items.append(dict(id=iid, probe="toe", task=task, ep=key, turn=n, kind=kind, obj=name, label=label,
                                      toe_end="A" if a_toe else "B", margin=round(margin, 1), held=held, near=near,
                                      gemini=claims.get(name)))
            if task in QR_TASKS:
                o = next((o for o in objs.values() if o["model"] == "070_paymentsign"), None)
                if o is None:
                    continue
                p = np.asarray(o["position_cm"])
                rel = rot(o["quat_wxyz"]) @ QR_TARGET.T
                yaw = math.degrees(math.atan2(rel[1, 0], rel[0, 0]))
                if p[2] < tz - 5 or not in_view((640, 480), project("agent_camera", p, tz)[0]):
                    continue
                label = "front" if abs(yaw) < 45 else "back" if abs(yaw) > 135 else "edge"
                held, near = held_or_near(st, p)
                iid = f"qr|{key}|{fid}"
                crop(imgs["agent_camera"], "agent_camera", p, 12, tz, 16).save(OUT/"img"/f"{_h(iid)}_agent.jpg")
                crop(imgs["top_camera"], "top_camera", p, 12, tz, 16).save(OUT/"img"/f"{_h(iid)}_top.jpg")
                items.append(dict(id=iid, probe="qr", task=task, ep=key, turn=n, kind=kind, label=label, yaw=round(yaw, 1),
                                  held=held, near=near))
            if task in REID_TASKS and kind == "turn" and n > 1 and n % 2 == 0:
                cands = {k: np.asarray(o["position_cm"]) for k, o in objs.items()
                         if k not in SKIP_OBJECTS and np.asarray(o["position_cm"])[2] > tz - 5
                         and in_view((640, 480), project("top_camera", o["position_cm"], tz)[0], 30)}
                for target in [k for k in cands if k in first["objects"] and _is_target(task, k)]:
                    p0 = np.asarray(first["objects"][target]["position_cm"])
                    order = sorted(cands, key=lambda k: _h(f"{key}|{n}|{target}|{k}"))[:6]
                    if target not in order or len(order) < 2:
                        continue
                    iid = f"reid|{key}|{fid}|{target}"
                    ref_img = Image.open(first_paths["top_camera"]).convert("RGB")
                    crop(ref_img, "top_camera", p0, 10, first["table_z_cm"]).save(OUT/"img"/f"{_h(iid)}_ref.jpg")
                    tile_sheet([crop(imgs["top_camera"], "top_camera", cands[k], 10, tz) for k in order]).save(
                        OUT/"img"/f"{_h(iid)}_tiles.jpg")
                    nearest = min(order, key=lambda k: float(np.linalg.norm(cands[k][:2] - p0[:2])))
                    held, near = held_or_near(st, cands[target])
                    items.append(dict(id=iid, probe="reid", task=task, ep=key, turn=n, obj=target,
                                      label=f"T{order.index(target) + 1}", k=len(order),
                                      moved=bool(np.linalg.norm(cands[target][:2] - p0[:2]) > 3.0),
                                      nearest_ok=bool(nearest == target), held=held, near=near,
                                      objects=[objs[k]["model"] for k in order]))
    (OUT/"items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    print(collections.Counter((i["probe"], i["task"]) for i in items))


def _is_target(task: str, name: str) -> bool:
    return {"pick_diverse_bottles": name in ("bottle1", "bottle2"),
            "place_object_scale": name == "object"}.get(task, name.startswith(("bottle", "bread")))


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


# ------------------------------------------------------------------ ask
def request(item, arm):
    img = lambda s: uri(Image.open(OUT/"img"/f"{_h(item['id'])}_{s}.jpg"))
    if item["probe"] == "toe" and arm == "ends":
        state = ("A robot is moving shoes on a table. The first picture looks straight down at the shoe in its centre; "
                 "the second shows the same shoe from a camera above and behind the robot. In both, the two ends of "
                 "the shoe are marked with magenta circles labelled A and B. The toe is the closed, rounded front end "
                 "of the shoe; the heel is the back end, next to the opening for the foot and the tongue.")
        q = {"toe": {"type": "choice", "criteria": {"A": "end A is the toe", "B": "end B is the toe"},
                     "instructions": "Which marked end of the shoe in the centre is its toe?"}}
        return state, q, [img("topAB"), img("agentAB")]
    if item["probe"] == "toe":
        views = {"top": ["top"], "both": ["top", "agent"]}[arm]
        state = ("A robot is moving shoes on a table. " +
                 ("The first picture looks straight down at the shoe in its centre; the robot stands at the bottom "
                  "edge of the picture. " if "top" in views else "") +
                 ("The second picture shows the same shoe from a camera above and behind the robot, with the same "
                  "left and right. " if "agent" in views else "") +
                 "The toe is the closed, rounded front end of the shoe; the heel is the back end, next to the "
                 "opening for the foot.")
        q = {"toe": {"type": "choice", "criteria": TOE_OPTIONS,
                     "instructions": "Which way does the toe of the shoe in the centre point, in the first picture?"}}
        return state, q, [img(v) for v in views]
    if item["probe"] == "qr":
        views = {"top": ["agent"], "both": ["agent", "top"]}[arm]
        state = ("A robot is turning a small payment sign: a slanted plate printed with a QR code on its front, held "
                 "up by a support at the back. The first picture is taken from above and behind the robot, looking "
                 "away from the robot." + (" The second picture looks straight down at the same sign; the robot "
                                           "stands at the bottom edge." if len(views) > 1 else ""))
        q = {"face": {"type": "choice", "instructions": "Which side of the sign faces the robot?",
                      "criteria": {"front": "the QR-code face faces the robot", "back": "the back with the support "
                                   "faces the robot", "edge": "the sign is turned sideways, edge-on to the robot"}}}
        return state, q, [img(v) for v in views]
    crit = {f"T{i}": f"tile {i}" for i in range(1, item["k"] + 1)} | {"none": "none of the tiles"}
    state = ("A robot works at a table. The REF picture shows one object at the start, from straight above. The next "
             f"picture shows {item['k']} numbered tiles of objects on the table now, from the same camera and scale.")
    q = {"which": {"type": "choice", "criteria": crit,
                   "instructions": "Which tile shows the same physical object as REF? It may have been moved, turned "
                                   "or picked up since the start, so match its colour, markings and shape."}}
    return state, q, [img("ref"), img("tiles")]


def ask(probes, arms, model):
    from branchlab.clef import Clef
    clef = Clef(cache_path=OUT/"cache.jsonl", backend="workers")
    items = [json.loads(l) for l in open(OUT/"items.jsonl")]
    jobs = [(i, a) for i in items if i["probe"] in probes for a in arms if i["probe"] != "reid" or a == arms[0]]

    def one(job):
        item, arm = job
        state, q, images = request(item, arm)
        r = clef.ask(state, q, images=images, model=model, key=f"{item['id']}|{arm}|{model}")
        (name, ans), = r["answers"].items()
        return dict(id=item["id"], arm=arm, model=model, probs=ans["probabilities"])

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    with open(OUT/"answers.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "answers")


# ------------------------------------------------------------------ report
def report():
    items = {i["id"]: i for i in map(json.loads, open(OUT/"items.jsonl"))}
    answers = {}
    for r in map(json.loads, open(OUT/"answers.jsonl")):
        answers[(r["id"], r["arm"], r["model"])] = r
    by = collections.defaultdict(list)
    for (iid, arm, model), r in answers.items():
        item = items[iid] if arm != "ends" else items[iid] | {"label": items[iid]["toe_end"]}
        by[(items[iid]["probe"], arm, model)].append((item, r["probs"]))
    for (probe, arm, model), rows in sorted(by.items()):
        print(f"\n== {probe} / {arm} / {model}: {len(rows)} items")
        strata = {"all": lambda i: True, "held": lambda i: i.get("held"), "on table": lambda i: not i.get("held"),
                  "near gripper": lambda i: i.get("near")}
        if probe == "toe":
            strata["clear (margin >= 10)"] = lambda i: i["margin"] >= 10
        if probe == "reid":
            strata["moved > 3 cm"] = lambda i: i["moved"]
        for sname, keep in strata.items():
            sub = [(i, p) for i, p in rows if keep(i)]
            if not sub:
                continue
            acc = np.mean([max(p, key=p.get) == i["label"] for i, p in sub])
            line = f"  {sname:22s} n={len(sub):4d}  Clef {acc:.3f}"
            if probe == "toe" and arm != "ends":
                axis = {"left": "right", "right": "left", "up": "down", "down": "up"}
                ax = np.mean([p[i["label"]] > p[axis[i["label"]]] for i, p in sub])
                line += f"  axis-only (vs depth's 0.5) {ax:.3f}"
            if probe == "reid":
                line += (f"  chance {np.mean([1 / i['k'] for i, _ in sub]):.3f}"
                         f"  nearest-to-start {np.mean([i['nearest_ok'] for i, _ in sub]):.3f}")
            print(line)
        if probe == "toe" and arm != "ends":
            claimed = [(i, p) for i, p in rows if i.get("gemini")]
            if claimed:
                g = np.mean([i["gemini"] == i["label"] for i, _ in claimed])
                c = np.mean([max(p, key=p.get) == i["label"] for i, p in claimed])
                print(f"  turns where Gemini states a toe direction: n={len(claimed)}  Gemini {g:.3f}  Clef {c:.3f}")
        conf = collections.Counter((i["label"], max(p, key=p.get)) for i, p in rows)
        print("  confusion (label, answer):", dict(sorted(conf.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "ask", "report"])
    ap.add_argument("--probes", nargs="+", default=["toe", "qr", "reid"])
    ap.add_argument("--arms", nargs="+", default=["top", "both"])
    ap.add_argument("--model", default="clef")
    ap.add_argument("--source", choices=["replay", "trace"], default="replay")
    a = ap.parse_args()
    {"build": lambda: build(a.source), "ask": lambda: ask(a.probes, a.arms, a.model), "report": report}[a.cmd]()


if __name__ == "__main__":
    main()
