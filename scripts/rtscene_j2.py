#!/usr/bin/env python3
"""J2: which detected object does the instruction mean? Compare information formats and models.

For every distinct scene (task, seed) of the replayed episodes, the first capture is parsed into objects
(rtscene: depth + colour, robot removed), numbered in a random order, and two questions are asked:
  target       which object must the robot pick up / act on;
  destination  which object is the place or reference for it (tasks with one).
The true answer is the object whose segmentation dominates the role's actor (labels only).

Arms (one factor changes at a time):
  code        word matching: instruction words vs colour / size / shape words (no model)
  words       numbered object list in code words, numbers in parentheses
  relations   words + cross-object facts ("the tallest", "the only red object")
  numbers     the same measurements as numbers only (Lab colour, cm)
  image       numbered markers on the top and agent camera images, no attributes
  words+image words + the numbered images
Models: Clef (cloudflare/clef), Gemini 3.1 Flash-Lite and Gemini 3.8 Flash on Vertex.

    PYTHONPATH=src python scripts/rtscene_j2.py build      # parse scenes, write items + images
    PYTHONPATH=src python scripts/rtscene_j2.py ask [--models clef lite flash] [--arms ...]
    PYTHONPATH=src python scripts/rtscene_j2.py report
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT/"outputs/analysis/j2"
ROOTS = [PROJECT/"outputs/replay_local/replay", *sorted((PROJECT/"outputs/robodawn_colab").glob("*/replay"))]
TARGET = {"adjust_bottle": "bottle", "click_bell": "bell", "lift_pot": "pot", "place_a2b_right": "object",
          "place_empty_cup": "cup", "place_fan": "fan", "handover_block": "box", "move_can_pot": "can",
          "open_laptop": "laptop", "place_object_stand": "object"}
DEST = {"place_a2b_right": "target_object", "place_empty_cup": "coaster", "place_fan": "pad",
        "handover_block": "target_box", "move_can_pot": "pot", "place_object_stand": "displaystand"}
# clef goes through OpenRouter, which (found 2026-10-09) does not pass images on to Clef; clefw calls Cloudflare
# Workers AI directly, where Clef does see them
MODELS = {"clef": "clef", "clefw": "clef", "lite": "google/gemini-3.1-flash-lite", "flash": "google/gemini-3.8-flash"}
ARMS = ("words", "relations", "numbers", "image", "words+image", "tiles", "tiles2", "tiles-yesno", "crop-yesno")
WORDING = {"target": "the object the instruction asks the robot to pick up, move, or act on",
           "destination": "the place or reference object in the instruction (what the object goes on, beside, or to the right of)"}
TILE_TEXT = {"tiles": "IMAGE 1: every object on the table as a zoomed tile seen from the front, labelled with its number.",
             "tiles2": "IMAGE 1: every object as a zoomed tile seen from the front; IMAGE 2: the same objects seen from straight "
                       "above, same numbers.",
             "tiles-yesno": "IMAGE 1: every object as a zoomed tile seen from the front; IMAGE 2: the same objects seen from "
                            "straight above, same numbers."}
QUESTION = {
    "target": "Which object (O1-O{n}) does the instruction ask the robot to pick up, move, or act on?",
    "destination": "Which object (O1-O{n}) is the place or the reference object for it in the instruction "
                   "(what it goes on, beside, or to the right of)?",
}


# ----------------------------------------------------------------------------- build
def scenes():
    from branchlab.rtscene.capture import episodes
    seen = {}
    for root in ROOTS:
        if root.is_dir():
            for ep in episodes(root):
                seen.setdefault((ep.task, ep.meta["seed"]), ep)
    return list(seen.values())


def draw_numbers(img: np.ndarray, uv: list[tuple[int, float, float]]) -> bytes:
    from PIL import Image, ImageDraw
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    for n, u, v in uv:
        if not (0 <= u < im.width and 0 <= v < im.height):
            continue
        d.ellipse([u - 4, v - 4, u + 4, v + 4], outline=(255, 0, 255), width=2)
        label = f"O{n}"
        d.rectangle([u + 5, v - 7, u + 7 + 7 * len(label), v + 7], fill=(0, 0, 0))
        d.text((u + 7, v - 6), label, fill=(255, 255, 0))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def build():
    from branchlab.rtscene import blobs as blobs_mod, colour, heightmap, labels, phrases, robot_model
    from branchlab.rtscene.geometry import project
    boxes = robot_model.load()
    (OUT/"images").mkdir(parents=True, exist_ok=True)
    items = []
    for ep in scenes():
        cap = ep.captures()[0]
        hm = heightmap.build(cap, boxes)
        bl = [b for b in blobs_mod.extract(hm, flat=colour.flat_mask(hm, colour.table_model(hm)))
              if b.robot_gap >= 1.0 and b.area >= 6.0]
        white = float(np.percentile(colour.to_lab(hm.rgb)[..., 0][hm.source == 1], 99))
        desc = phrases.describe(bl, white)
        actors = labels.blob_actors(ep, cap, bl)
        ids = labels.object_ids(ep)
        order = list(desc)
        random.Random(f"{ep.task}/{ep.meta['seed']}").shuffle(order)
        number = {bid: i + 1 for i, bid in enumerate(order)}
        truth = {}
        for q, roles in (("target", TARGET), ("destination", DEST)):
            role = roles.get(ep.task)
            if not role:
                continue
            cands = [b for b in bl if actors[b.id][0] in ids.get(role, []) and actors[b.id][1] >= 0.5]
            truth[q] = f"O{number[max(cands, key=lambda b: b.area).id]}" if cands else "none"
        name = f"{ep.task}_{ep.meta['seed']}"
        from branchlab.rtscene import render
        by_id = {b.id: b for b in bl}
        for cam, tag in (("agent_camera", "agent"), ("top_camera", "top")):
            tiles = [render.crop(cap, cam, by_id[b].centroid, max(by_id[b].length, by_id[b].width) / 2 + 3, by_id[b].h95,
                                 size=192, label=f"O{number[b]}") for b in sorted(order, key=lambda b: number[b])]
            render.tile_sheet(tiles, cols=4).save(OUT/"images"/f"{name}_tiles_{tag}.jpg", quality=90)
            for b in order:
                render.crop(cap, cam, by_id[b].centroid, max(by_id[b].length, by_id[b].width) / 2 + 3, by_id[b].h95,
                            size=320).save(OUT/"images"/f"{name}_O{number[b]}_{tag}.jpg", quality=90)
        for cam in ("top_camera", "agent_camera"):
            K = cap.K(cam).copy()
            K[:2] *= 2                                  # raw images are full resolution
            pts = [[desc[b]["x"], desc[b]["y"], cap.table_z + desc[b]["h95"]] for b in order]
            uv, _ = project(np.array(pts), K, cap.E(cam))
            (OUT/"images"/f"{name}_{cam}.jpg").write_bytes(
                draw_numbers(cap.rgb(cam), [(number[b], u, v) for b, (u, v) in zip(order, uv)]))
        items.append({"scene": name, "task": ep.task, "seed": ep.meta["seed"], "instruction": ep.instruction,
                      "objects": [[number[b], {k: v for k, v in desc[b].items()}] for b in order],
                      "truth": truth})
        print(name, len(order), "objects", truth, flush=True)
    (OUT/"items.json").write_text(json.dumps(items, default=float))
    print(len(items), "scenes")


# ----------------------------------------------------------------------------- code matcher
CATEGORY = {r"\b(cup|mug|bowl|pot|kitchenpot|basket|pencup)\b": {"open round container"},
            r"\b(block|box|cube|rubikscube|stapler|deck)\b": {"box-shaped object"},
            r"\b(can|bottle|cylinder)\b": {"round object", "box-shaped object"},
            r"\b(mat|pad|coaster|plate)\b": {"flat sheet", "flat round sheet"},
            r"\b(bell)\b": {"round object"}}
SIZE_WORDS = {"small": {"tiny", "small"}, "compact": {"tiny", "small"}, "hand-sized": {"small"}, "palm-sized": {"small"},
              "large": {"large", "very large"}, "big": {"large", "very large"}, "medium": {"medium"}}


def split_instruction(text: str) -> tuple[str, str]:
    m = re.search(r"\b(on top of|onto|on the right of|to the right of|on the right|beside|near|next to| on )\b", text, re.I)
    return (text[:m.start()], text[m.start():]) if m else (text, "")


def word_score(phrase: str, d: dict) -> float:
    from branchlab.rtscene import colour
    p = phrase.lower()
    s = 0.0
    names = {d["colour"]}
    for w, syn in colour.SYNONYMS.items():
        if re.search(rf"\b{w}\b", p):
            s += 2.0 if names & syn else 0.0
    for w in ("red", "orange", "yellow", "lime", "green", "teal", "blue", "purple", "pink", "white", "grey", "black", "brown"):
        if re.search(rf"\b{w}\b", p.replace("gray", "grey")):
            s += 3.0 if w in d["colour"] else -1.0
    for pattern, shapes in CATEGORY.items():
        if re.search(pattern, p):
            s += 2.0 if d["shape"] in shapes else 0.0
    for w, sizes in SIZE_WORDS.items():
        if re.search(rf"\b{w}\b", p):
            s += 1.0 if d["size"] in sizes else 0.0
    return s


def code_answer(item: dict, q: str) -> str:
    first, second = split_instruction(item["instruction"])
    phrase = first if q == "target" else second
    if not phrase:
        return "none"
    scored = sorted(((word_score(phrase, d), n) for n, d in item["objects"]), reverse=True)
    if q == "destination":
        tgt = code_answer(item, "target")
        scored = [(s, n) for s, n in scored if f"O{n}" != tgt]
    return f"O{scored[0][1]}" if scored and scored[0][0] > 0 else "none"


# ----------------------------------------------------------------------------- ask
def image_uris(item: dict) -> list[str]:
    return ["data:image/jpeg;base64," + base64.b64encode((OUT/"images"/f"{item['scene']}_{cam}.jpg").read_bytes()).decode()
            for cam in ("top_camera", "agent_camera")]


def crop_request(item: dict, k: int) -> tuple[str, dict, list[str]]:
    """One object alone: its zoomed front and top crops, and one yes/no question per role."""
    state = (f"TASK INSTRUCTION: {item['instruction']}\n\nIMAGE 1 shows one object on the table, zoomed, seen from the front; "
             "IMAGE 2 shows the same object from straight above.")
    qs = {q: {"type": "noul", "instructions": f"Is this object {WORDING[q]}?"} for q in item["truth"]}
    imgs = ["data:image/jpeg;base64," + base64.b64encode((OUT/"images"/f"{item['scene']}_O{k}_{t}.jpg").read_bytes()).decode()
            for t in ("agent", "top")]
    return state, qs, imgs


def tile_uris(item: dict, both: bool) -> list[str]:
    tags = ("agent", "top") if both else ("agent",)
    return ["data:image/jpeg;base64," + base64.b64encode((OUT/"images"/f"{item['scene']}_tiles_{t}.jpg").read_bytes()).decode()
            for t in tags]


def request(item: dict, arm: str) -> tuple[str, dict, list[str]]:
    from branchlab.rtscene import phrases
    if arm.startswith("tiles"):
        n = len(item["objects"])
        state = f"TASK INSTRUCTION: {item['instruction']}\n\n{TILE_TEXT[arm]}"
        if arm == "tiles-yesno":
            wording = {"target": "the object the instruction asks the robot to pick up, move, or act on",
                       "destination": "the place or reference object in the instruction (what the object goes on, beside, or to "
                                      "the right of)"}
            qs = {f"{q}.O{k}": {"type": "noul", "instructions": f"Is tile O{k} {wording[q]}?"}
                  for q in item["truth"] for k in range(1, n + 1)}
            return state, qs, tile_uris(item, True)
        criteria = {f"O{k}": f"the object in tile O{k}" for k in range(1, n + 1)}
        criteria["none"] = "none of the tiles"
        qs = {q: {"type": "choice", "instructions": QUESTION[q].format(n=n).replace("Which object", "Which tile"),
                  "criteria": criteria} for q in item["truth"]}
        return state, qs, tile_uris(item, arm == "tiles2")
    fmt = {"words": "words", "relations": "relations", "numbers": "numbers", "image": "minimal",
           "words+image": "words"}[arm]
    state = (f"TASK INSTRUCTION: {item['instruction']}\n\nOBJECTS ON THE TABLE (detected by the robot's depth camera):\n"
             + phrases.scene_text([(n, d) for n, d in sorted(item["objects"])], fmt))
    if "image" in arm:
        state += "\n\nIMAGES: 1. straight-down view of the table; 2. view from in front of the robot. Markers O1.. show the objects."
    n = len(item["objects"])
    criteria = {f"O{k}": f"object O{k}" for k in range(1, n + 1)}
    criteria["none"] = "none of the listed objects"
    qs = {q: {"type": "choice", "instructions": QUESTION[q].format(n=n), "criteria": criteria} for q in item["truth"]}
    return state, qs, image_uris(item) if "image" in arm else []


def ask(models, arms, workers):
    from branchlab.clef import Clef
    from branchlab.vlm import VLM
    items = json.loads((OUT/"items.json").read_text())
    clients = {"clef": Clef(cache_path=OUT/"responses_clef.jsonl"),
               "clefw": Clef(cache_path=OUT/"responses_clefw.jsonl", backend="workers"),
               "lite": VLM(cache_path=OUT/"responses_lite.jsonl", backend="vertex"),
               "flash": VLM(cache_path=OUT/"responses_flash.jsonl", backend="vertex")}
    jobs = [(m, arm, it, None) for m in models for arm in arms if arm != "crop-yesno" for it in items]
    if "crop-yesno" in arms:
        jobs += [(m, "crop-yesno", it, k) for m in models for it in items for k in range(1, len(it["objects"]) + 1)]

    def run(job):
        m, arm, it, k = job
        state, qs, imgs = crop_request(it, k) if k else request(it, arm)
        digest = hashlib.sha1(json.dumps([state, qs, imgs]).encode()).hexdigest()[:10]
        key = f"{m}|{arm}|{it['scene']}|{f'O{k}|' if k else ''}{digest}"     # a changed request is asked again
        try:
            return key, clients[m].ask(state, qs, imgs, model=MODELS[m], key=key)
        except Exception as exc:  # noqa: BLE001
            return key, {"error": f"{type(exc).__name__}: {exc}"[:300]}
    with ThreadPoolExecutor(workers) as pool:
        for i, (key, res) in enumerate(pool.map(run, jobs), 1):
            if "error" in res:
                print("ERROR", key, res["error"], flush=True)
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}", flush=True)


def report():
    import collections
    items = json.loads((OUT/"items.json").read_text())
    res = {}
    for f in OUT.glob("responses_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            res[r["key"]] = r["result"]
    rows = collections.defaultdict(lambda: collections.Counter())
    for it in items:
        for q, truth in it["truth"].items():
            if q == "destination" and not split_instruction(it["instruction"])[1]:
                continue                     # the instruction names no destination ("Grab the red block ...")
            if truth == "none":
                rows[("parser", q)]["missed"] += 1
                continue
            rows[("code", q)]["n"] += 1
            rows[("code", q)]["ok"] += code_answer(it, q) == truth
            for m in MODELS:
                for arm in ARMS:
                    if arm != "crop-yesno":
                        state, qs, imgs = request(it, arm)
                        digest = hashlib.sha1(json.dumps([state, qs, imgs]).encode()).hexdigest()[:10]
                        r = res.get(f"{m}|{arm}|{it['scene']}|{digest}")
                        if not r:
                            continue
                    if arm == "tiles-yesno":
                        scores = {k.split(".")[1]: (a.get("noul") or 0) for k, a in r["answers"].items() if k.startswith(q + ".")}
                        pick = max(scores, key=scores.get) if scores else None
                    elif arm == "crop-yesno":
                        scores = {}
                        for k in range(1, len(it["objects"]) + 1):
                            state, qs, imgs = crop_request(it, k)
                            dg = hashlib.sha1(json.dumps([state, qs, imgs]).encode()).hexdigest()[:10]
                            rr = res.get(f"{m}|crop-yesno|{it['scene']}|O{k}|{dg}")
                            if rr and q in rr["answers"]:
                                scores[f"O{k}"] = rr["answers"][q].get("noul") or 0
                        pick = max(scores, key=scores.get) if len(scores) == len(it["objects"]) else None
                        if pick is None:
                            continue
                    else:
                        pick = (r["answers"].get(q) or {}).get("choice")
                    rows[(f"{m}:{arm}", q)]["n"] += 1
                    rows[(f"{m}:{arm}", q)]["ok"] += pick == truth
    print(f"{'arm':24} {'target':>14} {'destination':>14}")
    for name in sorted({k[0] for k in rows}):
        cells = []
        for q in ("target", "destination"):
            c = rows.get((name, q))
            cells.append(f"{c['ok']}/{c['n']} ({c['ok'] / c['n']:.0%})" if c and c["n"] else
                         (f"missed {c['missed']}" if c else "-"))
        print(f"{name:24} {cells[0]:>14} {cells[1]:>14}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("build")
    a = sub.add_parser("ask")
    a.add_argument("--models", nargs="+", default=["clef", "lite"])
    a.add_argument("--arms", nargs="+", default=list(ARMS))
    a.add_argument("--workers", type=int, default=8)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.step == "build":
        build()
    elif args.step == "ask":
        ask(args.models, args.arms, args.workers)
    else:
        report()


if __name__ == "__main__":
    main()
