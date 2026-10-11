#!/usr/bin/env python3
"""Does Clef judge better from several separate images, or from ONE composed image with the same content?

Paired design: the same items, the same question wording except for the sentence that describes the image layout,
and the same pixels. A composed image is the separate images pasted side by side (or stacked) with a thin white gap,
at full size (no shrinking), with the short black label bars already used by the tile sheets ("T1".."Tk"; the REF
crop gets the same kind of bar, "REF", only where REF is not a picture of its own and so cannot be named by order).

  reid  re-identification (items from scripts/clef_semantic_offline.py, probe "reid"): REF = the target's top-camera
        crop at turn 1, tiles = every candidate object now. Which tile is REF? Options T1..Tk + none.
          sep2    REF image + tile-sheet image (two images): the existing request of clef_semantic_offline.py, arm "top"
          sepK    REF image + each tile as its own image (1 + k <= 4 images), tiles named by order
          one_lr  one image: REF (labelled) on the left, the tile sheet on the right
          one_tb  one image: REF (labelled) on top, the tile sheet below

    python scripts/clef_one_vs_many.py reid              # ask every arm (cached), then report
    python scripts/clef_one_vs_many.py reid --report     # report only

  topple  "has this object fallen over since the start?" (items from the replay captures, top camera): START crop at
        the first turn capture, NOW crop now; Clef noul, score = P(yes). See the topple section below.
          sep2    START image + NOW image
          one_lr  one image: START | NOW
          one_tb  one image: START over NOW

    python scripts/clef_one_vs_many.py topple --build    # items + crops only
    python scripts/clef_one_vs_many.py topple            # build if missing, ask every arm (cached), report
    python scripts/clef_one_vs_many.py topple --report   # report only

Answers: outputs/analysis/clef_one_vs_many/answers_<probe>.jsonl (one row per item and arm, with the request's image
sizes and Clef's input-token count; the latter shows whether Clef charges, and so maybe sees, a large image as more).
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import io
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"src"))
SEM = PROJECT/"outputs/analysis/clef_semantic"
OUT = PROJECT/"outputs/analysis/clef_one_vs_many"
TILE, PAD, GAP = 256, 8, 8          # tile size and the sheet's white padding (clef_semantic_offline.tile_sheet); gap


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def uri(img: Image.Image) -> str:
    """The same encoding as clef_semantic_offline.uri (JPEG q90), so that sep2 is byte-identical to the stored arm."""
    buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def label_bar(img: Image.Image, text: str, x: int = 0, y: int = 0) -> Image.Image:
    """The tile sheet's label: a 35x27 black bar at the corner with white default-font text."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + 34, y + 26], fill=(0, 0, 0))
    draw.text((x + 5, y + 5), text, fill=(255, 255, 255))
    return img


def compose(parts: list[Image.Image], horizontal: bool) -> Image.Image:
    """Paste the parts at full size, top-left aligned, with a GAP-pixel white gap between them."""
    W = sum(p.width for p in parts) + GAP * (len(parts) - 1) if horizontal else max(p.width for p in parts)
    H = max(p.height for p in parts) if horizontal else sum(p.height for p in parts) + GAP * (len(parts) - 1)
    out, at = Image.new("RGB", (W, H), (255, 255, 255)), 0
    for p in parts:
        out.paste(p, (at, 0) if horizontal else (0, at))
        at += (p.width if horizontal else p.height) + GAP
    return out


# ------------------------------------------------------------------ statistics
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: binomial(b + c, 1/2) on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def auc(pos, neg) -> float | None:
    """Mann-Whitney AUC, ties counted half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ------------------------------------------------------------------ reid
REID_ARMS = ("sep2", "sepK", "one_lr", "one_tb")
REID_INSTR = ("Which tile shows the same physical object as REF? It may have been moved, turned or picked up since "
              "the start, so match its colour, markings and shape.")


def reid_layout(arm: str, k: int) -> str:
    """The only sentence(s) that differ between arms: where REF and the tiles are."""
    return {
        "sep2": ("The REF picture shows one object at the start, from straight above. The next picture shows "
                 f"{k} numbered tiles of objects on the table now, from the same camera and scale."),
        "sepK": ("The first picture, REF, shows one object at the start, from straight above. The next "
                 f"{k} pictures are {k} numbered tiles, T1 to T{k} in order, of objects on the table now, from the "
                 "same camera and scale."),
        "one_lr": ("The picture has two parts. On the left, the part labelled REF shows one object at the start, from "
                   f"straight above. On the right are {k} numbered tiles of objects on the table now, from the same "
                   "camera and scale."),
        "one_tb": ("The picture has two parts. At the top, the part labelled REF shows one object at the start, from "
                   f"straight above. Below it are {k} numbered tiles of objects on the table now, from the same "
                   "camera and scale."),
    }[arm]


def reid_images(item: dict, arm: str) -> list[Image.Image]:
    h = _h(item["id"])
    ref = Image.open(SEM/"img"/f"{h}_ref.jpg").convert("RGB")
    sheet = Image.open(SEM/"img"/f"{h}_tiles.jpg").convert("RGB")
    assert ref.size == (TILE, TILE) and sheet.size == (3 * (TILE + PAD), TILE + PAD) and item["k"] <= 3
    if arm == "sep2":
        return [ref, sheet]
    if arm == "sepK":                     # each tile cut from the sheet, with its own T-label bar
        return [ref] + [sheet.crop((i * (TILE + PAD), 0, i * (TILE + PAD) + TILE, TILE)) for i in range(item["k"])]
    ref_l = label_bar(ref, "REF")
    return [compose([ref_l, sheet], horizontal=(arm == "one_lr"))]


def reid_request(item: dict, arm: str):
    state = "A robot works at a table. " + reid_layout(arm, item["k"])
    crit = {f"T{i}": f"tile {i}" for i in range(1, item["k"] + 1)} | {"none": "none of the tiles"}
    q = {"which": {"type": "choice", "criteria": crit, "instructions": REID_INSTR}}
    return state, q, reid_images(item, arm)


def reid_ask(arms, model: str, workers: int):
    from branchlab.clef import Clef
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"img").mkdir(exist_ok=True)
    clef = Clef(cache_path=OUT/"cache.jsonl", backend="workers")
    items = [i for i in map(json.loads, open(SEM/"items.jsonl")) if i["probe"] == "reid"]
    # one example of each composed layout, for looking at
    for arm in ("one_lr", "one_tb"):
        for it in (next(i for i in items if i["k"] == 2), next(i for i in items if i["k"] == 3)):
            reid_images(it, arm)[0].save(OUT/"img"/f"example_{arm}_k{it['k']}.jpg", quality=90)

    def one(job):
        item, arm = job
        state, q, imgs = reid_request(item, arm)
        r = clef.ask(state, q, images=[uri(im) for im in imgs], model=model, key=f"{item['id']}|{arm}|{model}")
        (_, ans), = r["answers"].items()
        return dict(id=item["id"], arm=arm, model=model, probs=ans["probabilities"],
                    sizes=[list(im.size) for im in imgs], input_tokens=r.get("usage", {}).get("input_tokens"),
                    latency=round(r.get("latency", 0.0), 3))

    jobs = [(i, a) for i in items for a in arms]
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, jobs))
    (OUT/"answers_reid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "answers ->", OUT/"answers_reid.jsonl")


def _argmax(p: dict) -> str:
    return max(p, key=p.get)


def reid_report(model: str):
    items = {i["id"]: i for i in map(json.loads, open(SEM/"items.jsonl")) if i["probe"] == "reid"}
    rows = [r for r in map(json.loads, open(OUT/"answers_reid.jsonl")) if r["model"] == model]
    A = collections.defaultdict(dict)                      # arm -> id -> row
    for r in rows:
        A[r["arm"]][r["id"]] = r
    arms = [a for a in REID_ARMS if a in A]
    ids = sorted(set.intersection(*(set(A[a]) for a in arms)))
    print(f"reid: {len(ids)} items answered by every arm ({', '.join(arms)}), model {model}")

    # reproduction of the stored 2-image arm
    stored = {r["id"]: r for r in map(json.loads, open(SEM/"answers.jsonl"))
              if r["id"] in items and r["arm"] == "top" and r["model"] == model}
    if "sep2" in A and stored:
        common = [i for i in ids if i in stored]
        same_arg = sum(_argmax(stored[i]["probs"]) == _argmax(A["sep2"][i]["probs"]) for i in common)
        dmax = max(max(abs(stored[i]["probs"][o] - A["sep2"][i]["probs"].get(o, 0)) for o in stored[i]["probs"])
                   for i in common)
        acc_s = np.mean([_argmax(stored[i]["probs"]) == items[i]["label"] for i in common])
        print(f"sep2 vs stored arm 'top': same argmax {same_arg}/{len(common)}, max |dp| {dmax:.4f}, "
              f"stored accuracy {acc_s:.3f}")

    correct = {a: {i: _argmax(A[a][i]["probs"]) == items[i]["label"] for i in ids} for a in arms}

    def metrics(a, sub):
        n = len(sub)
        k = sum(correct[a][i] for i in sub)
        lo, hi = wilson(k, n)
        tiles_only = np.mean([_argmax({o: p for o, p in A[a][i]["probs"].items() if o != "none"}) == items[i]["label"]
                              for i in sub])
        none = np.mean([_argmax(A[a][i]["probs"]) == "none" for i in sub])
        pos = [A[a][i]["probs"][items[i]["label"]] for i in sub]
        neg = [p for i in sub for o, p in A[a][i]["probs"].items() if o not in ("none", items[i]["label"])]
        conf = [max(A[a][i]["probs"].values()) for i in sub]
        auc_c = auc([c for c, i in zip(conf, sub) if correct[a][i]], [c for c, i in zip(conf, sub) if not correct[a][i]])
        # per-item ranking: P(true tile) above every distractor tile's P
        rank = np.mean([all(A[a][i]["probs"][items[i]["label"]] > p for o, p in A[a][i]["probs"].items()
                            if o not in ("none", items[i]["label"])) for i in sub])
        return dict(n=n, acc=k / n, lo=lo, hi=hi, tiles_only=tiles_only, none=none, auc_tile=auc(pos, neg),
                    auc_conf=auc_c, rank=rank, p_true=float(np.mean(pos)))

    strata = {"all": lambda i: True,
              "moved > 3 cm": lambda i: i["moved"], "not moved": lambda i: not i["moved"],
              "near (fingertip < 8 cm)": lambda i: i["near"], "not near": lambda i: not i["near"],
              "k = 2": lambda i: i["k"] == 2, "k = 3": lambda i: i["k"] == 3,
              "held": lambda i: i["held"],
              "nearest-to-start wrong": lambda i: not i["nearest_ok"]}
    for task in sorted({i["task"] for i in items.values()}):
        strata[task] = (lambda t: lambda i: i["task"] == t)(task)

    print("\nacc = argmax over T1..Tk+none equals the label; tiles = argmax over tiles only; none = share answering "
          "none;\nAUC_tile = P(true tile) vs P(distractor tiles), pooled; AUC_conf = max p separates right from wrong; "
          "rank = P(true) above every distractor")
    out = {}
    for sname, keep in strata.items():
        sub = [i for i in ids if keep(items[i])]
        if not sub:
            continue
        chance = np.mean([1 / items[i]["k"] for i in sub])
        near0 = np.mean([items[i]["nearest_ok"] for i in sub])
        print(f"\n[{sname}] n={len(sub)}  chance {chance:.3f}  nearest-to-start {near0:.3f}")
        for a in arms:
            m = metrics(a, sub)
            out[(sname, a)] = m
            ac = "  -  " if m["auc_conf"] is None else f"{m['auc_conf']:.3f}"
            print(f"  {a:7s} acc {m['acc']:.3f} [{m['lo']:.3f},{m['hi']:.3f}]  tiles {m['tiles_only']:.3f}  "
                  f"none {m['none']:.3f}  AUC_tile {m['auc_tile']:.3f}  AUC_conf {ac}  rank {m['rank']:.3f}  "
                  f"P(true) {m['p_true']:.3f}")
        for x in range(len(arms)):
            for y in range(x + 1, len(arms)):
                a, b = arms[x], arms[y]
                ao = sum(correct[a][i] and not correct[b][i] for i in sub)
                bo = sum(correct[b][i] and not correct[a][i] for i in sub)
                print(f"    {a} vs {b}: {a} only right {ao}, {b} only right {bo}, McNemar exact p {mcnemar_exact(ao, bo):.4f}")

    print("\nrequest size per arm (first item): image sizes, Clef input tokens (median over items)")
    for a in arms:
        r0 = A[a][ids[0]]
        toks = [A[a][i]["input_tokens"] for i in ids if A[a][i].get("input_tokens") is not None]
        by_k = {k: int(np.median([A[a][i]["input_tokens"] for i in ids if items[i]["k"] == k])) for k in (2, 3)}
        print(f"  {a:7s} sizes {r0['sizes']}  input tokens median {int(np.median(toks))} (k=2: {by_k[2]}, k=3: {by_k[3]})"
              f"  latency median {np.median([A[a][i]['latency'] for i in ids]):.2f} s")
    conf = {a: collections.Counter((items[i]["label"], _argmax(A[a][i]["probs"])) for i in ids) for a in arms}
    for a in arms:
        print(f"  confusion {a}: {dict(sorted(conf[a].items()))}")
    return out


# ------------------------------------------------------------------ topple
# Judgment 2: "has this object fallen over since the start?" on replay captures (top camera, turn captures only).
#   items  every (turn capture after the first, task object) with the arms clear: no fingertip within ARM_XY_CM in
#          xy (the monitor's arm-clear rule; it also excludes held objects, which are skipped first: within
#          HELD_CM of a fingertip and > HELD_RISE_CM above the table), on the table (z >= table - 5).
#          tilt = angle between the start's up direction carried by the relative rotation and the start's up
#          direction (R_now R_start^T z vs z). toppled >= 60 deg, upright <= 15 deg, between: skipped.
#          At most one item per (episode, object, label) per 5 turns; all toppled + up to 3x as many upright
#          (sha1 order of the id).
#   crops  release_gate.crop (crop_box + projector, the npz camera matrices, K doubled to 640x480): +-12 cm about
#          the object's START position in the first turn capture and its CURRENT position now, z in [-3, +12] cm
#          about its origin, 256 px. The simulator's truth places the crops here; at run time the monitor's depth
#          tracker places them.
#   arms   sep2 (START image, NOW image), one_lr (START | NOW), one_tb (START over NOW). Every crop carries its
#          label bar (START / NOW) in every arm, so the pixels are the same; composed images are 520x256 / 256x520
#          (no shrinking; GAP-pixel white gap).
#
#     python scripts/clef_one_vs_many.py topple --build    # items + crops only
#     python scripts/clef_one_vs_many.py topple            # build if missing, ask every arm (cached), report
#     python scripts/clef_one_vs_many.py topple --report   # report only
REPLAY = PROJECT/"outputs/robodawn_colab/rt8d/replay/eval_baseline"
TOPPLE_ARMS = ("sep2", "one_lr", "one_tb")
TOPPLE_IMG = OUT/"topple_img"
SKIP_NAMES = ("wall", "table")
TOPPLED_DEG, UPRIGHT_DEG = 60.0, 15.0
HELD_CM, HELD_RISE_CM, ARM_XY_CM, OFF_TABLE_CM = 10.0, 3.0, 15.0, 5.0
CROP_HALF_CM, CROP_RISE_CM = 12.0, 12.0
EVERY_TURNS, UPRIGHT_PER_TOPPLED = 5, 3
FALSE_ALARM = 0.05
TOPPLE_Q = "Has the object in the NOW picture fallen over or tipped onto its side, compared with the START picture?"


def topple_layout(arm: str) -> str:
    """The only sentence that differs between arms."""
    return {
        "sep2": "The first picture, labelled START, shows it at the start; the second picture, labelled NOW, shows it now.",
        "one_lr": ("The picture has two parts side by side: on the left, the part labelled START shows it at the start; "
                   "on the right, the part labelled NOW shows it now."),
        "one_tb": ("The picture has two parts, one above the other: at the top, the part labelled START shows it at the "
                   "start; below it, the part labelled NOW shows it now."),
    }[arm]


def topple_state(arm: str) -> str:
    return ("One object on a robot's table, in the centre of each view, seen from straight above at the same scale. "
            + topple_layout(arm))


def quat_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tilt_deg(q_start, q_now) -> float:
    up = quat_R(q_now) @ quat_R(q_start).T @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.acos(float(np.clip(up[2] / np.linalg.norm(up), -1.0, 1.0))))


def family(model: str) -> str:
    """'041_shoe' -> 'shoe'."""
    head, _, tail = model.partition("_")
    return tail if head.isdigit() and tail else model


def label_bar_fit(img: Image.Image, text: str) -> Image.Image:
    """label_bar's kind (black bar 27 px high at the top-left corner, white default-font text), widened to the text."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w = max(35, int(math.ceil(draw.textlength(text))) + 10)
    draw.rectangle([0, 0, w - 1, 26], fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    return img


def _top_project(npz, idx: int):
    from branchlab.rtscene.release_gate import projector
    K = np.asarray(npz["top_camera_K"][idx], dtype=float).copy()
    K[:2] *= 2                                   # stored for 320x240; the raw jpg is 640x480
    return projector({"intrinsic_cv": K, "extrinsic_cv": npz["top_camera_extrinsic"][idx]})


def _in_view(project, centre, w=640, h=480) -> bool:
    u, v = project([centre])[0]
    return 0 <= u < w and 0 <= v < h


def topple_build() -> list[dict]:
    from branchlab.rtscene.release_gate import crop
    TOPPLE_IMG.mkdir(parents=True, exist_ok=True)
    cands, skips = [], collections.Counter()
    for meta_path in sorted(REPLAY.glob("*/episode_*/meta.json")):
        ep_dir = meta_path.parent
        meta = json.loads(meta_path.read_text())
        caps = [c for c in meta["captures"] if c["kind"] == "turn"]
        if len(caps) < 2:
            continue
        npz = np.load(ep_dir/"captures.npz")
        first = caps[0]
        s0, tz0 = first["state"]["objects"], first["state"]["table_z_cm"]
        last_kept: dict = {}
        for cap in caps[1:]:
            st = cap["state"]
            tz = st["table_z_cm"]
            tips = {a: np.asarray(st[a]["position_cm"], float) for a in ("left", "right")}
            for name, o in st["objects"].items():
                if name in SKIP_NAMES or name not in s0 or "quat_wxyz" not in o:
                    continue
                p, p0 = np.asarray(o["position_cm"], float), np.asarray(s0[name]["position_cm"], float)
                if p[2] < tz - OFF_TABLE_CM or p0[2] < tz0 - OFF_TABLE_CM:
                    skips["off table"] += 1
                    continue
                d3 = min(float(np.linalg.norm(p - t)) for t in tips.values())
                dxy = min(float(np.linalg.norm(p[:2] - t[:2])) for t in tips.values())
                if d3 < HELD_CM and p[2] - tz > HELD_RISE_CM:
                    skips["held"] += 1
                    continue
                if dxy <= ARM_XY_CM:
                    skips["fingertip within 15 cm xy"] += 1
                    continue
                tilt = tilt_deg(s0[name]["quat_wxyz"], o["quat_wxyz"])
                if tilt >= TOPPLED_DEG:
                    label = "toppled"
                elif tilt <= UPRIGHT_DEG:
                    label = "upright"
                else:
                    skips["tilt 15-60"] += 1
                    continue
                if not (_in_view(_top_project(npz, first["idx"]), p0) and _in_view(_top_project(npz, cap["idx"]), p)):
                    skips["out of view"] += 1
                    continue
                k = (meta["episode"], name, label)
                if k in last_kept and cap["turn"] - last_kept[k] < EVERY_TURNS:
                    skips["within 5 turns of a kept item"] += 1
                    continue
                last_kept[k] = cap["turn"]
                iid = f"{meta['task']}/{meta['episode']}/{name}/t{cap['turn']}"
                cands.append(dict(
                    id=iid, task=meta["task"], episode=meta["episode"], ep_dir=str(ep_dir.relative_to(PROJECT)),
                    obj=name, model=o.get("model"), family=family(o.get("model") or name),
                    role="clutter" if name.startswith("cluttered") else "task", turn=cap["turn"],
                    start_idx=first["idx"], now_idx=cap["idx"], label=label, y=int(label == "toppled"),
                    tilt_deg=round(tilt, 1), moved_cm=round(float(np.linalg.norm(p[:2] - p0[:2])), 1),
                    tip_xy_cm=round(dxy, 1), start_cm=[round(v, 2) for v in p0], now_cm=[round(v, 2) for v in p],
                    episode_success=meta.get("success")))
    top = [c for c in cands if c["y"]]
    up = sorted((c for c in cands if not c["y"]), key=lambda c: _h(c["id"]))[:UPRIGHT_PER_TOPPLED * len(top)]
    items = sorted(top + up, key=lambda c: (c["task"], c["episode"], c["obj"], c["turn"]))
    # crops (PNG, no label bar; the bars are drawn at ask time, the same way in every arm)
    cache: dict = {}
    for it in items:
        ep_dir = PROJECT/it["ep_dir"]
        if it["ep_dir"] not in cache:
            cache = {it["ep_dir"]: np.load(ep_dir/"captures.npz")}
        npz = cache[it["ep_dir"]]
        h = _h(it["id"])
        for tag, idx, centre in (("start", it["start_idx"], it["start_cm"]), ("now", it["now_idx"], it["now_cm"])):
            img = Image.open(ep_dir/"images"/f"c{idx:03d}_top_camera_raw.jpg").convert("RGB")
            assert img.size == (640, 480)
            crop(img, _top_project(npz, idx), centre, CROP_HALF_CM, CROP_RISE_CM).save(TOPPLE_IMG/f"{h}_{tag}.png")
        it["img"] = h
    (OUT/"items_topple.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    print(f"topple: {len(cands)} candidates ({len(top)} toppled), {len(items)} items "
          f"({len(top)} toppled + {len(up)} upright) -> {OUT/'items_topple.jsonl'}")
    print("  skipped:", dict(skips))
    return items


def topple_images(item: dict, arm: str) -> list[Image.Image]:
    h = item["img"]
    start = label_bar_fit(Image.open(TOPPLE_IMG/f"{h}_start.png").convert("RGB"), "START")
    now = label_bar_fit(Image.open(TOPPLE_IMG/f"{h}_now.png").convert("RGB"), "NOW")
    if arm == "sep2":
        return [start, now]
    return [compose([start, now], horizontal=(arm == "one_lr"))]


def topple_request(item: dict, arm: str):
    return topple_state(arm), {"fallen": {"type": "noul", "instructions": TOPPLE_Q}}, topple_images(item, arm)


def topple_ask(arms, model: str, workers: int):
    from branchlab.clef import Clef
    clef = Clef(cache_path=OUT/"cache_topple.jsonl", backend="workers")
    items = [json.loads(l) for l in open(OUT/"items_topple.jsonl")]
    (OUT/"img").mkdir(exist_ok=True)
    ex = [next(i for i in items if i["y"]), next(i for i in items if not i["y"])]
    for arm in ("one_lr", "one_tb"):
        for it in ex:
            topple_images(it, arm)[0].save(OUT/"img"/f"example_topple_{arm}_{it['label']}.jpg", quality=90)

    def one(job):
        item, arm = job
        state, q, imgs = topple_request(item, arm)
        try:
            r = clef.ask(state, q, images=[uri(im) for im in imgs], model=model, key=f"{item['id']}|{arm}|{model}")
        except Exception as exc:  # noqa: BLE001  (record and go on; a rerun asks again)
            return dict(id=item["id"], arm=arm, model=model, error=f"{type(exc).__name__}: {str(exc)[:300]}")
        return dict(id=item["id"], arm=arm, model=model, p_yes=float(r["answers"]["fallen"]["noul"]),
                    sizes=[list(im.size) for im in imgs], input_tokens=r.get("usage", {}).get("input_tokens"),
                    usage=r.get("usage"), latency=round(r.get("latency", 0.0), 3))

    jobs = [(i, a) for i in items for a in arms]
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, jobs))
    (OUT/"answers_topple.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    errs = [r for r in rows if "error" in r]
    print(len(rows) - len(errs), "answers,", len(errs), "errors ->", OUT/"answers_topple.jsonl")
    if errs:
        print("  first error:", errs[0]["error"])


def threshold_at_fa(neg, fa: float = FALSE_ALARM) -> float:
    """The lowest threshold t with mean(neg >= t) <= fa (scores >= t say 'toppled')."""
    neg = np.sort(np.asarray(neg, float))
    for t in np.unique(np.concatenate([neg, [np.inf]])):
        if np.mean(neg >= t) <= fa:
            return float(t)
    return float("inf")


def topple_report(model: str):
    items = {i["id"]: i for i in map(json.loads, open(OUT/"items_topple.jsonl"))}
    A = collections.defaultdict(dict)
    for r in map(json.loads, open(OUT/"answers_topple.jsonl")):
        if r["model"] == model and "p_yes" in r:
            A[r["arm"]][r["id"]] = r
    arms = [a for a in TOPPLE_ARMS if a in A]
    ids = sorted(set.intersection(*(set(A[a]) for a in arms)))
    y = {i: items[i]["y"] for i in ids}
    s = {a: {i: A[a][i]["p_yes"] for i in ids} for a in arms}
    npos = sum(y.values())
    print(f"topple: {len(ids)} items answered by every arm ({', '.join(arms)}), {npos} toppled / "
          f"{len(ids) - npos} upright, model {model}; {len(items) - len(ids)} items missing an arm")
    thr = {a: threshold_at_fa([s[a][i] for i in ids if not y[i]]) for a in arms}
    print("thresholds at <= 5% false alarms on upright items (fitted on all items, in-sample): "
          + ", ".join(f"{a} {thr[a]:.4f}" for a in arms))
    correct = {(a, t): {i: (s[a][i] >= (0.5 if t == "0.5" else thr[a])) == bool(y[i]) for i in ids}
               for a in arms for t in ("0.5", "fa5")}

    def metrics(a, sub):
        pos = [s[a][i] for i in sub if y[i]]
        neg = [s[a][i] for i in sub if not y[i]]
        out = dict(n=len(sub), npos=len(pos), auc=auc(pos, neg))
        for t, tv in (("0.5", 0.5), ("fa5", thr[a])):
            out[f"acc_{t}"] = float(np.mean([correct[(a, t)][i] for i in sub]))
            out[f"tpr_{t}"] = float(np.mean([p >= tv for p in pos])) if pos else None
            out[f"fa_{t}"] = float(np.mean([p >= tv for p in neg])) if neg else None
        out["med_pos"] = float(np.median(pos)) if pos else None
        out["med_neg"] = float(np.median(neg)) if neg else None
        return out

    def f(v, w=5):
        return " " * (w - 1) + "-" if v is None else f"{v:.3f}".rjust(w)

    strata = {"all": lambda i: True}
    for task in sorted({items[i]["task"] for i in ids}):
        strata[f"task {task}"] = (lambda t: lambda i: items[i]["task"] == t)(task)
    for fam in sorted({items[i]["family"] for i in ids}):
        strata[f"family {fam}"] = (lambda t: lambda i: items[i]["family"] == t)(fam)
    for role in ("task", "clutter"):
        strata[f"role {role}"] = (lambda t: lambda i: items[i]["role"] == t)(role)
    strata["upright, moved > 3 cm (vs toppled)"] = lambda i: y[i] or items[i]["moved_cm"] > 3
    strata["upright, never moved (vs toppled)"] = lambda i: y[i] or items[i]["moved_cm"] <= 1

    print("\nacc/tpr/fa at 0.5 and at each arm's 5%-false-alarm threshold; AUC = P(yes) toppled vs upright")
    results = {}
    for sname, keep in strata.items():
        sub = [i for i in ids if keep(i)]
        if not sub:
            continue
        print(f"\n[{sname}] n={len(sub)} (toppled {sum(y[i] for i in sub)})")
        for a in arms:
            m = metrics(a, sub)
            results[(sname, a)] = m
            print(f"  {a:7s} AUC {f(m['auc'])}  @0.5 acc {f(m['acc_0.5'])} tpr {f(m['tpr_0.5'])} fa {f(m['fa_0.5'])}"
                  f"  @fa5 acc {f(m['acc_fa5'])} tpr {f(m['tpr_fa5'])} fa {f(m['fa_fa5'])}"
                  f"  median P(yes) toppled {f(m['med_pos'])} upright {f(m['med_neg'])}")
        for t in ("0.5", "fa5"):
            for x in range(len(arms)):
                for z in range(x + 1, len(arms)):
                    a, b = arms[x], arms[z]
                    ao = sum(correct[(a, t)][i] and not correct[(b, t)][i] for i in sub)
                    bo = sum(correct[(b, t)][i] and not correct[(a, t)][i] for i in sub)
                    if sname == "all" or (t == "0.5" and (ao or bo)):
                        print(f"    @{t:3s} {a} vs {b}: {a} only right {ao}, {b} only right {bo}, "
                              f"McNemar exact p {mcnemar_exact(ao, bo):.4f}")

    # paired AUC differences, bootstrap over episodes (items of one episode and object are not independent)
    rng = np.random.default_rng(0)
    eps = sorted({(items[i]["task"], items[i]["episode"]) for i in ids})
    by_ep = collections.defaultdict(list)
    for i in ids:
        by_ep[(items[i]["task"], items[i]["episode"])].append(i)
    print(f"\npaired AUC difference, bootstrap over {len(eps)} episodes (2000 resamples, 95% percentile interval)")
    for x in range(len(arms)):
        for z in range(x + 1, len(arms)):
            a, b = arms[x], arms[z]
            d0 = results[("all", a)]["auc"] - results[("all", b)]["auc"]
            ds = []
            for _ in range(2000):
                sub = [i for k in rng.integers(len(eps), size=len(eps)) for i in by_ep[eps[k]]]
                pa = auc([s[a][i] for i in sub if y[i]], [s[a][i] for i in sub if not y[i]])
                pb = auc([s[b][i] for i in sub if y[i]], [s[b][i] for i in sub if not y[i]])
                if pa is not None and pb is not None:
                    ds.append(pa - pb)
            lo, hi = np.percentile(ds, [2.5, 97.5])
            print(f"  AUC {a} - {b} = {d0:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print("paired accuracy difference at 0.5, same bootstrap over episodes")
    for x in range(len(arms)):
        for z in range(x + 1, len(arms)):
            a, b = arms[x], arms[z]
            d0 = results[("all", a)]["acc_0.5"] - results[("all", b)]["acc_0.5"]
            ds = []
            for _ in range(2000):
                sub = [i for k in rng.integers(len(eps), size=len(eps)) for i in by_ep[eps[k]]]
                ds.append(np.mean([correct[(a, "0.5")][i] for i in sub]) - np.mean([correct[(b, "0.5")][i] for i in sub]))
            lo, hi = np.percentile(ds, [2.5, 97.5])
            print(f"  acc {a} - {b} = {d0:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    corr = {(a, b): float(np.corrcoef([s[a][i] for i in ids], [s[b][i] for i in ids])[0, 1])
            for x, a in enumerate(arms) for b in arms[x + 1:]}
    print("score correlation between arms: " + ", ".join(f"{a}~{b} {v:.3f}" for (a, b), v in corr.items()))
    print("\nrequest size per arm: image sizes, Clef input tokens (median), latency (median)")
    for a in arms:
        toks = [A[a][i]["input_tokens"] for i in ids if A[a][i].get("input_tokens") is not None]
        print(f"  {a:7s} sizes {A[a][ids[0]]['sizes']}  input tokens median {int(np.median(toks)) if toks else '-'}"
              f"  latency median {np.median([A[a][i]['latency'] for i in ids]):.2f} s")
    return results, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reid", "topple"])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--model", default="clef")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report", action="store_true", help="report from the answers file only")
    ap.add_argument("--build", action="store_true", help="topple: (re)build the items and crops only")
    a = ap.parse_args()
    if a.cmd == "reid":
        if not a.report:
            reid_ask(a.arms or list(REID_ARMS), a.model, a.workers)
        reid_report(a.model)
    elif a.cmd == "topple":
        if a.build:
            topple_build()
            return
        if not a.report:
            if not (OUT/"items_topple.jsonl").is_file():
                topple_build()
            topple_ask(a.arms or list(TOPPLE_ARMS), a.model, a.workers)
        topple_report(a.model)


if __name__ == "__main__":
    main()
