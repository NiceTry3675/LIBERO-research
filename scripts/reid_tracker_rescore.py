#!/usr/bin/env python3
"""Rescore the static re-identification test with a run-time position tracker in place of the true start position.

No model calls: Clef's stored answers only (sep2 and one_tb in outputs/analysis/clef_one_vs_many/answers_reid.jsonl;
sep2 is checked against the 'top' arm of outputs/analysis/clef_semantic/answers.jsonl).

The earlier comparison (reports/Clef_한장_대_여러장.md section 4: nearest-to-start 0.900, Clef fused with a distance
prior 0.902, tile-only fused 0.933) centred the distance prior on the target's TRUE turn-1 position, which a robot
does not have at run time. Here the prior is centred on an image-free depth tracker's position instead.

Tracker (per episode and target; the 'run-time' tracker of
outputs/analysis/next_exp_2026-10-12/next_exp/d_clef_share/track_ceiling.py, on the same depth blobs as
g1_candidates/scene.py, recomputed here for EVERY turn-start capture, since the g1 pickle stops at the turn where
the replay diverged from the original run while the re-ID items run to turn 44):
  bind    turn 1: the target's segmentation id (the same-model entity whose top-surface centre is nearest the true
          origin, <= 15 cm; track_ceiling.role_sids), its largest non-robot blob with that majority id; the track is
          that blob's centroid. Ground truth picks the blob here only (it stands for the planner's choice).
  update  every later turn-start capture, in order:
            hold/cover (track_ceiling's rule): the target is > 3 cm above its turn-1 height, or a fingertip is within
              12 cm (xy) of its true position, or a fingertip is within 12 cm (xy) of the track -> no update (the
              first two use ground truth; track_ceiling justifies this as gripper feedback at run time; it measured
              the third on its oracle-updated track, here it is the run-time track);
            otherwise the track moves to the nearest non-robot blob whose centroid is within 30 cm of it (wrong moves
              cascade); none -> the track is kept. 'Robot' blob: its majority segmentation name is a robot link
              (track_ceiling / step3b).
          variant tc_exact: also no update when the target has no blob of its own (track_ceiling does this; it needs
            ground truth). variant gap_robot: robot blobs by the run-time rule of monitor.parse (robot_gap < 0.5 cm
            or area < 6 cm^2) instead of segmentation names.
          POST-HOC variants carry (the monitor's own grasp/release handling, no ground truth after turn 1) and naive
            (no hold rule), added after the pre-registered tracker turned out frozen at most item turns; see
            run_tracker.
  An item's tracker position is the track BEFORE that turn's update (the last place seen: the monitor's prior centre).

Tiles -> blobs: each tile's object -> its segmentation id (same binding) -> its largest non-robot blob with that
majority id at the item's capture; no such blob -> the object's true position, flagged ('gt_fallback').

Scores per item and layout (sep2, one_tb):
  trk_only        the tile nearest the tracker position
  trk_fuse        monitor.fuse(Clef probabilities, distances from the tracker position rounded to cm), D = 10
  trk_fuse_tile   the same with 'none' dropped and the tile probabilities renormalised
  gt_*            the same three with the old prior: true turn-1 origin to the tiles' true positions (reproduces
                  0.900 / 0.902 / 0.933 for sep2)
  clef, clef_tile Clef's argmax over T1..Tk+none, and over the tiles only

PRE-REGISTERED READING (fixed before any tracker number was computed):
  primary  sep2, trk_fuse_tile minus trk_only, paired over the 389 items; 95% interval = percentile interval of
           10,000 bootstrap resamples of the 22 episodes (seed 0).
  'Clef adds'  difference >= +3 pp and the interval excludes 0;  'Clef hurts'  <= -3 pp and excludes 0;
  otherwise 'no detectable difference'.
  Everything else (full-probability fusion, Clef alone, the old prior, one_tb, strata by moved / twin present,
  the tracker's own errors, the variants) is secondary.

    python scripts/reid_tracker_rescore.py            # build the scene cache if missing, track, score
    python scripts/reid_tracker_rescore.py --rebuild  # recompute the depth blobs

Writes outputs/analysis/reid_tracker/: scenes.pkl (cache), track.jsonl (tracker state per target and turn),
rows.jsonl (per item), summary.json; prints the tables.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"src"))
sys.path.insert(0, str(PROJECT/"scripts"))
G1 = PROJECT/"outputs/analysis/next_exp_2026-10-12/next_exp/g1_candidates"
OUT = PROJECT/"outputs/analysis/reid_tracker"
REPLAY = PROJECT/"outputs/robodawn_colab/rt8d/replay"
SEM = PROJECT/"outputs/analysis/clef_semantic"
OVM = PROJECT/"outputs/analysis/clef_one_vs_many"

import clef_semantic_offline as cso                    # noqa: E402  (the items' own candidate rule and hash order)
from branchlab.rtscene import monitor                    # noqa: E402

CLEAR = 12.0          # cm (xy), track_ceiling's hold/cover radius
RADIUS = 30.0         # cm, search radius (track_ceiling, monitor.CAND_RADIUS)
LIFT = 3.0            # cm above the turn-1 height: held (track_ceiling)
BIND = 15.0           # cm, segmentation centre to the true origin (track_ceiling.role_sids)
ROBOTNAME = re.compile(r"(link\d*|wheel|castor|camera|base_link|footprint)", re.I)   # step3b_replay_all
LAYOUTS = ("sep2", "one_tb")
METHODS = ("trk_only", "trk_fuse", "trk_fuse_tile", "gt_only", "gt_fuse", "gt_fuse_tile", "clef", "clef_tile")
VARIANTS = ("primary", "tc_exact", "gap_robot", "carry", "naive")   # the last two are post hoc
BOOT = 10000


# ------------------------------------------------------------------ depth scenes (run-time blobs + offline labels)
def build_episode(ep_dir: Path) -> dict:
    """Every turn-start capture: depth blobs labelled with their majority segmentation id (as step3b_replay_all
    does, without its divergence cut), the top-surface centres of the entities, true poses and the arms."""
    sys.path.insert(0, str(G1))
    import scene                                          # g1_candidates/scene.py
    from branchlab.rtscene import capture, heightmap
    ep = capture.Episode(ep_dir)
    names = scene.entity_names(ep)
    out = {}
    for cap in ep.captures():
        if cap.kind != "turn":
            continue
        hm = heightmap.build(cap, scene.BOXES, with_rgb=False)
        B = scene.blobs(hm)
        segmap = scene.seg_cells(cap, ep, hm)
        keep = []
        for b in B:
            ids, cnt = np.unique(segmap[b["iy"], b["ix"]], return_counts=True)
            share = {int(i): float(c / cnt.sum()) for i, c in zip(ids, cnt)}
            major = max(share, key=share.get)
            mname = names.get(major, "")
            keep.append(dict(id=b["id"], cx=b["cx"], cy=b["cy"], area=b["area"], h95=b["h95"],
                             robot_gap=b["robot_gap"], major=major, major_name=mname, purity=share[major],
                             members={i: s for i, s in share.items() if s >= 0.15},
                             robot=bool(ROBOTNAME.search(mname)) and mname != ""))
        st = ep.meta["captures"][cap.idx]["state"]
        if cap.turn in out:
            raise RuntimeError(f"two turn captures for turn {cap.turn} in {ep_dir}")
        out[cap.turn] = dict(idx=cap.idx, blobs=keep, cents=scene.object_centres(cap, ep, names),
                             objects={r: o["position_cm"] for r, o in st["objects"].items()},
                             models={r: o.get("model") for r, o in st["objects"].items()},
                             arms={a: dict(pos=st[a]["position_cm"], gripper=st[a].get("gripper"),
                                           gripper_real=st[a].get("gripper_real")) for a in ("left", "right")},
                             table_z=st["table_z_cm"])
    return out


def load_scenes(eps: list[str], rebuild: bool) -> dict:
    path = OUT/"scenes.pkl"
    if path.exists() and not rebuild:
        S = pickle.load(open(path, "rb"))
        if all(e in S for e in eps):
            return S
    from multiprocessing import Pool
    with Pool(6) as pool:
        res = pool.map(build_episode, [REPLAY/e for e in eps])
    S = dict(zip(eps, res))
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(S, open(path, "wb"))
    return S


def check_against_g1(S: dict) -> str:
    """The recomputed blobs must equal the g1 pickle's where both exist (turns up to the divergence)."""
    G = pickle.load(open(G1/"replay_scenes_all.pkl", "rb"))
    n = same = 0
    for e, turns in S.items():
        task, epi = e.split("/")[1], int(e.split("_")[-1])
        for t, d in turns.items():
            g = G.get((task, epi), {}).get(t)
            if g is None:
                continue
            n += 1
            a = sorted((round(b["cx"], 3), round(b["cy"], 3), b["major"], b["robot"]) for b in d["blobs"])
            b_ = sorted((round(b["cx"], 3), round(b["cy"], 3), b["major"], b["robot"]) for b in g["blobs"])
            same += a == b_
    return f"{same}/{n} captures identical to the g1 pickle"


# ------------------------------------------------------------------ binding and blobs
def bind_sids(turns: dict) -> dict:
    """role -> (segmentation id, turn used): the same-model entity whose top-surface centre is nearest the role's
    true origin (<= BIND cm), at the first turn where one is visible (turn 1 for all but a few)."""
    out = {}
    roles = {r for d in turns.values() for r, m in d["models"].items() if r not in ("wall", "table") and m}
    for role in sorted(roles):
        for t in sorted(turns):
            d = turns[t]
            if role not in d["objects"]:
                continue
            gx, gy = d["objects"][role][:2]
            cand = [(sid, c) for sid, c in d["cents"].items() if c["name"] == d["models"][role]]
            if not cand:
                continue
            sid, c = min(cand, key=lambda sc: math.hypot(sc[1]["x"] - gx, sc[1]["y"] - gy))
            if math.hypot(c["x"] - gx, c["y"] - gy) <= BIND:
                out[role] = (sid, t)
                break
    return out


def blob_of(d: dict, sid: int):
    bs = [b for b in d["blobs"] if not b["robot"] and b["major"] == sid]
    return max(bs, key=lambda b: b["area"]) if bs else None


def nonrobot(b: dict, variant: str) -> bool:
    if variant == "gap_robot":
        return b["robot_gap"] >= 0.5 and b["area"] >= 6.0       # monitor.parse
    return not b["robot"]


def holding(arm: dict) -> bool:
    """The harness's own reading of a gripper: commanded closed but stopped on something (monitor.holding)."""
    return (arm.get("gripper") if arm.get("gripper") is not None else 1.0) < 0.5 and (arm.get("gripper_real") or 0.0) > 0.02


def run_tracker(turns: dict, role: str, sid: int | None, variant: str, events: list | None = None) -> list[dict]:
    """One row per turn: the track before ('pre', the prior centre for an item at this turn) and after that turn's
    update, and why it did or did not move.

    POST-HOC variants (added after the pre-registered tracker turned out frozen at most item turns):
      naive  no hold/cover rule at all: every turn moves to the nearest non-robot blob within 30 cm.
      carry  run-time hold handling as in monitor.ObjectMonitor, no ground truth after turn 1: a close (preclose
             capture) whose fingertips are within 8 cm of the track and whose gripper then holds something (the
             harness's gripper reading) makes the target 'carried' by that arm; while carried, the track (and the
             prior centre at that turn) is that arm's fingertip position now; an open (preopen capture) of that arm
             puts the track where the fingers opened; a grip that ends without an open puts it where the fingers
             are. Not carried: a fingertip within 12 cm (xy) of the track freezes it (covered), else nearest blob."""
    ts = sorted(turns)
    d0 = turns[ts[0]]
    b0 = blob_of(d0, sid) if sid is not None else None
    if b0 is not None:
        track, bind_note = (b0["cx"], b0["cy"]), "blob"
    elif sid is not None and sid in d0["cents"]:
        track, bind_note = (d0["cents"][sid]["x"], d0["cents"][sid]["y"]), "seg_centre"
    else:
        track, bind_note = tuple(d0["objects"][role][:2]), "gt_origin"
    z0 = d0["objects"][role][2]
    rows = [dict(turn=ts[0], pre=track, post=track, status="bind", bind=bind_note, locked=b0 is not None)]
    locked = b0 is not None
    carried, pending = None, set()
    ev = sorted(events or [], key=lambda e: e["idx"])
    for t in ts[1:]:
        d = turns[t]
        g = d["objects"].get(role)
        if variant == "carry":                       # grasp / release events since the previous turn capture
            for e in [e for e in ev if turns[ts[ts.index(t) - 1]]["idx"] < e["idx"] < d["idx"]]:
                tcp = e["arms"][e["arm"]][:2]
                if e["kind"] == "preclose" and carried is None and math.dist(tcp, track) <= monitor.CARRY_RADIUS:
                    pending.add(e["arm"])
                elif e["kind"] == "preopen":
                    pending.discard(e["arm"])
                    if carried == e["arm"]:
                        carried, track = None, tuple(tcp)
            for arm in sorted(pending):
                if carried is None and holding(d["arms"][arm]):
                    carried = arm
            pending.clear()
        pre = track
        arms = [a["pos"][:2] for a in d["arms"].values()]
        tb = blob_of(d, sid) if sid is not None else None
        cands = [b for b in d["blobs"] if nonrobot(b, variant) and math.dist((b["cx"], b["cy"]), track) <= RADIUS]
        nearest = min(cands, key=lambda b: math.dist((b["cx"], b["cy"]), track)) if cands else None
        # blob-space correctness of a nearest-blob step from here (track_ceiling's ok_runtime), whether taken or not
        ok_step = nearest is not None and tb is not None and nearest["major"] == sid
        if variant == "carry" and carried is not None:
            tip = tuple(d["arms"][carried]["pos"][:2])
            status = "carried" if holding(d["arms"][carried]) else "slipped"
            if status == "slipped":
                carried = None
            track = pre = tip
        else:
            if variant == "naive":
                held = near = False
            elif variant == "carry":
                held, near = False, any(math.dist(a, track) < CLEAR for a in arms)
            else:
                held = g is not None and ((g[2] - z0) > LIFT or any(math.dist(a, g[:2]) < CLEAR for a in arms))
                near = any(math.dist(a, track) < CLEAR for a in arms)
            if held or near:
                status = "held" if held else "near"
            elif variant == "tc_exact" and tb is None:
                status = "missing"
            elif nearest is None:
                status = "nothing"
            else:
                status = "moved"
                track = (nearest["cx"], nearest["cy"])
                locked = nearest["major"] == sid
        rows.append(dict(turn=t, pre=pre, post=track, status=status, target_blob=tb is not None, ok_step=ok_step,
                         locked=locked, n_cands=len(cands)))
    return rows


def grip_events(ep: str) -> list[dict]:
    """preclose / preopen captures (the harness reports grasps and releases): arm, fingertip positions."""
    m = json.load(open(REPLAY/ep/"meta.json"))
    return [dict(idx=c["idx"], kind=c["kind"], arm=c["command"].split()[0],
                 arms={a: c["state"][a]["position_cm"] for a in ("left", "right")})
            for c in m["captures"] if c["kind"] in ("preclose", "preopen")]


# ------------------------------------------------------------------ items and answers
def load_items() -> list[dict]:
    items = [i for i in map(json.loads, open(SEM/"items.jsonl")) if i["probe"] == "reid"]
    metas = {}
    for it in items:
        ep = it["ep"]
        if ep not in metas:
            metas[ep] = json.load(open(REPLAY/ep/"meta.json"))
        m = metas[ep]
        idx = int(it["id"].split("|")[2][1:])
        c = next(c for c in m["captures"] if c["idx"] == idx)
        first = m["captures"][0]
        assert first["kind"] == "turn" and first["turn"] == 1
        st, n = c["state"], c["turn"]
        tz, objs = st["table_z_cm"], st["objects"]
        # the builder's own candidate rule and order (clef_semantic_offline.build)
        cands = {k: np.asarray(o["position_cm"]) for k, o in objs.items()
                 if k not in cso.SKIP_OBJECTS and np.asarray(o["position_cm"])[2] > tz - 5
                 and cso.in_view((640, 480), cso.project("top_camera", o["position_cm"], tz)[0], 30)}
        order = sorted(cands, key=lambda k: cso._h(f"{ep}|{n}|{it['obj']}|{k}"))[:6]
        if ([objs[k]["model"] for k in order] != it["objects"] or f"T{order.index(it['obj']) + 1}" != it["label"]
                or n != it["turn"] or c["kind"] != "turn"):
            raise RuntimeError(f"cannot reconstruct the tiles of {it['id']}")
        it.update(idx=idx, order=order, true_xy=[cands[k][:2].tolist() for k in order],
                  p0=first["state"]["objects"][it["obj"]]["position_cm"],
                  twin=sum(mo == objs[it["obj"]]["model"] for mo in it["objects"]) > 1)
    return items


def load_answers(ids: set) -> dict:
    A = collections.defaultdict(dict)
    for r in map(json.loads, open(OVM/"answers_reid.jsonl")):
        if r["model"] == "clef" and r["arm"] in LAYOUTS and r["id"] in ids:
            A[r["arm"]][r["id"]] = r["probs"]
    stored = {r["id"]: r["probs"] for r in map(json.loads, open(SEM/"answers.jsonl"))
              if r["id"] in ids and r["arm"] == "top" and r["model"] == "clef"}
    same = sum(max(stored[i], key=stored[i].get) == max(A["sep2"][i], key=A["sep2"][i].get) for i in stored)
    A["_check"] = f"sep2 vs stored 'top': same argmax {same}/{len(stored)}"
    return A


def argmax(p: dict) -> str:
    return max(p, key=p.get)


def tiles_only(p: dict) -> dict:
    t = {k: v for k, v in p.items() if k != "none"}
    s = sum(t.values())
    return {k: v / s for k, v in t.items()}


def nearest_tile(dists: list[float]) -> str:
    return f"T{int(np.argmin(dists)) + 1}"


def score_item(it: dict, probs: dict, trk_xy, tile_xy: list) -> dict:
    d_trk = [math.dist(trk_xy, xy) for xy in tile_xy]
    d_gt = [math.dist(it["p0"][:2], xy) for xy in it["true_xy"]]
    pick = {
        "trk_only": nearest_tile(d_trk),
        "trk_fuse": monitor.fuse(probs, [round(x) for x in d_trk])[0],
        "trk_fuse_tile": monitor.fuse(tiles_only(probs), [round(x) for x in d_trk])[0],
        "gt_only": nearest_tile(d_gt),
        "gt_fuse": monitor.fuse(probs, [round(x) for x in d_gt])[0],
        "gt_fuse_tile": monitor.fuse(tiles_only(probs), [round(x) for x in d_gt])[0],
        "clef": argmax(probs),
        "clef_tile": argmax(tiles_only(probs)),
    }
    return dict(pick=pick, ok={m: pick[m] == it["label"] for m in pick}, d_trk=[round(x, 1) for x in d_trk])


# ------------------------------------------------------------------ statistics
def boot_diff(rows: list[dict], a: str, b: str, seed: int = 0) -> tuple[float, float, float]:
    """Paired accuracy difference a - b and its 95% percentile interval over episode resamples."""
    eps = sorted({r["ep"] for r in rows})
    per = {e: [r for r in rows if r["ep"] == e] for e in eps}
    sa = np.array([sum(r["ok"][a] for r in per[e]) for e in eps], float)
    sb = np.array([sum(r["ok"][b] for r in per[e]) for e in eps], float)
    n = np.array([len(per[e]) for e in eps], float)
    diff = (sa.sum() - sb.sum()) / n.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(eps), size=(BOOT, len(eps)))
    bd = (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    lo, hi = np.percentile(bd, [2.5, 97.5])
    return float(diff), float(lo), float(hi)


def boot_acc(rows: list[dict], a: str, seed: int = 0) -> tuple[float, float]:
    eps = sorted({r["ep"] for r in rows})
    sa = np.array([sum(r["ok"][a] for r in rows if r["ep"] == e) for e in eps], float)
    n = np.array([sum(r["ep"] == e for r in rows) for e in eps], float)
    idx = np.random.default_rng(seed).integers(0, len(eps), size=(BOOT, len(eps)))
    lo, hi = np.percentile(sa[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])
    return float(lo), float(hi)


def reading(diff: float, lo: float, hi: float) -> str:
    if diff >= 0.03 and lo > 0:
        return "Clef adds"
    if diff <= -0.03 and hi < 0:
        return "Clef hurts"
    return "no detectable difference"


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    items = load_items()
    eps = sorted({i["ep"] for i in items})
    S = load_scenes(eps, a.rebuild)
    print(f"items {len(items)}, episodes {len(eps)}, turn captures {sum(len(S[e]) for e in eps)}")
    print("blob check:", check_against_g1(S))
    A = load_answers({i["id"] for i in items})
    print(A["_check"])

    sids = {e: bind_sids(S[e]) for e in eps}
    late_bind = [(e, r, t) for e in eps for r, (s, t) in sids[e].items() if t != 1]
    unbound = sorted({(i["ep"], k) for i in items for k in i["order"] if k not in sids[i["ep"]]})
    print(f"segmentation binding: late (not turn 1) {late_bind}; unbound tile objects {unbound}")

    # tracker per (episode, target) and variant
    events = {e: grip_events(e) for e in eps}
    tracks = {}
    with open(OUT/"track.jsonl", "w") as f:
        for v in VARIANTS:
            for e, role in sorted({(i["ep"], i["obj"]) for i in items}):
                sid = sids[e].get(role, (None, None))[0]
                rows = run_tracker(S[e], role, sid, v, events[e] if v == "carry" else None)
                tracks[(v, e, role)] = {r["turn"]: r for r in rows}
                for r in rows:
                    f.write(json.dumps(dict(variant=v, ep=e, role=role, sid=sid, **r)) + "\n")
    binds = collections.Counter(tracks[("primary", e, r)][min(S[e])]["bind"] for e, r in {(i["ep"], i["obj"]) for i in items})
    print("turn-1 binding of the targets:", dict(binds))

    # per item
    out_rows = collections.defaultdict(list)          # (variant, layout) -> rows
    with open(OUT/"rows.jsonl", "w") as f:
        for v in VARIANTS:
            for it in items:
                e, d = it["ep"], S[it["ep"]][it["turn"]]
                tr = tracks[(v, e, it["obj"])][it["turn"]]
                tile_xy, flags = [], []
                for k, txy in zip(it["order"], it["true_xy"]):
                    sid = sids[e].get(k, (None, None))[0]
                    b = blob_of(d, sid) if sid is not None else None
                    if b is None:
                        tile_xy.append(tuple(txy))
                        flags.append("gt_fallback" if sid is not None else "unbound")
                    else:
                        tile_xy.append((b["cx"], b["cy"]))
                        flags.append("blob")
                g = d["objects"][it["obj"]]
                for lay in LAYOUTS:
                    s = score_item(it, A[lay][it["id"]], tr["pre"], tile_xy)
                    row = dict(variant=v, layout=lay, id=it["id"], ep=e, task=it["task"], turn=it["turn"], obj=it["obj"],
                               label=it["label"], k=it["k"], moved=it["moved"], twin=it["twin"], held=it["held"],
                               near=it["near"], nearest_ok=it["nearest_ok"], trk_status=tr["status"],
                               trk_pre=[round(x, 2) for x in tr["pre"]], trk_post=[round(x, 2) for x in tr["post"]],
                               trk_err_cm=round(math.dist(tr["pre"], g[:2]), 1), trk_locked=tr["locked"],
                               ok_step=tr.get("ok_step"), tile_flags=flags,
                               tile_xy=[[round(x, 2) for x in xy] for xy in tile_xy], **s)
                    out_rows[(v, lay)].append(row)
                    f.write(json.dumps(row) + "\n")

    summary = dict(checks=dict(blobs=check_against_g1(S), sep2=A["_check"], late_bind=late_bind, unbound=unbound,
                               target_binding=dict(binds)))
    # reproduction of the old figures
    R = out_rows[("primary", "sep2")]
    repro = {m: float(np.mean([r["ok"][m] for r in R])) for m in ("gt_only", "gt_fuse", "gt_fuse_tile")}
    print("\nold prior, sep2 (expect 0.900 / 0.902 / 0.933):", {k: round(x, 3) for k, x in repro.items()})
    summary["repro_sep2"] = repro

    # primary
    diff, lo, hi = boot_diff(R, "trk_fuse_tile", "trk_only")
    ao = sum(r["ok"]["trk_fuse_tile"] and not r["ok"]["trk_only"] for r in R)
    bo = sum(r["ok"]["trk_only"] and not r["ok"]["trk_fuse_tile"] for r in R)
    verdict = reading(diff, lo, hi)
    summary["primary"] = dict(diff=diff, lo=lo, hi=hi, fuse_only_right=ao, trk_only_right=bo,
                              mcnemar=mcnemar_exact(ao, bo), reading=verdict)
    print(f"\nPRIMARY sep2 trk_fuse_tile - trk_only = {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]  discordant {ao}:{bo} "
          f"(McNemar p {mcnemar_exact(ao, bo):.3g})  -> {verdict}")

    strata = {"all": lambda r: True, "moved": lambda r: r["moved"], "not moved": lambda r: not r["moved"],
              "twin": lambda r: r["twin"], "no twin": lambda r: not r["twin"],
              "held/near (item flags)": lambda r: r["held"] or r["near"],
              "arm clear (item flags)": lambda r: not (r["held"] or r["near"]),
              "tracker frozen this turn": lambda r: r["trk_status"] in ("held", "near"),
              "tracker carried this turn": lambda r: r["trk_status"] in ("carried", "slipped"),
              "any tile on gt fallback": lambda r: any(fl != "blob" for fl in r["tile_flags"]),
              "all tiles on own blob": lambda r: all(fl == "blob" for fl in r["tile_flags"]),
              "trk_only wrong": lambda r: not r["ok"]["trk_only"],
              "gt_only wrong": lambda r: not r["ok"]["gt_only"]}
    summary["tables"] = {}
    for v in VARIANTS:
        for lay in LAYOUTS:
            rows = out_rows[(v, lay)]
            if v not in ("primary", "carry") and lay != "sep2":
                continue
            print(f"\n== variant {v}, layout {lay}")
            print(f"  {'stratum':26s} {'n':>4s} {'eps':>4s} " + " ".join(f"{m:>13s}" for m in METHODS)
                  + "   fuse_tile-trk_only [boot]")
            tab = {}
            for sname, keep in strata.items():
                sub = [r for r in rows if keep(r)]
                if not sub:
                    continue
                accs = {m: float(np.mean([r["ok"][m] for r in sub])) for m in METHODS}
                dd = boot_diff(sub, "trk_fuse_tile", "trk_only") if len({r["ep"] for r in sub}) > 1 else (
                    accs["trk_fuse_tile"] - accs["trk_only"], float("nan"), float("nan"))
                tab[sname] = dict(n=len(sub), eps=len({r["ep"] for r in sub}), acc=accs, diff=dd)
                print(f"  {sname:26s} {len(sub):4d} {len({r['ep'] for r in sub}):4d} "
                      + " ".join(f"{accs[m]:13.3f}" for m in METHODS)
                      + f"   {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]")
            summary["tables"][f"{v}/{lay}"] = tab

    # secondary contrasts with intervals (primary variant)
    sec = {}
    for v, lay in [("primary", "sep2"), ("primary", "one_tb"), ("carry", "sep2"), ("carry", "one_tb"), ("naive", "sep2")]:
        rows = out_rows[(v, lay)]
        lay = f"{v}/{lay}"
        for a_, b_ in (("trk_fuse", "trk_only"), ("trk_fuse_tile", "trk_only"), ("clef_tile", "trk_only"),
                       ("trk_fuse_tile", "gt_fuse_tile"), ("trk_only", "gt_only"), ("gt_fuse_tile", "gt_only")):
            sec[f"{lay}: {a_} - {b_}"] = boot_diff(rows, a_, b_)
        for m in METHODS:
            sec[f"{lay}: acc {m}"] = (float(np.mean([r["ok"][m] for r in rows])), *boot_acc(rows, m))
    summary["secondary"] = sec
    print("\nsecondary (diff or acc [episode bootstrap 95%]):")
    for k, (x, l, h) in sec.items():
        print(f"  {k:48s} {x:+.3f} [{l:+.3f}, {h:+.3f}]")

    # the tracker itself
    trk = {}
    for v in VARIANTS:
        rows = out_rows[(v, "sep2")]
        st = collections.Counter(r["trk_status"] for r in rows)
        trk[v] = dict(trk_only_wrong=sum(not r["ok"]["trk_only"] for r in rows),
                      trk_only_wrong_eps=len({r["ep"] for r in rows if not r["ok"]["trk_only"]}),
                      trk_only_wrong_targets=len({(r["ep"], r["obj"]) for r in rows if not r["ok"]["trk_only"]}),
                      lost_lock=sum(not r["trk_locked"] for r in rows),
                      err_cm_median=float(np.median([r["trk_err_cm"] for r in rows])),
                      err_cm_gt10=sum(r["trk_err_cm"] > 10 for r in rows),
                      status_at_item=dict(st),
                      gt_fallback_tiles=sum(fl != "blob" for r in rows for fl in r["tile_flags"]),
                      items_with_fallback=sum(any(fl != "blob" for fl in r["tile_flags"]) for r in rows),
                      target_tile_fallback=sum(r["tile_flags"][int(r["label"][1:]) - 1] != "blob" for r in rows))
        print(f"\ntracker ({v}): {trk[v]}")
    summary["tracker"] = trk
    # whole-episode tracker statistics (every turn, primary)
    allrows = [json.loads(l) for l in open(OUT/"track.jsonl")]
    summary["tracker_all_turns"] = {}
    for v in VARIANTS:
        vr = [r for r in allrows if r["variant"] == v and r["status"] != "bind"]
        summary["tracker_all_turns"][v] = dict(turns=len(vr), status=dict(collections.Counter(r["status"] for r in vr)),
                                               moved_wrong_blob=sum(r["status"] == "moved" and not r["locked"] for r in vr))
        print(f"tracker over all turns ({v}):", summary["tracker_all_turns"][v])

    # where the tracker loses the target: first unlocked turn per target
    lost = collections.defaultdict(dict)
    for (v, e, role), rows in tracks.items():
        first_lost = next((t for t, r in sorted(rows.items()) if not r["locked"]), None)
        if first_lost is not None:
            lost[v][f"{e}|{role}"] = first_lost
    summary["first_lost_turn"] = lost
    print("first turn each tracker sits on a wrong blob:", dict(lost))
    json.dump(summary, open(OUT/"summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
