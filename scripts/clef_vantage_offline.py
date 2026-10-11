#!/usr/bin/env python3
"""Where can Clef judge a shoe's toe well enough to act on? Items for several vantage points (moment + camera +
interface), each one shoe event with its truth, from the replay captures, the rt9b development runs and the live gate
pilot. No Clef call here: this fixes the events, the halves and the interfaces before any answer is seen.

Why: the release gate (src/branchlab/rtscene/release_gate.py) asked Clef, right before a holding gripper opened,
which way the toe points, from a top-camera crop at the opening arm's fingertips. In the live pilot
(outputs/robodawn_colab/rt9a/.../relay_baseline_gateclef_standard) all 3 vetoes were wrong: at the release moment
the toe end was under the arm and gripper, Clef took the visible heel for the toe, and Gemini turned a correct shoe.

Run-time rule. Events are SELECTED from proprioception and commands only (fingertip positions, commanded and real
gripper openings, the commands and their results); crops are placed the same way. The true poses label the events
(and give analysis-only fields); segmentation is analysis only. Nothing is ever dropped on the truth: an event with
no shoe within HOLD_CM of the fingertips is kept with label "none" (a veto there is a false alarm the gate would
make).

Sources.
  replay  outputs/robodawn_colab/rt8d/replay/eval_baseline/{place_shoe,place_dual_shoes}: raw images, camera matrices
          from captures.npz, segmentation (scripts/replay_capture.py). These ARE the rt8b gemini_flash_{flex,standard}
          rollouts, so rt8b traces add nothing (checked below: same seed and commands -> excluded).
  dev     outputs/robodawn_colab/rt9b/robodawn_dev/relay_baseline_standard/{place_shoe,place_dual_shoes}: 20 episodes,
          turn-start images only (annotated: grid and markers, as Gemini saw them), harness cameras
          (clef_semantic_offline.project), truth from the trace state. No release-moment images.
  pilot   the live gate pilot (place_dual_shoes 24-26; same seeds as dev place_dual_shoes 24-26, different rollouts).
          Never used to choose thresholds.

Vantages (replay; crops are the gate's: release_gate.crop_box, +-15 cm in x/y and [z-3, z+12] cm about the centre,
scaled to 256 px):
  P0   release moment: EVERY "preopen" capture (the replay captures right before every open of a gripper closed on
       something, i.e. the deployed gate's trigger): top crop at the opening arm's fingertips. The current gate.
  P1   same moment: agent (oblique) camera crop of the same box.
  P2   same moment: the opening arm's wrist camera, full 320x240.
  P3   same moment: top + agent + wrist in one request (the question names the first, top, picture).
       (P2c, the fixed wrist centre crop, is dropped: at every preopen the fingertip projects to about (321, 323),
       outside the 320x240 wrist frame, so no crop can be centred on the jaw; elsewhere it duplicated P2.)
  R1 / R2 / R3      after release, IMMEDIATE: the next capture of any kind; top / agent / top + agent crops centred on
                    the release fingertip xy at z = table + 4 cm.
  R1c / R2c / R3c  after release, CLEAR: the first capture after the open in which each arm's fingertips are >= 18 cm
                    (xy) from the release point or >= 25 cm above it; none when a preclose of any arm within 12 cm
                    (xy) of the release point comes first, or the episode ends.
  R1a / R2a / R3a  after release, ARM AWAY: the first turn or final capture after the open in which the releasing
                    fingertip is >= 10 cm from the release point and the other arm's fingertip >= 10 cm (xy) from it;
                    same stop rule.
  L1   carried: every "turn" capture where an arm has gripper < 0.5, gripper_real > 0.08 and fingertip z - table z
       >= 8 cm: top + agent crops at the fingertips (question about the first, top, picture).

Vantages from traces (dev and pilot; annotated turn-start images, crops on the harness cameras):
  T_P0  release moment at a turn start: an executed open of an arm that was holding at the turn start (gripper < 0.5,
        gripper_real > 0.02) with no command of that arm before the open in that turn, so the turn-start image shows
        the release pose: top crop at the turn-start fingertips.
  T_R1, T_R3     the turn start right after an executed open of a holding arm (top; top + agent), crop at the release
                 fingertip xy, z = table + 4.
  T_R1c, T_R3c   the first later turn start with both arms clear (rule of R1c); stop at a close (of any arm) within
                 12 cm (xy) of the release point.
  T_R1a          the first later turn start with the arm away (rule of R1a); same stop rule.
  T_L1  carried, the L1 rule at turn starts (top + agent).
Pilot only:
  LIVE_P0          the gate's saved crop with the live probabilities.
  LIVE_R1, LIVE_R3 the open that finally ran for a gated hold (the gate's turn when not vetoed, else the arm's next
                   executed open, searching only until the arm's next close or next gate entry), next turn start.

Release fingertip in traces: the turn-start fingertip plus the arm's moves sent before the open, using each move's
result (a failed move adds the "moved (dx, dy, dz)" it reports, or nothing when it did not move); when the open is the
arm's last motion in its turn, the next turn's fingertip (the open does not move the arm). fingertip_exact is False
when a command of unknown effect (point down, home, an unparsed failure) came first.

Events and units. Each item is one image set for one event. "event" identifies the shoe event: P/R events are one per
release (release capture/turn + arm), L1 events one per (episode, arm, hold segment, truth bucket, toe-yaw
cluster): the first capture of a hold is kept, and a later one only when no kept capture of that hold has the same
bucket and a toe yaw within 20 deg (keep_event; weight = 1 / captures in the hold). keep_hold_last marks the per-hold
version declared before any Clef call: the hold's last carried capture before the open. Counts are over keep_event
items.

Truth (labels and analysis only): toe = rot(quat) @ TOE_LOCAL; bucket() of its xy gives left/right/up/down in the
top picture and the margin to the nearest bucket edge (clear = margin >= 10 deg). "left" = correct-facing.
up_z = the shoe's up axis z in the checker's frame (upright = up_z >= 0.7). shoe_off_cm (R) = the released shoe's
xy distance from the release point (analysis only, never a filter). From the replay's top segmentation: vis_frac,
vis_frac_scaled, toe_visible / heel_visible (the point 9 cm along / against the toe lies on the shoe's own id).

Interface (frozen here, see interfaces.json): per image goal_pic = the picture direction of world -x at a run-time
anchor (crop centre for top/agent; for the wrist camera the point on its optical axis at the fingertip's depth),
from K and the extrinsic only, with goal_margin (flag < 10 deg). Two variants: "forced4" (left/right/up/down,
P(not left) = 1 - P(goal_pic)) and "abstain5" (adds "hidden": the toe end cannot be seen; P(not left) = 1 - P(goal)
- P(hidden), i.e. hidden is no veto). Release-moment and carried items ask about "the shoe held in the centre",
after-release items about "the shoe in the centre". Multi-image items ask about the first (top) picture.

Halves: A/B by sha1 of the episode key (even -> A), fixed before any Clef answer. Criterion (fixed): choose on A the
lowest P(not left) threshold with false alarms <= 5% of correct-facing events, test on B: PASS needs false alarms <=
5% and catches >= 50%, with >= 8 wrong-facing and >= 15 correct-facing events in B (else INCONCLUSIVE); and swapped.
QA sheets show truth, so they are built from half A and the pilot only.

    python scripts/clef_vantage_offline.py build      # items.jsonl, interfaces.json, img/, sheets/ under OUT

Needs numpy and Pillow.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"src"))
sys.path.insert(0, str(PROJECT/"scripts"))
from branchlab.rtscene.release_gate import TILE, TOE_OPTIONS, crop_box, projector  # noqa: E402
from clef_semantic_offline import TOE_LOCAL, bucket, project as fixed_project, rot  # noqa: E402

OUT = PROJECT/"outputs/analysis/clef_vantage"
RUNS = PROJECT/"outputs/robodawn_colab"
REPLAY = RUNS/"rt8d/replay/eval_baseline"
DEV = RUNS/"rt9b/robodawn_dev/relay_baseline_standard"
PILOT = RUNS/"rt9a/robodawn_dev/relay_baseline_gateclef_standard/place_dual_shoes"
RT8B = RUNS/"rt8b/robodawn"
TASKS = ("place_shoe", "place_dual_shoes")
SHOE = "041_shoe"
HALF_CM = 15.0
HOLD_CM = 12.0          # the shoe within this of the fingertips is the held one (label only)
CARRY_Z_CM = 8.0        # L1: fingertip this far above the table
HOLD_REAL = 0.02        # holding: gripper < 0.5 and gripper_real > this (replay_capture.holding)
CARRY_REAL = 0.08       # L1: gripper_real > this
REST_Z_CM = 4.0         # R*: crop height above the table at the release point
TOE_CM = 9.0            # toe / heel points along the toe direction
CLEAR_XY, CLEAR_Z = 18.0, 25.0     # R1c
AWAY_CM = 10.0                     # R1a
STOP_XY = 12.0                     # R1c / R1a: stop at a close this near the release point (xy)
TURN_DEG = 20.0                    # L1 event unit: a later capture of a hold counts when the toe turned this far
UPRIGHT = 0.7
VANTAGES = ("P0", "P1", "P2", "P3", "R1", "R2", "R3", "R1c", "R2c", "R3c", "R1a", "R2a", "R3a", "L1",
            "T_P0", "T_R1", "T_R3", "T_R1c", "T_R3c", "T_R1a", "T_L1", "LIVE_P0", "LIVE_R1", "LIVE_R3")
CARRIED = ("L1", "T_L1")
CHECKER_Q = [0.5, 0.5, -0.5, -0.5]

# ------------------------------------------------------------------ interfaces (frozen before any Clef call)
TOE_DEF = ("The toe is the closed, rounded front end of the shoe; the heel is the back end, next to the opening for "
           "the foot.")
VIEW_TEXT = {
    "top": "looks straight down at {subj}; the robot stands at the bottom edge of the picture.",
    "agent": "shows {subj} from a camera above and behind the robot, looking away from the robot.",
    "wrist": "is taken by a camera on the robot's wrist, looking past its gripper at {subj}.",
}
SUBJECT = {"held": "the shoe held in its centre", "centre": "the shoe in its centre"}
ORDINAL = ("first", "second", "third")
ABSTAIN = {"hidden": "the toe end of the shoe is hidden (for example under the gripper) or cannot be seen"}


def interface(views: tuple, subject: str, variant: str) -> dict:
    """(state, questions) for an image set; views name the pictures in order; the question is about picture 0."""
    subj = SUBJECT[subject] if views[0] != "wrist" else "the shoe its gripper holds"
    parts = ["A robot is moving shoes on a table."]
    for n, v in enumerate(views):
        lead = "The picture" if len(views) == 1 else f"The {ORDINAL[n]} picture"
        parts.append(f"{lead} " + VIEW_TEXT[v].format(subj=subj if n == 0 else "the same shoe"))
    parts.append(TOE_DEF)
    where = "in the picture" if len(views) == 1 else f"in the first ({views[0]}) picture"
    noun = "the shoe held by the gripper" if views[0] == "wrist" else \
        "the shoe held in the centre" if subject == "held" else "the shoe in the centre"
    crit = dict(TOE_OPTIONS) | (ABSTAIN if variant == "abstain5" else {})
    return {"state": " ".join(parts),
            "questions": {"toe": {"type": "choice", "criteria": crit,
                                  "instructions": f"Which way does the toe of {noun} point, {where}?"}}}


SCORING = {"forced4": "P(not left) = 1 - P(goal_pic[0])",
           "abstain5": "P(not left) = 1 - P(goal_pic[0]) - P('hidden')   ('hidden' counts as no veto)"}


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def half(key: str) -> str:
    """A/B by a fixed hash of the episode key."""
    return "A" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 2 == 0 else "B"


def toe_dir(quat) -> np.ndarray:
    return rot(quat) @ TOE_LOCAL


def up_z(quat) -> float:
    return float((rot(quat) @ rot(CHECKER_Q).T @ np.array([0.0, 0.0, 1.0]))[2])


def yaw(toe) -> float:
    return math.degrees(math.atan2(toe[1], toe[0]))


def truth(obj) -> dict:
    if obj is None:
        return dict(label="none", margin=None, toe_z=None, up_z=None, upright=None)
    toe = toe_dir(obj["quat_wxyz"])
    label, margin = bucket(toe[0], toe[1])
    u = up_z(obj["quat_wxyz"])
    return dict(label=label, margin=round(margin, 1), toe_z=round(float(toe[2]), 2), up_z=round(u, 2),
                upright=bool(u >= UPRIGHT))


def pic_dir(proj, p, d, length=TOE_CM) -> tuple | None:
    """Picture direction of world vector d at world point p in an image (image up = -v) and its margin."""
    uv = proj([p, p + length * np.asarray(d, float)])
    if not np.all(np.isfinite(uv)):
        return None
    v = uv[1] - uv[0]
    if np.linalg.norm(v) < 1e-6:
        return None
    name, margin = bucket(v[0], -v[1])
    return name, round(margin, 1)


MINUS_X = np.array([-1.0, 0.0, 0.0])


def in_front(params, pts) -> bool:
    E = np.asarray(params["extrinsic_cv"], float)[:3]
    cam = np.asarray(pts, float).reshape(-1, 3) / 100.0 @ E[:, :3].T + E[:, 3]
    return bool(np.all(cam[:, 2] > 0.02))


def axis_anchor(params, f) -> np.ndarray:
    """The world point (cm) on the camera's optical axis at the depth of point f (run time: K and extrinsic only)."""
    E = np.asarray(params["extrinsic_cv"], float)[:3]
    R, t = E[:, :3], E[:, 3]
    depth = float((R @ (np.asarray(f, float) / 100.0) + t)[2])
    cam_pt = np.array([0.0, 0.0, max(depth, 0.03)])
    return (R.T @ (cam_pt - t)) * 100.0


class Saver:
    """Saves images once (shared between vantages); returns a part dict: file, truth_uv, pic (the truth's picture
    direction, analysis only), goal_pic / goal_margin (run time)."""

    def __init__(self):
        (OUT/"img").mkdir(parents=True, exist_ok=True)
        self.done = set()

    def _part(self, name, view, uv, truthpic, goal):
        return dict(file=name, view=view, truth_uv=uv, pic=truthpic[0] if truthpic else None,
                    pic_margin=truthpic[1] if truthpic else None, goal_pic=goal[0] if goal else None,
                    goal_margin=goal[1] if goal else None)

    def crop(self, name, view, img, proj, centre, p, toe):
        box = crop_box(proj, centre, HALF_CM)
        if name not in self.done:
            img.crop(box).resize((TILE, TILE), Image.BICUBIC).save(OUT/"img"/name, quality=92)
            self.done.add(name)
        uv = None
        if p is not None:
            k = TILE / (box[2] - box[0])
            u = (proj([p, p + TOE_CM * toe]) - np.array(box[:2])) * k
            uv = {"centre": u[0].round(1).tolist(), "toe": u[1].round(1).tolist()}
        return self._part(name, view, uv, pic_dir(proj, p, toe) if p is not None else None,
                          pic_dir(proj, np.asarray(centre, float), MINUS_X, 2.0))

    def whole(self, name, view, img, proj, params, anchor, p, toe):
        if name not in self.done:
            img.save(OUT/"img"/name, quality=92)
            self.done.add(name)
        goal = pic_dir(proj, anchor, MINUS_X, 2.0) if in_front(params, [anchor, anchor + 2 * MINUS_X]) else None
        if p is None or not in_front(params, [p, p + TOE_CM * toe]):
            return self._part(name, view, None, None, goal)
        u = proj([p, p + TOE_CM * toe])
        return self._part(name, view, {"centre": u[0].round(1).tolist(), "toe": u[1].round(1).tolist()},
                          pic_dir(proj, p, toe), goal)


def item(vantage, base, parts, subject) -> dict:
    views = tuple(x["view"] for x in parts)
    return dict(id=f"{vantage}|{base['event']}", vantage=vantage, **base, views=list(views), subject=subject,
                images=[x["file"] for x in parts], goal_pic=[x["goal_pic"] for x in parts],
                goal_margin=[x["goal_margin"] for x in parts],
                goal_flag=bool(parts[0]["goal_margin"] is None or parts[0]["goal_margin"] < 10),
                pic=[x["pic"] for x in parts], pic_margin=[x["pic_margin"] for x in parts],
                truth_uv=[x["truth_uv"] for x in parts])


def nearest_shoe(objects, f, tz=None) -> tuple:
    """(name, object, distance) of the shoe nearest point f; (None, None, inf) without shoes."""
    best = (None, None, math.inf)
    for name, o in objects.items():
        if o.get("model") != SHOE or (tz is not None and o["position_cm"][2] < tz - 5):
            continue
        d = float(np.linalg.norm(np.asarray(o["position_cm"], float) - np.asarray(f, float)))
        if d < best[2]:
            best = (name, o, d)
    return best


def holding(arm_state, real=HOLD_REAL) -> bool:
    return arm_state["gripper"] < 0.5 and arm_state["gripper_real"] > real


def _angle(a, b) -> float:
    a, b = np.asarray(a[:2], float), np.asarray(b[:2], float)
    c = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def other(arm):
    return "right" if arm == "left" else "left"


def dedup_holds(items_of_vantage):
    """L1 event unit: one per (episode, arm, hold, truth bucket, toe-yaw cluster). In capture order, a capture is a
    new event unless an earlier kept capture of the same hold has the same bucket and a toe yaw within TURN_DEG
    (so a shoe that flips back and forth in one hold gives one event per bucket, and a stationary hold gives one).
    Also marks the per-hold version (the hold's last carried capture before the open) and weight = 1 / captures."""
    groups = collections.defaultdict(list)
    for i in items_of_vantage:
        groups[(i["ep"], i["arm"], i["hold"])].append(i)
    for g in groups.values():
        g.sort(key=lambda i: i["order"])
        kept = []
        for i in g:
            same = next((k for k in kept if k["label"] == i["label"] and (
                i["toe_yaw"] is None or abs((i["toe_yaw"] - k["toe_yaw"] + 180) % 360 - 180) < TURN_DEG)), None)
            i["keep_event"] = same is None
            if same is None:
                kept.append(i)
                i["event_n"] = len(kept)
            else:
                i["event_n"] = same["event_n"]
            i["n_in_hold"] = len(g)
            i["weight"] = round(1.0 / len(g), 4)
            i["keep_hold_last"] = i is g[-1]
        for i in g:
            i["event"] = f"{i['ep']}|{i['arm']}|hold{i['hold']}|e{i['event_n']}"
            i["id"] = f"{i['vantage']}|{i['ep']}|{i['arm']}|hold{i['hold']}|c{i['idx']}"


# ------------------------------------------------------------------ replay episodes
class Episode:
    def __init__(self, meta_path: Path):
        self.dir = meta_path.parent
        self.meta = json.loads(meta_path.read_text())
        self.key = self.meta["key"]
        self.task = self.meta["task"]
        self.caps = self.meta["captures"]
        self.pos = {c["idx"]: n for n, c in enumerate(self.caps)}
        z = np.load(self.dir/"captures.npz")
        self.npz = {k: z[k] for k in z.files}
        self.half_res = set(self.meta.get("half_resolution", ["top_camera", "agent_camera"]))
        self.first = next(c for c in self.caps if c["kind"] == "turn")
        self.seg_ids = self._shoe_ids()
        self.first_px = {n: self._count(self.first["idx"], i) for n, i in self.seg_ids.items()}
        self.first_dist = {n: self._cam_dist(self.first, n) for n in self.seg_ids}
        self.holds = self._holds()

    def _holds(self) -> dict:
        """(capture idx, arm) -> hold number: maximal runs of captures in which the arm is holding (proprioception)."""
        out, n, was = {}, {"left": 0, "right": 0}, {"left": False, "right": False}
        for c in self.caps:
            for arm in ("left", "right"):
                h = holding(c["state"][arm])
                if h and not was[arm]:
                    n[arm] += 1
                if h:
                    out[(c["idx"], arm)] = n[arm]
                was[arm] = h
        return out

    def params(self, cam: str, idx: int, raw: bool = True) -> dict:
        n = self.pos[idx]
        K = self.npz[f"{cam}_K"][n].copy()
        if raw and cam in self.half_res:
            K[:2] *= 2.0                               # K is for the stored half resolution
        return {"intrinsic_cv": K, "extrinsic_cv": self.npz[f"{cam}_extrinsic"][n]}

    def image(self, idx: int, cam: str) -> Image.Image | None:
        p = self.dir/"images"/f"c{idx:03d}_{cam}_raw.jpg"
        return Image.open(p).convert("RGB") if p.exists() else None

    def shoes(self, state) -> dict:
        return {n: o for n, o in state["objects"].items() if o.get("model") == SHOE}

    def _shoe_ids(self) -> dict:
        ids = sorted(int(k) for k, v in self.meta["entity_names"].items() if v == SHOE)
        seg = self.npz["top_camera_seg"][self.pos[self.first["idx"]]]
        proj = projector(self.params("top_camera", self.first["idx"], raw=False))
        out = {}
        for n, (name, o) in enumerate(self.shoes(self.first["state"]).items()):
            u, v = np.round(proj([o["position_cm"]])[0]).astype(int)
            win = seg[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4].ravel()
            votes = collections.Counter(int(i) for i in win if int(i) in ids)
            out[name] = votes.most_common(1)[0][0] if votes else ids[min(n, len(ids) - 1)]
        return out

    def _count(self, idx: int, sid: int) -> int:
        return int((self.npz["top_camera_seg"][self.pos[idx]] == sid).sum())

    def _cam_dist(self, cap, name) -> float:
        E = np.asarray(self.params("top_camera", cap["idx"], raw=False)["extrinsic_cv"])[:3]
        p = np.asarray(cap["state"]["objects"][name]["position_cm"]) / 100.0
        return float((E[:, :3] @ p + E[:, 3])[2])

    def seg_facts(self, cap, name) -> dict:
        """Visibility from the top camera's segmentation (analysis only)."""
        sid = self.seg_ids.get(name)
        if sid is None:
            return {}
        seg = self.npz["top_camera_seg"][self.pos[cap["idx"]]]
        proj = projector(self.params("top_camera", cap["idx"], raw=False))
        o = cap["state"]["objects"][name]
        p, toe = np.asarray(o["position_cm"], float), toe_dir(o["quat_wxyz"])
        n = self._count(cap["idx"], sid)
        first = max(self.first_px[name], 1)
        scale = (self._cam_dist(cap, name) / self.first_dist[name]) ** 2
        out = dict(seg_px=n, vis_frac=round(n / first, 3), vis_frac_scaled=round(n * scale / first, 3))
        H, W = seg.shape
        for tag, s in (("toe", TOE_CM), ("heel", -TOE_CM)):
            u, v = proj([p + s * toe])[0]
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < W and 0 <= vi < H):
                out[f"{tag}_visible"], out[f"{tag}_win"] = False, 0.0
                continue
            win = seg[max(vi - 2, 0):vi + 3, max(ui - 2, 0):ui + 3]
            out[f"{tag}_visible"] = bool(seg[vi, ui] == sid)
            out[f"{tag}_win"] = round(float((win == sid).mean()), 2)
        return out

    def wrist_facts(self, cap, name, arm) -> dict:
        cam, sid = f"{arm}_camera", self.seg_ids.get(name)
        if sid is None or f"{cam}_seg" not in self.npz:
            return {}
        params = self.params(cam, cap["idx"], raw=False)
        seg = self.npz[f"{cam}_seg"][self.pos[cap["idx"]]]
        o = cap["state"]["objects"][name]
        p, toe = np.asarray(o["position_cm"], float), toe_dir(o["quat_wxyz"])
        out = dict(wrist_shoe_frac=round(float((seg == sid).mean()), 3))
        H, W = seg.shape
        for tag, s in (("toe", TOE_CM), ("heel", -TOE_CM)):
            pt = p + s * toe
            u, v = projector(params)([pt])[0]
            ui, vi = int(round(u)), int(round(v))
            out[f"wrist_{tag}_visible"] = bool(in_front(params, [pt]) and 0 <= ui < W and 0 <= vi < H
                                               and seg[vi, ui] == sid)
        return out


def after_release(ep: Episode, n: int, arm: str, f: np.ndarray, rule: str):
    """The capture after the open at caps[n] for an R moment (proprioception only), or (None, reason)."""
    if rule == "immediate":
        return (ep.caps[n + 1], None) if n + 1 < len(ep.caps) else (None, "no capture after the open")
    for c in ep.caps[n + 1:]:
        st = c["state"]
        if c["kind"] == "preclose":
            g = c["command"].split()[0]
            if float(np.linalg.norm(np.asarray(st[g]["position_cm"], float)[:2] - f[:2])) < STOP_XY:
                return None, "a close near the release point came first"
        fa = {a: np.asarray(st[a]["position_cm"], float) for a in ("left", "right")}
        if rule == "clear":
            if all(float(np.linalg.norm(fa[a][:2] - f[:2])) >= CLEAR_XY or fa[a][2] - f[2] >= CLEAR_Z
                   for a in ("left", "right")):
                return c, None
        elif rule == "away" and c["kind"] in ("turn", "final"):
            if float(np.linalg.norm(fa[arm] - f)) >= AWAY_CM and \
                    float(np.linalg.norm(fa[other(arm)][:2] - f[:2])) >= AWAY_CM:
                return c, None
    return None, "the episode ended first"


def replay_items(saver: Saver, skips: collections.Counter) -> list[dict]:
    items = []
    for meta_path in sorted(p for t in TASKS for p in (REPLAY/t).glob("episode_*/meta.json")):
        ep = Episode(meta_path)
        hv = half(ep.key)
        common = dict(source="replay", task=ep.task, ep=ep.key, half=hv, annotated=False)
        for n, cap in enumerate(ep.caps):
            st, tz = cap["state"], cap["state"]["table_z_cm"]
            # ---------------- P*: every preopen (the gate's trigger), and R*: after it
            if cap["kind"] == "preopen":
                arm = cap["command"].split()[0]
                f = np.asarray(st[arm]["position_cm"], float)
                name, o, dist = nearest_shoe(st["objects"], f)
                if dist > HOLD_CM:
                    name, o = None, None
                p = np.asarray(o["position_cm"], float) if o else None
                toe = toe_dir(o["quat_wxyz"]) if o else None
                wrist = f"{arm}_camera"
                imgs = {c: ep.image(cap["idx"], c) for c in ("top_camera", "agent_camera", wrist)}
                if any(v is None for v in imgs.values()):
                    skips["P: missing image"] += 1
                    continue
                m = _h(f"{ep.key}|c{cap['idx']}|{arm}")
                pr = {c: ep.params(c, cap["idx"]) for c in imgs}
                top = saver.crop(f"{m}_top.jpg", "top", imgs["top_camera"], projector(pr["top_camera"]), f, p, toe)
                agent = saver.crop(f"{m}_agent.jpg", "agent", imgs["agent_camera"], projector(pr["agent_camera"]), f,
                                   p, toe)
                wfull = saver.whole(f"{m}_wrist.jpg", "wrist", imgs[wrist], projector(pr[wrist]), pr[wrist],
                                    axis_anchor(pr[wrist], f), p, toe)
                wf = pic_dir(projector(pr[wrist]), f, MINUS_X, 2.0) if in_front(pr[wrist], [f, f + 2 * MINUS_X]) \
                    else None
                event = f"{ep.key}|rel c{cap['idx']}|{arm}"
                base = dict(common, event=event, hold=ep.holds.get((cap["idx"], arm)), idx=cap["idx"],
                            turn=cap["turn"], kind=cap["kind"], command=cap["command"], arm=arm, obj=name, **truth(o),
                            lift_cm=round(float(p[2] - tz), 1) if o else None,
                            dist_cm=round(dist, 1) if math.isfinite(dist) else None,
                            gripper_real=st[arm]["gripper_real"], fingertip_cm=f.round(2).tolist(),
                            wrist_goal_at_fingertip=wf[0] if wf else None,
                            wrist_goal_at_fingertip_margin=wf[1] if wf else None,
                            **(ep.seg_facts(cap, name) if o else {}), **(ep.wrist_facts(cap, name, arm) if o else {}))
                for v, parts in (("P0", [top]), ("P1", [agent]), ("P2", [wfull]), ("P3", [top, agent, wfull])):
                    items.append(item(v, base, parts, "held"))
                for rule, (v1, v2, v3) in (("immediate", ("R1", "R2", "R3")), ("clear", ("R1c", "R2c", "R3c")),
                                           ("away", ("R1a", "R2a", "R3a"))):
                    nxt, why = after_release(ep, n, arm, f, rule)
                    if nxt is None:
                        skips[f"{v1}: {why}"] += 1
                        continue
                    st2, tz2 = nxt["state"], nxt["state"]["table_z_cm"]
                    o2 = st2["objects"][name] if name else None
                    p2 = np.asarray(o2["position_cm"], float) if o2 else None
                    toe2 = toe_dir(o2["quat_wxyz"]) if o2 else None
                    centre = np.array([f[0], f[1], tz2 + REST_Z_CM])
                    rimgs = {c: ep.image(nxt["idx"], c) for c in ("top_camera", "agent_camera")}
                    if any(v is None for v in rimgs.values()):
                        skips[f"{v1}: missing image"] += 1
                        continue
                    rm = _h(f"{ep.key}|c{nxt['idx']}|rel{arm}|c{cap['idx']}")
                    rtop = saver.crop(f"{rm}_rtop.jpg", "top", rimgs["top_camera"],
                                      projector(ep.params("top_camera", nxt["idx"])), centre, p2, toe2)
                    ragent = saver.crop(f"{rm}_ragent.jpg", "agent", rimgs["agent_camera"],
                                        projector(ep.params("agent_camera", nxt["idx"])), centre, p2, toe2)
                    fa = {a: np.asarray(st2[a]["position_cm"], float) for a in ("left", "right")}
                    rbase = dict(common, event=event, hold=base["hold"], idx=nxt["idx"], release_idx=cap["idx"],
                                 turn=nxt["turn"], kind=nxt["kind"], next_kind=nxt["kind"],
                                 after_final=nxt["kind"] == "final", captures_after=ep.pos[nxt["idx"]] - n,
                                 command=cap["command"], arm=arm, obj=name, **truth(o2),
                                 release_label=base["label"],
                                 arm_moved_cm=round(float(np.linalg.norm(fa[arm] - f)), 1),
                                 arm_moved_xy_cm=round(float(np.linalg.norm(fa[arm][:2] - f[:2])), 1),
                                 other_arm_xy_cm=round(float(np.linalg.norm(fa[other(arm)][:2] - f[:2])), 1),
                                 shoe_off_cm=round(float(np.linalg.norm(p2[:2] - f[:2])), 1) if o2 else None,
                                 moved_cm=round(float(np.linalg.norm(p2[:2] - p[:2])), 1) if o2 else None,
                                 turned_deg=round(_angle(toe, toe2), 1) if o2 else None,
                                 lift_cm=round(float(p2[2] - tz2), 1) if o2 else None,
                                 release_fingertip_cm=f.round(2).tolist(), **(ep.seg_facts(nxt, name) if o2 else {}))
                    for v, parts in ((v1, [rtop]), (v2, [ragent]), (v3, [rtop, ragent])):
                        items.append(item(v, rbase, parts, "centre"))
            # ---------------- L1: carried (proprioception: closed, something between, fingertips 8 cm up)
            if cap["kind"] == "turn":
                for arm in ("left", "right"):
                    a = st[arm]
                    f = np.asarray(a["position_cm"], float)
                    if not holding(a, CARRY_REAL) or f[2] - tz < CARRY_Z_CM:
                        continue
                    name, o, dist = nearest_shoe(st["objects"], f)
                    if dist > HOLD_CM:
                        name, o = None, None
                    p = np.asarray(o["position_cm"], float) if o else None
                    toe = toe_dir(o["quat_wxyz"]) if o else None
                    imgs = {c: ep.image(cap["idx"], c) for c in ("top_camera", "agent_camera")}
                    if any(v is None for v in imgs.values()):
                        skips["L1: missing image"] += 1
                        continue
                    m = _h(f"{ep.key}|c{cap['idx']}|{arm}")
                    top = saver.crop(f"{m}_top.jpg", "top", imgs["top_camera"],
                                     projector(ep.params("top_camera", cap["idx"])), f, p, toe)
                    agent = saver.crop(f"{m}_agent.jpg", "agent", imgs["agent_camera"],
                                       projector(ep.params("agent_camera", cap["idx"])), f, p, toe)
                    base = dict(common, event="", hold=ep.holds.get((cap["idx"], arm)), order=n, idx=cap["idx"],
                                turn=cap["turn"], kind="turn", arm=arm, obj=name, **truth(o),
                                toe_yaw=round(yaw(toe), 1) if o else None,
                                lift_cm=round(float(p[2] - tz), 1) if o else None,
                                dist_cm=round(dist, 1) if math.isfinite(dist) else None,
                                gripper_real=a["gripper_real"], fingertip_cm=f.round(2).tolist(),
                                fingertip_up_cm=round(float(f[2] - tz), 1), **(ep.seg_facts(cap, name) if o else {}))
                    items.append(item("L1", base, [top, agent], "held"))
    return items


# ------------------------------------------------------------------ traces (dev runs and the pilot)
MOVE = re.compile(r"^(left|right) move ([xyz]) ([+-]?\d+(?:\.\d+)?)$")
MOVED = re.compile(r"moved \(([+-]?[\d.]+), ([+-]?[\d.]+), ([+-]?[\d.]+)\)")
GRIP = re.compile(r"^(left|right) gripper ([\d.]+)$")


def _result(turn_rec, i):
    res = turn_rec.get("results") or []
    return res[i] if i < len(res) and isinstance(res[i], dict) else None


def fingertip_at(turn_rec, arm, k, next_rec=None) -> tuple[np.ndarray, bool]:
    """The arm's fingertip right before command k of a turn: turn-start fingertip plus the arm's moves before k,
    from each move's result (a failed move adds what it reports it moved). When no motion of that arm follows
    command k's predecessor in the turn and next_rec is given, the next turn's fingertip (exact) is used."""
    cmds = turn_rec["commands"]
    later = [c for c in cmds[k:] if c.startswith(arm + " ") and not GRIP.match(c.strip())]
    if next_rec is not None and not later:
        return np.asarray(next_rec["state"][arm]["position_cm"], float).copy(), True
    f = np.asarray(turn_rec["state"][arm]["position_cm"], float).copy()
    exact = True
    for i, cmd in enumerate(cmds[:k]):
        cmd = cmd.strip()
        if not cmd.startswith(arm + " ") or GRIP.match(cmd) or " rotate " in cmd:
            continue
        r = _result(turn_rec, i)
        mm = MOVE.match(cmd)
        if r is not None and not r.get("ok", True):
            note = r.get("note") or ""
            got = MOVED.search(note)
            if got:
                f += np.array([float(x) for x in got.groups()])
            elif "did NOT m" not in note:
                exact = False
            continue
        if mm:
            f["xyz".index(mm.group(2))] += float(mm.group(3))
        else:
            exact = False
    return f, exact


def executed_opens(turn_rec):
    """(command index, arm, holding at the open) for each executed open in a turn; holding follows the arm through
    the turn: the turn-start state, then each close's result."""
    out = []
    hold = {a: holding(turn_rec["state"][a]) for a in ("left", "right")}
    for i, cmd in enumerate(turn_rec["commands"]):
        m = GRIP.match(cmd.strip())
        if not m:
            continue
        arm, val = m.group(1), float(m.group(2))
        r = _result(turn_rec, i)
        if val >= 0.5:
            if r is not None and r.get("ok") and r.get("note") == "gripper opened":
                out.append((i, arm, hold[arm]))
                hold[arm] = False
        elif r is not None and r.get("ok"):
            hold[arm] = "something" in (r.get("note") or "")
    return out


def closes(turn_rec, start=0):
    """(command index, arm) of the close commands in a turn from command `start`."""
    return [(i, m.group(1)) for i, c in enumerate(turn_rec["commands"]) if i >= start
            for m in [GRIP.match(c.strip())] if m and float(m.group(2)) < 0.5]


class Trace:
    def __init__(self, ep_dir: Path, key: str, hv: str, source: str, task: str):
        self.dir, self.key, self.half, self.source, self.task = ep_dir, key, hv, source, task
        self.trace = json.loads((ep_dir/"trace.json").read_text())
        self.by_turn = {int(t["turn"]): t for t in self.trace}
        self.turns = sorted(self.by_turn)

    def image(self, turn, cam):
        p = self.dir/f"turn{turn:03d}_{cam}.png"
        return Image.open(p).convert("RGB") if p.exists() else None

    def common(self):
        return dict(source=self.source, task=self.task, ep=self.key, half=self.half, annotated=True)


def trace_crop(saver, tr: Trace, turn, cam, tag, centre, p, toe):
    img = tr.image(turn, cam)
    if img is None:
        return None
    tz = tr.by_turn[turn]["state"]["table_z_cm"]
    proj = lambda pts, cam=cam, tz=tz, W=img.width, H=img.height: fixed_project(cam, pts, tz, W, H)
    view = "top" if cam == "top_camera" else "agent"
    return saver.crop(f"{_h(tr.key + '|' + str(turn) + '|' + tag)}_{tag}.jpg", view, img, proj, centre, p, toe)


def trace_after(tr: Trace, t0: int, k: int, arm: str, f: np.ndarray, rule: str):
    """The turn start after the open (command k of turn t0) for an R moment, or (None, reason). Stop rule: a close
    of any arm within STOP_XY (xy) of the release point before that turn start."""
    if rule == "immediate":
        return (t0 + 1, None) if t0 + 1 in tr.by_turn else (None, "no turn after the open")
    for t in [x for x in tr.turns if x > t0 - 1]:
        if t > t0:
            st = tr.by_turn[t]["state"]
            fa = {a: np.asarray(st[a]["position_cm"], float) for a in ("left", "right")}
            if rule == "clear" and all(float(np.linalg.norm(fa[a][:2] - f[:2])) >= CLEAR_XY
                                       or fa[a][2] - f[2] >= CLEAR_Z for a in ("left", "right")):
                return t, None
            if rule == "away" and float(np.linalg.norm(fa[arm] - f)) >= AWAY_CM and \
                    float(np.linalg.norm(fa[other(arm)][:2] - f[:2])) >= AWAY_CM:
                return t, None
        rec = tr.by_turn[t]
        for i, a in closes(rec, k + 1 if t == t0 else 0):
            fc, _ = fingertip_at(rec, a, i)
            if float(np.linalg.norm(fc[:2] - f[:2])) < STOP_XY:
                return None, "a close near the release point came first"
    return None, "the episode ended first"


def trace_items(saver: Saver, skips: collections.Counter, tr: Trace) -> list[dict]:
    items = []
    tag = "T"
    for t in tr.turns:
        rec = tr.by_turn[t]
        st, tz = rec["state"], rec["state"]["table_z_cm"]
        nxt = tr.by_turn.get(t + 1)
        # ---------------- releases
        for k, arm, held in executed_opens(rec):
            if not held:
                skips[f"{tag}_R ({tr.source}): open of an arm not holding"] += 1
                continue
            f, exact = fingertip_at(rec, arm, k, nxt)
            f0 = np.asarray(st[arm]["position_cm"], float)
            name, o, dist = nearest_shoe(st["objects"], f0, tz)      # the held shoe at the turn start (label only)
            if dist > HOLD_CM:
                name, o = None, None
            event = f"{tr.key}|rel t{t}|{arm}"
            hold_n = f"rel t{t}"                                       # one release per hold
            # T_P0: the open has no command of that arm before it, so the turn-start image shows the release pose
            if not any(c.startswith(arm + " ") for c in rec["commands"][:k]):
                p, toe = (np.asarray(o["position_cm"], float), toe_dir(o["quat_wxyz"])) if o else (None, None)
                part = trace_crop(saver, tr, t, "top_camera", f"p0{arm}", f0, p, toe)
                if part is not None:
                    base = dict(tr.common(), event=event, hold=hold_n, idx=t, turn=t, kind="turn", arm=arm, obj=name,
                                **truth(o), lift_cm=round(float(p[2] - tz), 1) if o else None,
                                dist_cm=round(dist, 1) if math.isfinite(dist) else None,
                                gripper_real=st[arm]["gripper_real"], fingertip_cm=f0.round(2).tolist(),
                                other_cmds_before=k)
                    items.append(item("T_P0", base, [part], "held"))
            else:
                skips[f"T_P0 ({tr.source}): the arm moved before the open"] += 1
            for rule, vs in (("immediate", ("T_R1", "T_R3")), ("clear", ("T_R1c", "T_R3c")), ("away", ("T_R1a",))):
                t2, why = trace_after(tr, t, k, arm, f, rule)
                if t2 is None:
                    skips[f"{vs[0]} ({tr.source}): {why}"] += 1
                    continue
                st2, tz2 = tr.by_turn[t2]["state"], tr.by_turn[t2]["state"]["table_z_cm"]
                o2 = st2["objects"].get(name) if name else None
                p2, toe2 = (np.asarray(o2["position_cm"], float), toe_dir(o2["quat_wxyz"])) if o2 else (None, None)
                centre = np.array([f[0], f[1], tz2 + REST_Z_CM])
                parts = [trace_crop(saver, tr, t2, "top_camera", f"r{arm}{t}", centre, p2, toe2),
                         trace_crop(saver, tr, t2, "agent_camera", f"ra{arm}{t}", centre, p2, toe2)]
                if any(x is None for x in parts):
                    skips[f"{vs[0]} ({tr.source}): missing image"] += 1
                    continue
                fa = {a: np.asarray(st2[a]["position_cm"], float) for a in ("left", "right")}
                rbase = dict(tr.common(), event=event, hold=hold_n, idx=t2, turn=t2, release_turn=t, kind="turn",
                             next_kind="turn", after_final=False, arm=arm, obj=name, **truth(o2),
                             release_label=truth(o)["label"],
                             arm_moved_cm=round(float(np.linalg.norm(fa[arm] - f)), 1),
                             arm_moved_xy_cm=round(float(np.linalg.norm(fa[arm][:2] - f[:2])), 1),
                             other_arm_xy_cm=round(float(np.linalg.norm(fa[other(arm)][:2] - f[:2])), 1),
                             shoe_off_cm=round(float(np.linalg.norm(p2[:2] - f[:2])), 1) if o2 else None,
                             lift_cm=round(float(p2[2] - tz2), 1) if o2 else None,
                             release_fingertip_cm=f.round(2).tolist(), fingertip_exact=exact)
                items.append(item(vs[0], rbase, parts[:1], "centre"))
                if len(vs) > 1:
                    items.append(item(vs[1], rbase, parts, "centre"))
        # ---------------- T_L1: carried at a turn start
        for arm in ("left", "right"):
            a = st[arm]
            f = np.asarray(a["position_cm"], float)
            if not holding(a, CARRY_REAL) or f[2] - tz < CARRY_Z_CM:
                continue
            name, o, dist = nearest_shoe(st["objects"], f, tz)
            if dist > HOLD_CM:
                name, o = None, None
            p, toe = (np.asarray(o["position_cm"], float), toe_dir(o["quat_wxyz"])) if o else (None, None)
            parts = [trace_crop(saver, tr, t, "top_camera", f"l{arm}", f, p, toe),
                     trace_crop(saver, tr, t, "agent_camera", f"la{arm}", f, p, toe)]
            if any(x is None for x in parts):
                skips[f"T_L1 ({tr.source}): missing image"] += 1
                continue
            base = dict(tr.common(), event="", hold=None, order=t, idx=t, turn=t, kind="turn", arm=arm, obj=name,
                        **truth(o), toe_yaw=round(yaw(toe), 1) if o else None,
                        lift_cm=round(float(p[2] - tz), 1) if o else None,
                        dist_cm=round(dist, 1) if math.isfinite(dist) else None, gripper_real=a["gripper_real"],
                        fingertip_cm=f.round(2).tolist(), fingertip_up_cm=round(float(f[2] - tz), 1))
            items.append(item("T_L1", base, parts, "held"))
    # hold numbers for T_L1: runs of consecutive turn starts where the arm is holding (proprioception)
    runs, n, was = {}, {"left": 0, "right": 0}, {"left": False, "right": False}
    for t in tr.turns:
        for arm in ("left", "right"):
            h = holding(tr.by_turn[t]["state"][arm])
            opened_before = any(a == arm for _, a, _ in executed_opens(tr.by_turn[t - 1])) if t - 1 in tr.by_turn \
                else False
            if h and (not was[arm] or opened_before):
                n[arm] += 1
            if h:
                runs[(t, arm)] = n[arm]
            was[arm] = h
    for i in items:
        if i["vantage"] == "T_L1":
            i["hold"] = runs.get((i["turn"], i["arm"]))
    return items


def live_gate_items(saver: Saver, skips: collections.Counter, tr: Trace) -> list[dict]:
    items, seen = [], set()
    for t in tr.turns:
        rec = tr.by_turn[t]
        for g in rec.get("gate") or []:
            arm, tz = g["arm"], rec["state"]["table_z_cm"]
            k = next((i for i, c in enumerate(rec["commands"]) if (m := GRIP.match(c.strip()))
                      and m.group(1) == arm and float(m.group(2)) >= 0.5), len(rec["commands"]))
            f, exact = fingertip_at(rec, arm, k, tr.by_turn.get(t + 1) if not g["vetoed"] else None)
            name, o, dist = nearest_shoe(g["objects"], f, tz)
            if dist > HOLD_CM:
                name, o = None, None
            crop_path = tr.dir/g["image"]
            if not crop_path.exists():
                skips["LIVE_P0: no crop"] += 1
                continue
            p, toe = (np.asarray(o["position_cm"], float), toe_dir(o["quat_wxyz"])) if o else (None, None)
            proj = lambda pts, tz=tz: fixed_project("top_camera", pts, tz)
            fname = f"{_h(tr.key + '|' + g['image'])}_livegate.jpg"
            if fname not in saver.done:
                Image.open(crop_path).convert("RGB").save(OUT/"img"/fname, quality=95)
                saver.done.add(fname)
            box = crop_box(proj, f, HALF_CM)
            uv = None
            if o:
                u = (proj([p, p + TOE_CM * toe]) - np.array(box[:2])) * (TILE / (box[2] - box[0]))
                uv = {"centre": u[0].round(1).tolist(), "toe": u[1].round(1).tolist()}
            part = saver._part(fname, "top", uv, pic_dir(proj, p, toe) if o else None, pic_dir(proj, f, MINUS_X, 2.0))
            event = f"{tr.key}|gate t{t}|{arm}"
            base = dict(tr.common(), annotated=False, event=event, hold=None, idx=t, turn=t, kind="gate", arm=arm,
                        obj=name, **truth(o), lift_cm=round(float(p[2] - tz), 1) if o else None,
                        dist_cm=round(dist, 1) if math.isfinite(dist) else None, fingertip_cm=f.round(2).tolist(),
                        fingertip_exact=exact, live_p_not_left=g["p_not_left"],
                        live_probabilities=g["probabilities"], live_vetoed=g["vetoed"])
            items.append(item("LIVE_P0", base, [part], "held"))
            # the open that ran for this hold: this turn when not vetoed, else the arm's next executed open, searched
            # only until the arm's next close or the next gate entry for that arm
            rel, k2 = None, None
            if not g["vetoed"]:
                hit = [x for x in executed_opens(rec) if x[1] == arm]
                rel, k2 = (t, hit[0][0]) if hit else (None, None)
            else:
                for t2 in [x for x in tr.turns if x > t]:
                    r2 = tr.by_turn[t2]
                    if any(gg["arm"] == arm for gg in r2.get("gate") or []) and not any(
                            x[1] == arm for x in executed_opens(r2)):
                        break
                    ops = [x for x in executed_opens(r2) if x[1] == arm]
                    cl = [i for i, a in closes(r2) if a == arm]
                    if ops and (not cl or ops[0][0] < cl[0]):
                        rel, k2 = t2, ops[0][0]
                        break
                    if cl:
                        break
            if rel is None or rel + 1 not in tr.by_turn:
                skips["LIVE_R: no executed open for the hold with a next turn"] += 1
                continue
            if (rel, arm) in seen:
                skips["LIVE_R: release already attributed"] += 1
                continue
            seen.add((rel, arm))
            fr, exact_r = fingertip_at(tr.by_turn[rel], arm, k2, tr.by_turn[rel + 1])
            nt = tr.by_turn[rel + 1]
            st2, tz2 = nt["state"], nt["state"]["table_z_cm"]
            o2 = st2["objects"].get(name) if name else None
            p2, toe2 = (np.asarray(o2["position_cm"], float), toe_dir(o2["quat_wxyz"])) if o2 else (None, None)
            centre = np.array([fr[0], fr[1], tz2 + REST_Z_CM])
            parts = [trace_crop(saver, tr, rel + 1, "top_camera", f"lrtop{arm}", centre, p2, toe2),
                     trace_crop(saver, tr, rel + 1, "agent_camera", f"lragent{arm}", centre, p2, toe2)]
            fa = {a: np.asarray(st2[a]["position_cm"], float) for a in ("left", "right")}
            rbase = dict(tr.common(), event=f"{tr.key}|rel t{rel}|{arm}", hold=None, idx=rel + 1, turn=rel + 1,
                         release_turn=rel, gate_turn=t, kind="turn", next_kind="turn", after_final=False, arm=arm,
                         obj=name, **truth(o2), release_label=base["label"],
                         arm_moved_cm=round(float(np.linalg.norm(fa[arm] - fr)), 1),
                         other_arm_xy_cm=round(float(np.linalg.norm(fa[other(arm)][:2] - fr[:2])), 1),
                         shoe_off_cm=round(float(np.linalg.norm(p2[:2] - fr[:2])), 1) if o2 else None,
                         lift_cm=round(float(p2[2] - tz2), 1) if o2 else None,
                         release_fingertip_cm=fr.round(2).tolist(), fingertip_exact=exact_r,
                         live_vetoed=g["vetoed"])
            items.append(item("LIVE_R1", rbase, parts[:1], "centre"))
            items.append(item("LIVE_R3", rbase, parts, "centre"))
    return items


def replay_duplicates() -> dict:
    """rt8b traces that are the source rollouts of replay episodes (same task, seed and commands) -> replay key."""
    reps = {}
    for t in TASKS:
        for p in (REPLAY/t).glob("episode_*/meta.json"):
            m = json.loads(p.read_text())
            reps[(m["task"], int(m["seed"]))] = (m["key"], [c["command"] for c in m["command_log"]])
    out = {}
    for cfg in sorted(RT8B.glob("*/place_*/shard_*/reproduction_config.json")):
        c = json.loads(cfg.read_text())
        for e in c["episodes"]:
            tp = cfg.parent/f"episode_{e['episode']:03d}/trace.json"
            if c["task"] not in TASKS or not tp.exists():
                continue
            cmds = [x for t in json.loads(tp.read_text()) for x in (t.get("commands") or [])]
            rep = reps.get((c["task"], int(e["seed"])))
            same = rep is not None and rep[1][:20] == cmds[:20]     # the replay log can skip a command later
            out[str(tp.relative_to(RUNS))] = rep[0] if same else None
    return out


def all_trace_items(saver, skips) -> list[dict]:
    items = []
    for task in TASKS:
        for tp in sorted((DEV/task).glob("shard_*/episode_*/trace.json")):
            key = f"rt9b/relay_baseline_standard/{task}/{tp.parent.name}"
            tr = Trace(tp.parent, key, half(key), "dev", task)
            items += trace_items(saver, skips, tr)
    for tp in sorted(PILOT.glob("shard_*/episode_*/trace.json")):
        key = f"pilot/place_dual_shoes/{tp.parent.name}"
        tr = Trace(tp.parent, key, "pilot", "pilot", "place_dual_shoes")
        items += trace_items(saver, skips, tr) + live_gate_items(saver, skips, tr)
    return items


# ------------------------------------------------------------------ QA sheets (half A and the pilot only)
def sheets(items, per_sheet: int = 6, n_sheets: int = 2):
    out_dir = OUT/"sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    font = ImageFont.load_default(size=15)
    made = []
    by = collections.defaultdict(list)
    for i in items:
        if i["half"] in ("A", "pilot") and i.get("keep_event", True):
            by[i["vantage"]].append(i)
    for v in VANTAGES:
        rows = sorted(by.get(v, []), key=lambda i: _h(i["id"]))
        wrong = [i for i in rows if i["label"] != "left"]
        right = [i for i in rows if i["label"] == "left"]
        order = [x for pair in zip(wrong, right) for x in pair] + wrong[len(right):] + right[len(wrong):]
        n = min(n_sheets, math.ceil(len(order) / per_sheet)) if order else 0
        for s in range(n):
            chunk = order[s * per_sheet:(s + 1) * per_sheet]
            tiles = [[_marked(name, uv) for name, uv in zip(i["images"], i["truth_uv"])] for i in chunk]
            W = max(sum(t.width + 6 for t in row) for row in tiles) + 6
            Hrow = TILE + 44
            sheet = Image.new("RGB", (max(W, 900), len(chunk) * Hrow + 6), (255, 255, 255))
            draw = ImageDraw.Draw(sheet)
            for r, (i, row) in enumerate(zip(chunk, tiles)):
                x, y = 6, 6 + r * Hrow
                for t in row:
                    sheet.paste(t, (x, y + 40))
                    x += t.width + 6
                vis = f" toe_vis={int(i['toe_visible'])} vis={i['vis_frac_scaled']:.2f}" if "toe_visible" in i else ""
                live = f" live P(not left)={i['live_p_not_left']:.2f}" if "live_p_not_left" in i else ""
                draw.text((6, y + 2), f"{i['ep']} c/t{i['idx']} {i['arm']} {i['obj']} half={i['half']} "
                                      f"goal={i['goal_pic']}", fill=(0, 0, 0), font=font)
                color = (0, 120, 0) if i["label"] == "left" else (200, 0, 0)
                draw.text((6, y + 20), f"truth={i['label']} margin={i['margin']} up_z={i['up_z']} pic={i['pic']}"
                                       f"{vis}{live}", fill=color, font=font)
            path = out_dir/f"{v}_{s + 1}.png"
            sheet.save(path)
            made.append(str(path))
    return made


def _marked(name, uv) -> Image.Image:
    img = Image.open(OUT/"img"/name).convert("RGB")
    k = TILE / img.height
    if img.height != TILE:
        img = img.resize((round(img.width * k), TILE), Image.BICUBIC)
    if not uv:
        return img
    draw = ImageDraw.Draw(img)
    (cu, cv), (tu, tv) = [(a * k, b * k) for a, b in (uv["centre"], uv["toe"])]
    draw.line([cu, cv, tu, tv], fill=(255, 0, 255), width=3)
    draw.ellipse([tu - 7, tv - 7, tu + 7, tv + 7], outline=(255, 0, 255), width=3)
    draw.ellipse([cu - 3, cv - 3, cu + 3, cv + 3], fill=(255, 0, 255))
    return img


# ------------------------------------------------------------------ counts
def counted(items, v, unit="event"):
    rows = [i for i in items if i["vantage"] == v]
    if v in CARRIED:
        rows = [i for i in rows if (i["keep_event"] if unit == "event" else i["keep_hold_last"])]
    return rows


def tally(rows) -> dict:
    c = collections.Counter()
    for i in rows:
        c[f"{i['half']}_{'none' if i['label'] == 'none' else 'correct' if i['label'] == 'left' else 'wrong'}"] += 1
    return c


def feasible(c) -> str:
    ok = lambda h: c[f"{h}_wrong"] >= 8 and c[f"{h}_correct"] >= 15
    return f"test B {'ok' if ok('B') else 'INCONCLUSIVE'} / test A {'ok' if ok('A') else 'INCONCLUSIVE'}"


def report(items, skips, dups):
    def row(name, rows):
        c = tally(rows)
        eps = len({i["ep"] for i in rows})
        holds = len({(i["ep"], i["arm"], i["hold"]) for i in rows if i.get("hold") is not None})
        tv = [i["toe_visible"] for i in rows if "toe_visible" in i]
        tvs = f"{np.mean(tv):.2f}" if tv else "-"
        print(f"{name:22s} {len(rows):4d} {c['A_wrong']:>4d}/{c['A_correct']:<4d}{c['A_none']:>3d} "
              f"{c['B_wrong']:>4d}/{c['B_correct']:<4d}{c['B_none']:>3d} "
              f"{c['pilot_wrong']:>4d}/{c['pilot_correct']:<4d}{c['pilot_none']:>3d} {eps:4d} {holds:5d} {tvs:>6s}  "
              f"{feasible(c) if any(i['half'] != 'pilot' for i in rows) else ''}")

    print(f"{'vantage':22s} {'n':>4s} {'A w/c none':>12s} {'B w/c none':>12s} {'pilot w/c none':>14s} {'eps':>4s} "
          f"{'holds':>5s} {'toevis':>6s}")
    for v in VANTAGES:
        rows = counted(items, v)
        if not rows:
            print(f"{v:22s}    0")
            continue
        row(v, rows)
        row(f"  {v} margin>=10", [i for i in rows if i["margin"] is not None and i["margin"] >= 10])
        if v in CARRIED:
            row(f"  {v} per-hold (last)", counted(items, v, "hold"))
            row(f"  {v} all captures", [i for i in items if i["vantage"] == v])
        if v.startswith(("R", "T_R", "LIVE_R")):
            row(f"  {v} arm moved>=10", [i for i in rows if i["arm_moved_cm"] >= 10])
            row(f"  {v} arm moved<10", [i for i in rows if i["arm_moved_cm"] < 10])
            if any(i["after_final"] for i in rows):
                row(f"  {v} without final", [i for i in rows if not i["after_final"]])
            if any(i.get("shoe_off_cm") is not None and i["shoe_off_cm"] > 10 for i in rows):
                row(f"  {v} shoe_off<=10", [i for i in rows if (i.get("shoe_off_cm") or 0) <= 10])
        if any(i["half"] == "pilot" for i in rows):
            row(f"  {v} upright", [i for i in rows if i["upright"]])
            row(f"  {v} not upright", [i for i in rows if i["upright"] is False])
        if v in ("P2", "P3"):
            row(f"  {v} goal margin>=10", [i for i in rows if not i["goal_flag"]])
    print("\npooled moment families (replay raw images + dev annotated turn-start images; same rule, other domain):")
    for name, vs in (("P0+T_P0", ("P0", "T_P0")), ("R1+T_R1", ("R1", "T_R1")), ("R1c+T_R1c", ("R1c", "T_R1c")),
                     ("R1a+T_R1a", ("R1a", "T_R1a")), ("L1+T_L1", ("L1", "T_L1"))):
        rows = [i for v in vs for i in counted(items, v) if i["half"] != "pilot"]
        row(name, rows)
        row(f"  {name} margin>=10", [i for i in rows if i["margin"] is not None and i["margin"] >= 10])
        if "L1" in vs:
            row(f"  {name} per-hold", [i for v in vs for i in counted(items, v, "hold") if i["half"] != "pilot"])
    print("\nskipped (no item):", dict(sorted(skips.items())))
    print("rt8b traces:", {k: (f"= replay {v}, excluded" if v else "independent") for k, v in dups.items()})


# ------------------------------------------------------------------ build
def build():
    for old in (OUT/"img").glob("*.jpg"):
        old.unlink()
    saver, skips = Saver(), collections.Counter()
    dups = replay_duplicates()
    items = replay_items(saver, skips) + all_trace_items(saver, skips)
    for v in CARRIED:
        dedup_holds([i for i in items if i["vantage"] == v])
    ids = collections.Counter(i["id"] for i in items)
    assert all(n == 1 for n in ids.values()), [k for k, n in ids.items() if n > 1][:5]
    (OUT/"items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    combos = sorted({(tuple(i["views"]), i["subject"]) for i in items})
    (OUT/"interfaces.json").write_text(json.dumps(
        {"scoring": SCORING, "goal": "goal_pic[0] of each item (world -x in the asked picture, run-time)",
         "interfaces": {f"{'+'.join(v)}|{s}|{var}": interface(v, s, var) for v, s in combos
                        for var in ("forced4", "abstain5")}}, indent=1))
    made = sheets(items)
    print(f"{len(items)} items -> {OUT/'items.jsonl'}; {len(combos) * 2} interfaces -> {OUT/'interfaces.json'}")
    report(items, skips, dups)
    keys = [json.loads(p.read_text())["key"] for t in TASKS for p in sorted((REPLAY/t).glob("episode_*/meta.json"))]
    keys += [f"rt9b/relay_baseline_standard/{t}/{p.name}" for t in TASKS
             for p in sorted((DEV/t).glob("shard_*/episode_*"))]
    print("halves (all episodes):", collections.Counter(half(k) for k in keys))
    print(f"{len(made)} sheets (half A and pilot only) -> {OUT/'sheets'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.parse_args()
    build()


if __name__ == "__main__":
    main()
