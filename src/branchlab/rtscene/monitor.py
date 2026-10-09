"""Object monitor: the objects Gemini located in its first reply are tracked turn by turn and checked for a fall.

Gemini's first reply locates the task objects in words with table coordinates ("Fan is at approx (-5.5, -12)",
"block at x ≈ -27, y ≈ +18"). Code binds each coordinate to the nearest raised depth blob of that turn's start
view (a later reply's, while none is bound). After every turn, for each
bound object that no arm is near:

  track  the blobs within 30 cm of where it was last seen that are tall enough to be it (a rigid object stands
         at least half its smallest side high, however it lies), nearest first, as numbered tiles from straight
         above at the object's own scale, against its crop from the first turn; Clef picks the tile (choice), fused with a distance
         prior. The pick moves the track when its fused probability is at least 0.5 and Clef's own choice is the
         same tile; a check counts towards a fall only then.
  check  the code's shape-change score of the pick against the first turn, and Clef on the start and now crops
         from the front with the depth numbers in the text: has it fallen over? Only objects that stand taller
         than their narrower side can fall; one that still stands at 90% of its height has not, whatever the
         scores say.

The harness also tells the monitor about grasps and releases: an object a closing gripper caught (fingertips
within 8 cm of it) is searched for, after the release, around where the fingers opened, not where it was.
No object is checked while a gripper holds something: a held object leaves the table, its blob is gone and
the track would jump to a neighbour. An object flagged on `need` checks in a row is reported as fallen.
These interfaces were the best found for Clef on the replayed episodes (scripts/rtscene_clef_iface.py);
scripts/rtscene_monitor_replay.py runs this module offline on them, in the harness's order.
"""
from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import blobs as blobs_mod
from . import heightmap, render
from .blobs import Blob, shape_change

TILE = 224
CAND_RADIUS = 30.0        # cm from the last position
MAX_CANDS = 6
D_PRIOR = 10.0            # cm
SURE = 0.5
ARM_NEAR = 10.0           # cm from a fingertip to the object's middle: held or covered, not checked
BIND_RADIUS = 12.0        # cm from Gemini's coordinate to the blob's centroid (its estimates are off by ~6 cm)
GRIPPER_POINT = 6.0       # cm: a coordinate this close to a fingertip names the gripper
CARRY_RADIUS = 8.0        # cm (x/y) from the fingertips of a close that caught something to the object
CODE_THR = 2.34           # shape-change score above which the code calls a fall (J1: 0 false alarms)
CLEF_THR = 0.5

T, A = "top_camera", "agent_camera"
_NUM = r"([+-]?\d+(?:\.\d+)?)"
_POINT = re.compile(r"\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"(?:\s*,\s*" + _NUM + r")?\s*\)")
_APPROX = r"\s*(?:≈|~|=|:|is|of)?\s*(?:approx\.?|about|around)?\s*"
_XY = re.compile(r"\bx" + _APPROX + _NUM + r"\s*(?:cm)?\s*(?:,|and)?\s*y" + _APPROX + _NUM, re.IGNORECASE)

Ask = Callable[[str, dict, list], dict]      # (state, questions, image data URIs) -> answers


def coordinates(text: str) -> list[tuple[float, float]]:
    """Every location in a text, as (x, y) in cm: "(x, y)", "(x, y, z)" and "x ≈ -27, y ≈ +18" forms, in order."""
    text = text or ""
    found = [(m.start(), float(m.group(1)), float(m.group(2))) for m in _POINT.finditer(text)]
    found += [(m.start(), float(m.group(1)), float(m.group(2))) for m in _XY.finditer(text)]
    return [(x, y) for _, x, y in sorted(found)]


def parse(cap, boxes: dict) -> list[Blob]:
    return [b for b in blobs_mod.extract(heightmap.build(cap, boxes)) if b.area >= 6.0 and b.robot_gap >= 0.5]


def half_for(*blobs: Blob) -> float:
    return float(np.clip(max(max(b.length, b.width) for b in blobs) / 2 + 4, 6, 16))


def arm_near(cap, blob: Blob) -> bool:
    mid = (*blob.centroid, cap.table_z + blob.h95 / 2)
    return any(math.dist(st["position_cm"], mid) < ARM_NEAR for st in cap.arms.values())


def holding(cap) -> bool:
    """A gripper is commanded closed but stopped on something (the harness's own reading of its fingers)."""
    return any(st.get("gripper", 1.0) < 0.5 and st.get("gripper_real", 1.0) > 0.02 for st in cap.arms.values())


def reid_request(k: int) -> tuple[str, dict]:
    """Clef's question for k candidate tiles (the 'top' interface)."""
    crit = {f"T{i}": f"tile {i}" for i in range(1, k + 1)}
    crit["none"] = "none of the tiles"
    state = ("A robot works at a table. The REF image shows one object at the start, from straight above. The next "
             f"image shows {k} numbered tiles, the same view now, one tile per thing found near where the object was "
             "last seen. All tiles are at the same scale as REF.")
    q = {"which": {"type": "choice", "criteria": crit,
                   "instructions": "Which tile shows the same physical object as REF? It may have been moved, turned or "
                                   "knocked over since the start, so match its colour, markings and shape."}}
    return state, q


def topple_request(ref: Blob, now: Blob) -> tuple[str, dict]:
    """Clef's question on the start and now crops (the 'agent+num' interface)."""
    state = ("One object on a robot's table, at the start (image 1, START) and now (image 2, NOW), at the same scale."
             f" Measured by depth: its top is {round(ref.h95, 1)} cm above the table at the start and "
             f"{round(now.h95, 1)} cm now; its footprint is {round(ref.length)} x {round(ref.width)} cm at the start "
             f"and {round(now.length)} x {round(now.width)} cm now.")
    q = {"fallen": {"type": "noul", "instructions": "Has the object fallen over or tipped onto its side since the start?"}}
    return state, q


def fuse(probabilities: dict, dists: list[float]) -> tuple[str, float]:
    """Clef's tile probabilities times a distance prior; the pick and its normalised probability."""
    score = {f"T{k + 1}": math.log(max(probabilities.get(f"T{k + 1}", 0), 1e-6)) - d / D_PRIOR for k, d in enumerate(dists)}
    score["none"] = math.log(max(probabilities.get("none", 0), 1e-6)) - CAND_RADIUS / D_PRIOR
    top = max(score.values())
    post = {k: math.exp(v - top) for k, v in score.items()}
    pick = max(post, key=post.get)
    return pick, post[pick] / sum(post.values())


@dataclass
class Tracked:
    at: tuple[float, float]          # Gemini's coordinate
    ref: Blob
    last: Blob
    streak: int = 0
    history: list = field(default_factory=list)
    carried_by: Optional[str] = None  # the arm whose close caught it
    search: Optional[tuple[float, float]] = None   # where to look next instead of `last` (after a release)
    picked: Optional[Blob] = None    # the blob the last check picked (offline labelling)

    def centre(self) -> tuple[float, float]:
        return self.search if self.search is not None else self.last.centroid


class ObjectMonitor:
    def __init__(self, ask: Ask, boxes: dict, *, policy: str = "both", need: int = 2,
                 code_thr: float = CODE_THR, clef_thr: float = CLEF_THR, workers: int = 4):
        if policy not in ("code", "clef", "both"):
            raise ValueError(f"unknown policy: {policy}")
        if ask is None and policy != "code":
            raise ValueError("without Clef (ask=None) only the code policy can run")
        self.ask, self.boxes, self.policy, self.need = ask, boxes, policy, need
        self.code_thr, self.clef_thr, self.workers = code_thr, clef_thr, workers
        self.first = None
        self.objects: list[Tracked] = []
        self.calls = 0

    def bind(self, cap, points: list[tuple[float, float]]) -> list[dict]:
        """Pin each coordinate to the nearest raised blob within BIND_RADIUS of the first capture. Coordinates on
        a gripper's fingertips (the reply also says where the grippers are) are not objects."""
        self.first = cap
        bl = [b for b in parse(cap, self.boxes) if b.h95 >= 1.5]       # flat mats and pads cannot fall over
        taken = set()
        tips = [st["position_cm"][:2] for st in cap.arms.values()]
        for p in points:
            if any(math.dist(p, t) <= GRIPPER_POINT for t in tips):
                continue
            near = [b for b in bl if math.dist(b.centroid, p) <= BIND_RADIUS and b.id not in taken]
            if near:
                b = min(near, key=lambda b: math.dist(b.centroid, p))
                taken.add(b.id)
                self.objects.append(Tracked(at=p, ref=b, last=b))
        return [{"at": o.at, "centroid": [round(v, 1) for v in o.ref.centroid], "h95": round(o.ref.h95, 1)}
                for o in self.objects]

    def note_close(self, arm: str, tcp, caught: bool) -> None:
        """A close of `arm` with the fingertips at `tcp` (cm) ended; `caught`: the fingers stopped on something."""
        if not caught:
            return
        free = [o for o in self.objects if o.carried_by is None and math.dist(o.centre(), tcp[:2]) <= CARRY_RADIUS]
        if free:
            min(free, key=lambda o: math.dist(o.centre(), tcp[:2])).carried_by = arm

    def note_open(self, arm: str, tcp) -> None:
        """`arm` opened with the fingertips at `tcp` (cm): what it carried is now around there."""
        for o in self.objects:
            if o.carried_by == arm:
                o.carried_by, o.search = None, (float(tcp[0]), float(tcp[1]))

    def _check_one(self, cap, bl: list[Blob], obj: Tracked) -> dict:
        row = {"at": obj.at}
        if obj.carried_by is not None:
            return {**row, "status": "carried"}
        if arm_near(cap, obj.last) and obj.search is None:
            return {**row, "status": "arm"}
        centre = obj.centre()
        # a rigid object stands at least as tall as its smallest side, however it lies: lower blobs are not it
        floor = 0.5 * min(obj.ref.length, obj.ref.width, obj.ref.h95)
        cands = sorted((b for b in bl if math.dist(b.centroid, centre) <= CAND_RADIUS and b.h95 >= floor),
                       key=lambda b: math.dist(b.centroid, centre))[:MAX_CANDS]
        if not cands:
            return {**row, "status": "nothing"}
        if self.ask is None:                      # code only (ablation): the nearest tall-enough blob is the object
            ans = {"choice": "T1", "probabilities": {"T1": 1.0}}
        else:
            half = half_for(obj.ref)              # the object's own scale: a big neighbour must not widen the tiles
            height = max(obj.ref.h95, 4.0)
            ref_img = render.crop(self.first, T, obj.ref.centroid, half, height, size=TILE, label="REF")
            tiles = [render.crop(cap, T, b.centroid, half, max(height, b.h95), size=TILE, label=str(k + 1))
                     for k, b in enumerate(cands)]
            state, q = reid_request(len(cands))
            ans = self.ask(state, q, [render.jpeg_uri(ref_img), render.jpeg_uri(render.tile_sheet(tiles, cols=3))])["which"]
            self.calls += 1
        dists = [round(math.dist(b.centroid, centre)) for b in cands]
        pick, p_pick = fuse(ans["probabilities"], dists)
        row.update(status="seen", pick=pick, p_pick=round(p_pick, 3), n_cands=len(cands),
                   clef_choice=ans.get("choice"), clef_p=round(ans["probabilities"].get(ans.get("choice"), 0.0), 4))
        if pick == "none":
            return row
        now = cands[int(pick[1:]) - 1]
        row["_blob"] = now                      # for offline labelling; dropped from the log
        if p_pick >= SURE and ans.get("choice") == pick:     # Clef itself must name the same tile
            obj.last, obj.search = now, None
        row["code"] = round(shape_change(now, obj.ref), 3)
        # only an object standing taller than its narrower side can fall over (a remote lying flat cannot); one
        # that still stands at its full height has not, whatever its footprint does (a neighbour it now touches
        # merges into its blob)
        can_fall = obj.ref.h95 > min(obj.ref.length, obj.ref.width)
        row.update(h_ref=round(obj.ref.h95, 1), h_now=round(now.h95, 1), can_fall=bool(can_fall),
                   upright=bool(can_fall and now.h95 >= 0.9 * obj.ref.h95))
        if self.ask is None:
            return row
        half_t = half_for(obj.ref, now)
        height_t = max(obj.ref.h95, now.h95, 4.0)
        start = render.crop(self.first, A, obj.ref.centroid, half_t, height_t, size=TILE, label="START")
        here = render.crop(cap, A, now.centroid, half_t, height_t, size=TILE, label="NOW")
        state, q = topple_request(obj.ref, now)
        row["clef"] = round(self.ask(state, q, [render.jpeg_uri(start), render.jpeg_uri(here)])["fallen"]["noul"], 4)
        self.calls += 1
        return row

    def flagged(self, row: dict) -> bool:
        if (row.get("status") != "seen" or "code" not in row or row.get("clef_choice") != row.get("pick")
                or not row.get("can_fall") or row.get("upright")):
            return False
        code = row["code"] > self.code_thr
        clef = row.get("clef", 0.0) >= self.clef_thr
        return {"code": code, "clef": clef, "both": code and clef}[self.policy]

    def check(self, cap) -> list[dict]:
        """Track and check every bound object at this capture; returns one row per object."""
        if not self.objects:
            return []
        if holding(cap):
            return [{"at": o.at, "status": "holding", "streak": o.streak} for o in self.objects]
        bl = parse(cap, self.boxes)
        with ThreadPoolExecutor(min(self.workers, len(self.objects))) as pool:
            rows = list(pool.map(lambda o: self._safe_check(cap, bl, o), self.objects))
        for obj, row in zip(self.objects, rows):
            obj.picked = row.pop("_blob", None)
            if row.get("status") == "seen" and "code" in row:      # a check that found the object
                obj.streak = obj.streak + 1 if self.flagged(row) else 0
            row["streak"] = obj.streak
            obj.history.append(row)
        return rows

    def _safe_check(self, cap, bl, obj) -> dict:
        try:
            return self._check_one(cap, bl, obj)
        except Exception as exc:  # noqa: BLE001  (a failed check must not end the episode)
            return {"at": obj.at, "status": "error", "error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    def fallen(self) -> Optional[str]:
        """A description of an object flagged `need` checks in a row, else None."""
        for obj in self.objects:
            if obj.streak >= self.need:
                x, y = obj.at
                return f"the object located at ({x:g}, {y:g}) on turn 1 has fallen over"
        return None
