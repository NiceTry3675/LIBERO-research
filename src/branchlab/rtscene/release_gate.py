"""Release gate: right before a gripper lets go of a shoe, Clef checks which way the shoe's toe points.

place_shoe and place_dual_shoes count a shoe as placed only with its toe at -x, which is left in the top camera
(place_dual_shoes says "tips left"; place_shoe only checks it, RoboTwin's envs/place_shoe.py target pose). Gemini
releases shoes without checking the toe, and depth gives a shoe's long axis but not which end is the toe
(reports/앞코_경고_라이브AB_설계.md). The harness calls the gate before an open of an arm that holds something:

  clef   the raw top camera image, cropped to a square around a box of +-15 cm in x/y and [z-3, z+12] cm about the
         opening arm's fingertips, scaled to 256 px; Clef picks the toe's direction in the picture (left, right,
         up, down). P(not left) >= threshold vetoes the open.
  nudge  no image and no Clef: the open is vetoed with a generic reminder (the control for the reminder alone).

The harness reports a veto to Gemini as the open's result (NOT executed, and why) and vetoes at most once per hold.
The crop is scripts/clef_semantic_offline.py's (its crop() calls crop_box here), so the offline numbers carry
over. Nothing here reads the simulator's truth: the image, the camera matrices and the fingertip position
(proprioception) are what the harness has at run time.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from PIL import Image

from .render import jpeg_uri

TASKS = ("place_dual_shoes", "place_shoe")
GOAL = "left"              # the toe at -x, for both tasks
CAMERA = "top_camera"
HALF_CM = 15.0             # the box about the fingertips: +-HALF_CM in x and y, [z - DROP_CM, z + RISE_CM]
DROP_CM, RISE_CM = 3.0, 12.0
TILE = 256
THRESHOLD = 0.8            # P(not left) that vetoes; fixed before the runs with the crop size
MODES = ("clef", "nudge")

STATE = ("A robot is moving shoes on a table. The first picture looks straight down at the shoe in its centre; the "
         "robot stands at the bottom edge of the picture. The toe is the closed, rounded front end of the shoe; the "
         "heel is the back end, next to the opening for the foot.")
TOE_OPTIONS = {"left": "the toe points toward the left edge of the picture",
               "right": "the toe points toward the right edge of the picture",
               "up": "the toe points toward the top edge of the picture",
               "down": "the toe points toward the bottom edge of the picture"}
QUESTIONS = {"toe": {"type": "choice",
                     "instructions": "Which way does the toe of the shoe held in the centre point, in the picture?",
                     "criteria": TOE_OPTIONS}}
# how the veto names the direction Clef saw (the top camera's bottom edge is the robot's side)
SEEN = {"right": "RIGHT", "down": "toward the robot", "up": "away from the robot"}
CLEF_NOTE = ('"{arm} gripper open" NOT executed (release check): the fast checker sees the toe of the held shoe '
             "pointing {seen} in the top camera, not left as the task needs. Turn the shoe before releasing, or send "
             "the open again to release anyway.")
NUDGE_NOTE = ('"{arm} gripper open" NOT executed (release check): Before releasing, check in the images that the '
              "shoe's toe points the way the task asks; send the open again to release.")

Ask = Callable[[str, dict, list], dict]      # (state, questions, image data URIs) -> answers


def projector(params: dict) -> Callable:
    """World points in cm -> pixels of a camera, from its intrinsic_cv and extrinsic_cv (world -> OpenCV camera,
    3x4 or 4x4): the harness's env.project_points, in cm."""
    K = np.asarray(params["intrinsic_cv"], dtype=float)
    E = np.asarray(params["extrinsic_cv"], dtype=float)[:3]

    def project(points_cm) -> np.ndarray:
        cam = np.asarray(points_cm, dtype=float).reshape(-1, 3) / 100.0 @ E[:, :3].T + E[:, 3]
        return ((cam / np.clip(cam[:, 2:3], 1e-6, None)) @ K.T)[:, :2]
    return project


def crop_box(project: Callable, centre_cm, half_cm: float = HALF_CM, rise_cm: float = RISE_CM) -> tuple:
    """The square (left, top, right, bottom) in pixels around the projected box of +-half_cm in x/y and
    [z - DROP_CM, z + rise_cm] about centre_cm; at least 24 px wide."""
    c = np.asarray(centre_cm, dtype=float)
    corners = [c + [dx, dy, dz] for dx in (-half_cm, half_cm) for dy in (-half_cm, half_cm) for dz in (-DROP_CM, rise_cm)]
    uv = project(corners)
    u0, v0 = uv.min(0)
    u1, v1 = uv.max(0)
    side = max(u1 - u0, v1 - v0, 24.0)
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    return tuple(int(round(v)) for v in (cu - side / 2, cv - side / 2, cu + side / 2, cv + side / 2))


def crop(img: Image.Image, project: Callable, centre_cm, half_cm: float = HALF_CM, rise_cm: float = RISE_CM) -> Image.Image:
    """The crop_box of an image scaled to TILE (outside the image: black)."""
    return img.crop(crop_box(project, centre_cm, half_cm, rise_cm)).resize((TILE, TILE), Image.BICUBIC)


class ReleaseGate:
    """Decides on one open; the harness decides when to ask (an arm that holds something, once per hold)."""

    def __init__(self, mode: str, ask: Optional[Ask] = None, threshold: float = THRESHOLD):
        if mode not in MODES:
            raise ValueError(f"unknown release gate mode {mode!r}")
        if mode == "clef" and ask is None:
            raise ValueError("the clef release gate needs a way to ask Clef")
        self.mode, self.ask, self.threshold = mode, ask, threshold

    @staticmethod
    def applies(task: Optional[str]) -> bool:
        return task in TASKS

    def check(self, arm: str, obs: Optional[dict] = None) -> tuple[dict, Optional[Image.Image]]:
        """({mode, arm, threshold, probabilities, p_not_left, seen, vetoed, note, error, seconds}, the crop or None).

        Never raises: a missing image or camera, a crop that fails or a Clef call that fails is an error, and the
        open goes ahead."""
        t0 = time.time()
        entry = {"mode": self.mode, "arm": arm, "threshold": self.threshold if self.mode == "clef" else None,
                 "probabilities": None, "p_not_left": None, "seen": None, "vetoed": False, "note": None, "error": None}
        tile = None
        if self.mode == "nudge":
            entry.update(vetoed=True, note=NUDGE_NOTE.format(arm=arm))
        else:
            try:
                image = Image.fromarray(np.asarray(obs["images"][CAMERA], dtype=np.uint8)).convert("RGB")
                tile = crop(image, projector(obs["camera_params"][CAMERA]), obs["state"][arm]["position_cm"])
                answers = self.ask(STATE, QUESTIONS, [jpeg_uri(tile)])
                probs = {k: round(float(p), 4) for k, p in answers["toe"]["probabilities"].items()}
                if GOAL not in probs:
                    raise ValueError(f"no probability for {GOAL!r} in {sorted(probs)}")
                p_not = round(1.0 - probs[GOAL], 4)
                seen = max((k for k in probs if k != GOAL), key=probs.get, default=None)
                vetoed = p_not >= self.threshold
                if vetoed and seen not in SEEN:
                    raise ValueError(f"unknown direction {seen!r}")
                entry.update(probabilities=probs, p_not_left=p_not, seen=seen, vetoed=vetoed,
                             note=CLEF_NOTE.format(arm=arm, seen=SEEN[seen]) if vetoed else None)
            except Exception as exc:  # noqa: BLE001  (the gate must never end the episode)
                entry.update(vetoed=False, note=None, error=f"{type(exc).__name__}: {str(exc)[:200]}")
        entry["seconds"] = round(time.time() - t0, 3)
        return entry, tile
