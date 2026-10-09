"""Load replayed episodes (scripts/replay_capture.py output) as run-time captures.

Capture exposes only run-time data: depth, camera calibration, raw RGB, the robot's own body and the table
height. Object poses, segmentation and the success flag stay in Episode.meta and are read by labels.py only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

CAMERAS = ("top_camera", "agent_camera", "left_camera", "right_camera")
ARM_KEYS = ("position_cm", "wrist_cm", "approach", "finger_axis", "quat_wxyz", "gripper", "gripper_real")


@dataclass
class Capture:
    episode: "Episode"
    idx: int
    kind: str                 # turn | preclose | preopen | final
    turn: int
    cmd_index: int | None
    command: str | None
    table_z: float            # cm
    arms: dict                # arm -> {position_cm, wrist_cm, approach, finger_axis, quat_wxyz, gripper, gripper_real}
    links: dict               # link name -> [x, y, z (m), qw, qx, qy, qz]
    extra: dict = field(default_factory=dict)

    def depth(self, cam: str) -> np.ndarray:
        return self.episode.arrays[f"{cam}_depth"][self.idx]

    def K(self, cam: str) -> np.ndarray:
        return self.episode.arrays[f"{cam}_K"][self.idx]

    def E(self, cam: str) -> np.ndarray:
        return self.episode.arrays[f"{cam}_extrinsic"][self.idx]

    def rgb(self, cam: str, raw: bool = True) -> np.ndarray:
        from PIL import Image
        name = f"c{self.idx:03d}_{cam}{'_raw' if raw else ''}.jpg"
        return np.asarray(Image.open(self.episode.dir/"images"/name).convert("RGB"))


class Episode:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir/"meta.json").read_text())

    @cached_property
    def arrays(self):
        data = np.load(self.dir/"captures.npz")
        return {k: data[k] for k in data.files if not k.endswith("_seg")}

    @property
    def key(self) -> str:
        return self.meta["key"]

    @property
    def task(self) -> str:
        return self.meta["task"]

    @property
    def instruction(self) -> str:
        return self.meta["instruction"]

    def captures(self) -> list[Capture]:
        out = []
        for c in self.meta["captures"]:
            st = c["state"]
            arms = {a: {k: st[a][k] for k in ARM_KEYS if k in st[a]} for a in ("left", "right")}
            robot = c.get("robot") or {}
            links = (robot.get("left") or {}).get("links", {})
            out.append(Capture(self, c["idx"], c["kind"], c["turn"], c.get("cmd_index"), c.get("command"),
                               float(st["table_z_cm"]), arms, links))
        return out


def episodes(root: Path) -> list[Episode]:
    return [Episode(p.parent) for p in sorted(Path(root).glob("*/*/episode_*/meta.json"))]
