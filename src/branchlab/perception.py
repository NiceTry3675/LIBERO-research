import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from branchlab.visibility import visible_fraction


@dataclass
class Percept:
    """What the executor is allowed to see at one step.

    Proprioception (`ee_pos`, `gripper_width`) is always present. `objects` is
    None on steps without a look; after a look it maps each detected object to
    its reported position, and an object absent from it was not detected.
    """

    ee_pos: np.ndarray
    gripper_width: float
    objects: Optional[dict] = None


def _proprio(env):
    e = env.env
    # After a step MuJoCo's kinematics lag qpos by one substep; without this a
    # freshly restored state would yield a slightly different percept.
    e.sim.forward()
    robot = env.robots[0]
    finger_qpos = e.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]
    ee_pos = e.sim.data.site_xpos[robot.eef_site_id].copy()
    return ee_pos, float(abs(finger_qpos[0] - finger_qpos[1]))


def _true_objects(env):
    e = env.env
    return {name: e.sim.data.body_xpos[i].copy() for name, i in e.obj_body_id.items()}


def observe(env) -> Percept:
    """Ground-truth look, read straight from the simulator."""
    ee_pos, width = _proprio(env)
    return Percept(ee_pos, width, _true_objects(env))


@dataclass
class PerceptionConfig:
    pos_noise: float = 0.0  # std (m) of per-look xy jitter on each detection
    miss_prob: float = 0.0  # per-look chance of dropping each object
    camera: str = "agentview"  # objects are detected only if this camera sees them
    min_visible: float = 0.2  # fraction of the object's facing surface that must be in sight


@dataclass
class Fault:
    """An injected recognition error that can outlast a single look.

    kind "miss":   `obj` is not reported.
    kind "offset": `obj` is reported at its true position plus `vector`.
    kind "ghost":  `obj` is reported at the absolute position `vector`,
                   whether or not such an object exists.
    After each look the fault survives with probability `persistence`
    (1 = permanent, 0 = affects exactly one look).
    """

    kind: str
    obj: str
    vector: Optional[np.ndarray] = None
    persistence: float = 1.0


class PerceptionModel:
    """Turns simulator state into noisy, fallible object detections."""

    def __init__(self, config: PerceptionConfig = None, seed: int = 0):
        self.config = config or PerceptionConfig()
        self.faults: list = []
        self.rng = np.random.default_rng(seed)

    def get_state(self):
        return copy.deepcopy(self.faults), copy.deepcopy(self.rng.bit_generator.state)

    def set_state(self, saved):
        faults, rng_state = saved
        self.faults = copy.deepcopy(faults)
        self.rng.bit_generator.state = copy.deepcopy(rng_state)

    def proprio(self, env) -> Percept:
        return Percept(*_proprio(env))

    def look(self, env) -> Percept:
        cfg = self.config
        ee_pos, width = _proprio(env)
        truth = _true_objects(env)

        detections = {}
        for name, pos in truth.items():
            missed = self.rng.random() < cfg.miss_prob
            jitter = self.rng.normal(0.0, cfg.pos_noise, 2)
            if not missed and visible_fraction(env, name, cfg.camera) >= cfg.min_visible:
                detections[name] = pos + np.array([jitter[0], jitter[1], 0.0])

        for fault in self.faults:
            if fault.kind == "miss":
                detections.pop(fault.obj, None)
            elif fault.kind == "offset":
                detections[fault.obj] = truth[fault.obj] + fault.vector
            elif fault.kind == "ghost":
                detections[fault.obj] = np.array(fault.vector, dtype=float)
            else:
                raise ValueError(f"unknown fault kind: {fault.kind}")
        self.faults = [f for f in self.faults if self.rng.random() < f.persistence]

        return Percept(ee_pos, width, detections)


@dataclass
class Track:
    pos: np.ndarray  # most recent reported position
    seen_step: int  # belief step of that detection
    prev_pos: Optional[np.ndarray] = None  # reported position at the detection before


@dataclass
class Belief:
    """What the agent currently believes about objects, built only from looks."""

    step: int = 0
    look_step: Optional[int] = None  # step of the most recent look
    detected: set = field(default_factory=set)  # objects reported in that look
    tracks: dict = field(default_factory=dict)

    def update(self, percept: Percept):
        self.step += 1
        if percept.objects is None:
            return
        self.look_step = self.step
        self.detected = set(percept.objects)
        for name, pos in percept.objects.items():
            old = self.tracks.get(name)
            self.tracks[name] = Track(pos.copy(), self.step, old.pos if old else None)

    def pos(self, name):
        track = self.tracks.get(name)
        return None if track is None else track.pos
