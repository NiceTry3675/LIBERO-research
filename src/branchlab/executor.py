import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from branchlab.perception import Belief, Percept

OPEN, CLOSE = -1.0, 1.0


class Phase(str, Enum):
    SEARCH = "search"  # visit waypoints and look down with the wrist camera
    REACH = "reach"  # move above the object
    DESCEND = "descend"
    GRASP = "grasp"  # close the gripper
    LIFT = "lift"
    TRANSPORT = "transport"  # move above the destination
    RELEASE = "release"
    DONE = "done"
    FAILED = "failed"  # gave up after max_retries


@dataclass
class ExecutorConfig:
    hover_z: float = 0.32  # travel height of the end effector
    grasp_dz: float = 0.02  # end-effector height relative to the object origin when grasping
    pos_tol: float = 0.01
    gain: float = 10.0  # normalized action per metre of error
    grasp_steps: int = 15
    release_steps: int = 15
    held_min_width: float = 0.01  # fingers wider than this after closing = holding something
    phase_timeout: int = 150
    max_retries: int = 2
    action_noise: float = 0.0  # std of Gaussian noise on the translation command
    wrist_camera: str = "robot0_eye_in_hand"
    wrist_look_steps: int = 10  # time to take one deliberate wrist-camera look


@dataclass
class ExecutorState:
    phase: Phase = Phase.REACH
    phase_steps: int = 0  # 0 means the phase was just entered
    retries: int = 0
    belief: Belief = field(default_factory=Belief)
    look_camera: Optional[str] = None  # request for a look from this camera at the next step
    waypoints: list = field(default_factory=list)  # search: floor points still to visit
    search_stage: str = "move"  # search: "move" -> "wait" -> "looked"
    search_steps: int = 0  # search: steps spent in the current stage


class PickPlaceExecutor:
    """Scripted pick-and-place that acts only on Percepts.

    Object positions come from the belief, which changes only when a percept
    carries a look. Between looks, and when an object goes undetected, the
    executor keeps using the last reported position, so "continue" means
    carrying on with a possibly stale or wrong belief.
    """

    def __init__(self, obj: str, dest: str, config: ExecutorConfig = None, seed: int = 0):
        self.obj = obj
        self.dest = dest
        self.config = config or ExecutorConfig()
        self.state = ExecutorState()
        self.rng = np.random.default_rng(seed)

    # --- snapshot support -------------------------------------------------
    def get_state(self):
        return copy.deepcopy(self.state), copy.deepcopy(self.rng.bit_generator.state)

    def set_state(self, saved):
        state, rng_state = saved
        self.state = copy.deepcopy(state)
        self.rng.bit_generator.state = copy.deepcopy(rng_state)

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    # --- control ----------------------------------------------------------
    @property
    def finished(self):
        return self.state.phase in (Phase.DONE, Phase.FAILED)

    def start_search(self, waypoints):
        """Abandon the current step and look for the object at each waypoint in turn."""
        s = self.state
        s.waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        s.search_stage, s.search_steps = "move", 0
        self._enter(Phase.SEARCH)

    def hold_action(self) -> np.ndarray:
        """Stay still, keeping hold of whatever is in the gripper."""
        closed = self.state.phase in (Phase.GRASP, Phase.LIFT, Phase.TRANSPORT)
        return self._hold(CLOSE if closed else OPEN)

    def act(self, percept: Percept) -> np.ndarray:
        self.perceive(percept)
        return self.command(percept)

    def perceive(self, percept: Percept):
        self.state.belief.update(percept)

    def command(self, percept: Percept) -> np.ndarray:
        cfg, s = self.config, self.state
        s.phase_steps += 1

        obj = s.belief.pos(self.obj)
        dest = s.belief.pos(self.dest)
        ee = percept.ee_pos

        if s.phase == Phase.SEARCH:
            return self._search(ee)

        if s.phase == Phase.REACH:
            if obj is None:
                return self._wait(OPEN)
            target = np.array([obj[0], obj[1], cfg.hover_z])
            if self._reached(ee, target):
                self._enter(Phase.DESCEND)
            return self._move(ee, target, OPEN)

        if s.phase == Phase.DESCEND:
            target = np.array([obj[0], obj[1], obj[2] + cfg.grasp_dz])
            if self._reached(ee, target):
                self._enter(Phase.GRASP)
            return self._move(ee, target, OPEN)

        if s.phase == Phase.GRASP:
            if s.phase_steps >= cfg.grasp_steps:
                self._enter(Phase.LIFT)
            return self._hold(CLOSE)

        if s.phase == Phase.LIFT:
            target = np.array([ee[0], ee[1], cfg.hover_z])
            if abs(ee[2] - cfg.hover_z) < cfg.pos_tol:
                if percept.gripper_width > cfg.held_min_width:
                    self._enter(Phase.TRANSPORT)
                else:
                    self._retry()
            else:
                self._timed_out()
            return self._move(ee, target, CLOSE)

        if s.phase == Phase.TRANSPORT:
            if percept.gripper_width <= cfg.held_min_width:  # dropped on the way
                self._retry()
                return self._hold(OPEN)
            if dest is None:
                return self._wait(CLOSE)
            target = np.array([dest[0], dest[1], cfg.hover_z])
            if self._reached(ee, target):
                self._enter(Phase.RELEASE)
            return self._move(ee, target, CLOSE)

        if s.phase == Phase.RELEASE:
            if s.phase_steps >= cfg.release_steps:
                self._enter(Phase.DONE)
            return self._hold(OPEN)

        return self._hold(OPEN)

    def _search(self, ee):
        cfg, s = self.config, self.state
        s.search_steps += 1
        if s.search_stage == "looked":  # the wrist look requested last step is now in the belief
            track = s.belief.tracks.get(self.obj)
            if track is not None and track.seen_step == s.belief.step:
                self._enter(Phase.REACH)
            else:
                s.waypoints.pop(0)
                s.search_stage, s.search_steps = "move", 0
                if not s.waypoints:
                    self._enter(Phase.FAILED)
        elif s.search_stage == "wait":
            if s.search_steps >= cfg.wrist_look_steps:
                s.look_camera = cfg.wrist_camera
                s.search_stage = "looked"
        else:
            target = np.array([s.waypoints[0][0], s.waypoints[0][1], cfg.hover_z])
            arrived = np.linalg.norm(target - ee) < cfg.pos_tol
            if arrived or s.search_steps > cfg.phase_timeout:
                s.search_stage, s.search_steps = "wait", 0
            else:
                return self._move(ee, target, OPEN)
        return self._hold(OPEN)

    # --- helpers ----------------------------------------------------------
    def _enter(self, phase: Phase):
        self.state.phase = phase
        self.state.phase_steps = 0

    def _retry(self):
        self.state.retries += 1
        self._enter(Phase.FAILED if self.state.retries > self.config.max_retries else Phase.REACH)

    def _timed_out(self):
        if self.state.phase_steps > self.config.phase_timeout:
            self._retry()
            return True
        return False

    def _reached(self, ee, target):
        return not self._timed_out() and np.linalg.norm(target - ee) < self.config.pos_tol

    def _wait(self, grip):
        """Hold still because a needed object has never been seen."""
        self._timed_out()
        return self._hold(grip)

    def _move(self, ee, target, grip):
        delta = np.clip(self.config.gain * (target - ee), -1, 1)
        if self.config.action_noise > 0:
            delta = np.clip(delta + self.rng.normal(0, self.config.action_noise, 3), -1, 1)
        return np.concatenate([delta, np.zeros(3), [grip]])

    def _hold(self, grip):
        return np.concatenate([np.zeros(6), [grip]])
