import copy
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from branchlab.perception import Percept

OPEN, CLOSE = -1.0, 1.0


class Phase(str, Enum):
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


@dataclass
class ExecutorState:
    phase: Phase = Phase.REACH
    phase_steps: int = 0
    retries: int = 0
    last_seen: dict = field(default_factory=dict)  # object name -> last perceived position


class PickPlaceExecutor:
    """Scripted pick-and-place that acts only on a Percept.

    When an object is not perceived it keeps using its last perceived
    position, so "continue" means carrying on with a possibly stale belief.
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

    # --- control ----------------------------------------------------------
    @property
    def finished(self):
        return self.state.phase in (Phase.DONE, Phase.FAILED)

    def act(self, percept: Percept) -> np.ndarray:
        cfg, s = self.config, self.state
        s.last_seen.update(percept.objects)
        s.phase_steps += 1

        obj = s.last_seen.get(self.obj)
        dest = s.last_seen.get(self.dest)
        ee = percept.ee_pos

        if s.phase == Phase.REACH:
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
            return self._move(ee, target, CLOSE)

        if s.phase == Phase.TRANSPORT:
            if percept.gripper_width <= cfg.held_min_width:  # dropped on the way
                self._retry()
                return self._hold(OPEN)
            target = np.array([dest[0], dest[1], cfg.hover_z])
            if self._reached(ee, target):
                self._enter(Phase.RELEASE)
            return self._move(ee, target, CLOSE)

        if s.phase == Phase.RELEASE:
            if s.phase_steps >= cfg.release_steps:
                self._enter(Phase.DONE)
            return self._hold(OPEN)

        return self._hold(OPEN)

    # --- helpers ----------------------------------------------------------
    def _enter(self, phase: Phase):
        self.state.phase = phase
        self.state.phase_steps = 0

    def _retry(self):
        self.state.retries += 1
        self._enter(Phase.FAILED if self.state.retries > self.config.max_retries else Phase.REACH)

    def _reached(self, ee, target):
        if self.state.phase_steps > self.config.phase_timeout:
            self._retry()
            return False
        return np.linalg.norm(target - ee) < self.config.pos_tol

    def _move(self, ee, target, grip):
        delta = np.clip(self.config.gain * (target - ee), -1, 1)
        if self.config.action_noise > 0:
            delta = np.clip(delta + self.rng.normal(0, self.config.action_noise, 3), -1, 1)
        return np.concatenate([delta, np.zeros(3), [grip]])

    def _hold(self, grip):
        return np.concatenate([np.zeros(6), [grip]])


def run_episode(env, executor: PickPlaceExecutor, observe, max_steps: int = 600):
    """Run until the executor finishes, the task succeeds, or max_steps."""
    for step in range(max_steps):
        if executor.finished:
            break
        env.step(executor.act(observe(env)))
    return bool(env.check_success()), step + 1
