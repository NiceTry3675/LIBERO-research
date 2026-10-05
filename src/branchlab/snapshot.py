import copy
from dataclasses import dataclass

import numpy as np

# Controller fields that carry over between steps (robosuite 1.4 OSC).
_CONTROLLER_FIELDS = (
    "goal_pos",
    "goal_ori",
    "torques",
    "new_update",
    "initial_joint",
    "initial_ee_pos",
    "initial_ee_ori_mat",
)


@dataclass
class Snapshot:
    sim_state: np.ndarray
    qacc_warmstart: np.ndarray
    ctrl: np.ndarray
    gripper_action: np.ndarray
    controller: dict
    timestep: int
    cur_time: float


def snapshot(env) -> Snapshot:
    """Capture everything needed to replay from this point bit-for-bit.

    The flattened MuJoCo state alone is not enough: the gripper integrates its
    command in Python (`current_action`), so a plain state restore drifts.
    """
    robot = env.robots[0]
    return Snapshot(
        sim_state=env.get_sim_state().copy(),
        qacc_warmstart=env.sim.data.qacc_warmstart.copy(),
        ctrl=env.sim.data.ctrl.copy(),
        gripper_action=np.array(robot.gripper.current_action).copy(),
        controller={
            k: copy.deepcopy(getattr(robot.controller, k)) for k in _CONTROLLER_FIELDS
        },
        timestep=env.env.timestep,
        cur_time=env.env.cur_time,
    )


def restore(env, snap: Snapshot):
    obs = env.set_init_state(snap.sim_state)
    robot = env.robots[0]
    env.sim.data.qacc_warmstart[:] = snap.qacc_warmstart
    env.sim.data.ctrl[:] = snap.ctrl
    robot.gripper.current_action = snap.gripper_action.copy()
    for k, v in snap.controller.items():
        setattr(robot.controller, k, copy.deepcopy(v))
    env.env.timestep = snap.timestep
    env.env.cur_time = snap.cur_time
    return obs
