from dataclasses import dataclass, field

import numpy as np


@dataclass
class Percept:
    """What the executor is allowed to see at one step.

    Proprioception (`ee_pos`, `gripper_width`) is always present. An object
    missing from `objects` means it was not perceived this step.
    """

    ee_pos: np.ndarray
    gripper_width: float
    objects: dict = field(default_factory=dict)


def observe(env) -> Percept:
    """Ground-truth percept read straight from the simulator."""
    e = env.env
    # After a step MuJoCo's kinematics lag qpos by one substep; without this a
    # freshly restored state would yield a slightly different percept.
    e.sim.forward()
    robot = env.robots[0]
    finger_qpos = e.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]
    return Percept(
        ee_pos=e.sim.data.site_xpos[robot.eef_site_id].copy(),
        gripper_width=float(abs(finger_qpos[0] - finger_qpos[1])),
        objects={
            name: e.sim.data.body_xpos[body_id].copy()
            for name, body_id in e.obj_body_id.items()
        },
    )
