"""External events: something outside the robot changes the scene."""

import numpy as np


def object_xy(env, name: str) -> np.ndarray:
    e = env.env
    return e.sim.data.body_xpos[e.obj_body_id[name]][:2].copy()


def move_object(env, name: str, xy):
    """Teleport an object to a new floor position, keeping height and orientation."""
    e = env.env
    joint = e.objects_dict[name].joints[0]
    qpos = e.sim.data.get_joint_qpos(joint).copy()
    qpos[:2] = xy
    e.sim.data.set_joint_qpos(joint, qpos)
    e.sim.data.set_joint_qvel(joint, np.zeros(6))
    e.sim.forward()
