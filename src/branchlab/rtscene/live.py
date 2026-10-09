"""A Capture taken from the running harness environment, for the scene parser and the object monitor.

It holds what scripts/replay_capture.py stores, minus the segmentation: depth of the top and agent cameras at
every second pixel (with the halved intrinsics), world-to-camera extrinsics (OpenCV), the raw camera images,
the table height, both arms' state and the robot's link poses (proprioception).
"""
from __future__ import annotations

import numpy as np

VIEWS = ("top_camera", "agent_camera")
ARM_KEYS = ("position_cm", "wrist_cm", "approach", "finger_axis", "quat_wxyz", "gripper", "gripper_real")


def depth_mm(position) -> np.ndarray:
    """SAPIEN 'Position' picture (OpenGL camera frame) -> depth along the optical axis in mm; 0 where invalid."""
    depth = -position[..., 2]
    valid = position[..., 3] < 1
    return np.where(valid, np.round(np.clip(depth, 0, 60) * 1000.0), 0).astype(np.uint16)


def robot_links(env) -> dict:
    """World poses [x, y, z (m), qw, qx, qy, qz] of every link of the robot."""
    robot = env.env.robot
    links = {}
    for entity in {id(robot.left_entity): robot.left_entity, id(robot.right_entity): robot.right_entity}.values():
        for link in entity.get_links():
            pose = link.get_pose()
            links[link.get_name()] = [*(float(v) for v in pose.p), *(float(v) for v in pose.q)]
    return links


class LiveCapture:
    def __init__(self, env, obs: dict, turn: int):
        self.turn = turn
        state = obs["state"]
        self.table_z = float(state["table_z_cm"])
        self.arms = {a: {k: state[a][k] for k in ARM_KEYS if k in state[a]} for a in ("left", "right")}
        self.links = robot_links(env)
        self._rgb = {v: np.asarray(obs["images"][v]) for v in VIEWS}
        self._depth, self._K, self._E = {}, {}, {}
        params = obs["camera_params"]
        for v in VIEWS:
            cam = env._extra_cameras[v]
            self._depth[v] = depth_mm(cam.get_picture("Position"))[::2, ::2]
            K = np.asarray(params[v]["intrinsic_cv"], dtype=np.float64).copy()
            K[:2] /= 2
            E = np.eye(4)
            E[:3] = np.asarray(params[v]["extrinsic_cv"], dtype=np.float64)[:3]
            self._K[v], self._E[v] = K, E

    def depth(self, cam: str) -> np.ndarray:
        return self._depth[cam]

    def K(self, cam: str) -> np.ndarray:
        return self._K[cam]

    def E(self, cam: str) -> np.ndarray:
        return self._E[cam]

    def rgb(self, cam: str, raw: bool = True) -> np.ndarray:
        return self._rgb[cam]
