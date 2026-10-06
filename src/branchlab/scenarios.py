"""Disturbances that all show up the same way in the fact list -- the target
is no longer detected -- but have different hidden causes.

Every scenario starts from the same layout: an occluder standing between the
target and the camera, just far enough away that the target is still seen.
Partway through the reach, one thing happens:

    recognition_miss  nothing moves; the detector stops reporting the target,
                      for one look (persistence 0) or for good (persistence 1)
    occlusion         the occluder is moved in front of the target
    nudge             the target is moved behind the occluder
    relocate          the target is moved behind the destination container
    none              nothing happens (control)

and one that happens later, as the arm starts carrying the target:

    held_miss         the detector stops reporting the target in the gripper
                      (the hand covers it); nothing is wrong with the task
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from branchlab.events import move_object, object_xy
from branchlab.executor import Phase
from branchlab.perception import Fault, PerceptionModel
from branchlab.visibility import visible_fraction

# The task every scenario is built on: "pick up the bbq sauce and place it in
# the basket". The bottle is narrow enough that a grasp tolerates about 2 cm of
# position error, so failures come from wrong beliefs rather than grasp luck.
SUITE, TASK_ID = "libero_object", 3
TARGET, OCCLUDER, DEST = "bbq_sauce_1", "ketchup_1", "basket_1"

CAUSES = ("none", "recognition_miss", "occlusion", "nudge", "relocate", "held_miss")


@dataclass
class Scenario:
    cause: str
    magnitude: float = 0.06  # m moved by the occluder (occlusion) or the target (nudge)
    event_step: int = 12  # control step of the event; the reach takes about 25
    persistence: float = 1.0  # recognition_miss: chance the miss survives each further look
    hidden_gap: float = 0.08  # occluder-to-target distance at which the target is hidden
    relocate_gap: float = 0.14  # distance behind the destination for relocate
    clear_radius: float = 0.15  # other objects are kept this far from the target's path
    target: str = TARGET
    occluder: str = OCCLUDER
    dest: str = DEST
    camera: str = "agentview"


def _toward_camera(env, xy, camera):
    """Unit floor vector from a point toward the camera."""
    model, data = env.env.sim.model._model, env.env.sim.data._data
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    v = data.cam_xpos[cam_id][:2] - xy
    return v / np.linalg.norm(v)


def setup_layout(env, sc: Scenario):
    """Place the occluder in front of the target, `magnitude` short of hiding it,
    and move any other object out of the way.

    Objects other than target, occluder and destination that stand within
    `clear_radius` of the segment from the target to the occluder are pushed
    straight away from it. Otherwise a tall bottle next to the spot the target
    is nudged to blocks the hand, and grasps there fail for reasons unrelated
    to what the robot knows.
    """
    target_xy = object_xy(env, sc.target)
    u = _toward_camera(env, target_xy, sc.camera)
    occluder_xy = target_xy + (sc.hidden_gap + sc.magnitude) * u
    move_object(env, sc.occluder, occluder_xy)
    for name in env.env.obj_body_id:
        if name in (sc.target, sc.occluder, sc.dest):
            continue
        xy = object_xy(env, name)
        along = np.clip(np.dot(xy - target_xy, u), 0.0, sc.hidden_gap + sc.magnitude)
        closest = target_xy + along * u
        away = xy - closest
        dist = np.linalg.norm(away)
        if dist < sc.clear_radius:
            move_object(env, name, closest + away / dist * sc.clear_radius)


def due(sc: Scenario, step: int, executor) -> bool:
    """Is this the step at which the scenario's event happens?"""
    if sc.cause == "held_miss":
        return executor.state.phase == Phase.TRANSPORT and executor.state.phase_steps == 0
    return step == sc.event_step


def trigger(env, perception: PerceptionModel, sc: Scenario) -> dict:
    """Apply the scenario's event. Returns the hidden cause, for analysis only."""
    target_xy = object_xy(env, sc.target)
    record = {"cause": sc.cause, "moved": None, "distance": 0.0}

    def move(name, new_xy):
        record.update(moved=name, distance=float(np.linalg.norm(new_xy - object_xy(env, name))))
        move_object(env, name, new_xy)

    if sc.cause == "recognition_miss":
        perception.faults.append(Fault("miss", sc.target, persistence=sc.persistence))
    elif sc.cause == "held_miss":
        perception.faults.append(Fault("miss", sc.target, persistence=1.0))
    elif sc.cause == "occlusion":
        u = _toward_camera(env, target_xy, sc.camera)
        move(sc.occluder, target_xy + sc.hidden_gap * u)
    elif sc.cause == "nudge":
        u = _toward_camera(env, target_xy, sc.camera)
        move(sc.target, target_xy + sc.magnitude * u)
    elif sc.cause == "relocate":
        dest_xy = object_xy(env, sc.dest)
        move(sc.target, dest_xy - sc.relocate_gap * _toward_camera(env, dest_xy, sc.camera))
    elif sc.cause != "none":
        raise ValueError(f"unknown cause: {sc.cause}")

    record["target_visible"] = visible_fraction(env, sc.target, sc.camera)
    return record
