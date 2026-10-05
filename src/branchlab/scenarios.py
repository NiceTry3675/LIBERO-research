"""Disturbances that all show up the same way in the fact list -- the target
is no longer detected -- but have different hidden causes.

Every scenario starts from the same layout: an occluder standing between the
target and the camera, just far enough away that the target is still seen.
Partway through the reach, one thing happens:

    recognition_miss  nothing moves; the detector stops reporting the target
    occlusion         the occluder is moved in front of the target
    nudge             the target is moved behind the occluder
    relocate          the target is moved behind the destination container
    none              nothing happens (control)
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from branchlab.events import move_object, object_xy
from branchlab.perception import Fault, PerceptionModel
from branchlab.visibility import visible_fraction

CAUSES = ("none", "recognition_miss", "occlusion", "nudge", "relocate")


@dataclass
class Scenario:
    cause: str
    magnitude: float = 0.06  # m moved by the occluder (occlusion) or the target (nudge)
    event_step: int = 12  # control step of the event; the reach takes about 25
    persistence: float = 1.0  # recognition_miss: chance the miss survives each further look
    hidden_gap: float = 0.10  # occluder-to-target distance at which the target is hidden
    relocate_gap: float = 0.14  # distance behind the destination for relocate
    target: str = "alphabet_soup_1"
    occluder: str = "milk_1"
    dest: str = "basket_1"
    camera: str = "agentview"


def _toward_camera(env, xy, camera):
    """Unit floor vector from a point toward the camera."""
    model, data = env.env.sim.model._model, env.env.sim.data._data
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    v = data.cam_xpos[cam_id][:2] - xy
    return v / np.linalg.norm(v)


def setup_layout(env, sc: Scenario):
    """Place the occluder in front of the target, `magnitude` short of hiding it."""
    target_xy = object_xy(env, sc.target)
    u = _toward_camera(env, target_xy, sc.camera)
    move_object(env, sc.occluder, target_xy + (sc.hidden_gap + sc.magnitude) * u)


def trigger(env, perception: PerceptionModel, sc: Scenario) -> dict:
    """Apply the scenario's event. Returns the hidden cause, for analysis only."""
    target_xy = object_xy(env, sc.target)
    record = {"cause": sc.cause, "moved": None, "distance": 0.0}

    def move(name, new_xy):
        record.update(moved=name, distance=float(np.linalg.norm(new_xy - object_xy(env, name))))
        move_object(env, name, new_xy)

    if sc.cause == "recognition_miss":
        perception.faults.append(Fault("miss", sc.target, persistence=sc.persistence))
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
