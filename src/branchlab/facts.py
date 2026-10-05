"""Named facts describing the situation, computed only from what the agent
has perceived (belief, proprioception, executor bookkeeping). This text is what
decision models and the planner get to read; numbers are bucketed in code.
"""

from dataclasses import dataclass

import numpy as np

from branchlab.executor import Phase, PickPlaceExecutor
from branchlab.perception import Percept

CONTROL_HZ = 20


@dataclass
class FactConfig:
    moved_small: float = 0.02  # m between two looks; below this counts as not moved
    moved_large: float = 0.08
    near_radius: float = 0.12
    dest_radius: float = 0.07  # footprint of the destination container
    recent_seconds: float = 3.0
    stall_steps: int = 80  # steps in one phase before it counts as stalled
    open_width: float = 0.07


def _region(pos):
    """Coarse location named as it appears in the agentview image."""
    x, y = pos[0], pos[1]
    row = "back" if x < -0.1 else "front" if x > 0.1 else "middle"
    col = "left" if y < -0.1 else "right" if y > 0.1 else "center"
    return f"{row}_{col}"


def _xy_dist(a, b):
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


def build_facts(executor: PickPlaceExecutor, percept: Percept, config: FactConfig = None) -> dict:
    cfg = config or FactConfig()
    s = executor.state
    belief = s.belief
    target, dest = belief.tracks.get(executor.obj), belief.tracks.get(executor.dest)
    just_looked = belief.look_step == belief.step

    def last_seen(track):
        if track is None:
            return "never"
        if just_looked and track.seen_step == belief.step:
            return "now"
        seconds = (belief.step - track.seen_step) / CONTROL_HZ
        return "recent" if seconds <= cfg.recent_seconds else "stale"

    def moved(track):
        if track is None or track.prev_pos is None:
            return "unknown"
        if track is target and carrying:
            return "carried"
        d = _xy_dist(track.pos, track.prev_pos)
        return "none" if d < cfg.moved_small else "small" if d < cfg.moved_large else "large"

    def just_moved(name):
        track = belief.tracks[name]
        return track.seen_step == belief.step and moved(track) in ("small", "large")

    if percept.gripper_width > cfg.open_width:
        gripper = "open"
    elif percept.gripper_width > executor.config.held_min_width:
        gripper = "holding"
    else:
        gripper = "closed_empty"

    carrying = gripper == "holding" and s.phase in (Phase.LIFT, Phase.TRANSPORT, Phase.RELEASE)
    target_detected = executor.obj in belief.detected
    others = {
        name: t.pos
        for name, t in belief.tracks.items()
        if name not in (executor.obj, executor.dest) and name in belief.detected
    }
    near_target = sorted(
        n for n, p in others.items()
        if target is not None and _xy_dist(p, target.pos) < cfg.near_radius
    )
    in_dest = sorted(
        n for n, p in others.items()
        if dest is not None and _xy_dist(p, dest.pos) < cfg.dest_radius
    )
    moved_others = sorted(n for n in others if just_moved(n))
    target_in_dest = (
        target is not None and dest is not None and gripper != "holding"
        and _xy_dist(target.pos, dest.pos) < cfg.dest_radius
    )

    # Disagreements between what the current plan step expects and the last look
    mismatch = []
    before_grasp = s.phase in (Phase.REACH, Phase.DESCEND, Phase.GRASP)
    if before_grasp and not target_detected:
        mismatch.append("target_missing")
    if before_grasp and target_detected and moved(target) in ("small", "large"):
        mismatch.append("target_displaced")
    if in_dest:
        mismatch.append("destination_occupied")
    if s.phase == Phase.TRANSPORT and gripper == "closed_empty":
        mismatch.append("grasp_lost")

    return {
        "task": f"put {executor.obj} in {executor.dest}",
        "step": s.phase.value,
        "step_progress": "stalled" if s.phase_steps >= cfg.stall_steps else "normal",
        "retries": s.retries,
        "gripper": gripper,
        "target_detected": target_detected,
        "target_last_seen": last_seen(target),
        "target_moved_since_previous_look": moved(target),
        "target_region": _region(target.pos) if target is not None else "unknown",
        "target_in_destination": target_in_dest,
        "objects_near_target": near_target,
        "other_objects_moved": moved_others,
        "destination_detected": executor.dest in belief.detected,
        "destination_region": _region(dest.pos) if dest is not None else "unknown",
        "objects_in_destination": in_dest,
        "mismatch": mismatch,
    }


def render_facts(facts: dict) -> str:
    def fmt(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (list, tuple)):
            return ", ".join(value) if value else "none"
        return str(value)

    return "\n".join(f"{name}: {fmt(value)}" for name, value in facts.items())
