"""The four responses available at a decision point.

    continue        keep the plan and the current motion
    reobserve       pause and measure again with the main camera
    local_recovery  keep the plan, redo the current step: look down with the
                    wrist camera where the object was believed to be, then grasp
    replan          wait for a new plan, then follow it. For now the plan is
                    fixed rather than produced by an LLM: scan the workspace
                    with the wrist camera, then pick and place.
"""

from dataclasses import dataclass

import numpy as np

from branchlab.perception import Percept
from branchlab.runner import Session

INTERVENTIONS = ("continue", "reobserve", "local_recovery", "replan")


@dataclass
class InterventionConfig:
    reobserve_looks: int = 3  # looks fused into one measurement
    reobserve_gap: int = 5  # steps between those looks
    plan_delay: int = 60  # steps spent waiting for the planner (3 s)
    scan_x: tuple = (-0.15, 0.10)  # floor grid the scan visits
    scan_y: tuple = (-0.25, 0.0, 0.25)


def _reobserve(session: Session, cfg: InterventionConfig):
    """Take several looks and fuse them: an object counts as detected if most
    looks report it, at the mean of the reported positions."""
    looks = []
    for _ in range(cfg.reobserve_looks):
        session.hold(cfg.reobserve_gap)
        looks.append(session.perception.look(session.env))
    fused = {}
    for name in set().union(*(look.objects for look in looks)):
        seen = [look.objects[name] for look in looks if name in look.objects]
        if 2 * len(seen) > len(looks):
            fused[name] = np.mean(seen, axis=0)
    last = looks[-1]
    session.percept = Percept(last.ee_pos, last.gripper_width, fused)
    session.executor.perceive(session.percept)


def _scan_waypoints(start_xy, cfg: InterventionConfig):
    """The scan grid, ordered so each waypoint is the nearest one not yet visited."""
    remaining = [np.array([x, y]) for x in cfg.scan_x for y in cfg.scan_y]
    order, here = [], np.asarray(start_xy)
    while remaining:
        i = int(np.argmin([np.linalg.norm(w - here) for w in remaining]))
        here = remaining.pop(i)
        order.append(here)
    return order


def apply(session: Session, name: str, cfg: InterventionConfig = None):
    """Apply an intervention to a session stopped at a decision point."""
    cfg = cfg or InterventionConfig()
    executor = session.executor
    if name == "continue":
        return
    if name == "reobserve":
        _reobserve(session, cfg)
    elif name == "local_recovery":
        believed = executor.state.belief.pos(executor.obj)
        here = believed if believed is not None else session.percept.ee_pos
        executor.start_search([here[:2]])
    elif name == "replan":
        session.hold(cfg.plan_delay)
        executor.start_search(_scan_waypoints(session.percept.ee_pos[:2], cfg))
    else:
        raise ValueError(f"unknown intervention: {name}")
