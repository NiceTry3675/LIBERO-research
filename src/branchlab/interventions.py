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

from branchlab.executor import Phase
from branchlab.facts import build_facts
from branchlab.perception import Percept
from branchlab.runner import Decision, Session

INTERVENTIONS = ("continue", "reobserve", "local_recovery", "replan")


@dataclass
class InterventionConfig:
    reobserve_looks: int = 3  # looks fused into one measurement
    reobserve_gap: int = 3  # steps between those looks
    plan_delay: int = 60  # steps spent waiting for the planner (3 s)
    scan_x: tuple = (-0.15, 0.05)  # floor grid the scan visits; further out the elbow hits its limit
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


def escalate(session: Session, decision: Decision, cfg: InterventionConfig = None,
             max_steps: int = 600) -> str:
    """Staged escalation: try the cheapest response first and move up only when
    it does not clear the symptom.

    With no symptom at the decision point it just continues. Otherwise it
    re-observes; if the symptom is still in the facts it does a local recovery;
    if the wrist look does not find the object either, it replans. Returns the
    last stage used. Nothing further is attempted once a stage has cleared the
    symptom, even if execution fails later.
    """
    cfg = cfg or InterventionConfig()
    executor = session.executor
    symptoms = set(decision.facts["mismatch"])
    if not symptoms:
        return "continue"

    apply(session, "reobserve", cfg)
    facts = build_facts(executor, session.percept, session.fact_config)
    if not symptoms & set(facts["mismatch"]):
        return "reobserve"

    apply(session, "local_recovery", cfg)
    session.run(max_steps, stop=lambda: executor.state.phase != Phase.SEARCH)
    if executor.state.phase != Phase.FAILED:
        return "local_recovery"

    apply(session, "replan", cfg)
    return "replan"
