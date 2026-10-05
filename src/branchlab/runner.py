import copy
from dataclasses import dataclass
from typing import Optional

from branchlab.executor import PickPlaceExecutor
from branchlab.facts import FactConfig, build_facts
from branchlab.perception import PerceptionModel
from branchlab.snapshot import restore, snapshot


@dataclass
class Decision:
    step: int
    facts: dict


@dataclass
class Episode:
    success: bool
    steps: int
    decisions: list


def is_decision_point(executor: PickPlaceExecutor, fact_config: FactConfig) -> bool:
    """A plan step was just entered, or the current one has stalled."""
    steps = executor.state.phase_steps
    return steps == 0 or steps == fact_config.stall_steps


class Session:
    """One episode in progress: environment, executor and perception together.

    Each control step is observe-then-act. The agent looks at objects only at
    decision points (and when the executor asks for a wrist look); in between
    it acts on proprioception and its belief. A session can stop right after
    observing a decision point, which is where it is snapshotted and branched.
    """

    def __init__(self, env, executor: PickPlaceExecutor, perception: PerceptionModel,
                 fact_config: FactConfig = None):
        self.env = env
        self.executor = executor
        self.perception = perception
        self.fact_config = fact_config or FactConfig()
        self.step = 0
        self.decisions = []
        self.percept = None
        self._observed = False  # the current step has been observed but not acted on

    def observe(self) -> Optional[Decision]:
        state = self.executor.state
        decision = None
        if state.look_camera is not None:
            self.percept = self.perception.look(self.env, state.look_camera)
            state.look_camera = None
            self.executor.perceive(self.percept)
        elif is_decision_point(self.executor, self.fact_config):
            self.percept = self.perception.look(self.env)
            self.executor.perceive(self.percept)
            decision = Decision(self.step, build_facts(self.executor, self.percept, self.fact_config))
            self.decisions.append(decision)
        else:
            self.percept = self.perception.proprio(self.env)
            self.executor.perceive(self.percept)
        self._observed = True
        return decision

    def act(self):
        self.env.step(self.executor.command(self.percept))
        self.step += 1
        self._observed = False

    def hold(self, steps: int):
        """Let time pass with the arm still (waiting for a planner, a sensor...)."""
        for _ in range(steps):
            self.env.step(self.executor.hold_action())
            self.step += 1
            self.percept = self.perception.proprio(self.env)
            self.executor.perceive(self.percept)
        self._observed = True

    def run(self, max_steps: int = 600, on_decision=None, on_step=None, until=None):
        """Run until the executor finishes or max_steps.

        `on_step(step)` is called at the start of every step, which is where
        external events are triggered; `on_decision(decision)` at each decision
        point before the executor acts on it. If `until(decision)` is true the
        session stops there, observed but not yet acted on, and returns the
        decision; calling run() again carries on from that point.
        """
        while self.step < max_steps and not self.executor.finished:
            if not self._observed:
                if on_step is not None:
                    on_step(self.step)
                decision = self.observe()
                if decision is not None:
                    if on_decision is not None:
                        on_decision(decision)
                    if until is not None and until(decision):
                        return decision
            self.act()
        return None

    @property
    def success(self) -> bool:
        return bool(self.env.check_success())

    def snapshot(self):
        return (
            snapshot(self.env),
            self.executor.get_state(),
            self.perception.get_state(),
            self.step,
            copy.deepcopy(self.percept),
            self._observed,
            len(self.decisions),
        )

    def restore(self, saved):
        env_snap, exec_state, perc_state, step, percept, observed, n_decisions = saved
        restore(self.env, env_snap)
        self.executor.set_state(exec_state)
        self.perception.set_state(perc_state)
        self.step = step
        self.percept = copy.deepcopy(percept)
        self._observed = observed
        del self.decisions[n_decisions:]


def run_episode(env, executor, perception, max_steps: int = 600, fact_config: FactConfig = None,
                on_decision=None, on_step=None) -> Episode:
    session = Session(env, executor, perception, fact_config)
    session.run(max_steps, on_decision=on_decision, on_step=on_step)
    return Episode(session.success, session.step, session.decisions)
