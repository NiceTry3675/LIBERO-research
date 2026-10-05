from dataclasses import dataclass, field

from branchlab.executor import PickPlaceExecutor
from branchlab.facts import FactConfig, build_facts
from branchlab.perception import PerceptionModel


@dataclass
class Decision:
    step: int
    facts: dict


@dataclass
class Episode:
    success: bool
    steps: int
    decisions: list = field(default_factory=list)


def is_decision_point(executor: PickPlaceExecutor, fact_config: FactConfig) -> bool:
    """A plan step was just entered, or the current one has stalled."""
    steps = executor.state.phase_steps
    return steps == 0 or steps == fact_config.stall_steps


def run_episode(
    env,
    executor: PickPlaceExecutor,
    perception: PerceptionModel,
    max_steps: int = 600,
    fact_config: FactConfig = None,
    on_decision=None,
) -> Episode:
    """Run until the executor finishes or max_steps.

    The agent looks at objects only at decision points; in between it acts on
    proprioception and its belief. `on_decision(step, facts)` is called at each
    decision point before the executor acts on it.
    """
    fact_config = fact_config or FactConfig()
    decisions = []
    step = 0
    while step < max_steps and not executor.finished:
        if is_decision_point(executor, fact_config):
            percept = perception.look(env)
            executor.perceive(percept)
            decision = Decision(step, build_facts(executor, percept, fact_config))
            decisions.append(decision)
            if on_decision is not None:
                on_decision(decision)
        else:
            percept = perception.proprio(env)
            executor.perceive(percept)
        env.step(executor.command(percept))
        step += 1
    return Episode(bool(env.check_success()), step, decisions)
