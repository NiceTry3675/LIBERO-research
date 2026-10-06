"""Run one episode and print the fact list at every decision point, optionally
with a recognition fault injected partway through.

    uv run python scripts/show_facts.py
    uv run python scripts/show_facts.py --fault miss --at-decision 1
    uv run python scripts/show_facts.py --fault offset --persistence 0
    uv run python scripts/show_facts.py --fault ghost
"""

import argparse

import numpy as np

from branchlab.env import load_init_states, make_env, reset_to
from branchlab.executor import PickPlaceExecutor
from branchlab.facts import render_facts
from branchlab.perception import Fault, PerceptionConfig, PerceptionModel, observe
from branchlab.runner import run_episode
from branchlab.scenarios import DEST, OCCLUDER, SUITE, TARGET, TASK_ID


def make_fault(kind, env, persistence):
    truth = observe(env).objects
    if kind == "miss":  # target not recognized
        return Fault("miss", TARGET, persistence=persistence)
    if kind == "offset":  # target reported 6 cm from where it really is
        return Fault("offset", TARGET, np.array([0.0, 0.06, 0.0]), persistence)
    if kind == "ghost":  # a distractor falsely reported inside the destination
        return Fault("ghost", OCCLUDER, truth[DEST] + np.array([0.02, 0.0, 0.03]), persistence)
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault", choices=["miss", "offset", "ghost"])
    parser.add_argument("--at-decision", type=int, default=1, help="decision index to inject at")
    parser.add_argument("--persistence", type=float, default=1.0)
    parser.add_argument("--pos-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env, _ = make_env(SUITE, TASK_ID, use_camera_obs=False)
    env.seed(0)
    reset_to(env, load_init_states(SUITE, TASK_ID)[0])
    executor = PickPlaceExecutor(TARGET, DEST, seed=args.seed)
    perception = PerceptionModel(PerceptionConfig(pos_noise=args.pos_noise), seed=args.seed)

    count = 0

    def on_decision(decision):
        nonlocal count
        print(f"\n--- decision {count} (step {decision.step}) ---")
        print(render_facts(decision.facts))
        count += 1
        # Injected after this look, so the fault first shows at the next one.
        if args.fault and count == args.at_decision:
            perception.faults.append(make_fault(args.fault, env, args.persistence))
            print(f"[injected fault: {args.fault}, persistence {args.persistence}]")

    episode = run_episode(env, executor, perception, on_decision=on_decision)
    print(
        f"\nsuccess {episode.success} | steps {episode.steps} | "
        f"retries {executor.state.retries} | final step {executor.state.phase.value}"
    )
    env.close()


if __name__ == "__main__":
    main()
