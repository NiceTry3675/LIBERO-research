"""Run each hidden cause with no intervention (the executor just continues)
and show that they produce the same symptom in the fact list.

    uv run python scripts/run_scenarios.py [--magnitude 0.06] [--pos-noise 0.005] [--seeds 5]
"""

import argparse

import imageio
import numpy as np

from branchlab import PROJECT_ROOT
from branchlab.env import load_init_states, make_env, render, reset_to
from branchlab.executor import PickPlaceExecutor
from branchlab.facts import render_facts
from branchlab.perception import PerceptionConfig, PerceptionModel
from branchlab.runner import run_episode
from branchlab.scenarios import CAUSES, Scenario, setup_layout, trigger

SUITE, TASK = "libero_object", 0
SHOWN = ("step", "target_detected", "target_last_seen", "objects_near_target",
         "other_objects_moved", "mismatch")


def run(env, init_state, scenario, pos_noise, seed):
    reset_to(env, init_state)
    setup_layout(env, scenario)
    executor = PickPlaceExecutor(scenario.target, scenario.dest, seed=seed)
    perception = PerceptionModel(PerceptionConfig(pos_noise=pos_noise), seed=seed)
    log = {"record": None, "symptom": None, "frame": None}

    def on_step(step):
        if step == scenario.event_step:
            log["record"] = trigger(env, perception, scenario)

    def on_decision(decision):
        # The first decision point after the event is where the symptom shows.
        if log["record"] is not None and log["symptom"] is None:
            log["symptom"] = decision
            log["frame"] = render(env)

    episode = run_episode(env, executor, perception, on_decision=on_decision, on_step=on_step)
    return episode, executor, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--magnitude", type=float, default=0.06)
    parser.add_argument("--persistence", type=float, default=1.0)
    parser.add_argument("--pos-noise", type=float, default=0.005)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    env, _ = make_env(SUITE, TASK, use_camera_obs=False)
    env.seed(0)
    init_state = load_init_states(SUITE, TASK)[0]
    out_dir = PROJECT_ROOT / "outputs" / "scenarios"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for cause in CAUSES:
        scenario = Scenario(cause, magnitude=args.magnitude, persistence=args.persistence)
        outcomes = []
        for seed in range(args.seeds):
            episode, executor, log = run(env, init_state, scenario, args.pos_noise, seed)
            outcomes.append((episode.success, episode.steps, executor.state.retries))
            if seed == 0:
                record, symptom = log["record"], log["symptom"]
                frames.append(log["frame"])
                imageio.imwrite(out_dir / f"{cause}.png", log["frame"])
                print(f"\n=== {cause} ===")
                print(
                    f"hidden cause: moved {record['moved']} by {record['distance'] * 100:.0f} cm, "
                    f"target visible fraction {record['target_visible']:.2f}"
                )
                print(f"facts at decision step {symptom.step}:")
                shown = {k: symptom.facts[k] for k in SHOWN}
                print("  " + render_facts(shown).replace("\n", "\n  "))
        success, steps, retries = map(np.array, zip(*outcomes))
        print(
            f"continue: success {success.sum()}/{len(success)} | "
            f"steps mean {steps.mean():.0f} | retries mean {retries.mean():.1f}"
        )
    imageio.imwrite(out_dir / "all_causes.png", np.concatenate(frames, axis=1))
    print(f"\nframes at the symptom: outputs/scenarios/all_causes.png ({', '.join(CAUSES)})")
    env.close()


if __name__ == "__main__":
    main()
