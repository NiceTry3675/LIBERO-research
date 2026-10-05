"""Success rate of the scripted executor over a task's init states, with no
disturbances, plus a check that a mid-episode snapshot replays exactly.

    uv run python scripts/eval_executor.py [--pos-noise 0.01] [--miss-prob 0.1] [--seeds 3]
"""

import argparse

import numpy as np

from branchlab.env import load_init_states, make_env, reset_to
from branchlab.executor import ExecutorConfig, PickPlaceExecutor
from branchlab.perception import PerceptionConfig, PerceptionModel
from branchlab.runner import run_episode
from branchlab.snapshot import restore, snapshot

SUITE, TASK = "libero_object", 0
OBJ, DEST = "alphabet_soup_1", "basket_1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos-noise", type=float, default=0.0, help="std in metres")
    parser.add_argument("--miss-prob", type=float, default=0.0)
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=None)
    args = parser.parse_args()

    env, task = make_env(SUITE, TASK, use_camera_obs=False)
    env.seed(0)
    init_states = load_init_states(SUITE, TASK)[: args.max_states]
    exec_config = ExecutorConfig(action_noise=args.action_noise)
    perc_config = PerceptionConfig(pos_noise=args.pos_noise, miss_prob=args.miss_prob)
    print(
        f"task: {task.language} | pos noise {args.pos_noise} | "
        f"miss prob {args.miss_prob} | action noise {args.action_noise}"
    )

    results = []
    for init_state in init_states:
        for seed in range(args.seeds):
            reset_to(env, init_state)
            executor = PickPlaceExecutor(OBJ, DEST, exec_config, seed=seed)
            perception = PerceptionModel(perc_config, seed=seed)
            episode = run_episode(env, executor, perception)
            results.append((episode.success, episode.steps, executor.state.retries))
    success, steps, retries = map(np.array, zip(*results))
    print(
        f"success {success.sum()}/{len(success)} | steps mean {steps.mean():.0f} "
        f"max {steps.max()} | episodes with retries {(retries > 0).sum()}"
    )

    # Snapshot mid-descent, finish, restore, finish again
    reset_to(env, init_states[0])
    executor = PickPlaceExecutor(OBJ, DEST, exec_config, seed=0)
    perception = PerceptionModel(perc_config, seed=0)
    run_episode(env, executor, perception, max_steps=40)
    saved = snapshot(env), executor.get_state(), perception.get_state()
    run_episode(env, executor, perception)
    final_a = env.get_sim_state().copy()
    restore(env, saved[0])
    executor.set_state(saved[1])
    perception.set_state(saved[2])
    run_episode(env, executor, perception)
    diff = np.abs(final_a - env.get_sim_state()).max()
    print(f"restore: max |state diff| at episode end = {diff:.3e}")
    assert diff == 0, "restored episode diverged from the original"

    env.close()


if __name__ == "__main__":
    main()
