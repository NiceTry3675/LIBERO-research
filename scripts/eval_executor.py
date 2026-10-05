"""Success rate of the scripted executor over a task's init states, with no
disturbances, plus a check that a mid-episode snapshot replays exactly.

    uv run python scripts/eval_executor.py [--action-noise 0.1] [--seeds 3]
"""

import argparse

import numpy as np

from branchlab.env import load_init_states, make_env, reset_to
from branchlab.executor import ExecutorConfig, PickPlaceExecutor, run_episode
from branchlab.perception import observe
from branchlab.snapshot import restore, snapshot

SUITE, TASK = "libero_object", 0
OBJ, DEST = "alphabet_soup_1", "basket_1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=None)
    args = parser.parse_args()

    env, task = make_env(SUITE, TASK, use_camera_obs=False)
    env.seed(0)
    init_states = load_init_states(SUITE, TASK)[: args.max_states]
    config = ExecutorConfig(action_noise=args.action_noise)
    print(f"task: {task.language} | action noise {args.action_noise}")

    results = []
    for i, init_state in enumerate(init_states):
        for seed in range(args.seeds):
            reset_to(env, init_state)
            executor = PickPlaceExecutor(OBJ, DEST, config, seed=seed)
            success, steps = run_episode(env, executor, observe)
            results.append((success, steps, executor.state.retries))
    success, steps, retries = map(np.array, zip(*results))
    print(
        f"success {success.sum()}/{len(success)} | steps mean {steps.mean():.0f} "
        f"max {steps.max()} | episodes with retries {(retries > 0).sum()}"
    )

    # Snapshot mid-descent, finish, restore, finish again
    reset_to(env, init_states[0])
    executor = PickPlaceExecutor(OBJ, DEST, config, seed=0)
    for _ in range(40):
        env.step(executor.act(observe(env)))
    env_snap, exec_snap = snapshot(env), executor.get_state()
    run_episode(env, executor, observe)
    final_a = env.get_sim_state().copy()
    restore(env, env_snap)
    executor.set_state(exec_snap)
    run_episode(env, executor, observe)
    diff = np.abs(final_a - env.get_sim_state()).max()
    print(f"restore: max |state diff| at episode end = {diff:.3e}")
    assert diff == 0, "restored episode diverged from the original"

    env.close()


if __name__ == "__main__":
    main()
