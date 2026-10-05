"""Environment smoke test: build a LIBERO task, step it, render a camera frame,
and check that restoring a snapshot reproduces a rollout exactly.

    uv run python scripts/smoke_test.py [--suite libero_object] [--task 0]
"""

import argparse
import time

import imageio
import numpy as np

from branchlab import PROJECT_ROOT
from branchlab.env import load_init_states, make_env
from branchlab.snapshot import restore, snapshot


def rollout(env, actions):
    for action in actions:
        env.step(action)
    return env.get_sim_state().copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    # Cameras are off during stepping; frames are rendered on demand.
    t0 = time.time()
    env, task = make_env(args.suite, args.task, use_camera_obs=False)
    env.seed(0)
    env.reset()
    init_states = load_init_states(args.suite, args.task)
    env.set_init_state(init_states[0])
    print(f"task: {task.language}")
    print(f"env built in {time.time() - t0:.1f}s, {len(init_states)} init states")

    rng = np.random.default_rng(0)
    warmup = rng.uniform(-0.5, 0.5, size=(30, 7))
    actions = rng.uniform(-0.5, 0.5, size=(args.steps, 7))

    t0 = time.time()
    rollout(env, warmup)
    print(f"step: {(time.time() - t0) / len(warmup) * 1000:.1f} ms")

    # robosuite frames come out vertically flipped
    t0 = time.time()
    frame = env.sim.render(camera_name="agentview", height=256, width=256)[::-1]
    render_ms = (time.time() - t0) * 1000
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    imageio.imwrite(out_dir / "smoke_agentview.png", frame)
    print(f"render: {frame.shape} in {render_ms:.0f} ms -> outputs/smoke_agentview.png")
    assert frame.std() > 1, "rendered frame is blank"

    # Snapshot mid-episode, roll out, restore, roll out again
    snap = snapshot(env)
    final_a = rollout(env, actions)
    restore(env, snap)
    final_b = rollout(env, actions)
    diff = np.abs(final_a - final_b).max()
    print(f"restore: max |state diff| after {args.steps} replayed steps = {diff:.3e}")
    assert diff == 0, "restored rollout diverged from the original"

    env.close()
    print("OK")


if __name__ == "__main__":
    main()
