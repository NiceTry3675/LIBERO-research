"""Pilot branching experiment for the symptom "target no longer detected".

For each state: run to the first decision point after the disturbance,
snapshot it, then from that same snapshot try every intervention several
times and record what happens. No further intervention is allowed afterwards.

    uv run python scripts/pilot.py [--reps 20] [--workers 5] [--out outputs/pilot]

Besides the four interventions there is an "escalation" arm (staged
escalation, see interventions.escalate). `--arms` and `--states` run a subset; with `--tag X`
the output goes to states_X.jsonl / runs_X.jsonl so it can sit next to an
earlier run of the same states.
"""

import argparse
import json
import multiprocessing as mp
from dataclasses import asdict

import imageio

from branchlab import PROJECT_ROOT
from branchlab.env import load_init_states, make_env, render, reset_to
from branchlab.executor import PickPlaceExecutor
from branchlab.interventions import INTERVENTIONS, apply, escalate
from branchlab.perception import PerceptionConfig, PerceptionModel
from branchlab.runner import Session
from branchlab.scenarios import SUITE, TASK_ID, Scenario, due, setup_layout, trigger

NOISE_LEVELS = (0.005, 0.010)
RELOCATE_GAPS = (0.13, 0.15)
MAX_STEPS = 800
ARMS = INTERVENTIONS + ("escalation",)


def build_states(inits_per_variant: int = 2):
    variants = [Scenario("none")]
    variants += [Scenario("recognition_miss", persistence=p) for p in (0.0, 1.0)]
    variants += [Scenario("occlusion", magnitude=m) for m in (0.04, 0.08)]
    variants += [Scenario("nudge", magnitude=m) for m in (0.04, 0.06, 0.08)]
    variants += [Scenario("relocate", relocate_gap=g) for g in RELOCATE_GAPS]
    variants += [Scenario("held_miss")]
    combos = [(sc, noise, k) for sc in variants for noise in NOISE_LEVELS
              for k in range(inits_per_variant)]
    # Each state gets its own LIBERO init state and seed.
    return [
        {"state_id": i, "scenario": sc, "pos_noise": noise, "init_index": i, "seed": i}
        for i, (sc, noise, _) in enumerate(combos)
    ]


_env = None


def _get_env():
    global _env
    if _env is None:
        _env, _ = make_env(SUITE, TASK_ID, use_camera_obs=False)
        _env.seed(0)
    return _env


def run_state(job):
    spec, reps, arms, out_dir = job
    env = _get_env()
    sc = spec["scenario"]
    reset_to(env, load_init_states(SUITE, TASK_ID)[spec["init_index"]])
    setup_layout(env, sc)
    session = Session(
        env,
        PickPlaceExecutor(sc.target, sc.dest, seed=spec["seed"]),
        PerceptionModel(PerceptionConfig(pos_noise=spec["pos_noise"]), seed=spec["seed"]),
    )

    fired = []

    def on_step(step):
        if not fired and due(sc, step, session.executor):
            fired.append(trigger(env, session.perception, sc))

    decision = session.run(MAX_STEPS, on_step=on_step, until=lambda d: bool(fired))
    saved = session.snapshot()
    frame_path = out_dir / "frames" / f"state_{spec['state_id']:02d}.png"
    imageio.imwrite(frame_path, render(env))

    state_row = {
        "state_id": spec["state_id"],
        "pos_noise": spec["pos_noise"],
        "init_index": spec["init_index"],
        "scenario": asdict(sc),
        "hidden": fired[0],
        "decision_step": decision.step,
        "facts": decision.facts,
        "frame": str(frame_path.relative_to(PROJECT_ROOT)),
    }

    run_rows = []
    for name in arms:
        for rep in range(reps):
            session.restore(saved)
            # Same seed for every arm, so they face the same luck.
            session.perception.reseed([spec["seed"], rep])
            session.executor.reseed([spec["seed"], rep])
            if name == "escalation":
                stage = escalate(session, decision, max_steps=MAX_STEPS)
            else:
                apply(session, name)
                stage = name
            session.run(MAX_STEPS)
            run_rows.append({
                "state_id": spec["state_id"],
                "intervention": name,
                "stage": stage,
                "rep": rep,
                "success": session.success,
                "steps_after": session.step - decision.step,
                "retries": session.executor.state.retries,
                "final_step": session.executor.state.phase.value,
            })
    return state_row, run_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out", default="outputs/pilot")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--tag", default="")
    parser.add_argument("--states", type=int, nargs="+", help="run only these state ids")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    specs = [s for s in build_states() if args.states is None or s["state_id"] in args.states]
    jobs = [(spec, args.reps, tuple(args.arms), out_dir) for spec in specs]
    suffix = f"_{args.tag}" if args.tag else ""

    with mp.get_context("spawn").Pool(args.workers) as pool, \
            open(out_dir / f"states{suffix}.jsonl", "w") as f_states, \
            open(out_dir / f"runs{suffix}.jsonl", "w") as f_runs:
        for state_row, run_rows in pool.imap_unordered(run_state, jobs):
            f_states.write(json.dumps(state_row) + "\n")
            for row in run_rows:
                f_runs.write(json.dumps(row) + "\n")
            f_states.flush()
            f_runs.flush()
            print(f"state {state_row['state_id']:2d} done ({state_row['scenario']['cause']})", flush=True)
    print(f"wrote states{suffix}.jsonl and runs{suffix}.jsonl in {out_dir}")


if __name__ == "__main__":
    main()
