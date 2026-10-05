"""Summarize a branching result table: loss per intervention in each state,
which intervention is best, and the regret of simple policies.

    uv run python scripts/analyze_pilot.py [--dir outputs/pilot] [--fail-penalty 60]

Loss of a run = seconds from the decision to the end + fail_penalty if it failed.
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from branchlab import PROJECT_ROOT
from branchlab.facts import CONTROL_HZ
from branchlab.interventions import INTERVENTIONS

SHORT = {"continue": "cont", "reobserve": "reobs", "local_recovery": "local", "replan": "replan"}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def describe(state):
    sc = state["scenario"]
    extra = {
        "recognition_miss": f"persist {sc['persistence']:.0f}",
        "occlusion": f"{sc['magnitude'] * 100:.0f}cm",
        "nudge": f"{sc['magnitude'] * 100:.0f}cm",
        "relocate": f"{state['hidden']['distance'] * 100:.0f}cm",
    }.get(sc["cause"], "")
    return f"{sc['cause']} {extra}".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs/pilot")
    parser.add_argument("--fail-penalty", type=float, default=60.0, help="seconds")
    args = parser.parse_args()

    base = PROJECT_ROOT / args.dir
    states = sorted(load(base / "states.jsonl"), key=lambda s: s["state_id"])
    runs = defaultdict(list)
    for row in load(base / "runs.jsonl"):
        runs[row["state_id"], row["intervention"]].append(row)

    def loss(row):
        return row["steps_after"] / CONTROL_HZ + args.fail_penalty * (not row["success"])

    print(f"loss = seconds after the decision + {args.fail_penalty:.0f} s if failed\n")
    header = f"{'id':>2} {'hidden cause':26} {'noise':>5} {'symptom':>8} {'risk':>5} | "
    header += " ".join(f"{SHORT[a]:>12}" for a in INTERVENTIONS) + " | best"
    print(header)
    print(" " * 51 + "| " + " ".join(f"{'loss (succ)':>12}" for _ in INTERVENTIONS) + " |")

    mean_loss = np.zeros((len(states), len(INTERVENTIONS)))
    risk = np.zeros(len(states))
    for i, state in enumerate(states):
        cells = []
        for j, name in enumerate(INTERVENTIONS):
            rows = runs[state["state_id"], name]
            mean_loss[i, j] = np.mean([loss(r) for r in rows])
            cells.append(f"{mean_loss[i, j]:6.1f} ({sum(r['success'] for r in rows)}/{len(rows)})")
        cont = runs[state["state_id"], "continue"]
        risk[i] = 1 - np.mean([r["success"] for r in cont])
        symptom = "missing" if "target_missing" in state["facts"]["mismatch"] else "-"
        best = INTERVENTIONS[int(np.argmin(mean_loss[i]))]
        print(
            f"{state['state_id']:>2} {describe(state):26} {state['pos_noise'] * 100:>4.1f}c "
            f"{symptom:>8} {risk[i]:>5.2f} | " + " ".join(f"{c:>12}" for c in cells) + f" | {SHORT[best]}"
        )

    best_loss = mean_loss.min(axis=1)
    print("\nbest intervention by hidden cause:")
    by_cause = defaultdict(lambda: defaultdict(int))
    for i, state in enumerate(states):
        by_cause[state["scenario"]["cause"]][INTERVENTIONS[int(np.argmin(mean_loss[i]))]] += 1
    for cause, counts in by_cause.items():
        print(f"  {cause:17} " + ", ".join(f"{SHORT[a]} x{n}" for a, n in counts.items()))

    print("\nmean regret (seconds) of simple policies over all states:")
    for j, name in enumerate(INTERVENTIONS):
        print(f"  always {name:15} {np.mean(mean_loss[:, j] - best_loss):6.1f}")
    # Risk-only policy: one intervention per risk bin, fitted on these same states.
    high = risk >= 0.5
    regret, picks = 0.0, []
    for mask in (~high, high):
        if mask.any():
            j = int(np.argmin(mean_loss[mask].mean(axis=0)))
            picks.append(INTERVENTIONS[j])
            regret += (mean_loss[mask, j] - best_loss[mask]).sum()
    print(
        f"  measured risk, 2 bins    {regret / len(states):6.1f}   "
        f"(risk<0.5 -> {picks[0]}, risk>=0.5 -> {picks[-1]}; fitted in-sample)"
    )
    print("  best per state            0.0   (chosen and scored on the same runs)")


if __name__ == "__main__":
    main()
