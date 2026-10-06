"""Summarize a branching result table: loss per intervention in each state,
and the loss of decision policies that know different things.

    uv run python scripts/analyze_pilot.py [--dir outputs/pilot] [--fail-penalty 60]

Loss of a run = seconds from the decision to the end + fail_penalty if it failed.
Risk of a state = failure rate of "continue" there, i.e. known perfectly.

Policies compared (each picks one intervention per state):
    always X           the same intervention everywhere
    risk only          one intervention per risk bin (risk < 0.5, risk >= 0.5)
    risk + escalation  continue if risk < 0.5, otherwise staged escalation
    escalation         staged escalation everywhere (measured as its own arm)
    cause known        one intervention per hidden cause
    best per state     the best single intervention in each state, scored on
                       the same runs it was chosen on: a lower bound that also
                       profits from luck
"risk only" and "cause known" choose their intervention per bin / per cause
from the other states (leave one state out); in-sample numbers are given too.
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from branchlab import PROJECT_ROOT
from branchlab.facts import CONTROL_HZ
from branchlab.interventions import INTERVENTIONS

SHORT = {"continue": "cont", "reobserve": "reobs", "local_recovery": "local", "replan": "replan",
         "escalation": "escal"}


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def cause_label(state):
    sc = state["scenario"]
    if sc["cause"] == "recognition_miss":
        return "miss_transient" if sc["persistence"] < 1 else "miss_persistent"
    return sc["cause"]


def describe(state):
    sc = state["scenario"]
    extra = {
        "occlusion": f"{sc['magnitude'] * 100:.0f}cm",
        "nudge": f"{sc['magnitude'] * 100:.0f}cm",
        "relocate": f"{state['hidden']['distance'] * 100:.0f}cm",
    }.get(sc["cause"], "")
    return f"{cause_label(state)} {extra}".strip()


def fit_choice(loss, groups, i=None):
    """Per group, the intervention with the lowest mean loss over the group's
    states, leaving out state i if given. Returns {group: column}."""
    choice = {}
    for g in set(groups):
        idx = [k for k, h in enumerate(groups) if h == g and k != i]
        if not idx:  # nothing left to fit on: fall back to all states
            idx = [k for k in range(len(groups)) if k != i]
        choice[g] = int(np.argmin(loss[idx].mean(axis=0)))
    return choice


def policy_loss(loss, groups):
    """Mean loss of a one-intervention-per-group policy: (leave one out, in-sample)."""
    n = len(groups)
    loo = np.mean([loss[i, fit_choice(loss, groups, i)[groups[i]]] for i in range(n)])
    fitted = fit_choice(loss, groups)
    inside = np.mean([loss[i, fitted[groups[i]]] for i in range(n)])
    return loo, inside, fitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs/pilot")
    parser.add_argument("--fail-penalty", type=float, default=60.0, help="seconds")
    args = parser.parse_args()

    base = PROJECT_ROOT / args.dir
    states = sorted(load(base / "states.jsonl"), key=lambda s: s["state_id"])
    runs = defaultdict(list)
    for path in sorted(base.glob("runs*.jsonl")):  # runs.jsonl plus any tagged extra arms
        for row in load(path):
            runs[row["state_id"], row["intervention"]].append(row)
    has_escalation = any(arm == "escalation" for _, arm in runs)
    arms = INTERVENTIONS + (("escalation",) if has_escalation else ())

    def loss(row):
        return row["steps_after"] / CONTROL_HZ + args.fail_penalty * (not row["success"])

    n_reps = len(runs[states[0]["state_id"], "continue"])
    print(f"{len(states)} states x {n_reps} runs per intervention")
    print(f"loss = seconds after the decision + {args.fail_penalty:.0f} s if failed\n")
    header = f"{'id':>2} {'hidden cause':22} {'noise':>5} {'symptom':>8} {'risk':>5} | "
    header += " ".join(f"{SHORT[a]:>13}" for a in arms) + " | best"
    print(header)
    print(" " * 47 + "| " + " ".join(f"{'loss (succ)':>13}" for _ in arms) + " |")

    all_loss = np.zeros((len(states), len(arms)))
    risk = np.zeros(len(states))
    for i, state in enumerate(states):
        cells = []
        for j, name in enumerate(arms):
            rows = runs[state["state_id"], name]
            all_loss[i, j] = np.mean([loss(r) for r in rows])
            cells.append(f"{all_loss[i, j]:6.1f} ({sum(r['success'] for r in rows):>2}/{len(rows)})")
        cont = runs[state["state_id"], "continue"]
        risk[i] = 1 - np.mean([r["success"] for r in cont])
        symptom = "missing" if "target_missing" in state["facts"]["mismatch"] else "-"
        best = INTERVENTIONS[int(np.argmin(all_loss[i, : len(INTERVENTIONS)]))]
        print(
            f"{state['state_id']:>2} {describe(state):22} {state['pos_noise'] * 100:>4.1f}c "
            f"{symptom:>8} {risk[i]:>5.2f} | " + " ".join(f"{c:>13}" for c in cells) + f" | {SHORT[best]}"
        )

    single = all_loss[:, : len(INTERVENTIONS)]
    causes = [cause_label(s) for s in states]
    bins = ["high" if r >= 0.5 else "low" for r in risk]

    print("\nby hidden cause: mean loss of each intervention (s), and risk")
    order = list(dict.fromkeys(causes))
    print(f"  {'cause':16} " + " ".join(f"{SHORT[a]:>7}" for a in INTERVENTIONS) + "   risk")
    for c in order:
        idx = [i for i, h in enumerate(causes) if h == c]
        print(f"  {c:16} " + " ".join(f"{v:7.1f}" for v in single[idx].mean(axis=0))
              + f"   {risk[idx].mean():.2f}")

    rows = []
    for j, name in enumerate(INTERVENTIONS):
        rows.append((f"always {name}", single[:, j].mean(), None, ""))
    loo, inside, fitted = policy_loss(single, bins)
    picks = ", ".join(f"risk {b} -> {SHORT[INTERVENTIONS[j]]}" for b, j in sorted(fitted.items(), reverse=True))
    rows.append(("risk only", loo, inside, picks))
    if has_escalation:
        esc = all_loss[:, -1]
        gated = np.where(risk >= 0.5, esc, single[:, 0])
        rows.append(("risk + escalation", gated.mean(), None, "continue if risk < 0.5"))
        rows.append(("escalation", esc.mean(), None, ""))
    loo, inside, fitted = policy_loss(single, causes)
    picks = ", ".join(f"{c} -> {SHORT[INTERVENTIONS[fitted[c]]]}" for c in order)
    rows.append(("cause known", loo, inside, picks))
    best = single.min(axis=1).mean()
    rows.append(("best per state", best, None, "lower bound"))

    cause_loss = rows[-2][1]
    print("\nmean loss per state (s); gap = loss minus that of the cause-known policy")
    print(f"  {'policy':22} {'loss':>6} {'gap':>6}  {'in-sample':>9}")
    for name, value, inside, note in rows:
        ins = f"{inside:9.1f}" if inside is not None else " " * 9
        print(f"  {name:22} {value:6.1f} {value - cause_loss:+6.1f}  {ins}  {note}")

    if has_escalation:
        print("\nstaged escalation by hidden cause: successes and the stage it stopped at")
        for c in order:
            idx = [i for i, h in enumerate(causes) if h == c]
            rs = [r for i in idx for r in runs[states[i]["state_id"], "escalation"]]
            stages = defaultdict(int)
            for r in rs:
                stages[SHORT[r["stage"]]] += 1
            print(f"  {c:16} {sum(r['success'] for r in rs):>3}/{len(rs):<3} "
                  + ", ".join(f"{k} x{v}" for k, v in stages.items()))


if __name__ == "__main__":
    main()
