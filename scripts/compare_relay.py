#!/usr/bin/env python3
"""Compare relay variants with the baseline on the same episodes.

    python scripts/compare_relay.py [--dev] [--runs ROOT ...] [--reference LABEL]
                                    [--empty-grasp-limit N ...] [task ...]

ROOT is an outputs directory holding robodawn/ (evaluation episodes 0-9: the baseline's
gemini_flash_<tier> and the relay_* runs) and robodawn_dev/ (development episodes 10+, relay_* runs only,
relay_baseline_* being the upstream loop); the default is this project's outputs/ plus every fetched
outputs/robodawn_colab/<session>/. A run is labelled by its directory name and merged over shards and
roots; an episode counts once. Only runs of --tier (default flex) are read.

Rows use only the episodes that every listed run finished, so they are paired. Per run: successes, turns,
big-model calls, relayed steps, how relays ended, and per episode the big-model time, checker time, wall
time and cost. Successes are compared with the reference (default: the baseline) by an exact sign test
on the episodes where the two differ. --empty-grasp-limit N re-reads each trace as if the episode had
ended at its N-th empty close: that rule only stops episodes, so the run up to there is the rule's run
(wall time is scaled by the share of turns kept, checker time and cost by the share of checks kept).
Press tasks close on nothing on purpose; as in robodawn_relay.PRESS_TASKS, their closes do not count.

Gemini prices are USD per 1M tokens for Gemini 3.8 Flash standard on Vertex as of 2026-10 (flex: half),
as in compare_robodawn.py. Per episode, the cached share of the input is the run directory's. Clef's cost
is what OpenRouter reported; the lite checker's needs --price-lite-input/--price-lite-output.
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PRESS_TASKS = {"click_bell", "click_alarmclock", "press_stapler", "turn_switch"}   # = robodawn_relay.PRESS_TASKS


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def empty_close(result):
    return result.get("kind") == "gripper" and str(result.get("note", "")).startswith("fingers closed fully")


def sign_test(b, c):
    """Two-sided exact binomial p for b vs c discordant pairs."""
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(roots, dev, tasks, args):
    """label -> {(task, episode): record}"""
    runs, repeated = defaultdict(dict), Counter()
    replies = defaultdict(Counter)       # label -> answered calls, calls cut off by the token budget, reasoning tokens
    for root in roots:
        base = root / ("robodawn_dev" if dev else "robodawn")
        for run_dir in sorted([*base.glob(f"gemini_flash_{args.tier}"), *base.glob(f"relay_*_{args.tier}")]) \
                if base.is_dir() else []:
            scale = 0.5 if args.tier == "flex" else 1.0
            for shard in sorted(run_dir.glob("*/shard_*")):
                task = shard.parent.name
                if tasks and task not in tasks or not (shard / "results.json").exists():
                    continue
                answered = [t for t in read_jsonl(shard / "transport.jsonl") if t.get("status") == 200]
                usage = [t.get("usage") or {} for t in answered]
                replies[run_dir.name].update(answered=len(answered),
                                             cut=sum(t.get("finish_reason") == "max_tokens" for t in answered),
                                             reasoning=sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
                                                           for u in usage))
                prompt = sum(u.get("prompt_tokens", 0) for u in usage)
                cached = sum((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) for u in usage)
                share = cached / prompt if prompt else 0.0
                calls = defaultdict(list)       # checker calls per episode (an aborted attempt's calls included)
                for c in read_jsonl(shard / "checker_calls.jsonl"):
                    calls[c.get("episode")].append(c)
                for e in json.loads((shard / "results.json").read_text())["episodes"]:
                    key = (task, e["episode_index"])
                    if key in runs[run_dir.name]:
                        repeated[run_dir.name] += 1
                        continue
                    name = f"episode_{e['episode_index']:03d}"
                    gemini = scale * (e["prompt_tokens"] * ((1 - share) * args.price_input + share * args.price_cached)
                                      + e["completion_tokens"] * args.price_output) / 1e6
                    # Clef reports its cost; the lite checker's comes from its tokens (results.json: this attempt only)
                    lite = (calls[name][0].get("checker") == "lite") if calls[name] else False
                    checker = (e.get("checker_cost_usd") or 0.0) + (lite and (
                        (e.get("checker_input_tokens") or 0) * (args.price_lite_input or 0)
                        + (e.get("checker_output_tokens") or 0) * (args.price_lite_output or 0)) / 1e6)
                    runs[run_dir.name][key] = {**e, "gemini_cost": gemini, "checker_cost": checker,
                                               "checks": calls[name], "trace": shard / name / "trace.json"}
    return runs, repeated, replies


def truncate(task, e, limit):
    """The episode as if it had ended at its limit-th empty close (None: the rule never fires)."""
    if task in PRESS_TASKS or not e["trace"].exists():
        return None
    trace = json.loads(e["trace"].read_text())
    count = 0
    for i, rec in enumerate(trace):
        count += sum(empty_close(r) for r in rec.get("results") or [])
        if count >= limit and i < len(trace) - 1:
            kept = trace[: i + 1]
            model = [r for r in kept if "source" not in r]
            relayed = [r for r in kept if r.get("source") == "relay"]
            checks = (sum(1 for r in relayed if r.get("checks"))
                      + sum(1 for r in kept if (r.get("relay_end") or {}).get("failed_check")))
            share = checks / e["checker_calls"] if e.get("checker_calls") else 0.0
            # the cut record's relay end is the original run's; the rule's run ends the episode there instead
            ends = Counter(r["relay_end"]["reason"] for r in kept[:-1] if r.get("relay_end"))
            return {**e, "success": False, "turns": rec["turn"], "model_calls": len(model),
                    "model_seconds": sum(r.get("latency_s") or 0 for r in model),
                    "seconds": e["seconds"] * rec["turn"] / max(e["turns"], 1),
                    "gemini_cost": e["gemini_cost"] * len(model) / max(e.get("model_calls") or 0, 1),
                    "relayed_steps": len(relayed), "checker_calls": checks, "relay_ends": dict(ends),
                    "checker_seconds": (e.get("checker_seconds") or 0) * share,
                    "checker_cost": e["checker_cost"] * share, "finished_reason": "empty_grasp_limit"}
    return None


def row(label, eps):
    n = len(eps)
    mean = lambda k: statistics.mean(e.get(k) or 0 for e in eps)
    ends = Counter()
    for e in eps:
        ends.update(e.get("relay_ends") or {})
    checks = sum(e.get("checker_calls") or 0 for e in eps)
    return (f"{label:30} {sum(e['success'] for e in eps):>3}/{n:<3} {mean('turns'):>6.1f} {mean('model_calls'):>6.1f} "
            f"{mean('relayed_steps'):>6.1f} {mean('model_seconds'):>8.0f} {mean('checker_seconds'):>7.1f} "
            f"{mean('seconds'):>7.0f} {mean('gemini_cost'):>7.3f} {mean('checker_cost'):>7.4f} "
            f"{checks:>6}  {dict(ends) if ends else ''}")


HEADER = (f"{'run':30} {'success':>7} {'turns':>6} {'calls':>6} {'relay':>6} {'model s':>8} {'check s':>7} "
          f"{'wall s':>7} {'$gem':>7} {'$check':>7} {'checks':>6}  relay ends")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--dev", action="store_true", help="development episodes (outputs/robodawn_dev)")
    parser.add_argument("--tier", default="flex")
    parser.add_argument("--runs", type=Path, nargs="+",
                        default=[PROJECT / "outputs", *sorted((PROJECT / "outputs/robodawn_colab").glob("*"))])
    parser.add_argument("--labels", nargs="+", help="only these runs (directory names)")
    parser.add_argument("--reference", help="run the others are compared with (default: the baseline)")
    parser.add_argument("--empty-grasp-limit", type=int, nargs="*", default=[])
    parser.add_argument("--price-input", type=float, default=0.75)
    parser.add_argument("--price-cached", type=float, default=0.075)
    parser.add_argument("--price-output", type=float, default=3.75)
    parser.add_argument("--price-lite-input", type=float)
    parser.add_argument("--price-lite-output", type=float)
    args = parser.parse_args()
    runs, repeated, replies = load([r for r in args.runs if r.is_dir()], args.dev, set(args.tasks), args)
    if args.labels:
        runs = {k: v for k, v in runs.items() if k in args.labels}
    if not runs:
        print("no finished episodes found")
        return
    common = sorted(set.intersection(*(set(v) for v in runs.values())))
    print("episodes per run: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(runs.items())))
    print(f"paired episodes (finished by every run): {len(common)} over "
          f"{len({t for t, _ in common})} tasks" + (f"; repeated copies ignored: {dict(repeated)}" if repeated else ""))
    if not common:
        return
    reference = args.reference or next((k for k in sorted(runs) if k.startswith(("gemini_flash_", "relay_baseline_"))),
                                       sorted(runs)[0])
    print("\nper episode means; model s = big-model time, $gem / $check = cost per episode in USD")
    print(HEADER)
    for label in sorted(runs, key=lambda k: (k != reference, k)):
        print(row(label, [runs[label][k] for k in common]))
    print("\nbig-model replies (all episodes of each run): cut off by the reply token budget / mean reasoning tokens")
    for label in sorted(runs):
        r = replies[label]
        if r["answered"]:
            print(f"  {label:30} {r['cut']}/{r['answered']} ({r['cut'] / r['answered']:.1%}), "
                  f"{r['reasoning'] / r['answered']:.0f}")
    print(f"\nsuccess vs {reference} on the paired episodes (sign test on the episodes that differ)")
    for label in sorted(runs):
        if label == reference:
            continue
        b = sum(runs[reference][k]["success"] and not runs[label][k]["success"] for k in common)
        c = sum(runs[label][k]["success"] and not runs[reference][k]["success"] for k in common)
        print(f"  {label:30} {c - b:+d} episodes (gained {c}, lost {b}, p = {sign_test(b, c):.2f})")
    print("\nper task successes")
    tasks = sorted({t for t, _ in common})
    print(f"  {'task':24} " + " ".join(f"{label[:22]:>22}" for label in sorted(runs)))
    for task in tasks:
        keys = [k for k in common if k[0] == task]
        print(f"  {task:24} " + " ".join(f"{sum(runs[l][k]['success'] for k in keys):>19}/{len(keys):<2}" for l in sorted(runs)))
    for limit in args.empty_grasp_limit:
        print(f"\nas if each episode had ended at its empty close number {limit}")
        print(HEADER)
        for label in sorted(runs, key=lambda k: (k != reference, k)):
            eps = [truncate(k[0], runs[label][k], limit) or runs[label][k] for k in common]
            print(row(label, eps))
    for label in sorted(runs):
        checks = [c for k in common for c in runs[label][k]["checks"]]
        if not checks:
            continue
        seconds = sorted(c["seconds"] for c in checks)
        print(f"\n{label} checker calls on the paired episodes: {len(checks)}, passed "
              f"{sum(1 for c in checks if c.get('passed')) / len(checks):.0%}, not reached "
              f"{sum(1 for c in checks if c.get('error'))}, answer missing {sum(1 for c in checks if c.get('note'))}, "
              f"median {statistics.median(seconds):.1f}s, 90th percentile {seconds[int(0.9 * (len(seconds) - 1))]:.1f}s")
        if checks[0].get("checker") == "lite" and args.price_lite_input is None:
            print("  lite checker cost not included: pass --price-lite-input and --price-lite-output")


if __name__ == "__main__":
    main()
