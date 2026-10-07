#!/usr/bin/env python3
"""Compare our RoboDawn baseline runs with the collected site results.

Per task (all its shards merged): successes on the episodes run so far against the site's on the same
episodes, episode-level agreement, mean turns, how episodes ended. Overall: the
difference in success rate with a 95% interval, the token use (fresh input,
cached input, output and its reasoning share), replies cut off by the token budget, the
traffic type and the median call time, and a rough cost.

    python scripts/compare_robodawn.py [--tier flex] [--runs DIR ...] [task ...]

Works on the VM and, after fetching, on this machine; --runs takes several
result directories (one per VM) and merges them. Prices are USD per 1M
tokens for Gemini 3.8 Flash standard on Vertex as of 2026-10 (flex: half);
pass --price-* when they change.
"""
import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def site_episodes(path):
    site = {}
    for row in csv.DictReader(open(path)):
        site[(row["task"], int(row["episode"]))] = (row["outcome"] == "success", int(row["turns"]))
    return site


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--tier", default="flex")
    parser.add_argument("--runs", type=Path, nargs="+", default=[PROJECT / "outputs/robodawn"])
    parser.add_argument("--site", type=Path, default=PROJECT / "robodawn_site/episodes.csv")
    parser.add_argument("--price-input", type=float, default=0.75)
    parser.add_argument("--price-cached", type=float, default=0.075)
    parser.add_argument("--price-output", type=float, default=3.75)
    args = parser.parse_args()
    site = site_episodes(args.site)
    bases = [runs / f"gemini_flash_{args.tier}" for runs in args.runs]
    tasks = args.tasks or sorted({p.name for base in bases if base.is_dir() for p in base.iterdir()
                                  if list(p.glob("shard_*/results.json"))})

    print(f"{'task':24} {'run':>6} {'site':>6} {'agree':>6} {'turns run/site':>15}  ended")
    pairs, calls, repeated = [], [], 0
    for task in tasks:
        by_index = {}
        for out in [out for base in bases for out in sorted((base / task).glob("shard_*"))]:
            if (out / "results.json").exists():
                for e in json.loads((out / "results.json").read_text())["episodes"]:
                    repeated += e["episode_index"] in by_index
                    by_index.setdefault(e["episode_index"], e)   # an episode counts once even if two shards ran it
            calls += read_jsonl(out / "transport.jsonl")
        eps = [by_index[k] for k in sorted(by_index)]
        mine = [(bool(e["success"]), site[(task, e["episode_index"])]) for e in eps]
        pairs += mine
        if not eps:
            print(f"{task:24} {'-':>6}")
            continue
        run_s = sum(s for s, _ in mine)
        site_s = sum(o[0] for _, o in mine)
        agree = sum(s == o[0] for s, o in mine)
        turns = f"{statistics.mean(e['turns'] for e in eps):.0f}/{statistics.mean(o[1] for _, o in mine):.0f}"
        ended = dict(Counter(e["finished_reason"] for e in eps))
        print(f"{task:24} {run_s:>3}/{len(eps):<2} {site_s:>3}/{len(eps):<2} {agree:>3}/{len(eps):<2} {turns:>15}  {ended}")

    n = len(pairs)
    if not n:
        print("no finished episodes yet")
        return
    p_run = sum(s for s, _ in pairs) / n
    p_site = sum(o[0] for _, o in pairs) / n
    # Both rates are measured on the same episodes, but the model's runs are noisy, so the two are
    # treated as independent samples: a conservative interval for the difference.
    half = 1.96 * math.sqrt(p_run * (1 - p_run) / n + p_site * (1 - p_site) / n)
    diff = p_run - p_site
    print(f"\n{n} episodes: run {p_run:.1%}, site {p_site:.1%}, difference {diff:+.1%} "
          f"(95% interval {diff - half:+.1%} to {diff + half:+.1%}) -> "
          f"{'consistent with the site' if abs(diff) <= half else 'differs from the site'}")
    print(f"episode agreement {sum(s == o[0] for s, o in pairs) / n:.1%}")
    if repeated:
        print(f"{repeated} episodes were run more than once; the first copy (first --runs directory) counts")

    answered = [c for c in calls if c.get("status") == 200]
    usage = [c.get("usage") or {} for c in answered]
    prompt = sum(u.get("prompt_tokens", 0) for u in usage)
    cached = sum((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) for u in usage)
    reasoning = sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in usage)
    output = sum(u.get("completion_tokens", 0) for u in usage)   # includes the reasoning
    scale = 0.5 if args.tier == "flex" else 1.0
    cost = scale * ((prompt - cached) * args.price_input + cached * args.price_cached
                    + output * args.price_output) / 1e6
    print(f"\ncalls {len(calls)} (answered {len(answered)}, failed attempts {len(calls) - len(answered)}), "
          f"traffic {dict(Counter(c.get('traffic_type') for c in answered))}")
    print(f"tokens: input {prompt / 1e6:.1f}M (cached {cached / max(prompt, 1):.0%}), "
          f"output {output / 1e6:.2f}M (reasoning {reasoning / 1e6:.2f}M, reply {(output - reasoning) / 1e6:.2f}M)")
    print(f"cut off by the token budget: {sum(c.get('finish_reason') == 'max_tokens' for c in answered)} replies")
    if answered:
        seconds = sorted(c["seconds"] for c in answered)
        print(f"call time: median {statistics.median(seconds):.1f}s, 90th percentile {seconds[int(0.9 * (len(seconds) - 1))]:.1f}s")
    print(f"rough cost ({args.tier}): ${cost:.2f}; check the GCP bill for the real amount")


if __name__ == "__main__":
    main()
