#!/usr/bin/env python3
"""Run a relay variant of RoboDawn's harness (scripts/robodawn_relay.py) on Gemini through Vertex AI.

Everything but the agent loop is the baseline's (run_robodawn_baseline.py): the same model, client,
tier, reasoning, demonstrations, scenes, cameras and limits. The variant decides what happens to the
big model's planned steps:

    baseline  upstream loop and prompt, through this runner (for development episodes and loop checks)
    prompt    the reply format asks for `expect` and `next`, but only `commands` run
    open      `next` runs without the big model, stopped by code rules only
    checked   `next` runs step by step while the fast checker (--checker clef|lite) confirms `expect`

Episodes 0-9 are the evaluation episodes: the baseline's seeds and demonstrations, compared with the
baseline by compare_relay.py. Episodes 10-49 are development episodes for tuning the prompt and the
checker; they go to outputs/robodawn_dev so they never mix with evaluation results. A development
episode whose seed is the source seed of one of the task's demonstrations is refused (those sit at
episodes 12-23 for the subset tasks; episodes 24-49 are free of them).

    python scripts/run_robodawn_relay.py --task place_empty_cup --variant checked --checker clef \\
        --start-episode 24 --episodes 5

The result directory is relay_<variant>[_<checker>][_t<threshold>][_s<max steps>][_g<empty grasp
limit>][_m<reply tokens>]_<tier>, the optional parts only when they differ from the defaults, so runs
with different settings never merge. --max-tokens raises the reply budget (default 8000, the baseline's):
the relay prompt makes Gemini think longer, and a reply cut off by the budget loses its turn.

The Clef checker reads the OpenRouter key from --openrouter-key-file (default /content/.openrouter_key on
the VM, which subset.sh setup uploads), else from OPENROUTER_API_KEY in the project .env; the lite
checker uses the Vertex service account key. Keys are never logged. Use --dry-run to inspect the
configuration without a GPU, key or model call.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path

from audit_robodawn import ENDPOINT, PROJECT, REASONING, ROBODAWN_COMMIT, ROBOTWIN_COMMIT, git
from run_robodawn_baseline import MODEL, TEMPERATURE, TIERS, api_base, credentials_project, vertex_client

VARIANTS = ("baseline", "prompt", "open", "checked")
CHECKERS = {"clef": "cloudflare/clef", "lite": "google/gemini-3.1-flash-lite (minimal thinking)"}
EVAL_EPISODES = 10      # episodes 0-9 are the evaluation episodes with site results
SEEDS = 50              # valid seeds shipped per task


DEFAULTS = {"threshold": 0.5, "max_steps": 3, "empty_grasp_limit": 0, "max_tokens": 8000}


def run_name(variant: str, checker: str | None, tier: str, threshold: float = 0.5, max_steps: int = 3,
             empty_grasp_limit: int = 0, max_tokens: int = 8000) -> str:
    return (f"relay_{variant}" + (f"_{checker}" if variant == "checked" else "")
            + (f"_t{threshold:g}" if variant == "checked" and threshold != DEFAULTS["threshold"] else "")
            + (f"_s{max_steps}" if max_steps != DEFAULTS["max_steps"] else "")
            + (f"_g{empty_grasp_limit}" if empty_grasp_limit != DEFAULTS["empty_grasp_limit"] else "")
            + (f"_m{max_tokens}" if max_tokens != DEFAULTS["max_tokens"] else "")
            + f"_{tier}")


def demo_seeds(repo: Path, task: str) -> set[int]:
    """Source seeds of the task's demonstrations (every bank entry)."""
    bank = repo/"demos/robotwin2/expert"
    return {json.loads((d/"demo.json").read_text())["source"]["seed"]
            for d in [bank/task, *bank.glob(f"{task}+*")] if (d/"demo.json").is_file()}


def episode_plan(args) -> tuple[list[dict], bool]:
    """The episodes to run with their seeds (and, for evaluation episodes, the manifest's demonstration)."""
    first, last = args.start_episode, args.start_episode + args.episodes - 1
    if args.episodes < 1 or first < 0 or last >= SEEDS:
        raise ValueError(f"choose episodes within 0..{SEEDS - 1}")
    evaluation = first < EVAL_EPISODES
    if evaluation and last >= EVAL_EPISODES:
        raise ValueError(f"a run is either evaluation episodes (0..{EVAL_EPISODES - 1}) or development episodes, not both")
    repo = args.repo.resolve()
    seeds = json.loads((repo/f"harness/valid_seeds/{args.task}__demo_randomized__seed0.json").read_text())["seeds"]
    if not evaluation:
        shown = demo_seeds(repo, args.task)
        overlap = [k for k in range(first, last + 1) if seeds[k] in shown]
        if overlap:
            raise ValueError(f"development episodes {overlap} use a demonstration's source seed; choose others (24-49 are free)")
        return [{"episode": k, "seed": seeds[k]} for k in range(first, last + 1)], False
    manifest = json.loads(args.manifest.read_text())
    if manifest["robodawn_commit"] != ROBODAWN_COMMIT or manifest["robotwin_commit"] != ROBOTWIN_COMMIT:
        raise ValueError("Manifest commit differs from the audited commits")
    task = next((t for t in manifest["tasks"] if t["task"] == args.task), None)
    if task is None:
        raise ValueError("Evaluation episodes need a collected task")
    selected = task["episodes"][first:last + 1]
    for episode in selected:
        if episode["seed"] != seeds[episode["episode"]]:
            raise ValueError("Shipped seed differs from manifest")
    return selected, True


def configuration(args) -> tuple[dict, list[str]]:
    repo = args.repo.resolve()
    if git(repo, "rev-parse", "HEAD") != ROBODAWN_COMMIT:
        raise ValueError("RoboDawn checkout differs from the audited commit")
    if git(repo, "diff", "--name-only", "HEAD", "--", "harness", "demos/robotwin2"):
        raise ValueError("Tracked harness or demonstration files were modified")
    if args.variant == "checked" and not args.checker:
        raise ValueError("--variant checked needs --checker clef or lite")
    if args.variant != "checked" and args.checker:
        raise ValueError("--checker applies to --variant checked only")
    episodes, evaluation = episode_plan(args)
    if any(not (repo/f"demos/robotwin2/primer/turn{n:03d}_agent_camera.png").is_file() for n in range(1, 7)):
        raise ValueError("Command primer images are missing")
    name = run_name(args.variant, args.checker, args.tier, args.threshold, args.max_steps, args.empty_grasp_limit,
                    args.max_tokens)
    output = (args.output.resolve() if args.output else
              PROJECT/("outputs/robodawn" if evaluation else "outputs/robodawn_dev")/name/args.task/f"shard_{args.start_episode}")
    relay = {"variant": args.variant, "checker": args.checker, "checker_model": CHECKERS.get(args.checker),
             "threshold": args.threshold if args.variant == "checked" else None, "max_steps": args.max_steps,
             "empty_grasp_limit": args.empty_grasp_limit, "turns_count_relayed_steps": True}
    config = {"robodawn_commit": ROBODAWN_COMMIT, "robotwin_commit": ROBOTWIN_COMMIT, "task": args.task,
              "episodes": episodes, "evaluation": evaluation, "output": str(output), "model": MODEL,
              "endpoint": ENDPOINT, "tier": args.tier, "timeout_s": TIERS[args.tier]["timeout_s"],
              "stall_timeout_s": TIERS[args.tier]["stall_timeout_s"], "reasoning": REASONING,
              "max_turns": 45, "max_commands_per_turn": 4, "max_tokens": args.max_tokens,
              "task_config": "demo_randomized", "instruction_type": "unseen", "seed_base": 0,
              "cameras": ["agent_camera", "top_camera", "wrist"], "temperature": TEMPERATURE,
              "relay": relay, "robotwin_root": str(args.robotwin_root.resolve())}
    config_path = output/"relay_config.json"
    if config_path.is_file():
        prior = json.loads(config_path.read_text())
        keys = ("robodawn_commit", "robotwin_commit", "task", "model", "endpoint", "tier", "reasoning", "max_turns",
                "max_commands_per_turn", "max_tokens", "task_config", "instruction_type", "cameras", "relay")
        if any(prior.get(k) != config[k] for k in keys):
            raise ValueError("Existing output uses different conditions; choose another output")
    elif (output/"results.json").exists():
        raise ValueError("Existing output lacks relay metadata; choose another output")
    flags = ["--task", args.task, "--episodes", str(args.episodes), "--start_episode", str(args.start_episode),
             "--seed", "0", "--task_config", "demo_randomized", "--instruction_type", "unseen",
             "--model", MODEL, "--max_turns", "45", "--max_commands_per_turn", "4",
             "--max_tokens", str(args.max_tokens), "--timeout_s", str(TIERS[args.tier]["timeout_s"]),
             "--stall_timeout", str(TIERS[args.tier]["stall_timeout_s"]),
             "--cameras", "agent_camera,top_camera,wrist", "--label", name, "--output", str(output)]
    if args.no_video:
        flags += ["--no_video"]
    return config, flags


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def check_results(config: dict) -> dict:
    """The baseline's condition checks (seed, demonstration entry, model, traffic, reasoning) on a relay run."""
    output = Path(config["output"])
    results = json.loads((output/"results.json").read_text())
    errors = []
    if results.get("task") != config["task"] or results.get("model") != MODEL or results.get("task_config") != "demo_randomized":
        errors.append("Result task/model/scene configuration differs")
    for expected in config["episodes"]:
        actual = next((e for e in results["episodes"] if e["episode_index"] == expected["episode"]), None)
        if actual is None or actual.get("seed") != expected["seed"]:
            errors.append(f"episode {expected['episode']}: missing or different seed")
            continue
        if actual.get("variant") != config["relay"]["variant"]:
            errors.append(f"episode {expected['episode']}: ran another variant")
        if "demo_path" in expected:
            trace = json.loads((output/f"episode_{expected['episode']:03d}/trace.json").read_text())
            shown = trace[0].get("demo", []) if trace else []
            if [Path(d).name for d in shown] != [Path(expected["demo_path"]).name]:
                errors.append(f"episode {expected['episode']}: different demonstration entry")
    calls = read_jsonl(output/"llm_calls.jsonl")
    reasoning = [(c.get("usage", {}).get("completion_tokens_details") or {}).get("reasoning_tokens") for c in calls]
    if not any(isinstance(n, (int, float)) and n > 0 for n in reasoning):
        errors.append("No positive reasoning token usage was reported; reasoning is unverified")
    answered = [t for t in read_jsonl(output/"transport.jsonl") if t["status"] == 200]
    if not answered or any(not str(t.get("response_model")).startswith(MODEL) for t in answered):
        errors.append("Returned model is missing or differs from the requested Vertex model")
    if any(t.get("traffic_type") != TIERS[config["tier"]]["traffic"] for t in answered):
        errors.append(f"Some requests were not served as {TIERS[config['tier']]['traffic']} traffic")
    if any(t.get("reasoning_fields") for t in read_jsonl(output/"transport.jsonl")):
        errors.append("Reasoning fields were sent; the runs use the model default")
    checks = read_jsonl(output/"checker_calls.jsonl")
    failed = sum(1 for c in checks if c.get("error"))
    if failed > len(checks) / 2:
        errors.append(f"The checker could not be reached in {failed} of {len(checks)} calls")
    report = {"task": config["task"], "variant": config["relay"]["variant"], "errors": errors,
              "checker_calls": len(checks), "checker_errors": failed,
              "run_success": sum(e["success"] for e in results["episodes"]
                                 if e["episode_index"] in {x["episode"] for x in config["episodes"]})}
    (output/"condition_check.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def openrouter_key(path: Path | None) -> str | None:
    """The OpenRouter key from a file (the VM) or the project .env (this machine); never printed."""
    key = None
    if path and path.is_file():
        key = path.read_text().strip()
    elif (PROJECT/".env").is_file():
        for line in (PROJECT/".env").read_text().splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
                break
    if key and not key.isprintable() or key and any(c.isspace() for c in key):
        raise ValueError("the OpenRouter key holds whitespace or control characters (two keys?); fix the key file")
    return key or None


def preflight(checker) -> None:
    """One question on a plain red image, so a bad key or model stops the run before any episode."""
    from PIL import Image
    verdict = checker.check(["Is this image mostly red?"], "A test of the checker.", [Image.new("RGB", (64, 48), (220, 20, 20))])
    if verdict["error"]:
        raise RuntimeError(f"the {checker.name} checker does not answer: {verdict['error']}")
    print(f"checker {checker.name} answers: P(yes) {verdict['p_yes']} in {verdict['seconds']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--checker", choices=sorted(CHECKERS))
    parser.add_argument("--threshold", type=float, default=0.5, help="P(yes) every expectation needs (checked)")
    parser.add_argument("--max-steps", type=int, default=3, help="planned steps accepted per reply")
    parser.add_argument("--empty-grasp-limit", type=int, default=0,
                        help="end an episode at this many empty closes (0: off; its effect can also be read off "
                             "the traces by truncation, compare_relay.py --empty-grasp-limit)")
    parser.add_argument("--max-tokens", type=int, default=8000, help="reply budget per call, reasoning included")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--repo", type=Path, default=PROJECT/"third_party/robodawn")
    parser.add_argument("--manifest", type=Path, default=PROJECT/"robodawn_site/reproduction_manifest.json")
    parser.add_argument("--robotwin-root", type=Path, default=Path(os.environ.get("ROBOTWIN_ROOT", "/content/RoboTwin")))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--credentials", type=Path, default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    parser.add_argument("--openrouter-key-file", type=Path, default=Path("/content/.openrouter_key"))
    parser.add_argument("--tier", choices=sorted(TIERS), default="flex")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, flags = configuration(args)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    if git(args.robotwin_root.resolve(), "rev-parse", "HEAD") != ROBOTWIN_COMMIT:
        raise ValueError("RoboTwin checkout differs from the pinned benchmark commit")
    if not (args.robotwin_root/"assets/background_texture").is_dir():
        raise ValueError("Randomized evaluation needs background textures; bootstrap with --textures")
    if not args.credentials or not Path(args.credentials).is_file():
        raise ValueError("Pass --credentials or set GOOGLE_APPLICATION_CREDENTIALS to the service account key")
    credentials = Path(args.credentials).resolve()
    os.environ["ROBOTWIN_ROOT"] = str(args.robotwin_root.resolve())
    sys.path.insert(0, str(args.repo.resolve()))
    from harness import run_robotwin_eval
    import robodawn_relay as relay
    checker = (relay.make_checker(args.checker, args.threshold, credentials, openrouter_key(args.openrouter_key_file))
               if args.variant == "checked" else None)
    if checker:
        preflight(checker)
    run_robotwin_eval.ChatClient = vertex_client(run_robotwin_eval.ChatClient, credentials, args.tier)
    run_robotwin_eval.MLLMDiscreteAgent = functools.partial(
        relay.RelayAgent, variant=args.variant, checker=checker, max_steps=args.max_steps,
        empty_grasp_limit=args.empty_grasp_limit)
    Path(config["output"]).mkdir(parents=True, exist_ok=True)
    (Path(config["output"])/"relay_config.json").write_text(json.dumps(config, indent=2)+"\n")
    sys.argv = [str(args.repo/"harness/run_robotwin_eval.py"), *flags,
                "--api_base", api_base(credentials_project(credentials))]
    run_robotwin_eval.main()
    report = check_results(config)
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
