#!/usr/bin/env python3
"""Run the pinned RoboDawn harness through OpenRouter's Vertex endpoint.

Use --dry-run to inspect the configuration without a GPU, API key or model call.
Simulation, prompts, demo selection and episode termination remain upstream.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from audit_robodawn import PROJECT, ROBODAWN_COMMIT, ROBOTWIN_COMMIT, git

MODEL = "google/gemini-3.8-flash"
API_BASE = "https://openrouter.ai/api/v1"
PROVIDER = {"only":["google-vertex"],"allow_fallbacks":False}


def openrouter_client(base):
    """Adapt transport fields and record metadata; never log credentials."""
    class VertexClient(base):
        def _body(self,messages):
            body = super()._body(messages)
            body["provider"] = {"only":["google-vertex"],"allow_fallbacks":False}
            body["reasoning"] = {"enabled":True}
            return body

        def _post(self,body,tag=""):
            started = time.monotonic()
            status,payload = super()._post(body,tag)
            if self.log_path:
                raw = payload if isinstance(payload,dict) else {}
                entry = {"tag":tag,"status":status,"seconds":round(time.monotonic()-started,3),
                    "model":body["model"],"requested_provider":body["provider"],
                    "requested_reasoning":body["reasoning"],"response_provider":raw.get("provider"),
                    "response_id":raw.get("id"),"usage":raw.get("usage"),
                    "finish_reason":next((c.get("finish_reason") for c in raw.get("choices",[])),None)}
                self.log_path.parent.mkdir(parents=True,exist_ok=True)
                with (self.log_path.parent/"transport.jsonl").open("a") as f:
                    f.write(json.dumps(entry)+"\n")
            return status,payload
    return VertexClient


def configuration(args) -> tuple[dict,list[str]]:
    repo = args.repo.resolve()
    if git(repo,"rev-parse","HEAD")!=ROBODAWN_COMMIT:
        raise ValueError("RoboDawn checkout differs from the audited commit")
    if git(repo,"diff","--name-only","HEAD","--","harness","demos/robotwin2"):
        raise ValueError("Tracked harness or demonstration files were modified")
    manifest = json.loads(args.manifest.read_text())
    if manifest["robodawn_commit"]!=ROBODAWN_COMMIT or manifest["robotwin_commit"]!=ROBOTWIN_COMMIT:
        raise ValueError("Manifest commit differs from the audited commits")
    task = next((t for t in manifest["tasks"] if t["task"]==args.task),None)
    if task is None or args.episodes<1 or args.start_episode<0 or args.start_episode+args.episodes>10:
        raise ValueError("Choose a collected task and an episode range within 0..9")
    selected = task["episodes"][args.start_episode:args.start_episode+args.episodes]
    seeds = json.loads((repo/f"harness/valid_seeds/{args.task}__demo_randomized__seed0.json").read_text())["seeds"]
    for episode in selected:
        if episode["seed"]!=seeds[episode["episode"]]:
            raise ValueError("Shipped seed differs from manifest")
        demo_dir = repo/episode["demo_path"]
        demo = json.loads((demo_dir/"demo.json").read_text())
        if demo["source"]["seed"]!=episode["demo_seed"]:
            raise ValueError("Demonstration source seed differs from manifest")
        if any(not (demo_dir/name).is_file() for frame in demo["frames"] for name in frame["images"]):
            raise ValueError("Demonstration images are missing; fetch demos/robotwin2")
    if any(not (repo/f"demos/robotwin2/primer/turn{n:03d}_agent_camera.png").is_file() for n in range(1,7)):
        raise ValueError("Command primer images are missing")
    output = args.output.resolve() if args.output else PROJECT/"outputs/robodawn/gemini_flash"/args.task/"shard_0"
    config_path = output/"reproduction_config.json"
    if config_path.is_file():
        prior = json.loads(config_path.read_text())
        expected = {"robodawn_commit":ROBODAWN_COMMIT,"robotwin_commit":ROBOTWIN_COMMIT,
                    "task":args.task,"model":MODEL,"api_base":API_BASE,"provider":PROVIDER,
                    "reasoning":{"enabled":True},"max_turns":45,"max_commands_per_turn":4,
                    "max_tokens":8000,"task_config":"demo_randomized","instruction_type":"unseen",
                    "cameras":["agent_camera","top_camera","wrist"],"temperature_requested":0.0}
        if any(prior.get(k)!=v for k,v in expected.items()):
            raise ValueError("Existing output uses different reproduction conditions; choose another output")
    elif (output/"results.json").exists():
        raise ValueError("Existing output lacks reproduction metadata; choose another output")
    flags = ["--task",args.task,"--episodes",str(args.episodes),"--start_episode",str(args.start_episode),
        "--seed","0","--task_config","demo_randomized","--instruction_type","unseen",
        "--model",MODEL,"--api_base",API_BASE,"--max_turns","45","--max_commands_per_turn","4",
        "--max_tokens","8000","--cameras","agent_camera,top_camera,wrist","--output",str(output)]
    if args.api_key_file:
        flags += ["--api_key_file",str(args.api_key_file.resolve())]
    if args.no_video:
        flags += ["--no_video"]
    config = {"robodawn_commit":ROBODAWN_COMMIT,"robotwin_commit":ROBOTWIN_COMMIT,
        "task":args.task,"episodes":selected,"output":str(output),"model":MODEL,
        "api_base":API_BASE,"provider":PROVIDER,"reasoning":{"enabled":True},
        "max_turns":45,"max_commands_per_turn":4,"max_tokens":8000,
        "task_config":"demo_randomized","instruction_type":"unseen","seed_base":0,
        "cameras":["agent_camera","top_camera","wrist"],"temperature_requested":0.0,
        "robotwin_root":str(args.robotwin_root.resolve()),
        "limitations":["Original gateway reasoning effort is unknown. OpenRouter reasoning.enabled uses its default.",
            "Vertex metadata does not list temperature; the upstream temperature=0 may not be honored by this endpoint."]}
    return config,flags


def check_results(config: dict) -> dict:
    output = Path(config["output"])
    results = json.loads((output/"results.json").read_text())
    errors = []
    if results.get("task")!=config["task"] or results.get("model")!=MODEL or results.get("task_config")!="demo_randomized":
        errors.append("Result task/model/scene configuration differs")
    for expected in config["episodes"]:
        actual = next((e for e in results["episodes"] if e["episode_index"]==expected["episode"]),None)
        if actual is None or actual.get("seed")!=expected["seed"]:
            errors.append(f"episode {expected['episode']}: missing or different seed")
            continue
        trace_path = output/f"episode_{expected['episode']:03d}/trace.json"
        trace = json.loads(trace_path.read_text())
        shown = trace[0].get("demo",[]) if trace else []
        if [Path(d).name for d in shown]!=[Path(expected["demo_path"]).name]:
            errors.append(f"episode {expected['episode']}: different demonstration entry")
    calls = [json.loads(line) for line in (output/"llm_calls.jsonl").read_text().splitlines() if line.strip()]
    token_counts = [(call.get("usage",{}).get("completion_tokens_details") or {}).get("reasoning_tokens") for call in calls]
    if not any(isinstance(n,(int,float)) and n>0 for n in token_counts):
        errors.append("No positive reasoning token usage was reported; reasoning is unverified")
    transport_path = output/"transport.jsonl"
    transport = [json.loads(line) for line in transport_path.read_text().splitlines() if line.strip()] if transport_path.exists() else []
    answered = [t for t in transport if t["status"]==200]
    if not answered or any(t.get("response_provider")!="Google" for t in answered):
        errors.append("Returned provider is missing or differs from the audited Google Vertex provider")
    report = {"task":config["task"],"errors":errors,"reasoning_tokens":token_counts,
        "response_providers":sorted({str(t.get("response_provider")) for t in answered}),
        "site_success":sum(e["site_outcome"]=="success" for e in config["episodes"]),
        "run_success":sum(e["success"] for e in results["episodes"] if e["episode_index"] in {x["episode"] for x in config["episodes"]}),
        "reproduction_validated":False,
        "note":"A smoke run checks conditions. Success-rate reproduction needs the chosen tasks' full ten episodes."}
    (output/"condition_check.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",required=True)
    parser.add_argument("--episodes",type=int,default=1)
    parser.add_argument("--start-episode",type=int,default=0)
    parser.add_argument("--repo",type=Path,default=PROJECT/"third_party/robodawn")
    parser.add_argument("--manifest",type=Path,default=PROJECT/"robodawn_site/reproduction_manifest.json")
    parser.add_argument("--robotwin-root",type=Path,default=Path(os.environ.get("ROBOTWIN_ROOT","/content/RoboTwin")))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--api-key-file",type=Path)
    parser.add_argument("--no-video",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args()
    config,flags = configuration(args)
    if args.dry_run:
        print(json.dumps(config,ensure_ascii=False,indent=2))
        return
    if git(args.robotwin_root.resolve(),"rev-parse","HEAD")!=ROBOTWIN_COMMIT:
        raise ValueError("RoboTwin checkout differs from the pinned benchmark commit")
    if not (args.robotwin_root/"assets/background_texture").is_dir():
        raise ValueError("Randomized evaluation needs background textures; bootstrap with --textures")
    if not args.api_key_file and not os.environ.get("LLM_API_KEY"):
        if os.environ.get("OPENROUTER_API_KEY"):
            os.environ["LLM_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        else:
            raise ValueError("Set OPENROUTER_API_KEY/LLM_API_KEY or pass --api-key-file")
    os.environ["ROBOTWIN_ROOT"] = str(args.robotwin_root.resolve())
    sys.path.insert(0,str(args.repo.resolve()))
    from harness import run_robotwin_eval
    run_robotwin_eval.ChatClient = openrouter_client(run_robotwin_eval.ChatClient)
    Path(config["output"]).mkdir(parents=True,exist_ok=True)
    (Path(config["output"])/"reproduction_config.json").write_text(json.dumps(config,indent=2)+"\n")
    sys.argv = [str(args.repo/"harness/run_robotwin_eval.py"),*flags]
    run_robotwin_eval.main()
    report = check_results(config)
    print(json.dumps(report,indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
