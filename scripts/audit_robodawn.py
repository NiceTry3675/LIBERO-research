#!/usr/bin/env python3
"""Check collected RoboDawn text/seeds/demos against a pinned public harness.

No simulator or model call is made. Later-turn memory is copied from Inspect;
this checks prompt formatting, not regeneration of the missing full history.
"""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROBODAWN_COMMIT = "9247f366cd31f278e10f2fbe5fe8469b5f1b5b94"
ROBOTWIN_COMMIT = "96c1feab536306b50c26af200044fcdf126e8904"
# Gemini is called on Vertex AI's native generateContent API with a service account
# key (Flex PayGo is not offered on the OpenAI-compatible endpoint). Upstream sends
# no reasoning fields for Gemini, so the model's default thinking is used.
ENDPOINT = "vertex-ai/generateContent/global"
REASONING = "model default (no reasoning fields sent)"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def parse_inspect(path: Path) -> tuple[str, dict, list]:
    prompt, rest = path.read_text().split("\n\nREPLY\n\n", 1)
    reply, execution = rest.split("\n\nEXECUTION\n\n", 1)
    return (prompt.removeprefix("PROMPT (sent with the images)\n\n"),
            json.loads(re.sub(r"^```json\s*|\s*```$", "", reply)), json.loads(execution))


def prompt_inputs(prompt: str) -> dict:
    parts = prompt.split("\n\n")
    instruction = parts[0].removeprefix("TASK: ")
    turn = int(re.fullmatch(r"TURN (\d+)\.", parts[1]).group(1))
    last_results = []
    state, facts = {}, {}
    memory_parts = []
    captions = []
    for part in parts[2:]:
        if part.startswith("RESULT OF YOUR LAST COMMANDS:\n"):
            for line in part.splitlines()[1:]:
                match = re.fullmatch(r"- (.*?): (ok|FAILED)(?: \((.*)\))?", line)
                last_results.append({"command": match[1], "ok": match[2]=="ok", "note": match[3] or ""})
        elif part.startswith("CURRENT STATE:\n"):
            for line in part.splitlines()[1:]:
                arm = re.fullmatch(r"(LEFT|RIGHT) gripper position \(fingertip centre\) = (\(.*?\)) cm, approach = (\(.*?\)), finger axis = (\(.*?\)), opening = (\S+)", line)
                if arm:
                    state[arm[1].lower()] = {"position_cm": ast.literal_eval(arm[2]),
                        "approach": ast.literal_eval(arm[3]), "finger_axis": ast.literal_eval(arm[4]),
                        "gripper_real": float(arm[5])}
                elif line.startswith("table top at z = "):
                    state["table_z_cm"] = float(line.split(" = ")[1])
                elif line.startswith("steps used: "):
                    used, limit = line.removeprefix("steps used: ").split(" / ")
                    state.update(steps_used=int(used), step_limit=int(limit))
                else:
                    grasp = re.match(r"(LEFT|RIGHT) arm closed on an object with the fingertips at z = (\S+) ", line)
                    if not grasp:
                        raise ValueError(f"Unrecognized state line: {line}")
                    facts[grasp[1].lower()] = {"grasp_z": ast.literal_eval(grasp[2])}
        elif part.startswith(("YOUR NOTES FROM PREVIOUS TURNS", "HISTORY OF YOUR COMMANDS AND THEIR OUTCOMES")):
            memory_parts.append(part)
        elif part.startswith("IMAGES ATTACHED (in order): "):
            # Captions contain semicolons internally; split only between camera names.
            captions = re.split(r"; (?=(?:agent|top|left|right|head)_camera \()", part.removeprefix("IMAGES ATTACHED (in order): "))
        elif part != "Reply with the JSON object.":
            raise ValueError(f"Unrecognized prompt section: {part[:80]}")
    return dict(turn=turn,instruction=instruction,state=state,last_results=last_results,
                memory_text="\n\n".join(memory_parts),image_captions=captions,grasp_facts=facts)


def audit(repo: Path, site: Path) -> tuple[dict, dict]:
    import yaml
    sys.path.insert(0, str(repo))
    from harness.agent.demos import build_demo, demo_dirs_for_task, load_saved_demo
    from harness.agent.memory import AgentMemory
    from harness.agent.prompts import system_prompt, turn_text

    commit = git(repo,"rev-parse","HEAD")
    if commit != ROBODAWN_COMMIT:
        raise ValueError(f"RoboDawn commit differs: {commit}")
    rt_commit = git(repo,"ls-tree","HEAD","RoboTwin").split()[2]
    if rt_commit != ROBOTWIN_COMMIT:
        raise ValueError(f"RoboTwin pin differs: {rt_commit}")
    tasks = list(csv.DictReader((site/"tasks.csv").open()))
    episodes = list(csv.DictReader((site/"episodes.csv").open()))
    if len(tasks)!=50 or len(episodes)!=500 or sum(int(t["success"]) for t in tasks)!=311:
        raise ValueError("Collected record count or success total differs")
    bank = repo/"demos/robotwin2/expert"
    primer = build_demo(repo/"demos/robotwin2/primer")
    checks, manifest_tasks = [], []
    for task in tasks:
        name = task["task"]
        seed_file = repo/f"harness/valid_seeds/{name}__demo_randomized__seed0.json"
        seeds = json.loads(seed_file.read_text())["seeds"][:10]
        dirs = demo_dirs_for_task(bank,name)
        demos = [load_saved_demo(d) for d in dirs]
        rows = sorted((e for e in episodes if e["task"]==name),key=lambda e:int(e["episode"]))
        if [int(e["episode"]) for e in rows]!=list(range(10)):
            raise ValueError(f"Duplicate or missing episode indices: {name}")
        if seeds != [int(e["seed"]) for e in rows] or seeds != [int(e["expected_seed"]) for e in rows]:
            raise ValueError(f"Seed mismatch: {name}")
        if sum(e["outcome"]=="success" for e in rows)!=int(task["success"]):
            raise ValueError(f"Success count mismatch: {name}")
        mapped = []
        for row in rows:
            entry = int(row["demo_entry"])
            if not 1<=entry<=len(demos):
                raise ValueError(f"Missing demo entry {entry}: {name}")
            demo = demos[entry-1]
            mapped.append({"episode":int(row["episode"]),"seed":int(row["seed"]),
                "demo_entry":entry,"demo_path":dirs[entry-1].relative_to(repo).as_posix(),
                "demo_seed":demo.source.get("seed"),"demo_side_key":demo.side_key(),
                "request_images":primer.n_images()+demo.n_images()+4,
                "site_turns":int(row["turns"]),"site_outcome":row["outcome"]})
        manifest_tasks.append({"task":name,"site_success":int(task["success"]),"episodes":mapped})
    for path in sorted((site/"inspect").glob("*__turn*.md")):
        captured, reply, execution = parse_inspect(path)
        inputs = prompt_inputs(captured)
        rendered = turn_text(**inputs)
        checks.append({"file":path.relative_to(site).as_posix(),"exact_match":rendered==captured,
            "initial_memory_matches":inputs["memory_text"]==AgentMemory().render() if inputs["turn"]==1 else None,
            "diff":list(difflib.unified_diff(captured.splitlines(),rendered.splitlines())) if rendered!=captured else []})
    required = {f"inspect/{task}__ep{ep}__turn{turn}.md" for task,ep,turns in
                [("adjust_bottle",0,[1,5,10]),("adjust_bottle",3,[1,11,21]),("handover_block",0,[1,12,20,24])]
                for turn in turns}
    if not required.issubset({check["file"] for check in checks}):
        raise ValueError("Required collected Inspect files are missing")
    profile = yaml.safe_load((repo/"harness/configs/robotwin2_profile.yaml").read_text())
    system_checks = [{"file":filename,"table_z":z,"exact_match":
        system_prompt(profile,4,True,{"table_z":z})==(site/"inspect"/filename).read_text()}
        for filename,z in [("system_prompt.md",74),("system_prompt__handover_block.md",71)]]
    report = {"robodawn_commit":commit,"robotwin_commit":rt_commit,
        "checks":{"seed_pairs":500,"demo_entries":500,"demo_images_loaded":True,
                  "all_user_prompts_exact":all(c["exact_match"] for c in checks),
                  "all_system_prompts_exact":all(c["exact_match"] for c in system_checks)},
        "prompt_checks":checks,"system_checks":system_checks,
        "request_image_range":[min(e["request_images"] for t in manifest_tasks for e in t["episodes"]),
                               max(e["request_images"] for t in manifest_tasks for e in t["episodes"])],
        "limits":["Later-turn memory and image captions were reused from Inspect; full history and camera rendering were not regenerated.",
                  "Demo directories and image bytes match the shipped bank. Selecting the entry from initial simulator object positions still requires a GPU run.",
                  "Upstream sends no reasoning fields for Gemini and relies on the default thinking; the original gateway and its reasoning token counts are unpublished."],
        "local_vlm_difference":{"role":"Clef offline comparison, not the RoboDawn baseline",
            "different_fields":["system prompt","question/choice user prompt","reply schema","reasoning effort","reply token limit","task demo and command primer context"]}}
    manifest = {"robodawn_commit":commit,"robotwin_commit":rt_commit,
        "task_config":"demo_randomized","seed_base":0,"instruction_type":"unseen",
        "max_turns":45,"max_commands_per_turn":4,"max_tokens":8000,
        "model":"google/gemini-3.8-flash","endpoint":ENDPOINT,
        "reasoning":REASONING,"cameras":["agent_camera","top_camera","wrist"],
        "smoke_tasks":["adjust_bottle","place_empty_cup"],"tasks":manifest_tasks}
    return report,manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=PROJECT/"third_party/robodawn")
    parser.add_argument("--site",type=Path,default=PROJECT/"robodawn_site")
    args = parser.parse_args()
    report,manifest = audit(args.repo.resolve(),args.site.resolve())
    for filename,value in [("harness_audit.json",report),("reproduction_manifest.json",manifest)]:
        (args.site/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report["checks"],ensure_ascii=False))
    if not report["checks"]["all_user_prompts_exact"] or not report["checks"]["all_system_prompts_exact"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
