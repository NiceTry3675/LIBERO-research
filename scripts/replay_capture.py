#!/usr/bin/env python3
"""Replay recorded RoboDawn episodes in RoboTwin and capture depth, segmentation and true object states.

Two steps:

    python scripts/replay_capture.py manifest            # here: list the recorded episodes and their commands
    python scripts/replay_capture.py replay --keys K ... # on a Colab VM: replay them and capture

replay resets each episode's valid seed exactly as the harness does (same scene, same expert replay), then
re-sends the controller's commands turn by turn, with no model call. It captures at the start of every turn,
right before every gripper close, right before every open of a gripper that is closed on something, and after
the last command:

  - the views the controller saw (annotated agent and top camera, contrast-enhanced wrist cameras; JPEG) and the
    raw camera images without drawings or contrast change (<view>_raw.jpg);
  - depth in mm (uint16) of the top and agent cameras (every second pixel, 320x240) and both wrist cameras
    (native 320x240), with each camera's intrinsics and world-to-camera extrinsics (OpenCV convention);
  - actor-level segmentation of the same cameras (uint16 ids, with the scene's id -> entity name table):
    evaluation labels only, never input to a model;
  - the true state: every tracked object's pose and functional points, both fingertips and openings, table
    height;
  - the robot's own body: joint positions and every link's world pose (proprioception, i.e. allowed at run time,
    e.g. to mask the arms out of the depth images).

After every command it also logs the result, the true object poses and the full arm state (pose, approach and
finger axes, opening), no images, so the command that changed an object can be found. meta.json also records
each camera's near and far clipping planes (the wrist cameras do not see closer than their near plane). A replay need not match the original run (the planner and contacts can differ); every
turn start records how far the tracked objects are from the original trace's positions.

Output: outputs/replay/<run>/<task>/episode_<k>/{captures.npz, meta.json, images/c<NNN>_<view>.jpg}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT/"outputs/replay/manifest.json"
OUT = PROJECT/"outputs/replay"
CAMERAS = ["agent_camera", "top_camera", "wrist"]
DEPTH_VIEWS = ("top_camera", "agent_camera", "left_camera", "right_camera")
HALF = ("top_camera", "agent_camera")          # 640x480 cameras stored at every second pixel


# ----------------------------------------------------------------------------- manifest (this machine)
def runs_on_disk(root: Path) -> list[tuple[str, Path]]:
    """(run label, results.json) for the evaluation baseline and every development run fetched here."""
    found = [("eval_baseline", p) for p in sorted(root.glob("*/robodawn/gemini_flash_flex/*/shard_*/results.json"))]
    found += [(p.parts[-4], p) for p in sorted(root.glob("*/robodawn_dev/relay_*_m16000_standard/*/shard_*/results.json"))]
    return found


def build_manifest(root: Path) -> list[dict]:
    episodes, seen = [], set()
    for run, results in runs_on_disk(root):
        task = results.parent.parent.name
        for e in json.loads(results.read_text())["episodes"]:
            key = f"{run}/{task}/{e['episode_index']}"
            trace_path = results.parent/f"episode_{e['episode_index']:03d}/trace.json"
            if key in seen or not trace_path.is_file():
                continue
            seen.add(key)
            trace = json.loads(trace_path.read_text())
            turns = [{"turn": t["turn"], "source": t.get("source", "model"), "commands": t.get("commands") or [],
                      "objects": {k: o["position_cm"] for k, o in t["state"]["objects"].items() if "position_cm" in o}}
                     for t in trace]
            episodes.append({"key": key, "run": run, "task": task, "episode": e["episode_index"], "seed": e["seed"],
                             "instruction": e["instruction"], "success": e["success"],
                             "finished_reason": e["finished_reason"], "turns": turns})
    return episodes


# ----------------------------------------------------------------------------- capture (VM)
def depth_mm(position):
    """SAPIEN 'Position' picture (OpenGL camera frame) -> depth along the optical axis in mm; 0 where invalid."""
    import numpy as np
    depth = -position[..., 2]
    valid = position[..., 3] < 1
    return np.where(valid, np.round(np.clip(depth, 0, 60) * 1000.0), 0).astype(np.uint16)


class Capturer:
    """Everything one capture stores, from the harness environment after env.observe()."""

    def __init__(self, env, out: Path):
        self.env, self.out = env, out
        self.arrays: dict[str, list] = {f"{v}_{k}": [] for v in DEPTH_VIEWS for k in ("depth", "seg", "K", "extrinsic")}
        self.captures: list[dict] = []
        (out/"images").mkdir(parents=True, exist_ok=True)

    def cameras(self) -> dict:
        cams = dict(self.env._extra_cameras)
        cams["left_camera"] = self.env.env.cameras.left_camera
        cams["right_camera"] = self.env.env.cameras.right_camera
        return cams

    def capture(self, kind: str, turn: int, cmd_index: int | None = None, command: str | None = None,
                reference: dict | None = None) -> dict:
        import numpy as np
        obs = self.env.observe()           # renders every camera once; Position/Segmentation come from the same render
        idx = len(self.captures)
        for view in self.env.views(obs, CAMERAS):
            view.image.convert("RGB").save(self.out/"images"/f"c{idx:03d}_{view.name}.jpg", quality=90)
        from PIL import Image
        for name in DEPTH_VIEWS:
            if name in obs.get("images", {}):
                Image.fromarray(obs["images"][name]).save(self.out/"images"/f"c{idx:03d}_{name}_raw.jpg", quality=95)
        params = obs["camera_params"]
        for name, cam in self.cameras().items():
            pos = cam.get_picture("Position")
            seg = cam.get_picture("Segmentation")[..., 1]
            step = 2 if name in HALF else 1
            self.arrays[f"{name}_depth"].append(depth_mm(pos)[::step, ::step])
            self.arrays[f"{name}_seg"].append(seg[::step, ::step].astype(np.uint16))
            K = np.asarray(params[name]["intrinsic_cv"], dtype=np.float64).copy()
            K[:2] /= step                   # every second pixel: fx, fy, cx, cy halve
            ext = np.eye(4)
            ext[:3] = np.asarray(params[name]["extrinsic_cv"], dtype=np.float64)[:3]
            self.arrays[f"{name}_K"].append(K)
            self.arrays[f"{name}_extrinsic"].append(ext)
        state = obs["state"]
        entry = {"idx": idx, "kind": kind, "turn": turn, "cmd_index": cmd_index, "command": command,
                 "state": state, "robot": robot_body(self.env), "success": bool(self.env.success)}
        if reference:
            entry["divergence_cm"] = round(max((math.dist(state["objects"][k]["position_cm"], p) for k, p in reference.items()
                                                if k in state["objects"] and "position_cm" in state["objects"][k]), default=0.0), 2)
        self.captures.append(entry)
        return entry

    def save(self, meta: dict) -> None:
        import numpy as np
        np.savez_compressed(self.out/"captures.npz", **{k: np.stack(v) for k, v in self.arrays.items() if v})
        meta["captures"] = self.captures
        tmp = self.out/"meta.json.tmp"
        tmp.write_text(json.dumps(meta, default=_json_default))
        os.replace(tmp, self.out/"meta.json")        # meta.json marks a finished episode


def _json_default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def entity_names(env) -> dict:
    return {int(e.per_scene_id): e.get_name() for e in env.env.scene.get_entities()}


def robot_body(env) -> dict:
    """Joint positions and link world poses of both arms (what forward kinematics gives the robot itself)."""
    out = {}
    robot = env.env.robot
    for arm, entity in (("left", robot.left_entity), ("right", robot.right_entity)):
        try:
            links = {link.get_name(): [*(float(v) for v in link.get_pose().p), *(float(v) for v in link.get_pose().q)]
                     for link in entity.get_links()}
            out[arm] = {"qpos": [float(v) for v in entity.get_qpos()], "links": links}
        except Exception as exc:  # noqa: BLE001  (a missing accessor must not stop the replay)
            out[arm] = {"error": f"{type(exc).__name__}: {exc}"}
        if entity is robot.right_entity and robot.left_entity is robot.right_entity:
            out["right"] = out["left"]           # one articulation for both arms
    return out


def camera_planes(cameras: dict) -> dict:
    return {name: {"near": float(getattr(cam, "near", float("nan"))), "far": float(getattr(cam, "far", float("nan")))}
            for name, cam in cameras.items()}


def holding(env, arm: str) -> bool:
    """The arm's fingers are commanded closed but stopped on something."""
    st = env.arm_state(arm)
    return st.gripper < 0.5 and st.gripper_real > 0.02


def replay_episode(env, spec: dict, out: Path) -> dict:
    from harness.core.commands import parse_command
    started = time.time()
    info = env.reset_episode(spec["episode"])
    if int(info.seed) != int(spec["seed"]):
        raise RuntimeError(f"{spec['key']}: seed {info.seed} differs from the recorded {spec['seed']}")
    cap = Capturer(env, out)
    log = []
    names = entity_names(env)
    for t in spec["turns"]:
        cap.capture("turn", t["turn"], reference=t["objects"])
        for i, text in enumerate(t["commands"]):
            cmd = parse_command(text)
            if cmd.kind == "done":
                continue
            if cmd.kind == "gripper" and cmd.value < 0.5:
                cap.capture("preclose", t["turn"], i, text)
            elif cmd.kind == "gripper" and holding(env, cmd.arm):
                cap.capture("preopen", t["turn"], i, text)
            res = env.execute(cmd)
            st = env.state()
            log.append({"turn": t["turn"], "cmd_index": i, "command": text, "ok": res.get("ok"), "note": res.get("note"),
                        "objects": {k: {"position_cm": o["position_cm"], "quat_wxyz": o["quat_wxyz"]}
                                    for k, o in st["objects"].items() if "position_cm" in o},
                        "arms": {a: st[a] for a in ("left", "right")},
                        "success": bool(env.success)})
            if env.success or env.budget_exhausted:
                break
        if env.success or env.budget_exhausted or env.should_stop():
            break
    cap.capture("final", spec["turns"][-1]["turn"] if spec["turns"] else 0)
    meta = {k: spec[k] for k in ("key", "run", "task", "episode", "seed", "instruction", "success", "finished_reason")}
    meta.update(replay_success=bool(env.success), replay_stop=env.should_stop(), entity_names=names,
                camera_planes=camera_planes(cap.cameras()),
                command_log=log, seconds=round(time.time() - started, 1), depth_views=list(DEPTH_VIEWS),
                half_resolution=list(HALF), cameras_note="K is for the stored (possibly halved) resolution; "
                "extrinsic maps world to OpenCV camera coordinates")
    cap.save(meta)
    env.close_episode()
    return meta


def main_replay(args) -> None:
    manifest = {e["key"]: e for e in json.loads(args.manifest.read_text())}
    specs = [manifest[k] for k in args.keys]
    tasks = {s["task"] for s in specs}
    if len(tasks) != 1:
        raise ValueError("one task per process (the environment is built per task)")
    os.environ.setdefault("ROBOTWIN_ROOT", str(args.robotwin_root))
    sys.path.insert(0, str(args.repo.resolve()))
    from harness.core import watchdog
    from harness.robotwin.env import RoboTwinDiscreteEnv
    watchdog.start(stall_seconds=900)
    env = RoboTwinDiscreteEnv(tasks.pop(), task_config="demo_randomized", seed=0, instruction_type="unseen",
                              record_video=False)
    for spec in specs:
        out = args.out/spec["run"]/spec["task"]/f"episode_{spec['episode']:03d}"
        if (out/"meta.json").is_file():
            continue
        meta = replay_episode(env, spec, out)
        div = [c.get("divergence_cm", 0) for c in meta["captures"] if c["kind"] == "turn"]
        print(f"{spec['key']}: {len(meta['captures'])} captures, recorded success {spec['success']} / replay "
              f"{meta['replay_success']}, largest divergence {max(div, default=0):.1f} cm, {meta['seconds']}s", flush=True)
    env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--root", type=Path, default=PROJECT/"outputs/robodawn_colab")
    m.add_argument("--output", type=Path, default=MANIFEST)
    r = sub.add_parser("replay")
    r.add_argument("--manifest", type=Path, default=MANIFEST)
    r.add_argument("--keys", nargs="+", required=True)
    r.add_argument("--out", type=Path, default=OUT)
    r.add_argument("--repo", type=Path, default=PROJECT/"third_party/robodawn")
    r.add_argument("--robotwin-root", type=Path, default=Path(os.environ.get("ROBOTWIN_ROOT", "/content/RoboTwin")))
    args = parser.parse_args()
    if args.step == "manifest":
        episodes = build_manifest(args.root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(episodes))
        by = {}
        for e in episodes:
            by[e["run"]] = by.get(e["run"], 0) + 1
        print(f"{args.output}: {len(episodes)} episodes {by}, {sum(len(e['turns']) for e in episodes)} turns")
    else:
        main_replay(args)


if __name__ == "__main__":
    main()
