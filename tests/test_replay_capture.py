"""replay_capture on a fake environment: what is captured when, and in which format; no simulator."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"third_party/robodawn"))
import replay_capture as rc
from harness.core.env import View


class FakeCam:
    near, far = 0.1, 100.0

    def __init__(self, h, w, depth_m):
        self.h, self.w, self.depth_m = h, w, depth_m

    def get_picture(self, kind):
        if kind == "Position":
            pos = np.zeros((self.h, self.w, 4), np.float32)
            pos[..., 2] = -self.depth_m
            pos[0, 0, 3] = 1.0                       # one invalid pixel
            return pos
        seg = np.zeros((self.h, self.w, 4), np.uint32)
        seg[..., 1] = 7
        return seg


class FakeArm:
    def get_links(self):
        link = types.SimpleNamespace(get_name=lambda: "link1",
                                     get_pose=lambda: types.SimpleNamespace(p=[0.1, 0.2, 0.3], q=[1, 0, 0, 0]))
        return [link]

    def get_qpos(self):
        return np.array([0.5, -0.5])


class FakeEnv:
    def __init__(self):
        self._extra_cameras = {"agent_camera": FakeCam(480, 640, 1.2), "top_camera": FakeCam(480, 640, 0.9)}
        self.env = types.SimpleNamespace(
            cameras=types.SimpleNamespace(left_camera=FakeCam(240, 320, 0.3), right_camera=FakeCam(240, 320, 0.4)),
            scene=types.SimpleNamespace(get_entities=lambda: [types.SimpleNamespace(per_scene_id=7, get_name=lambda: "cup")]),
            robot=types.SimpleNamespace(left_entity=FakeArm(), right_entity=FakeArm()))
        self.cup = [10.0, -10.0, 75.0]
        self.executed, self.grip = [], {"left": (1.0, 1.0), "right": (1.0, 1.0)}

    def arm_state(self, arm):
        return types.SimpleNamespace(gripper=self.grip[arm][0], gripper_real=self.grip[arm][1])

    def reset_episode(self, k):
        return types.SimpleNamespace(seed=100017)

    def state(self):
        arm = {"position_cm": [0.0, 0.0, 90.0], "gripper_real": 1.0}
        return {"objects": {"cup": {"position_cm": list(self.cup), "quat_wxyz": [1, 0, 0, 0]}},
                "left": arm, "right": arm, "table_z_cm": 74.0}

    def observe(self):
        params = {n: {"intrinsic_cv": np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]), "extrinsic_cv": np.eye(4)[:3]}
                  for n in ("agent_camera", "top_camera", "left_camera", "right_camera")}
        images = {n: np.zeros((48, 64, 3), np.uint8) for n in ("agent_camera", "top_camera", "left_camera", "right_camera")}
        return {"camera_params": params, "state": self.state(), "images": images}

    def views(self, obs, cameras):
        return [View(n, Image.new("RGB", (8, 6)), n) for n in ("agent_camera", "top_camera", "left_camera", "right_camera")]

    def execute(self, cmd):
        self.executed.append(cmd.text())
        if cmd.kind == "move":
            self.cup[0] += 1.0
        if cmd.kind == "gripper":
            self.grip[cmd.arm] = (cmd.value, 0.3 if cmd.value < 0.5 else 1.0)    # the close grasps something
        return {"ok": True, "note": "fingers closed fully: nothing between them" if cmd.kind == "gripper" else ""}

    success = False
    budget_exhausted = False

    def should_stop(self):
        return None

    def close_episode(self):
        pass


SPEC = {"key": "eval_baseline/place_empty_cup/3", "run": "eval_baseline", "task": "place_empty_cup", "episode": 3,
        "seed": 100017, "instruction": "Put the cup on the coaster.", "success": False, "finished_reason": "max_turns",
        "turns": [{"turn": 1, "source": "model", "commands": ["left move x +1.0", "left gripper 0.00", "done"],
                   "objects": {"cup": [10.0, -10.0, 75.0]}},
                  {"turn": 2, "source": "relay", "commands": ["left gripper 1.00"], "objects": {"cup": [13.0, -10.0, 75.0]}}]}


class ReplayTests(unittest.TestCase):
    def test_captures_and_formats(self):
        env = FakeEnv()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"ep"
            meta = rc.replay_episode(env, SPEC, out)
            arrays = np.load(out/"captures.npz")
            saved = json.loads((out/"meta.json").read_text())
            images = sorted(p.name for p in (out/"images").iterdir())
        self.assertEqual(env.executed, ["left move x +1.0", "left gripper 0.00", "left gripper 1.00"])   # "done" skipped
        self.assertEqual([c["kind"] for c in meta["captures"]], ["turn", "preclose", "turn", "preopen", "final"])
        self.assertEqual(meta["captures"][1]["command"], "left gripper 0.00")
        self.assertEqual(meta["captures"][3]["command"], "left gripper 1.00")   # open of a gripper holding something
        self.assertEqual([c.get("divergence_cm") for c in meta["captures"]], [0.0, None, 2.0, None, None])
        self.assertEqual(meta["captures"][0]["robot"]["left"]["qpos"], [0.5, -0.5])
        self.assertEqual(meta["captures"][0]["robot"]["left"]["links"]["link1"], [0.1, 0.2, 0.3, 1, 0, 0, 0])
        self.assertEqual(saved["camera_planes"]["left_camera"], {"near": 0.1, "far": 100.0})
        self.assertEqual(arrays["top_camera_depth"].shape, (5, 240, 320))           # every second pixel
        self.assertEqual(arrays["left_camera_depth"].shape, (5, 240, 320))          # wrist at native resolution
        self.assertEqual(arrays["top_camera_depth"][0, 1, 1], 900)                  # mm
        self.assertEqual(arrays["top_camera_depth"][0, 0, 0], 0)                    # invalid pixel
        self.assertEqual(arrays["top_camera_seg"].dtype, np.uint16)
        np.testing.assert_allclose(arrays["top_camera_K"][0], [[250, 0, 160], [0, 250, 120], [0, 0, 1]])
        np.testing.assert_allclose(arrays["left_camera_K"][0], [[500, 0, 320], [0, 500, 240], [0, 0, 1]])
        self.assertEqual(arrays["top_camera_extrinsic"].shape, (5, 4, 4))
        self.assertEqual(saved["entity_names"], {"7": "cup"})
        self.assertEqual(len(saved["command_log"]), 3)
        self.assertEqual(len(images), 5 * 8)                                      # 4 views + 4 raw per capture


if __name__ == "__main__":
    unittest.main()
