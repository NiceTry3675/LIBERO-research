"""The release gate's crop and decision (branchlab.rtscene.release_gate); no simulator or API calls."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))
import clef_semantic_offline as offline
from branchlab.rtscene import release_gate as gate


def top_camera_params(table_z_cm: float, W: int = 640, H: int = 480) -> dict:
    """The offline script's top camera as the harness hands it over: intrinsic_cv and a 3x4 extrinsic_cv."""
    c = offline.CAMERAS["top_camera"]
    f = np.array(c["fwd"], float) / np.linalg.norm(c["fwd"])
    left = np.array(c["left"], float) / np.linalg.norm(c["left"])
    R = np.stack([-left, -np.cross(f, left), f])
    t = np.array(c["pos"], float) + [0.0, 0.0, (table_z_cm - 74.0) / 100.0]
    fy = (H / 2) / math.tan(math.radians(c["fovy"]) / 2)
    return {"intrinsic_cv": np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]]),
            "extrinsic_cv": np.hstack([R, (-R @ t)[:, None]])}


def obs_for(image: np.ndarray, tcp_cm, table_z_cm: float = 74.6) -> dict:
    return {"images": {"top_camera": image}, "camera_params": {"top_camera": top_camera_params(table_z_cm)},
            "state": {"table_z_cm": table_z_cm, "right": {"position_cm": list(tcp_cm)}}}


IMAGE = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
TCP = [12.0, -8.0, 86.0]


class CropTests(unittest.TestCase):
    def test_the_live_crop_is_the_offline_crop(self):
        img = Image.fromarray(IMAGE)
        live = gate.crop(img, gate.projector(top_camera_params(74.6)), TCP)
        off = offline.crop(img, "top_camera", TCP, 15, 74.6)
        self.assertEqual(live.size, (256, 256))
        self.assertEqual(live.tobytes(), off.tobytes())

    def test_the_projection_is_the_harness_one(self):
        params = top_camera_params(74.6)
        pts = np.array([TCP, [0.0, 0.0, 74.6], [-30.0, 20.0, 80.0]])
        np.testing.assert_allclose(gate.projector(params)(pts), offline.project("top_camera", pts, 74.6), atol=1e-6)
        # a 4x4 extrinsic gives the same
        four = {**params, "extrinsic_cv": np.vstack([params["extrinsic_cv"], [0, 0, 0, 1]])}
        np.testing.assert_allclose(gate.projector(four)(pts), gate.projector(params)(pts))

    def test_the_box_is_square_around_the_fingertips(self):
        project = gate.projector(top_camera_params(74.6))
        box = gate.crop_box(project, TCP)
        u, v = project([TCP])[0]
        self.assertAlmostEqual(box[2] - box[0], box[3] - box[1], delta=1)
        # off centre by perspective (the box rises 12 cm above the fingertips), but well inside
        self.assertTrue(box[0] + 40 < u < box[2] - 40 and box[1] + 40 < v < box[3] - 40)


class DecisionTests(unittest.TestCase):
    @staticmethod
    def ask(probs):
        calls = []

        def ask(state, questions, images):
            calls.append((state, questions, images))
            return {"toe": {"type": "choice", "probabilities": probs}}
        return ask, calls

    def test_veto_at_the_threshold_names_the_direction(self):
        for probs, vetoed, seen in (({"left": 0.2, "right": 0.7, "up": 0.05, "down": 0.05}, True, "RIGHT"),
                                    ({"left": 0.1, "right": 0.1, "up": 0.1, "down": 0.7}, True, "toward the robot"),
                                    ({"left": 0.05, "right": 0.05, "up": 0.85, "down": 0.05}, True, "away from the robot"),
                                    ({"left": 0.21, "right": 0.79, "up": 0.0, "down": 0.0}, False, None)):
            ask, calls = self.ask(probs)
            entry, tile = gate.ReleaseGate("clef", ask).check("right", obs_for(IMAGE, TCP))
            with self.subTest(probs):
                self.assertEqual((entry["vetoed"], entry["error"]), (vetoed, None))
                self.assertEqual(tile.size, (256, 256))
                self.assertTrue(calls[0][2][0].startswith("data:image/jpeg;base64,"))
                if vetoed:
                    self.assertIn(f"pointing {seen} in the top camera, not left", entry["note"])
                else:
                    self.assertIsNone(entry["note"])

    def test_failures_never_veto(self):
        def broken(state, questions, images):
            raise TimeoutError("timed out")
        cases = [(self.ask({"right": 1.0})[0], obs_for(IMAGE, TCP)),                    # no P(left)
                 (broken, obs_for(IMAGE, TCP)),
                 (self.ask({"left": 0.0, "right": 1.0})[0], {"images": {}, "state": {}}),  # no image
                 (self.ask({"left": 0.0, "right": 1.0})[0], None)]
        for ask, obs in cases:
            entry, _ = gate.ReleaseGate("clef", ask).check("right", obs)
            self.assertFalse(entry["vetoed"])
            self.assertTrue(entry["error"])

    def test_nudge_needs_no_image(self):
        entry, tile = gate.ReleaseGate("nudge").check("left")
        self.assertEqual((entry["vetoed"], tile, entry["threshold"]), (True, None, None))
        self.assertTrue(entry["note"].startswith('"left gripper open" NOT executed (release check): Before releasing'))

    def test_bad_settings_are_refused(self):
        with self.assertRaises(ValueError):
            gate.ReleaseGate("clef")
        with self.assertRaises(ValueError):
            gate.ReleaseGate("loud")
        self.assertEqual([t for t in ("place_shoe", "place_dual_shoes", "place_empty_cup") if gate.ReleaseGate.applies(t)],
                         ["place_shoe", "place_dual_shoes"])


if __name__ == "__main__":
    unittest.main()
