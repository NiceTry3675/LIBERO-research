"""The object monitor's interfaces and stop logic, with a scripted Clef and synthetic blobs; no simulator or API."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))
import rtscene_clef_iface as iface
from branchlab.rtscene import monitor
from branchlab.rtscene.blobs import Blob


def blob(id_, x, y, h95, length, width):
    return Blob(id=id_, iy=np.array([0]), ix=np.array([0]), centroid=(x, y), area=length * width, length=length,
                width=width, yaw=0.0, fill=1.0, circularity=0.8, h95=h95, hmed=h95 * 0.8, top_tilt=None, hollow=0.0,
                lab=(50.0, 0.0, 0.0), robot_gap=20.0, unknown_edge=0.0)


class FakeCap:
    table_z = 74.0
    arms = {"left": {"position_cm": [-40.0, -30.0, 95.0]}, "right": {"position_cm": [40.0, -30.0, 95.0]}}


class Interfaces(unittest.TestCase):
    def test_coordinates(self):
        text = "Fan is at approx (-5.5, -12), yellow mat is at (-25, -14). Left gripper at (-30, -19, 94.2)."
        self.assertEqual(monitor.coordinates(text), [(-5.5, -12.0), (-25.0, -14.0), (-30.0, -19.0)])
        self.assertEqual(monitor.coordinates(None), [])
        text = "Red tall block is at x ≈ -27, y ≈ +18, standing. Blue pad is at x ~ +23, y ~ +11. Cup x=4.5 cm, y=-3."
        self.assertEqual(monitor.coordinates(text), [(-27.0, 18.0), (23.0, 11.0), (4.5, -3.0)])

    def test_reid_text_is_the_tuned_interface(self):
        item = {"cands": [{"dist": 0, "h95": 5.0, "lw": [4, 4]}] * 3, "refs": {"top_camera": "r"},
                "sheets": {"top_camera": "s"}, "ref_num": {"h95": 5.0, "lw": [4, 4]}, "instruction": "x"}
        state, q, imgs = iface.request("reid", "top", item)
        self.assertEqual(monitor.reid_request(3), (state, q))
        self.assertEqual(imgs, ["r", "s"])

    def test_topple_text_is_the_tuned_interface(self):
        ref, now = blob(1, 0, 0, 10.04, 4.2, 3.9), blob(2, 1, 0, 3.96, 10.4, 4.1)
        item = {"imgs": {"ref_agent_camera": "a", "now_agent_camera": "b"},
                "num": {"h_ref": round(ref.h95, 1), "h_now": round(now.h95, 1), "lw_ref": [round(ref.length), round(ref.width)],
                        "lw_now": [round(now.length), round(now.width)]}}
        state, q, _ = iface.request("topple", "agent+num", item)
        self.assertEqual(monitor.topple_request(ref, now), (state, q))

    def test_fuse_prefers_the_nearer_tile_when_clef_hesitates(self):
        pick, p = monitor.fuse({"T1": 0.45, "T2": 0.5, "none": 0.05}, [1, 12])
        self.assertEqual(pick, "T1")
        pick, p = monitor.fuse({"T1": 0.02, "T2": 0.95, "none": 0.03}, [1, 12])
        self.assertEqual(pick, "T2")
        self.assertGreater(p, 0.5)


class StopLogic(unittest.TestCase):
    def setUp(self):
        self.p_fallen = 0.1
        self.calls = []

        def ask(state, q, images):
            self.calls.append(list(q))
            if "which" in q:
                return {"which": {"choice": "T1", "probabilities": {"T1": 0.9, "none": 0.1}}}
            return {"fallen": {"noul": self.p_fallen}}
        self.scenes = {}
        self.patches = [patch.object(monitor, "parse", lambda cap, boxes: self.scenes[id(cap)]),
                        patch.object(monitor.render, "crop", lambda *a, **k: Image.new("RGB", (8, 8)))]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])
        self.mon = monitor.ObjectMonitor(ask, {}, policy="both", need=2)

    def capture(self, *blobs):
        cap = FakeCap()
        self.scenes[id(cap)] = list(blobs)
        return cap

    def test_binds_only_raised_blobs_near_a_coordinate(self):
        cap = self.capture(blob(1, 0, 0, 10, 4, 4), blob(2, 20, 0, 0.6, 12, 12), blob(3, 40, 0, 8, 4, 4))
        bound = self.mon.bind(cap, [(1.0, 1.0), (21.0, 0.0), (80.0, 0.0)])
        self.assertEqual([b["centroid"] for b in bound], [[0.0, 0.0]])      # the flat mat and the far point are not bound

    def test_two_flagged_checks_in_a_row_stop(self):
        self.mon.bind(self.capture(blob(1, 0, 0, 10, 4, 4)), [(0.0, 0.0)])
        self.mon.check(self.capture(blob(5, 0.5, 0, 10, 4, 4)))               # standing
        self.assertIsNone(self.mon.fallen())
        self.p_fallen = 0.9
        rows = self.mon.check(self.capture(blob(6, 3, 0, 4, 10, 4)))         # lying: code and Clef agree
        self.assertEqual(rows[0]["streak"], 1)
        self.assertIsNone(self.mon.fallen())
        self.mon.check(self.capture(blob(7, 3, 0, 4, 10, 4)))
        self.assertIn("(0, 0)", self.mon.fallen())

    def test_both_policy_needs_the_code_too(self):
        self.mon.bind(self.capture(blob(1, 0, 0, 10, 4, 4)), [(0.0, 0.0)])
        self.p_fallen = 0.95                                                 # Clef alone says fallen
        for k in range(3):
            self.mon.check(self.capture(blob(10 + k, 0.5, 0, 10, 4, 4)))
        self.assertIsNone(self.mon.fallen())

    def test_arm_near_skips_without_calls_or_reset(self):
        self.mon.bind(self.capture(blob(1, 0, 0, 10, 4, 4)), [(0.0, 0.0)])
        self.p_fallen = 0.9
        self.mon.check(self.capture(blob(6, 3, 0, 4, 10, 4)))
        cap = self.capture(blob(7, 3, 0, 4, 10, 4))
        cap.arms = {"left": {"position_cm": [3.0, 0.0, 76.0]}, "right": {"position_cm": [40.0, -30.0, 95.0]}}
        n = len(self.calls)
        row = self.mon.check(cap)[0]
        self.assertEqual((row["status"], row["streak"], len(self.calls)), ("arm", 1, n))
        self.mon.check(self.capture(blob(8, 3, 0, 4, 10, 4)))
        self.assertIsNotNone(self.mon.fallen())

    def test_no_checks_while_a_gripper_holds_something(self):
        self.mon.bind(self.capture(blob(1, 0, 0, 10, 4, 4)), [(0.0, 0.0)])
        self.p_fallen = 0.9
        self.mon.check(self.capture(blob(6, 3, 0, 4, 10, 4)))
        cap = self.capture(blob(7, 3, 0, 4, 10, 4))
        cap.arms = {"left": {"position_cm": [-40.0, -30.0, 95.0], "gripper": 0.0, "gripper_real": 0.3},
                    "right": {"position_cm": [40.0, -30.0, 95.0], "gripper": 1.0, "gripper_real": 0.9}}
        n = len(self.calls)
        row = self.mon.check(cap)[0]
        self.assertEqual((row["status"], row["streak"], len(self.calls)), ("holding", 1, n))
        cap.arms["left"]["gripper_real"] = 0.0                               # closed on nothing: not holding
        self.assertEqual(self.mon.check(cap)[0]["streak"], 2)

    def test_a_carried_object_is_looked_for_where_it_was_released(self):
        self.mon.bind(self.capture(blob(1, 0, 0, 10, 4, 4)), [(0.0, 0.0)])
        self.mon.note_close("left", [2.0, 1.0, 80.0], caught=False)            # an empty close carries nothing
        self.assertIsNone(self.mon.objects[0].carried_by)
        self.mon.note_close("left", [2.0, 1.0, 80.0], caught=True)
        self.assertEqual(self.mon.check(self.capture(blob(5, 0, 0, 10, 4, 4)))[0]["status"], "carried")
        self.mon.note_open("left", [25.0, 5.0, 80.0])
        # the cup now stands 25 cm away; a look-alike stays at the old spot, within reach of the old search
        self.mon.check(self.capture(blob(6, 25, 5, 10, 4, 4), blob(7, 1, 0, 10, 4, 4)))
        self.assertEqual(self.mon.objects[0].last.id, 6)
        self.assertIsNone(self.mon.objects[0].search)

    def test_no_fall_counts_when_clef_does_not_name_the_pick(self):
        answers = {"which": {"choice": "none", "probabilities": {"T1": 0.45, "none": 0.55}}}
        mon = monitor.ObjectMonitor(lambda s, q, i: answers if "which" in q else {"fallen": {"noul": 0.95}}, {})
        mon.bind(self.capture(blob(1, 0, 0, 10, 4, 4)), [(0.0, 0.0)])
        for k in range(3):
            row = mon.check(self.capture(blob(10 + k, 1, 0, 4, 10, 4)))[0]  # the distance prior still picks T1
        self.assertEqual((row["pick"], row["streak"], mon.objects[0].last.id), ("T1", 0, 1))

    def test_blobs_too_low_to_be_the_object_are_not_candidates(self):
        self.mon.bind(self.capture(blob(1, 0, 0, 10, 4, 4)), [(0.0, 0.0)])
        row = self.mon.check(self.capture(blob(5, 0.5, 0, 0.9, 12, 12), blob(6, 6, 0, 10, 4, 4)))[0]
        self.assertEqual((row["n_cands"], self.mon.objects[0].last.id), (1, 6))   # the coaster is no cup

    def test_a_tall_object_at_full_height_has_not_fallen(self):
        self.mon.bind(self.capture(blob(1, 0, 0, 10, 4, 4)), [(0.0, 0.0)])
        self.p_fallen = 0.9
        for k in range(3):                         # merged with a mat: the footprint grows, the height stays
            row = self.mon.check(self.capture(blob(10 + k, 2, 0, 10, 14, 6)))[0]
        self.assertTrue(row["upright"] and row["code"] > monitor.CODE_THR)
        self.assertIsNone(self.mon.fallen())

    def test_coordinates_on_a_fingertip_are_not_bound(self):
        cap = self.capture(blob(1, 0, 0, 10, 4, 4), blob(2, -40, -28, 3, 4, 4))
        self.assertEqual(len(self.mon.bind(cap, [(0.0, 0.0), (-40.0, -30.0)])), 1)

    def test_a_failed_call_is_logged_not_raised(self):
        def broken(state, q, images):
            raise RuntimeError("HTTP 500")
        mon = monitor.ObjectMonitor(broken, {}, policy="both")
        mon.bind(self.capture(blob(1, 0, 0, 10, 4, 4)), [(0.0, 0.0)])
        row = mon.check(self.capture(blob(2, 0, 0, 10, 4, 4)))[0]
        self.assertEqual(row["status"], "error")
        self.assertIsNone(mon.fallen())


if __name__ == "__main__":
    unittest.main()
