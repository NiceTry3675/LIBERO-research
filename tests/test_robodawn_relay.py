"""The relay agent on a scripted environment and model; no GPU, simulator or API calls."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"third_party/robodawn"))
sys.path.insert(0, str(ROOT/"src"))
import compare_relay
import robodawn_relay as relay
import run_robodawn_relay as runner
from harness.agent.llm_client import ChatResult
from harness.agent.mllm_agent import AgentConfig, MLLMDiscreteAgent
from harness.core.env import DiscreteEnvBase, View

TASK = "place_empty_cup"


class FakeEnv(DiscreteEnvBase):
    """A cup at CUP; a close within 2 cm (xy) and 3 cm (z) of it grasps it, and a held cup lifted to z >= 90
    is a success. Moves below z = 75 fail without moving, like the planner refusing to hit the table."""
    CUP = (10.0, -10.0, 79.0)

    def __init__(self, step_limit=60):
        self.arms = {"left": {"position_cm": [-20.0, -20.0, 95.0], "gripper_real": 1.0},
                     "right": {"position_cm": [20.0, -20.0, 95.0], "gripper_real": 1.0}}
        self.cup = list(self.CUP)
        self.held_by = None
        self._steps, self._limit, self.drop_at_step = 0, step_limit, None

    def reset_episode(self, episode_index=None):
        raise NotImplementedError

    def arm(self, name):
        a = self.arms[name]
        return {"position_cm": list(a["position_cm"]), "approach": [0.0, 0.0, -1.0], "finger_axis": [1.0, 0.0, 0.0],
                "gripper_real": a["gripper_real"], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}

    def state(self):
        return {"steps_used": self._steps, "step_limit": self._limit, "success": self.success, "table_z_cm": 74.0,
                "left": self.arm("left"), "right": self.arm("right"),
                "objects": {"cup": {"position_cm": list(self.cup)}}}

    def observe(self):
        shade = int(self._steps * 7 % 255)
        images = {n: np.full((8, 8, 3), shade, dtype=np.uint8)
                  for n in ("agent_camera", "top_camera", "left_camera", "right_camera")}
        return {"images": images, "state": self.state()}

    def views(self, obs, cameras):
        return [View(n, Image.fromarray(img), f"{n} caption") for n, img in obs["images"].items()]

    @property
    def success(self):
        return self.held_by is not None and self.cup[2] >= 90.0

    @property
    def steps_used(self):
        return self._steps

    @property
    def step_limit(self):
        return self._limit

    def prompt_context(self):
        return {"table_z": 74}

    def execute(self, cmd):
        before = {n: self.arm(n) for n in self.arms}
        result = {"command": cmd.text(), "kind": cmd.kind, "arm": cmd.arm, "ok": True, "note": ""}
        self._steps += 1
        a = self.arms.get(cmd.arm)
        if cmd.kind == "move":
            target = list(a["position_cm"])
            target["xyz".index(cmd.axis)] += cmd.value
            if target[2] < 75.0:
                result.update(ok=False, note="motion planner could not reach the target: the arm did NOT move")
            else:
                a["position_cm"] = target
                if self.held_by == cmd.arm:
                    self.cup = [target[0], target[1], target[2] - 2.0]
        elif cmd.kind == "gripper":
            if cmd.value < 0.5:
                p = a["position_cm"]
                near = abs(p[0] - self.cup[0]) <= 2 and abs(p[1] - self.cup[1]) <= 2 and abs(p[2] - self.cup[2]) <= 3
                a["gripper_real"] = 0.4 if near else 0.0
                self.held_by = cmd.arm if near else None
                result["note"] = "fingers stopped at opening 0.40: something is between them (probably grasped)" if near \
                    else "fingers closed fully: nothing between them"
            else:
                a["gripper_real"], self.held_by = 1.0, None
                result["note"] = "gripper opened"
        if self.drop_at_step == self._steps and self.held_by:
            self.arms[self.held_by]["gripper_real"], self.held_by = 0.0, None
        result.update(before=before, after={n: self.arm(n) for n in self.arms}, seconds=0.5,
                      steps_used_after=self._steps, success=self.success)
        return result


class ScriptedClient:
    """Replies from a list (a string, or a dict sent as JSON, or an Exception-like error marker)."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, messages, tag=""):
        self.prompts.append((tag, messages[0]["content"], messages[-1]["content"][-1]["text"]))
        reply = self.replies.pop(0) if self.replies else {"commands": ["wait"]}
        if reply == "ERROR":
            return ChatResult(text="", latency_s=2.0, error="HTTP 503: unavailable")
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return ChatResult(text=text, latency_s=2.0, usage={"prompt_tokens": 100, "completion_tokens": 10})


class FakeChecker(relay.Checker):
    name, model = "fake", "fake-model"

    def __init__(self, answers):
        super().__init__(threshold=0.5, max_retries=1)
        self.answers, self.asked = list(answers), []

    def _ask(self, questions, context, images):
        self.asked.append((questions, context, len(images)))
        p = self.answers.pop(0)
        return 200, [p] * len(questions), {"input_tokens": 10, "output_tokens": 0}, 0.001, None, None

    tokens = staticmethod(relay.ClefChecker.tokens)


def reply(commands, expect=None, next_=None, **extra):
    out = {"scene": "s", "progress": "p", "memory": "m", "plan": "q", "commands": commands, **extra}
    if expect is not None:
        out["expect"] = expect
    if next_ is not None:
        out["next"] = next_
    return out


GRASP_AND_LIFT = [
    reply(["right point down", "right move x -10"]),                      # to (10, -20, 95)
    "this is not json",
    "ERROR",
    reply(["right move y +10", "right move z -15", "right fly away"]),   # invalid third command
    reply(["done"]),
    reply(["right move z -20"]),                                          # fails: below the table
    reply(["right gripper close", "right move z +12"]),                  # grasp at z 80, lift to 92 -> success
]


def run(agent_cls, replies, max_turns=45, env=None, tmp=None, task=TASK, instruction="Put the cup on the coaster.",
        **kwargs):
    cfg = AgentConfig(max_turns=max_turns)
    client = ScriptedClient(replies)
    env = env or FakeEnv()
    out = Path(tmp)
    agent = agent_cls(cfg, client, out/"run", task=task, **kwargs)
    result = agent.run_episode(env, instruction, out/"run/episode_000")
    trace = json.loads((out/"run/episode_000/trace.json").read_text())
    memory = json.loads((out/"run/episode_000/memory.json").read_text())
    return result, trace, memory, client, env


class LoopEquivalenceTests(unittest.TestCase):
    def test_baseline_variant_reproduces_upstream_loop(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            up = run(MLLMDiscreteAgent, GRASP_AND_LIFT, tmp=a)
            ours = run(relay.RelayAgent, GRASP_AND_LIFT, tmp=b, variant="baseline")
            self.assertTrue(up[0].success)
            self.assertEqual(up[1], ours[1])                      # trace
            self.assertEqual(up[2], ours[2])                      # memory
            self.assertEqual(up[3].prompts, ours[3].prompts)      # every system prompt and turn text sent
            skip = {"seconds"}
            self.assertEqual({k: v for k, v in asdict(up[0]).items() if k not in skip},
                             {k: v for k, v in asdict(ours[0]).items() if k in asdict(up[0]) and k not in skip})
            for name in ("system_prompt.txt", "turn001_agent_camera.png", "turn007_right_camera.png"):
                self.assertEqual((Path(a)/"run/episode_000"/name).read_bytes(), (Path(b)/"run/episode_000"/name).read_bytes())

    def test_baseline_variant_matches_upstream_at_the_turn_limit(self):
        script = [reply(["right move z +1"])] * 6
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            up = run(MLLMDiscreteAgent, script, max_turns=4, tmp=a)
            ours = run(relay.RelayAgent, script, max_turns=4, tmp=b, variant="baseline")
            self.assertEqual((up[0].turns, up[0].finished_reason), (4, "max_turns"))
            self.assertEqual((ours[0].turns, ours[0].finished_reason), (4, "max_turns"))
            self.assertEqual(up[1], ours[1])


# Approach, then plan: descend, close, lift. Each planned step's expectation is checked before it runs.
PLANNED = [
    reply(["right point down", "right move x -10", "right move y +10"], expect=["Is the right gripper above the cup?"],
          next_=[{"commands": ["right move z -15"], "expect": ["Is the cup between the right fingers?"]},
                 {"commands": ["right gripper close", "right move z +12"], "expect": ["Is the cup lifted?"]}]),
]


class RelayTests(unittest.TestCase):
    def test_open_variant_runs_planned_steps_without_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, trace, memory, client, _ = run(relay.RelayAgent, PLANNED, tmp=tmp, variant="open")
        self.assertTrue(result.success)
        self.assertEqual((result.model_calls, result.turns, result.relayed_steps, result.planned_steps), (1, 3, 2, 2))
        self.assertEqual([t.get("source") for t in trace], [None, "relay", "relay"])
        self.assertEqual(trace[2]["relay_from_turn"], 1)
        self.assertIsNone(trace[1]["checks"])
        self.assertEqual(memory["turns"][1]["turn"], "2 (planned step 1, run without asking you)")
        self.assertIn("PLANNED STEPS: you may plan ahead.", client.prompts[0][1])
        self.assertIn('"next": [{"commands": [...], "expect": [...]}, ...]', client.prompts[0][1])

    def test_checked_variant_stops_at_a_failed_check_and_reports_it(self):
        checker = FakeChecker([0.9, 0.2])
        script = PLANNED + [reply(["right gripper close"])]
        with tempfile.TemporaryDirectory() as tmp:
            result, trace, memory, client, _ = run(relay.RelayAgent, script, max_turns=3, tmp=tmp, variant="checked",
                                                   checker=checker)
            logged = [json.loads(l) for l in (Path(tmp)/"run/checker_calls.jsonl").read_text().splitlines()]
        self.assertEqual((result.relayed_steps, result.checker_calls, result.relay_ends), (1, 2, {"check_failed": 1}))
        self.assertEqual([c[0] for c in checker.asked], [["Is the right gripper above the cup?"],
                                                         ["Is the cup between the right fingers?"]])
        self.assertEqual(checker.asked[0][2], 4)                  # all four views
        self.assertIn("NEXT STEP IF EVERY ANSWER IS YES: right move z -15.0", checker.asked[0][1])
        second = client.prompts[1][2]
        self.assertIn("Planned step 1 ran in turn 2 after the checker answered YES to \"Is the right gripper above the cup?\"", second)
        self.assertIn("Planned step 2 did NOT run: the checker did not answer YES to \"Is the cup between the right fingers?\"", second)
        self.assertIn("TURN 3.", second)
        # the RESULT section lists the model's three commands and the relayed descent
        self.assertIn("- right move z -15.0: ok", second.split("PLANNED STEPS SINCE")[0])
        self.assertIn("- right point down: ok", second.split("PLANNED STEPS SINCE")[0])
        self.assertLess(second.index("PLANNED STEPS SINCE"), second.index("CURRENT STATE:"))
        self.assertEqual(trace[1]["relay_end"]["reason"], "check_failed")
        self.assertEqual(trace[1]["relay_end"]["failed_check"]["p_yes"], [0.2])
        self.assertEqual([l["turn"] for l in logged], [2, 3])
        self.assertEqual((result.turns, result.model_calls, result.success), (3, 2, False))

    def test_code_rules_stop_the_relay(self):
        cases = {
            "command_failed": [reply(["right move z -20", "right move z -5"], next_=[{"commands": ["right gripper close"], "expect": ["x?"]}])],
            "empty_close": [reply(["right gripper close"], next_=[{"commands": ["right move z +5"], "expect": ["x?"]}])],
        }
        for reason, script in cases.items():
            with self.subTest(reason), tempfile.TemporaryDirectory() as tmp:
                result, trace, _, client, _ = run(relay.RelayAgent, script, max_turns=2, tmp=tmp, variant="open")
                self.assertEqual(result.relayed_steps, 0)
                self.assertEqual(result.relay_ends, {reason: 1})
                self.assertIn(f"Planned step 1 did NOT run: {relay.STOP_TEXT[reason]}", client.prompts[1][2])

    def test_lost_object_stops_the_relay(self):
        env = FakeEnv()
        env.drop_at_step = 6      # the cup slips during the lift of planned step 1
        script = [reply(["right point down", "right move x -10", "right move y +10", "right move z -15"],
                        next_=[{"commands": ["right gripper close", "right move z +5"], "expect": ["x?"]},
                               {"commands": ["right move z +5"], "expect": ["y?"]}])]
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, _, _ = run(relay.RelayAgent, script, max_turns=3, env=env, tmp=tmp, variant="open")
        self.assertEqual(result.relay_ends, {"object_lost": 1})
        self.assertEqual(result.relayed_steps, 1)

    def test_relayed_steps_count_against_the_turn_limit(self):
        script = [reply(["right move z +1"], next_=[{"commands": ["right move z +1"], "expect": ["a?"]}] * 3)]
        with tempfile.TemporaryDirectory() as tmp:
            result, trace, _, _, _ = run(relay.RelayAgent, script, max_turns=2, tmp=tmp, variant="open")
        self.assertEqual((result.turns, result.model_calls, result.relayed_steps), (2, 1, 1))
        self.assertEqual(result.relay_ends, {"turn_limit": 1})
        self.assertEqual(result.finished_reason, "max_turns")

    def test_checked_variant_needs_expectations(self):
        script = [reply(["right move z +1"], next_=[{"commands": ["right move z +1"]}])]
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, _, _ = run(relay.RelayAgent, script, max_turns=2, tmp=tmp, variant="checked",
                                     checker=FakeChecker([]))
        self.assertEqual((result.relay_ends, result.checker_calls), ({"no_expect": 1}, 0))

    def test_prompt_variant_never_relays_but_sends_the_same_system_prompt(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            prompt = run(relay.RelayAgent, PLANNED, max_turns=2, tmp=a, variant="prompt")
            open_ = run(relay.RelayAgent, PLANNED, max_turns=2, tmp=b, variant="open")
        self.assertEqual(prompt[0].relayed_steps, 0)
        self.assertEqual(prompt[3].prompts[0][1], open_[3].prompts[0][1])
        self.assertIn("none of your planned steps ran", prompt[3].prompts[1][2])

    def test_malformed_plans_never_end_the_episode_and_are_reported(self):
        for bad in ([{"commands": 5, "expect": ["a?"]}], [{"commands": True}], "soon", [{"commands": ["right fly"]}]):
            with self.subTest(bad), tempfile.TemporaryDirectory() as tmp:
                result, trace, _, client, _ = run(relay.RelayAgent, [reply(["right move z +1"], expect=["a?"], next_=bad)],
                                                  max_turns=2, tmp=tmp, variant="open")
                self.assertEqual(result.relayed_steps, 0)
                self.assertTrue(trace[0]["plan_notes"])
                self.assertIn("PLANNED STEPS: none of your planned steps ran: ", client.prompts[1][2])

    def test_an_object_lost_before_does_not_block_the_other_arm(self):
        env = FakeEnv()
        env.drop_at_step = 6      # the right arm drops the cup in turn 2
        script = [reply(["right point down", "right move x -10", "right move y +10", "right move z -15"]),
                  reply(["right gripper close", "right move z +5"]),
                  reply(["left move z +1"], expect=["a?"], next_=[{"commands": ["left move z +1"], "expect": ["b?"]}])]
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, client, _ = run(relay.RelayAgent, script, max_turns=4, env=env, tmp=tmp, variant="open")
        self.assertEqual((result.relayed_steps, result.relay_ends), (1, {"all_ran": 1}))
        self.assertIn("RIGHT arm closed on an object", client.prompts[2][2])    # the loss is in upstream's state text

    def test_a_loss_during_a_step_names_the_arm(self):
        env = FakeEnv()
        env.drop_at_step = 6
        script = [reply(["right point down", "right move x -10", "right move y +10", "right move z -15"], expect=["a?"],
                        next_=[{"commands": ["right gripper close", "right move z +5"], "expect": ["b?"]},
                               {"commands": ["right move z +5"]}])]
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, client, _ = run(relay.RelayAgent, script, max_turns=3, env=env, tmp=tmp, variant="open")
        self.assertEqual(result.relay_ends, {"object_lost": 1})
        self.assertIn("Planned step 2 did NOT run: the right arm lost the object it held.", client.prompts[1][2])

    def test_press_tasks_do_not_count_closes_on_nothing(self):
        script = [reply(["right gripper close"], expect=["a?"], next_=[{"commands": ["right move z -5"]}])]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AgentConfig(max_turns=2)
            agent = relay.RelayAgent(cfg, ScriptedClient(script), Path(tmp)/"run", task="click_bell", variant="open",
                                     empty_grasp_limit=1)
            result = agent.run_episode(FakeEnv(), "Press the bell.", Path(tmp)/"run/episode_000")
        self.assertEqual((result.empty_closes, result.relayed_steps, result.finished_reason), (0, 1, "max_turns"))
        self.assertNotIn("a close finds nothing", relay.relay_system_prompt(
            relay.FORMAT_LINE.format(n=4), 4, press_task=True))
        self.assertIn("a close finds nothing", relay.relay_system_prompt(relay.FORMAT_LINE.format(n=4), 4))

    def test_checker_tokens_and_unanswered_questions(self):
        checker = FakeChecker([None])
        with tempfile.TemporaryDirectory() as tmp:
            result, trace, _, client, _ = run(relay.RelayAgent, PLANNED, max_turns=2, tmp=tmp, variant="checked",
                                              checker=checker)
        self.assertEqual((result.relay_ends, result.checker_input_tokens), ({"check_failed": 1}, 10))
        self.assertIn("the checker did not answer YES to \"Is the right gripper above the cup?\"", client.prompts[1][2])

    def test_empty_grasp_limit_ends_the_episode(self):
        script = [reply(["right gripper close", "right gripper open", "right gripper close"])]
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, _, _ = run(relay.RelayAgent, script, tmp=tmp, variant="baseline", empty_grasp_limit=2)
        self.assertEqual((result.finished_reason, result.empty_closes, result.turns), ("empty_grasp_limit", 2, 1))


class FakeMonitor:
    """Binds whatever coordinates it gets and reports a fall from `fall_at` (a turn) on."""

    def __init__(self, fall_at=99):
        self.objects, self.calls, self.fall_at, self.points, self.turns = [], 0, fall_at, None, []
        self.grippers = []

    def note_close(self, arm, tcp, caught):
        self.grippers.append(("close", arm, tuple(tcp), caught))

    def note_open(self, arm, tcp):
        self.grippers.append(("open", arm, tuple(tcp)))

    def bind(self, cap, points):
        self.points, self.objects = points, ["object"] * len(points)
        return [{"at": p} for p in points]

    def check(self, cap):
        self.turns.append(cap.turn)
        self.calls += 2
        return [{"status": "seen", "streak": 1}]

    def fallen(self):
        return "the object located at (10, -10) on turn 1 has fallen over" if self.turns and self.turns[-1] >= self.fall_at else None


class FakeLiveCapture:
    def __init__(self, env, obs, turn):
        self.turn = turn


SCENE = [reply(["right move z +1"], scene="Cup at approx (10, -10), coaster at (-20, 5).")] + [reply(["right move z +1"])] * 6


class MonitorTests(unittest.TestCase):
    def setUp(self):
        from branchlab.rtscene import live
        patcher = patch.object(live, "LiveCapture", FakeLiveCapture)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_fall_ends_the_episode_before_that_turns_reply_runs(self):
        mon = FakeMonitor(fall_at=3)
        with tempfile.TemporaryDirectory() as tmp:
            result, trace, _, client, env = run(relay.RelayAgent, SCENE, tmp=tmp, variant="baseline", monitor=lambda: mon)
            log = json.loads((Path(tmp)/"run/episode_000/monitor.json").read_text())
        self.assertEqual(mon.points, [(10.0, -10.0), (-20.0, 5.0)])
        self.assertEqual((result.model_calls, result.turns, result.finished_reason), (3, 3, "monitor_fallen"))
        self.assertEqual(result.monitor_stop["turn"], 3)
        self.assertEqual((result.monitor_bound, result.monitor_calls), (2, 4))
        self.assertEqual(trace[0]["monitor_bound"], [{"at": [10.0, -10.0]}, {"at": [-20.0, 5.0]}])
        self.assertEqual((trace[-1]["commands"], trace[-1]["monitor_stop"][:10]), ([], "the object"))
        self.assertEqual(env.arms["right"]["position_cm"][2], 97.0)          # two of the three replies ran
        self.assertEqual([e["turn"] for e in log], [2, 3])

    def test_a_quiet_monitor_changes_nothing_the_model_sees(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            base = run(relay.RelayAgent, SCENE, max_turns=5, tmp=a, variant="baseline")
            mon = run(relay.RelayAgent, SCENE, max_turns=5, tmp=b, variant="baseline", monitor=FakeMonitor)
        self.assertEqual(base[3].prompts, mon[3].prompts)
        self.assertEqual((mon[0].turns, mon[0].finished_reason), (5, "max_turns"))

    def test_grasps_and_releases_reach_the_monitor(self):
        mon = FakeMonitor()
        script = [reply(["right point down", "right move x -10", "right move y +10"], scene="Cup at (10, -10)."),
                  reply(["right move z -15", "right gripper close", "right gripper open"])]
        with tempfile.TemporaryDirectory() as tmp:
            run(relay.RelayAgent, script, max_turns=2, tmp=tmp, variant="baseline", monitor=lambda: mon)
        self.assertEqual([g[0] for g in mon.grippers], ["close", "open"])
        self.assertTrue(mon.grippers[0][3])                                  # the close caught the cup
        self.assertEqual(mon.grippers[0][2][:2], (10.0, -10.0))

    def test_off_for_tasks_that_turn_objects_on_purpose(self):
        cfg = AgentConfig(max_turns=3)
        agent = relay.RelayAgent(cfg, ScriptedClient([]), None, task="open_laptop", variant="baseline", monitor=FakeMonitor)
        self.assertIsNone(agent.monitor_factory)

    def test_a_broken_capture_is_logged_and_the_episode_goes_on(self):
        from branchlab.rtscene import live

        def broken(env, obs, turn):
            raise RuntimeError("no Position picture")
        with patch.object(live, "LiveCapture", broken), tempfile.TemporaryDirectory() as tmp:
            result, *_ = run(relay.RelayAgent, SCENE, max_turns=3, tmp=tmp, variant="baseline", monitor=FakeMonitor)
        self.assertEqual((result.turns, result.monitor_bound), (3, 0))


class ShoeEnv(FakeEnv):
    """FakeEnv (the cup stands in for a shoe) with a 640x480 top camera looking straight down and its matrices."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observed = 0

    def observe(self):
        self.observed += 1
        obs = super().observe()
        rng = np.random.default_rng(self._steps)
        obs["images"]["top_camera"] = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        E = np.hstack([R, (-R @ np.array([0.0, -0.10, 1.65]))[:, None]])
        obs["camera_params"] = {"top_camera": {"intrinsic_cv": np.array([[514.68, 0, 320], [0, 514.68, 240], [0, 0, 1]]),
                                               "extrinsic_cv": E}}
        return obs


class FakeClef:
    """Clef's answers to the toe question, in order: (direction, p) gives the direction p and left the rest (or
    left p and the rest spread evenly), or an error to raise."""

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def __call__(self, state, questions, images):
        self.asked.append((state, questions, images))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        direction, p = answer
        probs = {k: 0.0 for k in ("left", "right", "up", "down")} | {"left": 1 - p, direction: p}
        if direction == "left":
            probs |= {k: (1 - p) / 3 for k in ("right", "up", "down")}
        return {"toe": {"type": "choice", "probabilities": probs}}


SHOE = "place_dual_shoes"
REACH = reply(["right point down", "right move x -10", "right move y +10", "right move z -15"])    # to the cup, z 80
HOLD_AND_OPEN = reply(["right gripper close", "right move z +5", "right gripper open", "right move z +5"])


class ReleaseGateTests(unittest.TestCase):
    def gate(self, mode="clef", *answers, threshold=0.8):
        from branchlab.rtscene import release_gate
        clef = FakeClef(*answers)
        return release_gate.ReleaseGate(mode, clef if mode == "clef" else None, threshold), clef

    def test_off_reproduces_upstream_in_a_shoe_task(self):
        script = [REACH, HOLD_AND_OPEN, reply(["right gripper open"])]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            up = run(MLLMDiscreteAgent, script, max_turns=3, tmp=a, task=SHOE, env=ShoeEnv())
            ours = run(relay.RelayAgent, script, max_turns=3, tmp=b, task=SHOE, env=ShoeEnv(), variant="baseline")
        self.assertEqual(up[1], ours[1])
        self.assertEqual(up[3].prompts, ours[3].prompts)
        self.assertEqual((ours[0].gate_calls, ours[4].observed), (0, 4))   # turn starts and the final views

    def test_a_confident_wrong_toe_holds_the_open_back_once(self):
        gate, clef = self.gate("clef", ("right", 0.9))
        script = [REACH, HOLD_AND_OPEN, reply(["right gripper open", "right move z +5"])]
        with tempfile.TemporaryDirectory() as tmp:
            result, trace, memory, client, env = run(relay.RelayAgent, script, max_turns=3, tmp=tmp, task=SHOE,
                                                     env=ShoeEnv(), variant="baseline", release_gate=gate)
            crop = Image.open(Path(tmp)/"run/episode_000/turn002_gate_right.jpg")
            self.assertEqual(crop.size, (256, 256))
        self.assertEqual((result.gate_calls, result.gate_vetoes, result.gate_errors), (1, 1, 0))
        self.assertEqual(len(clef.asked), 1)
        state, questions, images = clef.asked[0]
        self.assertEqual(questions["toe"]["instructions"],
                         "Which way does the toe of the shoe held in the centre point, in the picture?")
        self.assertEqual(sorted(questions["toe"]["criteria"]), ["down", "left", "right", "up"])
        self.assertTrue(state.startswith("A robot is moving shoes on a table.") and len(images) == 1)
        # the open and the move after it did not run; the result tells Gemini why
        self.assertEqual([r["ok"] for r in trace[1]["results"]], [True, True, False])
        self.assertIn('"right gripper open" NOT executed (release check): the fast checker sees the toe of the held '
                      "shoe pointing RIGHT in the top camera", trace[1]["results"][2]["note"])
        self.assertIn("NOT executed (release check)", client.prompts[2][2])
        gated = trace[1]["gate"][0]
        self.assertEqual((gated["turn"], gated["arm"], gated["mode"], gated["vetoed"], gated["seen"]),
                         (2, "right", "clef", True, "right"))
        self.assertAlmostEqual(gated["p_not_left"], 0.9)
        self.assertEqual(gated["image"], "turn002_gate_right.jpg")
        # the second open of the same hold runs unchecked
        self.assertNotIn("gate", trace[2])
        self.assertEqual(trace[2]["results"][0]["note"], "gripper opened")
        self.assertEqual(env.arms["right"]["position_cm"][2], 90.0)

    def test_a_new_hold_rearms_the_gate_and_a_reclose_does_not(self):
        gate, clef = self.gate("clef", ("up", 0.95), ("down", 0.85))
        script = [REACH,
                  reply(["right gripper close", "right gripper open"]),       # vetoed
                  reply(["right gripper close", "right gripper open"]),       # closes again on the same hold: runs
                  reply(["right gripper close", "right gripper open"])]       # a new hold: vetoed again
        with tempfile.TemporaryDirectory() as tmp:
            result, trace, _, _, _ = run(relay.RelayAgent, script, max_turns=4, tmp=tmp, task=SHOE, env=ShoeEnv(),
                                         variant="baseline", release_gate=gate)
        self.assertEqual((result.gate_calls, result.gate_vetoes, len(clef.asked)), (2, 2, 2))
        self.assertEqual([("gate" in t) for t in trace], [False, True, False, True])
        self.assertIn("pointing away from the robot", trace[1]["results"][1]["note"])
        self.assertIn("pointing toward the robot", trace[3]["results"][1]["note"])

    def test_a_toe_seen_left_or_an_unsure_answer_lets_the_open_run(self):
        for answer in (("left", 0.9), ("right", 0.75)):
            gate, clef = self.gate("clef", answer)
            with self.subTest(answer), tempfile.TemporaryDirectory() as tmp:
                result, trace, _, _, env = run(relay.RelayAgent, [REACH, HOLD_AND_OPEN], max_turns=2, tmp=tmp,
                                               task=SHOE, env=ShoeEnv(), variant="baseline", release_gate=gate)
            self.assertEqual((result.gate_calls, result.gate_vetoes), (1, 0))
            self.assertEqual([r["ok"] for r in trace[1]["results"]], [True] * 4)
            self.assertFalse(trace[1]["gate"][0]["vetoed"])

    def test_a_failing_gate_lets_the_open_run(self):
        class NoCamera(ShoeEnv):
            def observe(self):
                obs = super().observe()
                del obs["camera_params"]
                return obs
        for env, answers in ((ShoeEnv(), [RuntimeError("Clef HTTP 503")]), (NoCamera(), [("right", 0.99)])):
            gate, clef = self.gate("clef", *answers)
            with self.subTest(type(env).__name__), tempfile.TemporaryDirectory() as tmp:
                result, trace, _, _, _ = run(relay.RelayAgent, [REACH, HOLD_AND_OPEN], max_turns=2, tmp=tmp,
                                             task=SHOE, env=env, variant="baseline", release_gate=gate)
            self.assertEqual((result.gate_calls, result.gate_vetoes, result.gate_errors), (1, 0, 1))
            self.assertEqual([r["ok"] for r in trace[1]["results"]], [True] * 4)
            self.assertTrue(trace[1]["gate"][0]["error"])

    def test_a_lost_object_is_not_checked(self):
        env = ShoeEnv()
        env.drop_at_step = 6          # the cup slips during the lift
        gate, clef = self.gate("clef", ("right", 0.99))
        with tempfile.TemporaryDirectory() as tmp:
            result, *_ = run(relay.RelayAgent, [REACH, HOLD_AND_OPEN], max_turns=2, tmp=tmp, task=SHOE, env=env,
                             variant="baseline", release_gate=gate)
        self.assertEqual((result.gate_calls, len(clef.asked)), (0, 0))

    def test_other_tasks_are_never_gated(self):
        gate, clef = self.gate("clef", ("right", 0.99))
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            base = run(relay.RelayAgent, [REACH, HOLD_AND_OPEN], max_turns=2, tmp=a, env=ShoeEnv(), variant="baseline")
            gated = run(relay.RelayAgent, [REACH, HOLD_AND_OPEN], max_turns=2, tmp=b, env=ShoeEnv(), variant="baseline",
                        release_gate=gate)
        self.assertEqual(base[1], gated[1])
        self.assertEqual((gated[0].gate_calls, len(clef.asked)), (0, 0))

    def test_nudge_holds_back_once_without_clef_or_a_render(self):
        gate, _ = self.gate("nudge")
        script = [REACH, HOLD_AND_OPEN, reply(["right gripper open"])]
        with tempfile.TemporaryDirectory() as tmp:
            result, trace, _, _, env = run(relay.RelayAgent, script, max_turns=3, tmp=tmp, task="place_shoe",
                                           env=ShoeEnv(), variant="baseline", release_gate=gate)
            saved = sorted(p.name for p in (Path(tmp)/"run/episode_000").glob("*gate*"))
        self.assertEqual((result.gate_calls, result.gate_vetoes, env.observed), (1, 1, 4))   # turn starts, final views
        self.assertEqual(trace[1]["results"][2]["note"],
                         '"right gripper open" NOT executed (release check): Before releasing, check in the images '
                         "that the shoe's toe points the way the task asks; send the open again to release.")
        self.assertEqual((trace[1]["gate"][0]["probabilities"], saved), (None, []))
        self.assertEqual(trace[2]["results"][0]["note"], "gripper opened")

    def test_a_veto_in_a_planned_step_stops_the_relay(self):
        gate, _ = self.gate("nudge")
        script = [reply(REACH["commands"], next_=[{"commands": ["right gripper close", "right gripper open"]},
                                                  {"commands": ["right move z +5"]}])]
        with tempfile.TemporaryDirectory() as tmp:
            result, trace, _, client, _ = run(relay.RelayAgent, script, max_turns=3, tmp=tmp, task=SHOE, env=ShoeEnv(),
                                              variant="open", release_gate=gate)
        self.assertEqual((result.relayed_steps, result.relay_ends, result.gate_vetoes), (1, {"command_failed": 1}, 1))
        self.assertEqual(trace[1]["gate"][0]["turn"], 2)

    def test_task_note_only_for_place_shoe_and_only_when_asked(self):
        note = "The shoe counts as placed only with its toe pointing left (-x)."
        for task, task_note, shown in (("place_shoe", "shoe-left", True), ("place_shoe", None, False),
                                       (SHOE, "shoe-left", False)):
            with self.subTest(task=task, note=task_note), tempfile.TemporaryDirectory() as tmp:
                _, _, _, client, _ = run(relay.RelayAgent, [reply(["wait"])], max_turns=1, tmp=tmp, task=task,
                                         variant="baseline", task_note=task_note)
            self.assertEqual(note in client.prompts[0][2], shown)
            self.assertNotIn(note, client.prompts[0][1])                  # the system prompt is unchanged
        for instruction in ("Drop the shoe on the mat", "Drop the shoe on the mat. "):   # RoboTwin often has no period
            with self.subTest(instruction=instruction), tempfile.TemporaryDirectory() as tmp:
                _, _, _, client, _ = run(relay.RelayAgent, [reply(["wait"])], max_turns=1, tmp=tmp, task="place_shoe",
                                         variant="baseline", task_note="shoe-left", instruction=instruction)
            self.assertIn("TASK: Drop the shoe on the mat. " + note + "\n", client.prompts[0][2])
        with self.assertRaisesRegex(ValueError, "task note"):
            relay.RelayAgent(AgentConfig(), ScriptedClient([]), None, task="place_shoe", task_note="shoe-right")


class PlanParsingTests(unittest.TestCase):
    def test_plan_is_cut_at_the_first_unusable_step(self):
        good = {"commands": ["left move z -5"], "expect": ["a?", "", 3]}
        for bad, why in [({"commands": ["left fly"]}, "invalid command"), ({"commands": ["done"]}, "\"done\""),
                         ({"commands": ["wait"] * 5}, "more than 4"), ("left move z 1", "not an object"),
                         ({"commands": []}, "not an object")]:
            expect, steps, notes = relay.parse_plan({"expect": "q?", "next": [good, bad, good]}, 4)
            self.assertEqual(expect, ["q?"])
            self.assertEqual(len(steps), 1)
            self.assertEqual(steps[0].expect, ["a?"])
            self.assertIn(why, notes[0])

    def test_extra_steps_are_dropped_with_a_note(self):
        _, steps, notes = relay.parse_plan({"next": [{"commands": "wait"}] * 5}, 4, max_steps=3)
        self.assertEqual(len(steps), 3)
        self.assertIn("first 3", notes[0])
        self.assertEqual(relay.parse_plan({"next": "soon"}, 4)[2], ["\"next\" was not a list; no planned step was used"])

    def test_relay_prompt_requires_the_pinned_format_block(self):
        with self.assertRaisesRegex(ValueError, "RESPONSE FORMAT"):
            relay.relay_system_prompt("no format here", 4)


class CheckerTests(unittest.TestCase):
    IMAGES = [Image.new("RGB", (64, 48), (200, 10, 10))] * 2

    def test_clef_request_and_413_fallback(self):
        sent = []

        def post(url, body, headers, timeout_s):
            sent.append((url, body, headers))
            if len(sent) == 1:
                return 413, "too large"
            return 200, {"answers": {"q1": {"type": "noul", "noul": 0.8}, "q2": {"type": "noul", "noul": 0.4}},
                         "usage": {"cost": 0.0012}}
        checker = relay.ClefChecker("secret-key")
        with patch.object(relay, "_post_json", side_effect=post):
            verdict = checker.check(["a?", "b?"], "context", self.IMAGES)
        self.assertEqual((verdict["p_yes"], verdict["passed"], verdict["cost_usd"]), ([0.8, 0.4], False, 0.0012))
        url, body, headers = sent[1]
        self.assertEqual(url, relay.ClefChecker.URL)
        self.assertEqual(body["model"], "cloudflare/clef")
        self.assertEqual(body["questions"], {"q1": {"type": "noul", "instructions": "a?"},
                                             "q2": {"type": "noul", "instructions": "b?"}})
        self.assertEqual(body["state"][0], {"type": "text", "text": "context"})
        self.assertTrue(body["state"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertLess(len(body["state"][1]["image_url"]["url"]), len(sent[0][1]["state"][1]["image_url"]["url"]))
        self.assertEqual(headers, {"Authorization": "Bearer secret-key"})

    def test_checker_failure_does_not_pass(self):
        checker = relay.ClefChecker("k", max_retries=2)
        with patch.object(relay, "_post_json", return_value=(503, "busy")), patch.object(relay.time, "sleep") as sleep:
            verdict = checker.check(["a?"], "c", self.IMAGES)
        self.assertFalse(verdict["passed"])
        self.assertIn("HTTP 503", verdict["error"])
        self.assertEqual(sleep.call_count, 1)                    # no wait after the last attempt

    def test_checker_exceptions_become_errors(self):
        checker = relay.ClefChecker("k", max_retries=2)
        with patch.object(relay.ClefChecker, "_ask", side_effect=RuntimeError("token refresh failed")), \
                patch.object(relay.time, "sleep"):
            verdict = checker.check(["a?"], "c", self.IMAGES)
        self.assertEqual((verdict["passed"], verdict["error"]), (False, "RuntimeError: token refresh failed"))

    def test_errors_quoting_headers_are_withheld(self):
        status, text = relay._post_json("http://127.0.0.1:9/", {}, {"Authorization": "Bearer sk-SECRET\nsk-OTHER"}, 1)
        self.assertEqual(status, 0)
        self.assertNotIn("SECRET", text)

    def test_lite_reply_parsing(self):
        with patch("google.oauth2.service_account.Credentials.from_service_account_file") as creds, \
                tempfile.NamedTemporaryFile("w", suffix=".json") as key:
            key.write(json.dumps({"project_id": "demo"}))
            key.flush()
            creds.return_value.valid, creds.return_value.token = True, "tok"
            checker = relay.LiteChecker(Path(key.name))
            out = {"candidates": [{"content": {"parts": [{"text": '{"q1": "yes", "q2": "No"}'}]}}],
                   "usageMetadata": {"promptTokenCount": 900}}
            with patch.object(relay, "_post_json", return_value=(200, out)) as post:
                verdict = checker.check(["a?", "b?"], "c", self.IMAGES)
        body = post.call_args[0][1]
        self.assertEqual(body["generationConfig"]["thinkingConfig"], {"thinkingLevel": "minimal"})
        self.assertIn("q2: b?", body["contents"][0]["parts"][0]["text"])
        self.assertEqual((verdict["p_yes"], verdict["passed"]), ([1.0, 0.0], False))


class CompareTests(unittest.TestCase):
    def make_run(self, root, label, variant, script, **kwargs):
        shard = root/"robodawn_dev"/label/TASK/"shard_24"
        cfg = AgentConfig(max_turns=6)
        shard.mkdir(parents=True)
        agent = relay.RelayAgent(cfg, ScriptedClient(script), shard, task=TASK, variant=variant, **kwargs)
        result = agent.run_episode(FakeEnv(), "Put the cup on the coaster.", shard/"episode_024")
        (shard/"results.json").write_text(json.dumps({"episodes": [{"episode_index": 24, "seed": 1, **asdict(result)}]}))
        return result

    def test_paired_comparison_and_truncation(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = [reply(["right gripper close"]), reply(["right gripper open"]), reply(["right gripper close"])]
            self.make_run(root, "relay_baseline_flex", "baseline", empty + GRASP_AND_LIFT[:1])
            self.make_run(root, "relay_checked_clef_flex", "checked", PLANNED, checker=FakeChecker([0.9, 0.9]))
            out = io.StringIO()
            argv = ["compare_relay.py", "--dev", "--runs", str(root), "--empty-grasp-limit", "2"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
                compare_relay.main()
        text = out.getvalue()
        self.assertIn("paired episodes (finished by every run): 1 over 1 tasks", text)
        self.assertIn("relay_checked_clef_flex        +1 episodes (gained 1, lost 0", text)
        truncated = text.split("empty close number 2")[1]
        base_row = next(l for l in truncated.splitlines() if l.startswith("relay_baseline_flex"))
        self.assertEqual(base_row.split()[1:3], ["0/1", "3.0"])          # ended at turn 3, the second empty close
        self.assertIn("relay_checked_clef_flex checker calls on the paired episodes: 2, passed 100%", text)


class RunnerTests(unittest.TestCase):
    def args(self, **changes):
        fields = dict(repo=ROOT/"third_party/robodawn", manifest=ROOT/"robodawn_site/reproduction_manifest.json",
                      task=TASK, episodes=5, start_episode=24, output=None, variant="checked", checker="clef",
                      threshold=0.5, max_steps=3, empty_grasp_limit=0, max_tokens=8000, no_video=False, tier="flex",
                      robotwin_root=Path("/content/RoboTwin"))
        fields.update(changes)
        return argparse.Namespace(**fields)

    def test_development_and_evaluation_outputs_are_separate(self):
        dev, _ = runner.configuration(self.args())
        self.assertFalse(dev["evaluation"])
        self.assertTrue(dev["output"].endswith("outputs/robodawn_dev/relay_checked_clef_flex/place_empty_cup/shard_24"))
        self.assertEqual([e["episode"] for e in dev["episodes"]], [24, 25, 26, 27, 28])
        tuned, _ = runner.configuration(self.args(threshold=0.7, empty_grasp_limit=5))
        self.assertIn("/relay_checked_clef_t0.7_g5_flex/", tuned["output"])
        longer, flags = runner.configuration(self.args(max_tokens=16000, tier="standard"))
        self.assertIn("/relay_checked_clef_m16000_standard/", longer["output"])
        self.assertEqual((longer["max_tokens"], flags[flags.index("--max_tokens") + 1]), (16000, "16000"))
        names = set()
        for changes in ({}, dict(gate="clef"), dict(gate="clef", gate_threshold=0.9), dict(gate="nudge"),
                        dict(task_note="shoe-left"), dict(gate="clef", task_note="shoe-left"),
                        dict(gate="nudge", gate_threshold=0.9)):
            cfg, flags = runner.configuration(self.args(variant="baseline", checker=None, tier="standard", **changes))
            names.add(flags[flags.index("--label") + 1])
            self.assertEqual(Path(cfg["output"]).parents[1].name, flags[flags.index("--label") + 1])
            self.assertEqual(cfg["relay"].get("gate", {}).get("mode", "off"), changes.get("gate", "off"))
            self.assertEqual(cfg["relay"].get("task_note"), changes.get("task_note"))
        self.assertEqual(names, {"relay_baseline_standard", "relay_baseline_gateclef_standard",
                                 "relay_baseline_gateclef0.9_standard", "relay_baseline_gatenudge_standard",
                                 "relay_baseline_noteshoe-left_standard", "relay_baseline_gateclef_noteshoe-left_standard"})
        with self.assertRaisesRegex(ValueError, "gate-threshold"):
            runner.configuration(self.args(variant="baseline", checker=None, gate="clef", gate_threshold=1.5))
        ev, flags = runner.configuration(self.args(start_episode=0, variant="open", checker=None))
        self.assertTrue(ev["evaluation"])
        self.assertTrue(ev["output"].endswith("outputs/robodawn/relay_open_flex/place_empty_cup/shard_0"))
        self.assertIn("demo_path", ev["episodes"][0])
        self.assertEqual(flags[flags.index("--label") + 1], "relay_open_flex")

    def test_openrouter_key_with_two_lines_is_refused(self):
        with tempfile.NamedTemporaryFile("w") as f:
            f.write("sk-or-OLD\nsk-or-NEW\n")
            f.flush()
            with self.assertRaisesRegex(ValueError, "whitespace") as caught:
                runner.openrouter_key(Path(f.name))
        self.assertNotIn("sk-or", str(caught.exception))

    def test_bad_combinations_are_rejected(self):
        for changes, message in [(dict(start_episode=8), "not both"), (dict(start_episode=48), "within"),
                                 (dict(start_episode=15), r"development episodes \[17\] use a demonstration"),
                                 (dict(checker=None), "needs --checker"), (dict(variant="open"), "only")]:
            with self.subTest(changes), self.assertRaisesRegex(ValueError, message):
                runner.configuration(self.args(**changes))


if __name__ == "__main__":
    unittest.main()
