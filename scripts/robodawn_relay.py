"""RoboDawn's agent loop with the big model's planned next steps relayed through a fast checker.

The big model (Gemini, through the unchanged harness client) answers each turn as upstream does and may
add two fields: `expect`, yes/no questions about what the images will show once its `commands` have run,
and `next`, up to MAX_STEPS further steps, each with its own commands and expectations. Per variant:

    baseline  ignores both fields: upstream prompt and loop (checks that this loop matches upstream)
    prompt    asks for both fields but runs only `commands`: the effect of the reply format alone
    open      runs the steps of `next` in order without calling the big model, stopped by code rules only
    checked   runs a step only when the fast checker answers yes to every expectation of the step before

prompt, open and checked send the same system prompt, so they differ only in what the harness does with
the plan. Code rules stop a relay in open and checked alike: a command reported FAILED or invalid, a
close that found nothing between the fingers (except in press tasks, which close the fingers on purpose
to press), an object lost during the step just run, or the end of the episode (success, step budget,
turn limit, harness early stop). Every executed segment, the big model's commands or one
relayed step, counts as one turn against max_turns, so a variant never acts more often than the baseline
could; what it saves is big-model calls. The big model's next reply lists every command that ran since
its last reply and says how far the relay got and why it stopped.

Upstream code is imported, not modified. run_episode is upstream's loop (harness/agent/mllm_agent.py at
the pinned commit) with the relay added; in the baseline variant it produces upstream's trace, memory and
result (tests/test_robodawn_relay.py runs both on a scripted environment and compares them).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from harness.agent.memory import AgentMemory
from harness.agent.mllm_agent import (EpisodeResult, MLLMDiscreteAgent, _json_default, _update_grasp_facts,
                                      _where, encode_png)
from harness.agent.prompts import parse_agent_json, state_text, system_prompt, turn_text
from harness.agent.llm_client import text_part
from harness.core import watchdog
from harness.core.commands import parse_command_list

logger = logging.getLogger(__name__)

VARIANTS = ("baseline", "prompt", "open", "checked")
MAX_STEPS = 3            # planned steps accepted per reply
MAX_QUESTIONS = 4        # expectations per step
EMPTY_CLOSE_OPENING = 0.08   # the environment's own threshold for "fingers closed fully: nothing between them"
LOST_OPENING = 0.05          # prompts.state_text's threshold for a held object being LOST
# Tasks whose demonstrations close the fingers on nothing on purpose, to press or push with them (their demo.json
# effects say "nothing held"): there such a close is neither an empty grasp nor a reason to stop.
PRESS_TASKS = {"click_bell", "click_alarmclock", "press_stapler", "turn_switch"}

# Inserted into upstream's RESPONSE FORMAT block in place of its last line.
FORMAT_LINE = '  "commands": ["<command>", ...]   // 1 to {n} commands, executed in order\n}}\n'
RELAY_FORMAT = (
    '  "commands": ["<command>", ...],   // 1 to {n} commands, executed in order\n'
    '  "expect": ["<yes/no question>", ...],   // with "next": what the images will show once these commands worked (at most 4)\n'
    '  "next": [{{"commands": [...], "expect": [...]}}, ...]   // optional: up to {steps} further steps, see PLANNED STEPS\n'
    '}}\n'
)
# v2 (2026-10-08): v1 drew questions about positions, angles and openings, which code checks anyway and a checker
# cannot read from images, and "Leave next out when unsure" kept most replies without a plan.
RELAY_NOTE = (
    "\n\nPLANNED STEPS: you may plan ahead. \"expect\" lists yes/no questions about what the images will show once "
    "\"commands\" have run, if they had the intended effect. \"next\" lists up to {steps} further steps in order, each "
    "with 1 to {n} commands and its own \"expect\". After your commands run, the controller continues with the steps "
    "of \"next\" WITHOUT asking you, one step at a time. A step runs only if the step before it (or your \"commands\") "
    "has \"expect\" questions and a checker looking at the new images answers YES to every one of them, so give "
    "\"expect\" to your commands and to every step that another step follows. The controller stops and asks you again, "
    "telling you what ran and why it stopped, as soon as a command reports FAILED, {empty}a held object is lost, or the "
    "checker does not answer YES. Every step that runs uses one of your turns.\n"
    "Give \"next\" whenever you already know the moves that follow (e.g. descend to the grasp height, close, lift, "
    "carry to a position you have already read); end the list where the next move depends on something you must look "
    "at first, such as fine alignment in the wrist camera. Never put \"done\" in \"next\".\n"
    "The controller already checks the robot itself: whether each command reached its target, whether a close found "
    "something, whether a held object was lost. So do not ask about gripper positions, heights, angles, orientations "
    "or openings, and use no numbers. Ask only about what the images show of the task's objects and their contact "
    "with the gripper, one fact per question, phrased so that YES means \"as planned\". The checker sees only the "
    "current images, not earlier ones, so do not compare with before. Examples: \"In the left wrist camera, is the cup "
    "between the two fingers?\", \"Is the cup clearly lifted off the table?\", \"Is the block resting on the plate?\"."
)
STOP_TEXT = {
    "command_failed": "a command before it reported FAILED or was invalid",
    "empty_close": "a close before it found nothing between the fingers",
    "object_lost": "the {arm} arm lost the object it held",
    "no_expect": "the step before it had no \"expect\" questions to check",
    "checker_error": "the checker could not be reached",
}


def relay_system_prompt(base: str, max_commands: int, max_steps: int = MAX_STEPS, press_task: bool = False) -> str:
    """Upstream's system prompt with the relay fields; in press tasks a close on nothing is not a stop reason."""
    line = FORMAT_LINE.format(n=max_commands)
    if base.count(line) != 1:
        raise ValueError("upstream RESPONSE FORMAT block not found; the relay prompt needs the pinned harness")
    empty = "" if press_task else "a close finds nothing between the fingers, "
    return (base.replace(line, RELAY_FORMAT.format(n=max_commands, steps=max_steps))
            + RELAY_NOTE.format(n=max_commands, steps=max_steps, empty=empty))


def with_relay_report(text: str, report: str) -> str:
    """Upstream's turn text with the relay report placed right before the current state."""
    marker = "\n\nCURRENT STATE:\n"
    if not report:
        return text
    if marker not in text:
        raise ValueError("upstream turn text has no CURRENT STATE section")
    return text.replace(marker, "\n\n" + report + marker, 1)


# ----------------------------------------------------------------------------- planned steps
@dataclass
class Step:
    commands: list            # parsed harness Commands
    expect: list[str]

    def as_json(self) -> dict:
        return {"commands": [c.text() for c in self.commands], "expect": self.expect}


def questions(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [q.strip() for q in value if isinstance(q, str) and q.strip()][:MAX_QUESTIONS]


def parse_plan(parsed: dict, max_commands: int, max_steps: int = MAX_STEPS) -> tuple[list[str], list[Step], list[str]]:
    """(expectations of `commands`, valid steps of `next`, notes on what was dropped).

    Steps are taken in order up to the first unusable one: a step that is not an object, has no commands,
    an invalid command, "done", or more commands than one turn allows ends the plan there."""
    expect, steps, notes = questions(parsed.get("expect")), [], []
    raw = parsed.get("next")
    if raw is None or raw == []:
        return expect, steps, notes
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return expect, steps, ["\"next\" was not a list; no planned step was used"]
    for i, item in enumerate(raw[:max_steps], 1):
        commands = item.get("commands") if isinstance(item, dict) else None
        if isinstance(commands, str):
            commands = [commands]
        if not isinstance(commands, list):
            commands = None
        parsed_commands, errors = parse_command_list(commands or [])
        why = ("it is not an object with \"commands\"" if not isinstance(item, dict) or not commands else
               f"invalid command ({errors[0]})" if errors else
               "\"done\" is not allowed in a planned step" if any(c.kind == "done" for c in parsed_commands) else
               f"more than {max_commands} commands" if len(parsed_commands) > max_commands else None)
        if why:
            notes.append(f"planned step {i} and any after it were dropped: {why}")
            return expect, steps, notes
        steps.append(Step(parsed_commands, questions(item.get("expect"))))
    if len(raw) > max_steps:
        notes.append(f"only the first {max_steps} planned steps are used")
    return expect, steps, notes


def lost_arms(grasp_facts: dict, state: dict) -> set:
    """Arms that closed on an object and whose fingers are now fully closed: the object was lost."""
    return {arm for arm in grasp_facts if (state.get(arm) or {}).get("gripper_real", 1.0) <= LOST_OPENING}


def code_stop(results: list[dict], empty_closes: int, grasp_facts: dict, state: dict,
              lost_before: set) -> tuple[Optional[str], Optional[str]]:
    """(why the relay must hand back after a segment with these results, the arm concerned), by code alone.

    An object counts as lost only if it was lost during the segment: an arm already lost before it (and
    not opened since, which would clear its grasp fact) does not stop relays for the other arm."""
    if any(not r.get("ok") for r in results):
        return "command_failed", None
    if empty_closes:
        return "empty_close", None
    for arm in sorted(lost_arms(grasp_facts, state) - lost_before):
        return "object_lost", arm
    return None, None


# ----------------------------------------------------------------------------- checkers
def jpeg_bytes(image, scale: float = 1.0, quality: int = 85) -> bytes:
    from PIL import Image

    image = image.convert("RGB")
    if scale != 1.0:
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


CHECKER_INTRO = (
    "A robot controller planned its next step and wrote yes/no questions that must all be true before that step "
    "runs. Answer each question from the images; answer yes only if the images show it."
)


def checker_context(instruction: str, ran: list[str], upcoming: list[str], state: dict, grasp_facts: dict,
                    captions: list[str]) -> str:
    return "\n\n".join([
        CHECKER_INTRO, f"TASK: {instruction}",
        "COMMANDS JUST RUN: " + ("; ".join(ran) or "(none)"),
        "NEXT STEP IF EVERY ANSWER IS YES: " + "; ".join(upcoming),
        "ROBOT STATE:\n" + state_text(state, grasp_facts),
        "IMAGES (in order): " + "; ".join(f"{i}. {c}" for i, c in enumerate(captions, 1)),
    ])


def safe_error(exc: BaseException) -> str:
    """An exception as text for logs, withheld when it quotes a request header (it would carry the key)."""
    text = f"{type(exc).__name__}: {exc}"
    if re.search(r"bearer|authorization|header", text, re.I):
        return f"{type(exc).__name__} (message withheld: it quotes a request header)"
    return text[:300]


def _post_json(url: str, body: dict, headers: dict, timeout_s: float) -> tuple[int, object]:
    try:
        request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json", **headers})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return 200, json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read().decode(errors="replace")[:300]
        except Exception as inner:  # noqa: BLE001  (the error body itself could not be read)
            return exc.code, safe_error(inner)
    except Exception as exc:  # noqa: BLE001  (timeouts, connection resets, bad JSON, bad header values)
        return 0, safe_error(exc)


class Checker:
    """Answers yes/no questions about the current images; subclasses implement _ask."""
    name = ""
    model = ""
    RETRY_STATUS = (0, 408, 429, 500, 502, 503, 504, 529)

    def __init__(self, threshold: float = 0.5, timeout_s: float = 60.0, max_retries: int = 4):
        self.threshold = threshold
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def check(self, questions_: list[str], context: str, images: list) -> dict:
        """{"p_yes", "passed", "seconds", "usage", "tokens": [in, out], "cost_usd", "error", "note"}; never raises.

        error: the checker could not be reached (after retries); note: it answered, but not every question
        (a missing answer counts as not YES)."""
        started = time.monotonic()
        n = len(questions_)
        for attempt in range(1, self.max_retries + 1):
            watchdog.beat(f"checker {self.name} attempt {attempt}")
            try:
                status, p_yes, usage, cost, error, note = self._ask(questions_, context, images)
            except Exception as exc:  # noqa: BLE001  (e.g. a failed token refresh)
                status, p_yes, usage, cost, error, note = 0, [None] * n, {}, None, safe_error(exc), None
            if status == 200 or status not in self.RETRY_STATUS or attempt == self.max_retries:
                break
            time.sleep(min(30.0, 2.0 ** attempt))
        passed = error is None and all(p is not None and p >= self.threshold for p in p_yes)
        return {"p_yes": p_yes, "passed": passed, "seconds": round(time.monotonic() - started, 3),
                "usage": usage, "tokens": list(self.tokens(usage)), "cost_usd": cost, "error": error, "note": note}

    def _ask(self, questions_, context, images):
        """(HTTP status, P(yes) per question, usage, cost in USD or None, error text, note)."""
        raise NotImplementedError

    @staticmethod
    def tokens(usage: dict) -> tuple[int, int]:
        return 0, 0


class ClefChecker(Checker):
    """Cloudflare's Clef decision model on OpenRouter: P(yes) per question."""
    name = "clef"
    URL = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, api_key: str, model: str = "clef", **kwargs):
        super().__init__(**kwargs)
        self._key = api_key
        self.model = model

    def _ask(self, questions_, context, images):
        qs = {f"q{i}": {"type": "noul", "instructions": q} for i, q in enumerate(questions_, 1)}
        for scale in (1.0, 0.5):      # two full-size images have drawn HTTP 413 before; halve them once
            state = [{"type": "text", "text": context}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                                   + base64.b64encode(jpeg_bytes(img, scale)).decode()}}
                for img in images]
            status, out = _post_json(self.URL, {"model": f"cloudflare/{self.model}", "state": state, "questions": qs},
                                     {"Authorization": f"Bearer {self._key}"}, self.timeout_s)
            if status != 413:
                break
        if status != 200:
            return status, [None] * len(questions_), {}, None, f"HTTP {status}: {out}", None
        out = out.get("result", out) if isinstance(out, dict) else {}
        answers = out.get("answers") or {}
        p_yes = [(answers.get(f"q{i}") or {}).get("noul") for i in range(1, len(questions_) + 1)]
        p_yes = [float(p) if isinstance(p, (int, float)) else None for p in p_yes]
        usage = out.get("usage") or {}
        missing = None if all(p is not None for p in p_yes) else "answer missing"
        return 200, p_yes, usage, usage.get("cost"), None, missing

    @staticmethod
    def tokens(usage):
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)


class LiteChecker(Checker):
    """A small Gemini on Vertex AI (service account), asked for a yes/no word per question."""
    name = "lite"
    URL = ("https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/"
           "{model}:generateContent")

    def __init__(self, credentials_path: Path, model: str = "gemini-3.1-flash-lite", thinking: str = "minimal",
                 **kwargs):
        super().__init__(**kwargs)
        from google.oauth2 import service_account
        self._project = json.loads(Path(credentials_path).read_text())["project_id"]
        self._credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path), scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.model = model
        self.thinking = thinking

    def _token(self) -> str:
        from google.auth.transport.requests import Request
        if not self._credentials.valid:
            self._credentials.refresh(Request())
        return self._credentials.token

    def _ask(self, questions_, context, images):
        listed = "\n".join(f"q{i}: {q}" for i, q in enumerate(questions_, 1))
        keys = ", ".join(f'"q{i}": "yes" or "no"' for i in range(1, len(questions_) + 1))
        parts = [{"text": f"{context}\n\nQUESTIONS:\n{listed}\n\nReply with one JSON object only: {{{keys}}}"}]
        parts += [{"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg_bytes(img)).decode()}}
                  for img in images]
        config = {"maxOutputTokens": 1000, "responseMimeType": "application/json"}
        if self.thinking:
            config["thinkingConfig"] = {"thinkingLevel": self.thinking}
        status, out = _post_json(self.URL.format(project=self._project, model=self.model),
                                 {"contents": [{"role": "user", "parts": parts}], "generationConfig": config},
                                 {"Authorization": f"Bearer {self._token()}"}, self.timeout_s)
        if status != 200:
            return status, [None] * len(questions_), {}, None, f"HTTP {status}: {out}", None
        candidate = (out.get("candidates") or [{}])[0]
        raw = "".join(p.get("text", "") for p in (candidate.get("content") or {}).get("parts", []) if not p.get("thought"))
        try:
            reply = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except ValueError:
            reply = {}
        words = [str(reply.get(f"q{i}", "")).strip().lower() for i in range(1, len(questions_) + 1)]
        p_yes = [1.0 if w == "yes" else 0.0 if w == "no" else None for w in words]
        missing = None if all(p is not None for p in p_yes) else f"unusable reply: {raw[:120]!r}"
        return 200, p_yes, out.get("usageMetadata") or {}, None, None, missing

    @staticmethod
    def tokens(usage):
        return (int(usage.get("promptTokenCount") or 0),
                int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0))


# ----------------------------------------------------------------------------- the agent
@dataclass
class RelayEpisodeResult(EpisodeResult):
    variant: str = "baseline"
    planned_steps: int = 0        # valid steps the big model put in `next`
    relayed_steps: int = 0        # turns run from `next` without a big-model call
    checker_calls: int = 0
    checker_seconds: float = 0.0
    checker_cost_usd: float = 0.0
    checker_input_tokens: int = 0
    checker_output_tokens: int = 0
    empty_closes: int = 0         # closes that found nothing between the fingers, in any segment (none in press tasks)
    relay_ends: dict = field(default_factory=dict)   # why each relay ended: all_ran, check_failed, ...


class RelayAgent(MLLMDiscreteAgent):
    def __init__(self, cfg, client, out_dir=None, task=None, *, variant: str = "baseline",
                 checker: Optional[Checker] = None, max_steps: int = MAX_STEPS, empty_grasp_limit: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if variant == "checked" and checker is None:
            raise ValueError("the checked variant needs a checker")
        super().__init__(cfg, client, out_dir, task=task)
        self.variant = variant
        self.checker = checker if variant == "checked" else None
        self.max_steps = max_steps
        self.empty_grasp_limit = empty_grasp_limit     # 0: off; else end the episode at that many empty closes
        self.press_task = task in PRESS_TASKS

    # ------------------------------------------------------------------
    def _execute(self, env, commands, grasp_facts) -> tuple[list[dict], bool, int]:
        """Upstream's command loop: (results, whether "done" was sent, closes that found nothing)."""
        results, done, empty = [], False, 0
        for cmd in commands:
            if cmd.kind == "done":
                done = True
                results.append({
                    "command": "done", "kind": "done", "arm": None, "ok": False,
                    "note": ("NOT finished: the benchmark checker has not registered success, so the episode continues. "
                             "Re-read the instruction; typical reasons: the object is not where/how the task wants it "
                             "(wrong spot, not high enough, not far enough to the side, wrong orientation), a gripper "
                             "must be opened at the end, or a second object/arm is still required. Keep acting."),
                })
                break
            res = env.execute(cmd)
            results.append(res)
            _update_grasp_facts(grasp_facts, cmd, res)
            if (not self.press_task and cmd.kind == "gripper" and cmd.value < 0.5 and res.get("ok")
                    and ((res.get("after") or {}).get(cmd.arm) or {}).get("gripper_real", 1.0) <= EMPTY_CLOSE_OPENING):
                empty += 1
            if env.success or env.budget_exhausted:
                break
        return results, done, empty

    def _log_check(self, episode_dir, turn, step, questions_, verdict) -> None:
        if not self.out_dir:
            return
        entry = {"episode": episode_dir.name if episode_dir else None, "turn": turn, "step": step,
                 "checker": self.checker.name, "model": self.checker.model, "questions": questions_, **verdict}
        with open(self.out_dir / "checker_calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    def run_episode(self, env, instruction: str, episode_dir: Optional[Path] = None) -> RelayEpisodeResult:
        cfg = self.cfg
        memory = AgentMemory(use_history=cfg.memory_history, use_scratchpad=cfg.memory_scratchpad,
                             history_turns=cfg.history_turns)
        sys_prompt = system_prompt(cfg.profile, cfg.max_commands_per_turn, cfg.memory_scratchpad,
                                   context=env.prompt_context())
        if self.variant != "baseline":
            sys_prompt = relay_system_prompt(sys_prompt, cfg.max_commands_per_turn, self.max_steps, self.press_task)
        episode_dir = Path(episode_dir) if episode_dir else None
        if episode_dir:
            episode_dir.mkdir(parents=True, exist_ok=True)
            (episode_dir / "system_prompt.txt").write_text(sys_prompt)
        trace = []
        last_results: list[dict] = []
        grasp_facts: dict[str, dict] = {}
        t_start = time.time()
        model_seconds = 0.0
        calls = 0
        prompt_tokens = completion_tokens = 0
        parse_failures = 0
        done_count = 0
        finished_reason = "max_turns"
        turn = 0
        stats = {"planned_steps": 0, "relayed_steps": 0, "checker_calls": 0, "checker_seconds": 0.0,
                 "checker_cost_usd": 0.0, "checker_input_tokens": 0, "checker_output_tokens": 0,
                 "empty_closes": 0, "relay_ends": {}}
        report = ""          # what the relay did since the big model's last reply, shown in its next turn

        demo_msgs = self.demo_msgs
        while turn < cfg.max_turns:
            turn += 1
            obs = env.observe()
            state = obs["state"]
            if turn == 1 and self.task_demos:
                shown = self._select_demos(state)
                demo_msgs = self._messages_for(shown)
                logger.info("demonstration(s) for this episode: %s", [_where(d) for d in shown])
            img_parts, captions, saved = self._images_for_turn(env, obs)
            if episode_dir:
                for name, png in saved.items():
                    (episode_dir / f"turn{turn:03d}_{name}.png").write_bytes(png)
            text = turn_text(turn, instruction, state, last_results, memory.render(), captions,
                             grasp_facts=grasp_facts)
            text, report = with_relay_report(text, report), ""
            messages = [{"role": "system", "content": sys_prompt}] + demo_msgs + [
                {"role": "user", "content": img_parts + [text_part(text)]},
            ]
            reply = self.client.chat(messages, tag=f"turn{turn}")
            calls += 1
            model_seconds += reply.latency_s
            prompt_tokens += int(reply.usage.get("prompt_tokens") or 0)
            completion_tokens += int(reply.usage.get("completion_tokens") or 0)

            record = {"turn": turn, "state": state, "prompt": text, "reply": reply.text, "latency_s": round(reply.latency_s, 2),
                      "error": reply.error, "commands": [], "results": []}
            if turn == 1 and self.task_demos:
                record["demo"] = [_where(d) for d in shown]
            if reply.error:
                record["parse_error"] = f"model call failed: {reply.error}"
                trace.append(record)
                parse_failures += 1
                if parse_failures > cfg.consecutive_parse_failures_allowed:
                    finished_reason = "model_error"
                    break
                last_results = [{"command": "(no command)", "ok": False, "note": "your previous reply could not be obtained; reply with the JSON object"}]
                continue

            try:
                parsed = parse_agent_json(reply.text)
                raw_cmds = parsed.get("commands") or []
                if isinstance(raw_cmds, str):
                    raw_cmds = [raw_cmds]
                commands, errors = parse_command_list(raw_cmds)
                parse_failures = 0
            except Exception as exc:  # noqa: BLE001
                parse_failures += 1
                record["parse_error"] = str(exc)
                trace.append(record)
                logger.warning("turn %d: could not parse reply (%s)", turn, exc)
                if parse_failures > cfg.consecutive_parse_failures_allowed:
                    finished_reason = "parse_failure"
                    break
                last_results = [{"command": "(unparseable reply)", "ok": False,
                                 "note": f"your reply was not a valid JSON object ({exc}); reply with the JSON object only"}]
                continue

            memory.update_scratchpad(parsed.get("memory"))
            record.update(scene=parsed.get("scene"), progress=parsed.get("progress"), plan=parsed.get("plan"),
                          memory=parsed.get("memory"), command_errors=errors)
            commands = commands[: cfg.max_commands_per_turn]
            lost_before = lost_arms(grasp_facts, state)
            results, done, empty = self._execute(env, commands, grasp_facts)
            stats["empty_closes"] += empty
            for err in errors:
                results.append({"command": "(invalid)", "kind": "invalid", "arm": None, "ok": False, "note": err})
            record["commands"] = [c.text() for c in commands]
            record["results"] = [{k: v for k, v in r.items() if k not in ("before", "after")} for r in results]
            expect, steps, plan_notes = [], [], []
            if self.variant != "baseline":
                try:
                    expect, steps, plan_notes = parse_plan(parsed, cfg.max_commands_per_turn, self.max_steps)
                except Exception as exc:  # noqa: BLE001  (a plan that cannot be read must not end the episode)
                    expect, steps, plan_notes = [], [], [f"the plan could not be read ({type(exc).__name__})"]
                stats["planned_steps"] += len(steps)
                record.update(expect=expect, next=[s.as_json() for s in steps], plan_notes=plan_notes)
            trace.append(record)
            memory.record_turn(turn, record["commands"], results, env.state())
            last_results = results

            if env.success:
                finished_reason = "success"
                break
            if done:
                done_count += 1
                if done_count >= cfg.done_limit:
                    finished_reason = "agent_done"
                    break
            else:
                done_count = 0
            if env.budget_exhausted:
                finished_reason = "step_budget"
                break
            stop = env.should_stop()
            if stop:
                # harness-side early stop (bookkeeping only; never shown to the model)
                finished_reason = stop
                break
            if self.empty_grasp_limit and stats["empty_closes"] >= self.empty_grasp_limit:
                finished_reason = "empty_grasp_limit"
                break

            if self.variant == "prompt" and steps:
                report = "PLANNED STEPS: none of your planned steps ran; in this run you are asked again after every turn."
            elif self.variant in ("open", "checked") and plan_notes and not steps and not done:
                report = ("PLANNED STEPS: none of your planned steps ran: "
                          + "; ".join(plan_notes) + ". RESULT OF YOUR LAST COMMANDS above lists what ran.")
            if self.variant not in ("open", "checked") or not steps or done:
                continue
            # ---------------------------------------------------------- relay
            model_turn, ran, end, detail, failed_check = turn, [], "all_ran", "", None
            prev_results, prev_empty, prev_expect, prev_cmds = results, empty, expect, record["commands"]
            prev_lost = lost_before
            for k, step in enumerate(steps, 1):
                end, detail = code_stop(prev_results, prev_empty, grasp_facts, env.state(), prev_lost)
                if end:
                    break
                if turn >= cfg.max_turns:
                    end = "turn_limit"
                    break
                obs = env.observe()
                views = env.views(obs, list(cfg.cameras))
                verdict = None
                if self.variant == "checked":
                    if not prev_expect:
                        end = "no_expect"
                        break
                    context = checker_context(instruction, prev_cmds, [c.text() for c in step.commands], obs["state"],
                                              grasp_facts, [v.caption for v in views])
                    verdict = self.checker.check(prev_expect, context, [v.image for v in views])
                    stats["checker_calls"] += 1
                    stats["checker_seconds"] += verdict["seconds"]
                    stats["checker_cost_usd"] += verdict["cost_usd"] or 0.0
                    stats["checker_input_tokens"] += verdict["tokens"][0]
                    stats["checker_output_tokens"] += verdict["tokens"][1]
                    self._log_check(episode_dir, turn + 1, k, prev_expect, verdict)
                    if not verdict["passed"]:
                        end = "checker_error" if verdict["error"] else "check_failed"
                        failed = [q for q, p in zip(prev_expect, verdict["p_yes"]) if p is None or p < self.checker.threshold]
                        detail = "; ".join(f"\"{q}\"" for q in failed)
                        failed_check = {"questions": prev_expect, "p_yes": verdict["p_yes"], "error": verdict["error"],
                                        "note": verdict["note"]}
                        break
                turn += 1
                if episode_dir:
                    for view in views:
                        (episode_dir / f"turn{turn:03d}_{view.name}.png").write_bytes(encode_png(view.image))
                prev_lost = lost_arms(grasp_facts, obs["state"])
                results, _, empty = self._execute(env, step.commands, grasp_facts)
                stats["empty_closes"] += empty
                stats["relayed_steps"] += 1
                rec = {"turn": turn, "source": "relay", "relay_step": k, "relay_from_turn": model_turn,
                       "state": obs["state"], "prompt": "", "reply": "", "latency_s": 0.0, "error": None,
                       "commands": [c.text() for c in step.commands],
                       "results": [{kk: v for kk, v in r.items() if kk not in ("before", "after")} for r in results],
                       "expect": step.expect, "checks": verdict and {"questions": prev_expect, "p_yes": verdict["p_yes"]}}
                trace.append(rec)
                memory.record_turn(f"{turn} (planned step {k}, run without asking you)", rec["commands"], results,
                                   env.state())
                last_results = last_results + results
                ran.append((k, turn, prev_expect if verdict else None))
                prev_results, prev_empty, prev_expect, prev_cmds = results, empty, step.expect, rec["commands"]
                if env.success or env.budget_exhausted or env.should_stop():
                    end = "episode_end"
                    break
                if self.empty_grasp_limit and stats["empty_closes"] >= self.empty_grasp_limit:
                    end = "episode_end"
                    break
            else:
                # every step ran; the last one's results still decide nothing more (the big model looks next)
                end = "all_ran"
            stats["relay_ends"][end] = stats["relay_ends"].get(end, 0) + 1
            trace[-1]["relay_end"] = {"reason": end, "steps_ran": len(ran), "steps_planned": len(steps),
                                      **({"failed_check": failed_check} if failed_check else {})}
            if env.success:
                finished_reason = "success"
                break
            if env.budget_exhausted:
                finished_reason = "step_budget"
                break
            stop = env.should_stop()
            if stop:
                finished_reason = stop
                break
            if self.empty_grasp_limit and stats["empty_closes"] >= self.empty_grasp_limit:
                finished_reason = "empty_grasp_limit"
                break
            report = self._report(model_turn, ran, end, detail, len(steps), plan_notes)

        if episode_dir:
            # final observation (after the last command)
            try:
                for view in env.views(env.observe(), list(cfg.cameras)):
                    (episode_dir / f"final_{view.name}.png").write_bytes(encode_png(view.image))
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not save the final observation: %s", exc)
        result = RelayEpisodeResult(
            success=env.success, turns=turn, steps_used=env.steps_used, step_limit=env.step_limit,
            finished_reason=finished_reason, seconds=round(time.time() - t_start, 1), model_seconds=round(model_seconds, 1),
            model_calls=calls, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            variant=self.variant, **{**stats, "checker_seconds": round(stats["checker_seconds"], 1),
                                     "checker_cost_usd": round(stats["checker_cost_usd"], 6)},
        )
        if episode_dir:
            (episode_dir / "trace.json").write_text(json.dumps(trace, indent=1, default=_json_default))
            (episode_dir / "memory.json").write_text(json.dumps(memory.to_json(), indent=1, default=_json_default))
        return result

    def _report(self, model_turn: int, ran: list, end: str, detail: str, planned: int, notes: list[str]) -> str:
        """The relay paragraph of the big model's next turn text."""
        lines = [f"PLANNED STEPS SINCE YOUR LAST REPLY: your commands ran in turn {model_turn}."]
        for k, turn, checked in ran:
            why = (" after the checker answered YES to " + "; ".join(f"\"{q}\"" for q in checked)) if checked else ""
            lines.append(f"Planned step {k} ran in turn {turn}{why}.")
        nxt = len(ran) + 1
        if end == "all_ran":
            lines.append("All your planned steps ran.")
        elif end == "check_failed":
            lines.append(f"Planned step {nxt} did NOT run: the checker did not answer YES to {detail}. "
                         "Look at the images yourself and decide.")
        elif end in STOP_TEXT:
            lines.append(f"Planned step {nxt} did NOT run: {STOP_TEXT[end].format(arm=detail)}.")
        elif end == "turn_limit":
            lines.append(f"Planned step {nxt} did NOT run: no turns left.")
        lines += [n[0].upper() + n[1:] + "." for n in notes]
        lines.append("RESULT OF YOUR LAST COMMANDS above lists every command that ran since your last reply.")
        return "\n".join(lines)


def make_checker(kind: str, threshold: float, credentials: Optional[Path], openrouter_key: Optional[str]) -> Checker:
    if kind == "clef":
        if not openrouter_key:
            raise ValueError("the clef checker needs an OpenRouter key")
        return ClefChecker(openrouter_key, threshold=threshold)
    if kind == "lite":
        if not credentials:
            raise ValueError("the lite checker needs the Vertex service account key")
        return LiteChecker(credentials, threshold=threshold)
    raise ValueError(f"unknown checker {kind!r}")
