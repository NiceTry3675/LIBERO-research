"""The explicit prompt cache of the Vertex client (run_robodawn_baseline.vertex_client): no network, key or GPU."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"third_party/robodawn"))
import compare_relay
import run_robodawn_baseline as rb
import run_robodawn_relay as runner
from harness.agent.llm_client import ChatClient

BASE = "https://aiplatform.googleapis.com/v1/projects/p/locations/global"
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def messages(turn, demo="demo"):
    """The harness's request shape: system prompt, demonstration, acknowledgement, then the turn's message."""
    return [{"role": "system", "content": "You control a robot."},
            {"role": "user", "content": [{"type": "text", "text": demo}, IMAGE]},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": [IMAGE, {"type": "text", "text": f"TURN {turn}."}]}]


def reply(cached=0):
    return {"candidates": [{"content": {"parts": [{"text": '{"commands": []}'}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 5, "thoughtsTokenCount": 7,
                              "cachedContentTokenCount": cached, "trafficType": "ON_DEMAND"},
            "modelVersion": "gemini-3.8-flash", "responseId": "r"}


class FakeAPI:
    """Stands in for http_json: records each request and answers from per-kind scripts (default: success)."""
    def __init__(self, **scripts):
        self.calls, self.scripts, self.created = [], {k: list(v) for k, v in scripts.items()}, 0

    def __call__(self, url, token, method="GET", body=None, headers=None, timeout=300.0):
        kind = ("create" if url.endswith("/cachedContents") else "generate" if url.endswith(":generateContent")
                else "delete" if method == "DELETE" else "renew")
        self.calls.append((kind, url, body))
        if self.scripts.get(kind):
            return self.scripts[kind].pop(0)
        if kind == "create":
            self.created += 1
            return 200, {"name": f"projects/p/locations/global/cachedContents/{self.created}",
                         "usageMetadata": {"totalTokenCount": 100}}
        if kind == "generate":
            return 200, reply(100 if "cachedContent" in body else 0)
        return 200, {}

    def kinds(self):
        return [k for k, _, _ in self.calls]


class FakeCredentials:
    valid, token = True, "token"


def client(tmp, prompt_cache=True):
    with patch("google.oauth2.service_account.Credentials.from_service_account_file", return_value=FakeCredentials()):
        cls = rb.vertex_client(ChatClient, Path("key.json"), "standard", prompt_cache=prompt_cache)
    with patch("atexit.register"):
        return cls(model=rb.MODEL, base_url=BASE, max_tokens=8000, temperature=None,
                   log_path=Path(tmp)/"llm_calls.jsonl")


def cache_log(tmp):
    path = Path(tmp)/"prompt_cache.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class PromptCacheTests(unittest.TestCase):
    def test_prefix_is_everything_before_the_last_message(self):
        one, key = rb.split_prefix(rb.to_native({"model": rb.MODEL, "messages": messages(1), "max_tokens": 10}))
        two, key2 = rb.split_prefix(rb.to_native({"model": rb.MODEL, "messages": messages(2), "max_tokens": 10}))
        self.assertEqual((one, key), (two, key2))
        self.assertEqual(len(one["contents"]), 2)
        self.assertIn("systemInstruction", one)
        other = rb.split_prefix(rb.to_native({"model": rb.MODEL, "messages": messages(1, "another demo"), "max_tokens": 10}))
        self.assertNotEqual(other[1], key)
        self.assertIsNone(rb.split_prefix({"contents": [{"role": "user", "parts": []}]}))

    def test_the_model_reads_the_same_tokens_and_one_cache_serves_an_episode(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as tmp, patch.object(rb, "http_json", api):
            c = client(tmp)
            first = c.chat(messages(1), tag="turn1")
            c.chat(messages(2), tag="turn2")
            c.chat(messages(1, "next episode's demo"), tag="turn1")
            c._drop_cache()
            log = cache_log(tmp)
            transport = [json.loads(line) for line in (Path(tmp)/"transport.jsonl").read_text().splitlines()]
        self.assertEqual(api.kinds(), ["create", "generate", "generate", "delete", "create", "generate", "delete"])
        plain = rb.to_native({"model": rb.MODEL, "messages": messages(1), "max_tokens": 8000})
        _, _, created = api.calls[0]
        _, _, sent = api.calls[1]
        self.assertNotIn("systemInstruction", sent)
        self.assertEqual(created["systemInstruction"], plain["systemInstruction"])
        self.assertEqual(created["contents"] + sent["contents"], plain["contents"])
        self.assertEqual(sent["generationConfig"], plain["generationConfig"])
        self.assertTrue(created["model"].endswith("/locations/global/publishers/google/models/gemini-3.8-flash"))
        self.assertEqual(sent["cachedContent"], "projects/p/locations/global/cachedContents/1")
        self.assertEqual(api.calls[3][1], f"{rb.API_ROOT}/projects/p/locations/global/cachedContents/1")
        self.assertEqual(first.usage["prompt_tokens_details"]["cached_tokens"], 100)
        self.assertEqual([e["tag"] for e in log], ["cache_create", "cache_delete", "cache_create", "cache_delete"])
        self.assertEqual([t["tag"] for t in transport], ["turn1", "turn2", "turn1"])   # one line per model call

    def test_off_sends_the_plain_request(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as tmp, patch.object(rb, "http_json", api):
            client(tmp, prompt_cache=False).chat(messages(1), tag="turn1")
            self.assertEqual(cache_log(tmp), [])
        self.assertEqual(api.kinds(), ["generate"])
        self.assertEqual(api.calls[0][2], rb.to_native({"model": rb.MODEL, "messages": messages(1), "max_tokens": 8000}))

    def test_a_refused_cache_falls_back_to_the_plain_request_for_that_prefix(self):
        api = FakeAPI(create=[(400, "cachedContent too small")])
        with tempfile.TemporaryDirectory() as tmp, patch.object(rb, "http_json", api):
            c = client(tmp)
            result = c.chat(messages(1), tag="turn1")
            c.chat(messages(2), tag="turn2")
            log = cache_log(tmp)
        self.assertEqual(result.error, None)
        self.assertEqual(api.kinds(), ["create", "generate", "generate"])
        self.assertTrue(all("cachedContent" not in body for kind, _, body in api.calls if kind == "generate"))
        self.assertEqual([(e["tag"], e["status"]) for e in log], [("cache_create", 400)])

    def test_a_busy_api_is_tried_again_on_the_next_call(self):
        api = FakeAPI(create=[(503, "unavailable")])
        with tempfile.TemporaryDirectory() as tmp, patch.object(rb, "http_json", api):
            c = client(tmp)
            c.chat(messages(1), tag="turn1")
            c.chat(messages(2), tag="turn2")
        self.assertEqual(api.kinds(), ["create", "generate", "create", "generate"])
        self.assertIn("cachedContent", api.calls[3][2])

    def test_an_expired_cache_is_dropped_and_the_call_goes_out_plain(self):
        api = FakeAPI(generate=[(404, "CachedContent not found")])
        with tempfile.TemporaryDirectory() as tmp, patch.object(rb, "http_json", api):
            c = client(tmp)
            result = c.chat(messages(1), tag="turn1")
            c.chat(messages(2), tag="turn2")
            log = cache_log(tmp)
        self.assertEqual(result.error, None)
        self.assertEqual(api.kinds(), ["create", "generate", "delete", "generate", "generate"])
        self.assertNotIn("cachedContent", api.calls[3][2])
        self.assertNotIn("cachedContent", api.calls[4][2])
        self.assertEqual([e["tag"] for e in log], ["cache_create", "cache_refused", "cache_delete"])

    def test_a_busy_model_keeps_the_cache_for_the_retry(self):
        api = FakeAPI(generate=[(429, "Resource exhausted")])
        with tempfile.TemporaryDirectory() as tmp, patch.object(rb, "http_json", api), patch("time.sleep"):
            client(tmp).chat(messages(1), tag="turn1")
        self.assertEqual(api.kinds(), ["create", "generate", "generate"])
        self.assertTrue(all("cachedContent" in body for kind, _, body in api.calls if kind == "generate"))

    def test_a_long_episode_renews_the_cache(self):
        api = FakeAPI()
        with tempfile.TemporaryDirectory() as tmp, patch.object(rb, "http_json", api):
            c = client(tmp)
            c.chat(messages(1), tag="turn1")
            c.cache["renewed"] -= rb.CACHE_TTL_S
            c.chat(messages(2), tag="turn2")
        self.assertEqual(api.kinds(), ["create", "generate", "renew", "generate"])
        self.assertEqual(api.calls[2][2], {"ttl": f"{rb.CACHE_TTL_S}s"})

    def test_cache_cost_and_run_name(self):
        events = [{"tag": "cache_create", "status": 200, "key": "a", "cache_tokens": 1000},
                  {"tag": "cache_delete", "status": 200, "key": "a", "held_s": 1800},
                  {"tag": "cache_create", "status": 400, "key": "b"},
                  {"tag": "cache_create", "status": 200, "key": "c", "cache_tokens": 2000}]   # never deleted
        self.assertAlmostEqual(compare_relay.cache_overhead(events, 0.75, 1.0),
                               (1000 * 0.75 + 1000 * 0.5 + 2000 * 0.75 + 2000 * 1.0) / 1e6)
        args = argparse.Namespace(repo=ROOT/"third_party/robodawn", manifest=ROOT/"robodawn_site/reproduction_manifest.json",
                                  task="place_empty_cup", episodes=1, start_episode=24, output=None, variant="baseline",
                                  checker=None, threshold=0.5, max_steps=3, empty_grasp_limit=0, max_tokens=8000,
                                  no_video=False, tier="standard", robotwin_root=Path("/content/RoboTwin"), prompt_cache=True,
                                  monitor="both")
        cfg, flags = runner.configuration(args)
        self.assertEqual(flags[flags.index("--label") + 1], "relay_baseline_monboth2_pc_standard")
        self.assertEqual(cfg["prompt_cache"], "explicit")
        args.prompt_cache = False
        self.assertNotIn("prompt_cache", runner.configuration(args)[0])


if __name__ == "__main__":
    unittest.main()
