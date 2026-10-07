"""Offline transport and reproduction-condition checks; no GPU or API calls."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"third_party/robodawn"))
import run_robodawn_baseline as baseline
from harness.agent.llm_client import ChatClient


class BaselineTests(unittest.TestCase):
    def args(self,**changes):
        fields = dict(repo=ROOT/"third_party/robodawn",manifest=ROOT/"robodawn_site/reproduction_manifest.json",
                      task="adjust_bottle",episodes=1,start_episode=0,output=None,api_key_file=None,
                      no_video=False,robotwin_root=Path("/content/RoboTwin"))
        fields.update(changes)
        return argparse.Namespace(**fields)

    def test_vertex_routing_cannot_be_overridden(self):
        client = baseline.openrouter_client(ChatClient)(model=baseline.MODEL,base_url=baseline.API_BASE,
            api_key="fake-secret",max_tokens=8000,extra_body={"provider":{"only":["google-ai-studio"]},
                                                           "reasoning":{"effort":"none"}})
        body = client._body([{"role":"system","content":"unchanged prompt"}])
        self.assertEqual(body["provider"],baseline.PROVIDER)
        self.assertEqual(body["reasoning"],{"enabled":True})
        self.assertEqual(body["messages"][0]["content"],"unchanged prompt")
        self.assertEqual(body["max_tokens"],8000)

    def test_transport_log_has_usage_and_no_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = baseline.openrouter_client(ChatClient)(model=baseline.MODEL,base_url=baseline.API_BASE,
                api_key="fake-secret",log_path=Path(tmp)/"llm_calls.jsonl")
            response = {"provider":"Google","usage":{"completion_tokens_details":{"reasoning_tokens":100}},
                        "choices":[{"finish_reason":"stop"}]}
            with patch.object(ChatClient,"_post",return_value=(200,response)):
                client._post(client._body([]),"turn1")
            text = (Path(tmp)/"transport.jsonl").read_text()
            self.assertNotIn("fake-secret",text)
            self.assertEqual(json.loads(text)["usage"]["completion_tokens_details"]["reasoning_tokens"],100)

    def test_dry_configuration_matches_site_seed_and_demo(self):
        config,flags = baseline.configuration(self.args())
        self.assertEqual(config["episodes"][0]["seed"],100000)
        self.assertEqual(config["episodes"][0]["demo_path"],"demos/robotwin2/expert/adjust_bottle")
        self.assertIn("demo_randomized",flags)

    def test_invalid_episode_range_is_rejected(self):
        with self.assertRaises(ValueError):
            baseline.configuration(self.args(start_episode=9,episodes=2))

    def test_changed_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.loads(self.args().manifest.read_text())
            manifest["tasks"][0]["episodes"][0]["seed"] = 999999
            path = Path(tmp)/"manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"seed differs"):
                baseline.configuration(self.args(manifest=path))

    def test_resume_without_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/"results.json").write_text("{}")
            with self.assertRaisesRegex(ValueError,"lacks reproduction metadata"):
                baseline.configuration(self.args(output=Path(tmp)))

    def write_result(self,tmp,seed=100000,demo="adjust_bottle",reasoning=100,provider="Google"):
        output = Path(tmp)
        expected = {"episode":0,"seed":100000,"demo_path":"demos/robotwin2/expert/adjust_bottle","site_outcome":"success"}
        (output/"results.json").write_text(json.dumps({"task":"adjust_bottle","model":baseline.MODEL,
            "task_config":"demo_randomized","episodes":[{"episode_index":0,"seed":seed,"success":True}]}))
        ep = output/"episode_000"
        ep.mkdir()
        (ep/"trace.json").write_text(json.dumps([{"demo":["/repo/demos/robotwin2/expert/"+demo]}]))
        (output/"llm_calls.jsonl").write_text(json.dumps({"usage":{"completion_tokens_details":{"reasoning_tokens":reasoning}}})+"\n")
        (output/"transport.jsonl").write_text(json.dumps({"status":200,"response_provider":provider})+"\n")
        return {"output":str(output),"task":"adjust_bottle","episodes":[expected]}

    def test_matching_runtime_conditions_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(baseline.check_results(self.write_result(tmp))["errors"],[])

    def test_wrong_demo_and_unverified_reasoning_are_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = baseline.check_results(self.write_result(tmp,demo="adjust_bottle+2",reasoning=None))
            self.assertEqual(len(report["errors"]),2)
            self.assertFalse(report["reproduction_validated"])

    def test_wrong_seed_and_ai_studio_provider_are_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = baseline.check_results(self.write_result(tmp,seed=999999,provider="Google AI Studio"))
            self.assertEqual(len(report["errors"]),2)


if __name__=="__main__":
    unittest.main()
