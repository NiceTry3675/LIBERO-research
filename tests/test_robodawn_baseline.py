"""Offline transport and reproduction-condition checks; no GPU or API calls."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"third_party/robodawn"))
import run_robodawn_baseline as baseline
from harness.agent.llm_client import ChatClient


class BaselineTests(unittest.TestCase):
    def args(self,**changes):
        fields = dict(repo=ROOT/"third_party/robodawn",manifest=ROOT/"robodawn_site/reproduction_manifest.json",
                      task="adjust_bottle",episodes=1,start_episode=0,output=None,credentials=None,
                      no_video=False,tier="flex",robotwin_root=Path("/content/RoboTwin"))
        fields.update(changes)
        return argparse.Namespace(**fields)

    def client(self,tier="flex",**kwargs):
        credentials = MagicMock(valid=True,token="fake-token")
        with patch("google.oauth2.service_account.Credentials.from_service_account_file",return_value=credentials):
            return baseline.vertex_client(ChatClient,Path("key.json"),tier)(model=baseline.MODEL,
                base_url=baseline.api_base("demo-project"),api_key=None,max_tokens=8000,**kwargs)

    MESSAGES = [{"role":"system","content":"unchanged prompt"},
                {"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},
                                          {"type":"text","text":"demo"}]},
                {"role":"assistant","content":"Understood."},
                {"role":"user","content":[{"type":"text","text":"TURN 1."}]}]

    def test_native_request_keeps_text_images_and_settings(self):
        body = self.client()._body(self.MESSAGES)
        self.assertEqual(set(body),{"model","messages","max_tokens","temperature"})
        native = baseline.to_native(body)
        self.assertEqual(native["systemInstruction"],{"parts":[{"text":"unchanged prompt"}]})
        self.assertEqual([c["role"] for c in native["contents"]],["user","model","user"])
        self.assertEqual(native["contents"][0]["parts"],[{"inlineData":{"mimeType":"image/png","data":"AAAA"}},
                                                         {"text":"demo"}])
        self.assertEqual(native["generationConfig"],{"maxOutputTokens":8000})
        self.assertNotIn("thinkingConfig",native["generationConfig"])

    def test_unknown_request_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"no native equivalent"):
            baseline.to_native({"model":baseline.MODEL,"messages":[],"max_tokens":8000,"reasoning_effort":"high"})

    def test_native_reply_is_mapped_for_the_harness(self):
        reply = baseline.from_native({"responseId":"r1","modelVersion":"gemini-3.8-flash",
            "candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"hidden","thought":True},{"text":"{}"}]}}],
            "usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2,"thoughtsTokenCount":50,
                             "cachedContentTokenCount":8,"totalTokenCount":62,"trafficType":"ON_DEMAND_FLEX"}})
        self.assertEqual(reply["choices"][0]["message"]["content"],"{}")
        self.assertEqual(reply["model"],baseline.MODEL)
        self.assertEqual(reply["usage"]["completion_tokens_details"]["reasoning_tokens"],50)
        self.assertEqual(reply["usage"]["completion_tokens"],52)   # reply 2 + reasoning 50, billed as output
        self.assertEqual(reply["usage"]["total_tokens"],62)
        self.assertEqual(reply["usage"]["prompt_tokens_details"]["cached_tokens"],8)
        self.assertEqual(baseline.traffic_type(reply["usage"]),"ON_DEMAND_FLEX")

    def test_flex_headers_and_native_url(self):
        for tier,flex in [("flex",True),("standard",False)]:
            client = self.client(tier)
            response = MagicMock()
            response.__enter__.return_value = response
            with patch("urllib.request.urlopen",return_value=response) as urlopen, \
                 patch("json.load",return_value={"modelVersion":"gemini-3.8-flash","candidates":[]}):
                status,_ = client._send(client._body(self.MESSAGES))
            request = urlopen.call_args[0][0]
            self.assertEqual(status,200)
            self.assertTrue(request.full_url.endswith("/locations/global/publishers/google/models/gemini-3.8-flash:generateContent"))
            self.assertEqual(request.get_header("X-vertex-ai-llm-shared-request-type"),"flex" if flex else None)

    def test_transport_log_has_usage_and_no_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client(log_path=Path(tmp)/"llm_calls.jsonl")
            self.assertEqual(client.api_key,"fake-token")
            response = baseline.from_native({"modelVersion":"gemini-3.8-flash","candidates":[{"finishReason":"STOP"}],
                "usageMetadata":{"thoughtsTokenCount":100,"trafficType":"ON_DEMAND_FLEX"}})
            with patch.object(client,"_send",return_value=(200,response)):
                client._post(client._body([]),"turn1")
            text = (Path(tmp)/"transport.jsonl").read_text()
            self.assertNotIn("fake-token",text)
            entry = json.loads(text)
            self.assertEqual(entry["usage"]["completion_tokens_details"]["reasoning_tokens"],100)
            self.assertEqual((entry["tier"],entry["traffic_type"],entry["reasoning_fields"]),("flex","ON_DEMAND_FLEX",{}))

    def test_dry_configuration_matches_site_seed_and_demo(self):
        config,flags = baseline.configuration(self.args())
        self.assertEqual(config["episodes"][0]["seed"],100000)
        self.assertEqual(config["episodes"][0]["demo_path"],"demos/robotwin2/expert/adjust_bottle")
        self.assertIn("demo_randomized",flags)
        self.assertEqual((config["tier"],flags[flags.index("--timeout_s")+1]),("flex","900"))
        self.assertGreater(int(flags[flags.index("--stall_timeout")+1]),int(flags[flags.index("--timeout_s")+1]))
        self.assertTrue(config["output"].endswith("gemini_flash_flex/adjust_bottle/shard_0"))
        config,flags = baseline.configuration(self.args(start_episode=5,episodes=5))
        self.assertTrue(config["output"].endswith("gemini_flash_flex/adjust_bottle/shard_5"))
        self.assertEqual([e["episode"] for e in config["episodes"]],[5,6,7,8,9])
        self.assertEqual(flags[flags.index("--start_episode")+1],"5")

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

    def write_result(self,tmp,seed=100000,demo="adjust_bottle",reasoning=100,traffic="ON_DEMAND_FLEX"):
        output = Path(tmp)
        expected = {"episode":0,"seed":100000,"demo_path":"demos/robotwin2/expert/adjust_bottle","site_outcome":"success"}
        (output/"results.json").write_text(json.dumps({"task":"adjust_bottle","model":baseline.MODEL,
            "task_config":"demo_randomized","episodes":[{"episode_index":0,"seed":seed,"success":True}]}))
        ep = output/"episode_000"
        ep.mkdir()
        (ep/"trace.json").write_text(json.dumps([{"demo":["/repo/demos/robotwin2/expert/"+demo]}]))
        (output/"llm_calls.jsonl").write_text(json.dumps({"usage":{"completion_tokens_details":{"reasoning_tokens":reasoning}}})+"\n")
        (output/"transport.jsonl").write_text(json.dumps({"status":200,"response_model":baseline.MODEL,
            "traffic_type":traffic,"reasoning_fields":{}})+"\n")
        return {"output":str(output),"task":"adjust_bottle","tier":"flex","episodes":[expected]}

    def test_matching_runtime_conditions_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(baseline.check_results(self.write_result(tmp))["errors"],[])

    def test_wrong_demo_and_unverified_reasoning_are_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = baseline.check_results(self.write_result(tmp,demo="adjust_bottle+2",reasoning=None))
            self.assertEqual(len(report["errors"]),2)
            self.assertFalse(report["reproduction_validated"])

    def test_wrong_seed_and_standard_traffic_in_flex_run_are_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = baseline.check_results(self.write_result(tmp,seed=999999,traffic="ON_DEMAND"))
            self.assertEqual(len(report["errors"]),2)


if __name__=="__main__":
    unittest.main()
