#!/usr/bin/env python3
"""Run the pinned RoboDawn harness on Gemini through Vertex AI directly.

The model is called with a service account key on Vertex AI's native
generateContent API, because Flex PayGo (half price, slower, lower priority) is
not offered on the OpenAI-compatible endpoint. The harness still builds
OpenAI-style requests; the client translates each one to the native format and
the reply back, without changing any text or image. The request stays the
upstream one (no thinking settings, so the model's default thinking, as upstream
sends for Gemini; 8000 reply tokens) except for upstream's temperature 0, which
Gemini 3.8 Flash ignores and is therefore not sent. Access tokens expire after
an hour and are refreshed before each request.

--tier flex is for success-rate runs; turn latency must be measured on
--tier standard, since flex requests wait in a lower-priority queue.

Use --dry-run to inspect the configuration without a GPU, key or model call.
Simulation, prompts, demo selection and episode termination remain upstream.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from audit_robodawn import ENDPOINT, PROJECT, REASONING, ROBODAWN_COMMIT, ROBOTWIN_COMMIT, git

MODEL = "google/gemini-3.8-flash"
LOCATION = "global"
API_ROOT = "https://aiplatform.googleapis.com/v1"
CACHE_TTL_S = 3600      # prompt cache lifetime, renewed while in use (an episode on flex can outlast an hour)
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
TEMPERATURE = "not sent (upstream 0; Gemini 3.8 Flash ignores temperature)"
# Flex only: "shared" keeps the request off provisioned throughput, "flex" picks Flex PayGo.
# The harness watchdog kills a process that shows no progress for stall_timeout_s; a model call
# is one step, so the watchdog must outlast the request timeout (upstream: 300 s and 900 s).
TIERS = {"flex":{"headers":{"X-Vertex-AI-LLM-Request-Type":"shared","X-Vertex-AI-LLM-Shared-Request-Type":"flex"},
                 "traffic":"ON_DEMAND_FLEX","timeout_s":900,"stall_timeout_s":1200},
         "standard":{"headers":{},"traffic":"ON_DEMAND","timeout_s":300,"stall_timeout_s":900}}


def api_base(project: str) -> str:
    return f"{API_ROOT}/projects/{project}/locations/{LOCATION}"


def traffic_type(usage) -> str | None:
    return (((usage or {}).get("extra_properties") or {}).get("google") or {}).get("traffic_type")


def native_parts(content) -> list[dict]:
    if isinstance(content,str):
        return [{"text":content}]
    parts = []
    for part in content:
        if part["type"]=="text":
            parts.append({"text":part["text"]})
        elif part["type"]=="image_url":
            url = part["image_url"]["url"]
            if not url.startswith("data:") or ";base64," not in url:
                raise ValueError("Only inline base64 images are supported")
            mime,data = url.removeprefix("data:").split(";base64,",1)
            parts.append({"inlineData":{"mimeType":mime,"data":data}})
        else:
            raise ValueError(f"Unsupported content part: {part['type']}")
    return parts


def to_native(body: dict) -> dict:
    """OpenAI chat body -> generateContent body. Text and images pass through unchanged."""
    unknown = set(body)-{"model","messages","max_tokens","temperature"}
    if unknown:
        raise ValueError(f"Request fields with no native equivalent: {sorted(unknown)}")
    system,contents = [],[]
    for message in body["messages"]:
        if message["role"]=="system":
            system += native_parts(message["content"])
        else:
            contents.append({"role":"model" if message["role"]=="assistant" else "user",
                             "parts":native_parts(message["content"])})
    # Upstream's temperature is dropped: Gemini 3.8 Flash ignores it.
    native = {"contents":contents,"generationConfig":{"maxOutputTokens":body["max_tokens"]}}
    if system:
        native["systemInstruction"] = {"parts":system}
    return native


def from_native(out: dict) -> dict:
    """generateContent reply -> the OpenAI-shaped payload the harness reads."""
    candidate = (out.get("candidates") or [{}])[0]
    text = "".join(p.get("text","") for p in (candidate.get("content") or {}).get("parts",[]) if not p.get("thought"))
    usage = out.get("usageMetadata") or {}
    # OpenAI convention: completion tokens include the reasoning (both are billed as output), so the
    # harness's per-episode completion_tokens is the billed output; the reasoning share is in details.
    thoughts = usage.get("thoughtsTokenCount",0)
    return {"id":out.get("responseId"),"model":"google/"+str(out.get("modelVersion")),
        "choices":[{"index":0,"message":{"role":"assistant","content":text},
                    "finish_reason":str(candidate.get("finishReason","")).lower() or None}],
        "usage":{"prompt_tokens":usage.get("promptTokenCount",0),
                 "completion_tokens":usage.get("candidatesTokenCount",0)+thoughts,
                 "total_tokens":usage.get("totalTokenCount",0),
                 "prompt_tokens_details":{"cached_tokens":usage.get("cachedContentTokenCount",0)},
                 "completion_tokens_details":{"reasoning_tokens":thoughts},
                 "extra_properties":{"google":{"traffic_type":usage.get("trafficType")}}}}


def split_prefix(native: dict) -> tuple[dict,str] | None:
    """A generateContent body's fixed part (the system instruction and every message before the last: within an
    episode the harness repeats the system prompt, the demonstration and its acknowledgement unchanged and puts the
    turn's images and history in the last message) and its hash; None with fewer than two messages."""
    if len(native["contents"])<2:
        return None
    prefix = {"contents":native["contents"][:-1],
              **({"systemInstruction":native["systemInstruction"]} if "systemInstruction" in native else {})}
    return prefix,hashlib.sha256(json.dumps(prefix,sort_keys=True).encode()).hexdigest()[:16]


def http_json(url: str, token: str, method: str = "GET", body: dict | None = None, headers: dict | None = None,
              timeout: float = 300.0) -> tuple[int,dict | str]:
    """(200, JSON) / (HTTP status, error text) / (0, error), like upstream _post."""
    request = urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,method=method,
        headers={"Authorization":f"Bearer {token}","Content-Type":"application/json",**(headers or {})})
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:
            raw = response.read()
            return 200,json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        return exc.code,exc.read().decode(errors="replace")[:500]
    except Exception as exc:  # noqa: BLE001  (timeouts, connection resets, bad JSON)
        return 0,f"{type(exc).__name__}: {exc}"


def vertex_client(base, credentials_path: Path, tier: str = "flex", prompt_cache: bool = False):
    """Authenticate with a service account and record metadata; never log credentials.

    prompt_cache puts each episode's fixed prefix (split_prefix) in a Vertex context cache and sends only the last
    message with a reference to it, so the prefix is billed at the cached rate on every call instead of whenever
    implicit caching happens to hit (it missed about 5,500 tokens per call in the baseline runs). The model reads
    the same tokens in the same order. A cache is created when a new prefix appears, its lifetime is renewed while in
    use and it is deleted when the prefix changes or the process ends. Should a cache fail (creation refused, expired,
    or a cached request answered 400/404), that call and the rest of its prefix go out uncached as before.
    prompt_cache.jsonl (next to transport.jsonl, which keeps one line per model call) records every cache operation
    (cache_create / cache_delete / cache_refused) for the cost accounting (compare_relay.cache_overhead)."""
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    credentials = service_account.Credentials.from_service_account_file(str(credentials_path),scopes=SCOPES)
    headers = TIERS[tier]["headers"]

    def token():
        if not credentials.valid:
            credentials.refresh(Request())
        return credentials.token

    class VertexClient(base):
        def __init__(self,*args,**kwargs):
            kwargs["api_key"] = token()
            super().__init__(*args,**kwargs)
            self.cache = None           # {"key", "name", "tokens", "created", "renewed"} of the live cache
            self.cache_failed = set()   # prefixes that go out uncached
            if prompt_cache:
                atexit.register(self._drop_cache)

        def _cache_log(self,entry):
            if self.log_path:
                self.log_path.parent.mkdir(parents=True,exist_ok=True)
                with (self.log_path.parent/"prompt_cache.jsonl").open("a") as f:
                    f.write(json.dumps({"time":time.strftime("%Y-%m-%d %H:%M:%S"),**entry})+"\n")

        def _drop_cache(self):
            if not self.cache:
                return
            cache,self.cache = self.cache,None
            status,payload = http_json(f"{API_ROOT}/{cache['name']}",token(),"DELETE",timeout=60)
            self._cache_log({"tag":"cache_delete","status":status,"key":cache["key"],"cache_tokens":cache["tokens"],
                             "held_s":round(time.monotonic()-cache["created"],1),
                             **({"error":str(payload)[:300]} if status!=200 else {})})

        def _cached(self,native,model):
            """The body with its prefix replaced by the live cache (creating or renewing it), or None (send uncached)."""
            split = split_prefix(native) if prompt_cache else None
            if split is None or split[1] in self.cache_failed:
                return None
            prefix,key = split
            if self.cache and self.cache["key"]!=key:
                self._drop_cache()
            if not self.cache:
                started = time.monotonic()
                resource = self.base_url.split("/v1/",1)[1]
                status,payload = http_json(f"{self.base_url}/cachedContents",self.api_key,"POST",
                    {"model":f"{resource}/publishers/google/models/{model}",**prefix,"ttl":f"{CACHE_TTL_S}s"},
                    timeout=self.timeout_s)
                ok = status==200 and isinstance(payload,dict) and payload.get("name")
                tokens = ((payload.get("usageMetadata") or {}).get("totalTokenCount") if ok else None)
                self._cache_log({"tag":"cache_create","status":status,"key":key,"cache_tokens":tokens,
                                 "seconds":round(time.monotonic()-started,3),**({} if ok else {"error":str(payload)[:300]})})
                if not ok:
                    if status not in (0,429) and status<500:    # refused; a busy or unreachable API is tried again
                        self.cache_failed.add(key)
                    return None
                now = time.monotonic()
                self.cache = {"key":key,"name":payload["name"],"tokens":tokens,"created":now,"renewed":now}
            elif time.monotonic()-self.cache["renewed"]>CACHE_TTL_S/2:
                status,_ = http_json(f"{API_ROOT}/{self.cache['name']}?updateMask=ttl",self.api_key,"PATCH",
                                     {"ttl":f"{CACHE_TTL_S}s"},timeout=60)
                if status==200:
                    self.cache["renewed"] = time.monotonic()
            rest = {k:v for k,v in native.items() if k not in ("systemInstruction","contents")}
            return {"cachedContent":self.cache["name"],"contents":native["contents"][-1:],**rest}

        def _send(self,body):
            """Same contract as upstream _post: (200, payload) / (HTTP status, error text) / (0, error)."""
            model = body['model'].removeprefix('google/')
            url = f"{self.base_url}/publishers/google/models/{model}:generateContent"
            native = to_native(body)
            cached = self._cached(native,model)
            status,payload = http_json(url,self.api_key,"POST",cached or native,headers,self.timeout_s)
            if cached and status in (400,404):
                # an expired or refused cache: this prefix goes out uncached from here on
                self._cache_log({"tag":"cache_refused","status":status,"key":self.cache["key"],"error":str(payload)[:300]})
                self.cache_failed.add(self.cache["key"])
                self._drop_cache()
                status,payload = http_json(url,self.api_key,"POST",native,headers,self.timeout_s)
            if status!=200:
                return status,payload
            try:
                return 200,from_native(payload)
            except Exception as exc:  # noqa: BLE001  (a 200 that is not a generateContent reply)
                return 0,f"{type(exc).__name__}: {exc}"

        def _post(self,body,tag=""):
            self.api_key = token()
            started = time.monotonic()
            status,payload = self._send(body)
            if self.log_path:
                raw = payload if isinstance(payload,dict) else {}
                entry = {"tag":tag,"status":status,"seconds":round(time.monotonic()-started,3),"tier":tier,
                    "model":body["model"],"response_model":raw.get("model"),
                    "reasoning_fields":{k:body[k] for k in ("reasoning","reasoning_effort") if k in body},
                    "traffic_type":traffic_type(raw.get("usage")),"response_id":raw.get("id"),
                    "usage":raw.get("usage"),
                    "finish_reason":next((c.get("finish_reason") for c in raw.get("choices",[])),None)}
                if status!=200:
                    entry["error"] = str(payload)[:300]
                self.log_path.parent.mkdir(parents=True,exist_ok=True)
                with (self.log_path.parent/"transport.jsonl").open("a") as f:
                    f.write(json.dumps(entry)+"\n")
            return status,payload
    return VertexClient


def credentials_project(path: Path) -> str:
    key = json.loads(path.read_text())
    if key.get("type")!="service_account" or not key.get("project_id"):
        raise ValueError("Credentials must be a service account key with a project_id")
    return key["project_id"]


def configuration(args) -> tuple[dict,list[str]]:
    repo = args.repo.resolve()
    if git(repo,"rev-parse","HEAD")!=ROBODAWN_COMMIT:
        raise ValueError("RoboDawn checkout differs from the audited commit")
    if git(repo,"diff","--name-only","HEAD","--","harness","demos/robotwin2"):
        raise ValueError("Tracked harness or demonstration files were modified")
    manifest = json.loads(args.manifest.read_text())
    if manifest["robodawn_commit"]!=ROBODAWN_COMMIT or manifest["robotwin_commit"]!=ROBOTWIN_COMMIT:
        raise ValueError("Manifest commit differs from the audited commits")
    task = next((t for t in manifest["tasks"] if t["task"]==args.task),None)
    if task is None or args.episodes<1 or args.start_episode<0 or args.start_episode+args.episodes>10:
        raise ValueError("Choose a collected task and an episode range within 0..9")
    selected = task["episodes"][args.start_episode:args.start_episode+args.episodes]
    seeds = json.loads((repo/f"harness/valid_seeds/{args.task}__demo_randomized__seed0.json").read_text())["seeds"]
    for episode in selected:
        if episode["seed"]!=seeds[episode["episode"]]:
            raise ValueError("Shipped seed differs from manifest")
        demo_dir = repo/episode["demo_path"]
        demo = json.loads((demo_dir/"demo.json").read_text())
        if demo["source"]["seed"]!=episode["demo_seed"]:
            raise ValueError("Demonstration source seed differs from manifest")
        if any(not (demo_dir/name).is_file() for frame in demo["frames"] for name in frame["images"]):
            raise ValueError("Demonstration images are missing; fetch demos/robotwin2")
    if any(not (repo/f"demos/robotwin2/primer/turn{n:03d}_agent_camera.png").is_file() for n in range(1,7)):
        raise ValueError("Command primer images are missing")
    # One output directory per starting episode, so a task can run as several shards at once.
    output = (args.output.resolve() if args.output else
              PROJECT/f"outputs/robodawn/gemini_flash_{args.tier}"/args.task/f"shard_{args.start_episode}")
    config_path = output/"reproduction_config.json"
    if config_path.is_file():
        prior = json.loads(config_path.read_text())
        expected = {"robodawn_commit":ROBODAWN_COMMIT,"robotwin_commit":ROBOTWIN_COMMIT,
                    "task":args.task,"model":MODEL,"endpoint":ENDPOINT,"tier":args.tier,
                    "reasoning":REASONING,"max_turns":45,"max_commands_per_turn":4,
                    "max_tokens":8000,"task_config":"demo_randomized","instruction_type":"unseen",
                    "cameras":["agent_camera","top_camera","wrist"],"temperature":TEMPERATURE}
        if any(prior.get(k)!=v for k,v in expected.items()):
            raise ValueError("Existing output uses different reproduction conditions; choose another output")
    elif (output/"results.json").exists():
        raise ValueError("Existing output lacks reproduction metadata; choose another output")
    flags = ["--task",args.task,"--episodes",str(args.episodes),"--start_episode",str(args.start_episode),
        "--seed","0","--task_config","demo_randomized","--instruction_type","unseen",
        "--model",MODEL,"--max_turns","45","--max_commands_per_turn","4",
        "--max_tokens","8000","--timeout_s",str(TIERS[args.tier]["timeout_s"]),
        "--stall_timeout",str(TIERS[args.tier]["stall_timeout_s"]),
        "--cameras","agent_camera,top_camera,wrist","--output",str(output)]
    if args.no_video:
        flags += ["--no_video"]
    config = {"robodawn_commit":ROBODAWN_COMMIT,"robotwin_commit":ROBOTWIN_COMMIT,
        "task":args.task,"episodes":selected,"output":str(output),"model":MODEL,
        "endpoint":ENDPOINT,"tier":args.tier,"timeout_s":TIERS[args.tier]["timeout_s"],
        "stall_timeout_s":TIERS[args.tier]["stall_timeout_s"],"reasoning":REASONING,
        "max_turns":45,"max_commands_per_turn":4,"max_tokens":8000,
        "task_config":"demo_randomized","instruction_type":"unseen","seed_base":0,
        "cameras":["agent_camera","top_camera","wrist"],"temperature":TEMPERATURE,
        "robotwin_root":str(args.robotwin_root.resolve()),
        "limitations":["The original gateway is unknown. Like upstream, no reasoning fields are sent, so the "
                       "model's default thinking on Vertex is used; the original thinking level is not published."]
                      +(["Flex PayGo queues requests at lower priority: use these runs for success rates, "
                         "not for turn latency."] if args.tier=="flex" else [])}
    return config,flags


def check_results(config: dict) -> dict:
    output = Path(config["output"])
    results = json.loads((output/"results.json").read_text())
    errors = []
    if results.get("task")!=config["task"] or results.get("model")!=MODEL or results.get("task_config")!="demo_randomized":
        errors.append("Result task/model/scene configuration differs")
    for expected in config["episodes"]:
        actual = next((e for e in results["episodes"] if e["episode_index"]==expected["episode"]),None)
        if actual is None or actual.get("seed")!=expected["seed"]:
            errors.append(f"episode {expected['episode']}: missing or different seed")
            continue
        trace_path = output/f"episode_{expected['episode']:03d}/trace.json"
        trace = json.loads(trace_path.read_text())
        shown = trace[0].get("demo",[]) if trace else []
        if [Path(d).name for d in shown]!=[Path(expected["demo_path"]).name]:
            errors.append(f"episode {expected['episode']}: different demonstration entry")
    calls = [json.loads(line) for line in (output/"llm_calls.jsonl").read_text().splitlines() if line.strip()]
    token_counts = [(call.get("usage",{}).get("completion_tokens_details") or {}).get("reasoning_tokens") for call in calls]
    if not any(isinstance(n,(int,float)) and n>0 for n in token_counts):
        errors.append("No positive reasoning token usage was reported; reasoning is unverified")
    transport_path = output/"transport.jsonl"
    transport = [json.loads(line) for line in transport_path.read_text().splitlines() if line.strip()] if transport_path.exists() else []
    answered = [t for t in transport if t["status"]==200]
    if not answered or any(not str(t.get("response_model")).startswith(MODEL) for t in answered):
        errors.append("Returned model is missing or differs from the requested Vertex model")
    if any(t.get("traffic_type")!=TIERS[config["tier"]]["traffic"] for t in answered):
        errors.append(f"Some requests were not served as {TIERS[config['tier']]['traffic']} traffic")
    if any(t.get("reasoning_fields") for t in transport):
        errors.append("Reasoning fields were sent; the reproduction uses the model default")
    report = {"task":config["task"],"errors":errors,"reasoning_tokens":token_counts,
        "traffic_types":sorted({str(t.get("traffic_type")) for t in answered}),
        "site_success":sum(e["site_outcome"]=="success" for e in config["episodes"]),
        "run_success":sum(e["success"] for e in results["episodes"] if e["episode_index"] in {x["episode"] for x in config["episodes"]}),
        "reproduction_validated":False,
        "note":"A smoke run checks conditions. Success-rate reproduction needs the chosen tasks' full ten episodes."}
    (output/"condition_check.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",required=True)
    parser.add_argument("--episodes",type=int,default=1)
    parser.add_argument("--start-episode",type=int,default=0)
    parser.add_argument("--repo",type=Path,default=PROJECT/"third_party/robodawn")
    parser.add_argument("--manifest",type=Path,default=PROJECT/"robodawn_site/reproduction_manifest.json")
    parser.add_argument("--robotwin-root",type=Path,default=Path(os.environ.get("ROBOTWIN_ROOT","/content/RoboTwin")))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--credentials",type=Path,
                        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
                        help="Vertex AI service account key (default: $GOOGLE_APPLICATION_CREDENTIALS)")
    parser.add_argument("--tier",choices=sorted(TIERS),default="flex",
                        help="flex: half price, for success rates; standard: for latency measurements")
    parser.add_argument("--no-video",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args()
    config,flags = configuration(args)
    if args.dry_run:
        print(json.dumps(config,ensure_ascii=False,indent=2))
        return
    if git(args.robotwin_root.resolve(),"rev-parse","HEAD")!=ROBOTWIN_COMMIT:
        raise ValueError("RoboTwin checkout differs from the pinned benchmark commit")
    if not (args.robotwin_root/"assets/background_texture").is_dir():
        raise ValueError("Randomized evaluation needs background textures; bootstrap with --textures")
    if not args.credentials or not Path(args.credentials).is_file():
        raise ValueError("Pass --credentials or set GOOGLE_APPLICATION_CREDENTIALS to the service account key")
    credentials = Path(args.credentials).resolve()
    os.environ["ROBOTWIN_ROOT"] = str(args.robotwin_root.resolve())
    sys.path.insert(0,str(args.repo.resolve()))
    from harness import run_robotwin_eval
    run_robotwin_eval.ChatClient = vertex_client(run_robotwin_eval.ChatClient,credentials,args.tier)
    Path(config["output"]).mkdir(parents=True,exist_ok=True)
    (Path(config["output"])/"reproduction_config.json").write_text(json.dumps(config,indent=2)+"\n")
    sys.argv = [str(args.repo/"harness/run_robotwin_eval.py"),*flags,"--api_base",api_base(credentials_project(credentials))]
    run_robotwin_eval.main()
    report = check_results(config)
    print(json.dumps(report,indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
