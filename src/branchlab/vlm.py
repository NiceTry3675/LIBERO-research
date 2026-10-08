"""Generative vision-language models through OpenRouter, asked the same typed
questions as Clef so the two can be compared on equal terms.

The state, questions and options are written out as text, the images are
attached, and the model must reply with one JSON object mapping each question
id to an option key (score questions: the level index). Answers come back in
Clef's format with all probability on the chosen option; an unparsable or
invalid answer is returned as choice None. Reasoning is off unless `think`;
models that refuse to switch it off get the minimal effort they allow.

backend="vertex" calls Gemini on Vertex AI directly (native generateContent,
service account key from $GOOGLE_APPLICATION_CREDENTIALS) instead of going
through OpenRouter. It sends no thinking settings, so the model thinks at its
default level, as in the RoboDawn baseline; `think` is ignored there.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from branchlab.clef import _load_env

URL = "https://openrouter.ai/api/v1/chat/completions"
VERTEX_URL = ("https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/"
              "{model}:generateContent")
# Gemini is served on OpenRouter by Google Vertex and Google AI Studio. The BYOK
# key is registered for Vertex only; a request routed to AI Studio (by load
# balancing, or as a fallback when Vertex errors) runs on OpenRouter's own key
# and is billed to OpenRouter credits. So Gemini requests are pinned to Vertex
# with fallbacks off; a Vertex error is then retried instead of rerouted.
PROVIDERS = {"google/": {"only": ["google-vertex"], "allow_fallbacks": False}}
SYSTEM = (
    "You are the fast executor of a robot. You get the current state, images and a list of questions, "
    "each with a fixed set of options. Answer every question by picking exactly one option key. "
    "Reply with a single JSON object that maps each listed question id to the chosen option key, and nothing "
    "else: no other keys, no nulls."
)


def render(state, questions):
    lines = ["STATE:", state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1), "",
             "QUESTIONS:"]
    for qid, q in questions.items():
        lines.append(f"- {qid}: {q['instructions']}")
        if q["type"] == "noul":
            options = {"yes": "yes", "no": "no"}
        elif q["type"] == "score":
            options = {str(i): text for i, text in enumerate(q["criteria"])}
        else:
            options = q["criteria"]
        for key, text in options.items():
            lines.append(f"    {key}: {text}")
    lines.append("")
    lines.append("Reply with JSON only, e.g. {" + ", ".join(f'"{qid}": "<key>"' for qid in questions) + "}")
    return "\n".join(lines)


def to_answers(reply: dict, questions):
    answers = {}
    for qid, q in questions.items():
        pick = reply.get(qid)
        pick = None if pick is None else str(pick).strip()
        if q["type"] == "score":
            valid = pick in {str(i) for i in range(len(q["criteria"]))}
            answers[qid] = {"type": "score", "probabilities": {pick: 1.0} if valid else {"-1": 1.0}}
        elif q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 1.0 if pick == "yes" else 0.0 if pick == "no" else 0.5}
        else:
            valid = pick in q["criteria"]
            answers[qid] = {"type": "choice", "choice": pick if valid else None,
                            "probabilities": {pick: 1.0} if valid else {}}
    return answers


class VLM:
    def __init__(self, cache_path=None, max_retries=6, backend="openrouter"):
        self.backend = backend
        if backend == "vertex":
            import os
            from google.oauth2 import service_account
            key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not key:
                raise RuntimeError("backend='vertex' needs GOOGLE_APPLICATION_CREDENTIALS (service account key)")
            self._project = json.loads(Path(key).read_text())["project_id"]
            self._credentials = service_account.Credentials.from_service_account_file(
                key, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        else:
            self._token = _load_env()["OPENROUTER_API_KEY"]
        self.max_retries = max_retries
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = {}
        self._lock = threading.Lock()
        if self.cache_path and self.cache_path.exists():
            for line in open(self.cache_path):
                row = json.loads(line)
                self._cache[row["key"]] = row["result"]

    def ask(self, state, questions, images=(), model="google/gemini-3.1-flash-lite", key=None, think=False):
        if key is not None and key in self._cache:
            return self._cache[key]
        if self.backend == "vertex":
            return self._store(key, self._ask_vertex(state, questions, images, model))
        content = [{"type": "text", "text": render(state, questions)}]
        content += [{"type": "image_url", "image_url": {"url": uri}} for uri in images]
        body = {"model": model, "temperature": 0, "max_tokens": 6000 if think else 800,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
                "reasoning": {"effort": "medium"} if think else {"effort": "none"}}
        for prefix, provider in PROVIDERS.items():
            if model.startswith(prefix):
                body["provider"] = provider
        data = json.dumps(body).encode()
        raw, out, effort = "", {}, body["reasoning"]["effort"]
        for attempt in range(self.max_retries):
            request = urllib.request.Request(URL, data=data, headers={
                "Authorization": f"Bearer {self._token}", "Content-Type": "application/json"})
            start = time.time()
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    out = json.load(response)
                if "error" in out:
                    raise RuntimeError(str(out["error"])[:300])
                raw = out["choices"][0]["message"].get("content") or ""
                break
            except urllib.error.HTTPError as e:
                message = e.read().decode()[:500]
                if e.code == 400 and "Reasoning is mandatory" in message and effort == "none":
                    # Some models cannot switch reasoning off; use the least they allow.
                    effort = body["reasoning"]["effort"] = "minimal"
                    data = json.dumps(body).encode()
                    continue
                if e.code in (408, 429, 500, 502, 503, 504, 529) and attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"OpenRouter HTTP {e.code}: {message}") from None
            except (urllib.error.URLError, TimeoutError, RuntimeError):
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        try:
            text = raw[raw.index("{"): raw.rindex("}") + 1]
            reply = json.loads(text)
        except ValueError:
            reply = {}
        result = {"answers": to_answers(reply, questions), "usage": out.get("usage", {}),
                  "latency": time.time() - start, "raw": raw[:500], "reasoning": effort,
                  "provider": out.get("provider")}
        return self._store(key, result)

    def _store(self, key, result):
        if key is not None and self.cache_path:
            with self._lock:
                self._cache[key] = result
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"key": key, "result": result}) + "\n")
        return result

    def _ask_vertex(self, state, questions, images, model):
        from google.auth.transport.requests import Request
        parts = [{"text": SYSTEM + "\n\n" + render(state, questions)}]
        for uri in images:
            mime, data = uri.removeprefix("data:").split(";base64,", 1)
            parts.append({"inlineData": {"mimeType": mime, "data": data}})
        body = json.dumps({"contents": [{"role": "user", "parts": parts}],
                           "generationConfig": {"maxOutputTokens": 8000, "responseMimeType": "application/json"}}).encode()
        url = VERTEX_URL.format(project=self._project, model=model.removeprefix("google/"))
        out, start = {}, time.time()
        for attempt in range(self.max_retries):
            with self._lock:
                if not self._credentials.valid:
                    self._credentials.refresh(Request())
                token = self._credentials.token
            request = urllib.request.Request(url, data=body, method="POST", headers={
                "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            start = time.time()
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    out = json.load(response)
                break
            except urllib.error.HTTPError as e:
                message = e.read().decode()[:300]
                if e.code in (408, 429, 500, 502, 503, 504) and attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Vertex HTTP {e.code}: {message}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        candidate = (out.get("candidates") or [{}])[0]
        raw = "".join(p.get("text", "") for p in (candidate.get("content") or {}).get("parts", []) if not p.get("thought"))
        try:
            reply = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except ValueError:
            reply = {}
        usage = out.get("usageMetadata", {})
        return {"answers": to_answers(reply, questions), "usage": usage, "latency": time.time() - start,
                "raw": raw[:500], "reasoning": "default", "provider": "vertex",
                "traffic_type": usage.get("trafficType"), "finish_reason": candidate.get("finishReason")}
