"""Minimal client for Cloudflare's Clef decision models (Workers AI).

Clef does not generate text. A request carries a `state` (text or JSON), up
to 4 images as base64 data URIs, and up to 64 typed questions:

    noul    yes/no            -> answer["noul"] = P(yes)
    choice  one of `criteria` -> answer["probabilities"] = {option: p}
    score   ordinal levels    -> answer["probabilities"] = {"0": p, "1": p, ...}

Two backends reach the same models:

    workers     Cloudflare Workers AI directly (CLOUDFLARE_ACCOUNT_ID,
                CLOUDFLARE_AUTH_TOKEN in .env). The default. The free plan stops
                after 10,000 neurons a day, about 1,800 calls of the size used here.
    openrouter  OpenRouter's decisions endpoint (OPENROUTER_API_KEY). Text only:
                it never delivers images to Clef, neither a top-level `images`
                field nor image parts in the state (solid red and solid blue got
                the same answer, 10-09), so this client refuses images there.

The first call with images on a client first asks Clef the colour of a solid red
and a solid blue image and raises if either answer is wrong, so a backend that
drops images fails loudly instead of returning answers from text alone.

Responses can be cached in a JSONL file keyed by a caller-chosen id, so
interrupted runs resume without paying twice.
"""

import base64
import io
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from branchlab import PROJECT_ROOT

MODELS = ("clef", "clef-flash")


def _load_env(path=PROJECT_ROOT / ".env"):
    env = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def image_uri(path) -> str:
    path = Path(path)
    mime = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}[path.suffix.lower()]
    return f"data:image/{mime};base64," + base64.b64encode(path.read_bytes()).decode()


class Clef:
    def __init__(self, cache_path=None, max_retries=6, backend="workers", env_path=None, check_images=True):
        env = _load_env(env_path) if env_path else _load_env()
        self.backend = backend
        self._images_ok = not check_images
        if backend == "workers":
            self._token = env["CLOUDFLARE_AUTH_TOKEN"]
            self._url = (f"https://api.cloudflare.com/client/v4/accounts/{env['CLOUDFLARE_ACCOUNT_ID']}"
                         "/ai/run/@cf/cloudflare/{model}")
        elif backend == "openrouter":
            self._token = env["OPENROUTER_API_KEY"]
            self._url = "https://openrouter.ai/api/alpha/decisions"
        else:
            raise ValueError(f"unknown backend: {backend}")
        self.max_retries = max_retries
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = {}
        self._lock = threading.Lock()
        if self.cache_path and self.cache_path.exists():
            for line in open(self.cache_path):
                row = json.loads(line)
                self._cache[row["key"]] = row["result"]

    def ask(self, state, questions: dict, images=(), model="clef", key=None) -> dict:
        """Returns {"answers": ..., "usage": ..., "latency": seconds}."""
        if key is not None and key in self._cache:
            return self._cache[key]
        if images and self.backend != "workers":
            raise ValueError(f"the {self.backend} backend does not deliver images to Clef; use backend='workers'")
        if images and not self._images_ok:
            self.check_images(model)
        if self.backend == "workers":
            body = {"model": model, "state": state, "questions": questions}
            if images:
                body["images"] = list(images)
        else:
            text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            parts = [{"type": "text", "text": text}]
            parts += [{"type": "image_url", "image_url": {"url": uri}} for uri in images]
            body = {"model": f"cloudflare/{model}", "state": parts, "questions": questions}
        url = self._url.format(model=model)
        data = json.dumps(body).encode()
        for attempt in range(self.max_retries):
            request = urllib.request.Request(url, data=data, headers={
                "Authorization": f"Bearer {self._token}", "Content-Type": "application/json"})
            start = time.time()
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    out = json.load(response)
                out = out.get("result", out)
                break
            except urllib.error.HTTPError as e:
                if e.code in (408, 429, 500, 502, 503, 504, 529) and attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Clef HTTP {e.code}: {e.read().decode()[:500]}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        result = {"answers": out["answers"], "usage": out.get("usage", {}), "latency": time.time() - start}
        if key is not None and self.cache_path:
            with self._lock:
                self._cache[key] = result
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"key": key, "result": result}) + "\n")
        return result

    def check_images(self, model="clef"):
        """Raises unless Clef tells a solid red image from a solid blue one."""
        from PIL import Image
        for colour in ("red", "blue"):
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), colour).save(buf, "PNG")
            uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            self._images_ok = True        # let the probe itself through
            try:
                out = self.ask("A single image.", {"colour": {"type": "choice", "instructions": "What colour is the image?",
                                                               "criteria": {"red": "the image is red",
                                                                            "blue": "the image is blue"}}}, images=[uri], model=model)
            finally:
                self._images_ok = False
            probs = out["answers"]["colour"]["probabilities"]
            if max(probs, key=probs.get) != colour:
                raise RuntimeError(f"Clef ({self.backend}, {model}) did not see the image: solid {colour} -> {probs}")
        self._images_ok = True
