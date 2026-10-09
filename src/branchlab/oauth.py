"""Direct OAuth HTTP transport for multimodal experiments (no inference CLI).

Backends: openai-oauth (Sign in with ChatGPT Responses API),
openai-codex-oauth (Codex token/backend compatibility), anthropic-oauth
(Messages with Bearer auth; subscription-token compatibility is experimental).

Credentials are read from an explicitly supplied private JSON file containing
access_token and optionally refresh_token, expires_at (Unix seconds), client_id,
and account_id (Codex). CLI credential files are deliberately not modified.
Alternatively set OPENAI_OAUTH_ACCESS_TOKEN / ANTHROPIC_OAUTH_ACCESS_TOKEN;
Codex also needs OPENAI_OAUTH_ACCOUNT_ID. Environment tokens are not refreshed.
No fallback to API keys, a different endpoint, or a different model is made.
"""
from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.request

ENDPOINTS = {
    "openai-oauth": "https://api.openai.com/v1/responses",
    "openai-codex-oauth": "https://chatgpt.com/backend-api/codex/responses",
    "anthropic-oauth": "https://api.anthropic.com/v1/messages?beta=true",
}
TOKEN_URLS = {
    "openai": "https://auth.openai.com/oauth/token",
    "anthropic": "https://platform.claude.com/v1/oauth/token",
}
DEFAULT_CLAUDE_MAX_TOKENS = 64000
# Request profile from the user's claude-oauth-provider.js v0.1.7.
# Update its version field to the locally verified Claude Code 2.1.295:
# Opus 5.5 rejected the reference's 2.1.87 and requires >=2.1.280.
# Pin it for reproducibility rather than silently adopting future CLI updates.
CLAUDE_CODE_VERSION = "2.1.295"
CLAUDE_COMPAT_VERSION = f"risu-0.1.7-cc-{CLAUDE_CODE_VERSION}"
CLAUDE_BETAS = "oauth-2025-04-20,interleaved-thinking-2025-05-14"


def _js_utf8(text):
    # JavaScript TextEncoder replaces lone UTF-16 surrogates with U+FFFD.
    return text.encode("utf-16-le", errors="surrogatepass").decode("utf-16-le", errors="replace").encode("utf-8")


def claude_billing_marker(messages):
    """Match reference firstUserText/computeCch/computeVersionSuffix exactly."""
    first_user = next((m for m in messages if m["role"] == "user"), {})
    first_text = next((p for p in first_user.get("content", []) if p.get("type") == "text"), {})
    text = first_text.get("text", "")
    utf16 = text.encode("utf-16-le", errors="surrogatepass")
    sampled = b"".join(utf16[i * 2:i * 2 + 2] or b"0\x00" for i in (4, 7, 20))
    sampled = sampled.decode("utf-16-le", errors="surrogatepass")
    suffix = hashlib.sha256(_js_utf8("59cf53e54c78" + sampled + CLAUDE_CODE_VERSION)).hexdigest()[:3]
    cch = hashlib.sha256(_js_utf8(text)).hexdigest()[:5]
    return (f"x-anthropic-billing-header: cc_version={CLAUDE_CODE_VERSION}.{suffix}; "
            f"cc_entrypoint=sdk-cli; cch={cch};")


class OAuthError(RuntimeError):
    """Only safe local diagnostics, never response bodies or credential values."""
    def __init__(self, message, status=0):
        super().__init__(message)
        self.status = status


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise OAuthError("OAuth redirects are disabled", code)


def open_request(request, timeout):
    return urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout)


def private_json(path):
    """Reject shared, linked or non-regular files before reading secrets."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd) as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or info.st_mode & 0o077):
                raise OAuthError("OAuth file must be owned by you, regular, unlinked and mode 0600")
            result = json.load(stream)
        if not isinstance(result, dict):
            raise ValueError
        return result
    except OAuthError:
        raise
    except (OSError, ValueError):
        raise OAuthError("Cannot read private OAuth credential JSON") from None


class Credentials:
    def __init__(self, backend, path=None):
        if backend not in ENDPOINTS:
            raise ValueError("Unknown OAuth backend")
        self.backend = backend
        self.family = "anthropic" if backend == "anthropic-oauth" else "openai"
        self.path = Path(path).absolute() if path else None
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def locked(self):
        with self._lock:
            if not self.path:
                yield
                return
            # A separate lock survives atomic replacement. Multiple experiment
            # workers must use this same file instead of copying refresh tokens.
            try:
                fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            except OSError:
                raise OAuthError("Cannot lock OAuth credential file") from None
            with os.fdopen(fd, "w") as lock:
                info = os.fstat(lock.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_nlink != 1 or info.st_mode & 0o077):
                    raise OAuthError("Unsafe OAuth lock file")
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self):
        if self.path:
            data = private_json(self.path)
        else:
            prefix = self.family.upper() + "_OAUTH_"
            data = {"access_token": os.environ.get(prefix + "ACCESS_TOKEN"),
                    "account_id": os.environ.get(prefix + "ACCOUNT_ID")}
        if not isinstance(data.get("access_token"), str) or not data["access_token"].strip():
            raise OAuthError("OAuth access token missing; supply --oauth-credentials or OAuth environment variables", 401)
        if any(c.isspace() for c in data["access_token"]):
            raise OAuthError("Invalid OAuth access token", 401)
        if self.backend == "openai-codex-oauth" and not data.get("account_id"):
            raise OAuthError("Codex OAuth requires account_id", 401)
        if data.get("backend", self.backend) != self.backend:
            raise OAuthError("Credential backend differs from requested backend", 401)
        return data

    def get(self, rejected_token=None):
        with self.locked():
            data = self._read()
            try:
                expired = data.get("expires_at") is not None and float(data["expires_at"]) <= time.time() + 60
            except (TypeError, ValueError):
                raise OAuthError("Invalid OAuth expires_at; expected Unix seconds", 401) from None
            # Another worker may already have replaced a rejected token.
            if expired or (rejected_token is not None and data["access_token"] == rejected_token):
                data = self._refresh(data)
            return data

    def _refresh(self, data):
        if not self.path or not data.get("refresh_token") or not data.get("client_id"):
            raise OAuthError("OAuth token expired/rejected; reauthenticate or provide refresh_token and client_id", 401)
        body = {"grant_type": "refresh_token", "refresh_token": data["refresh_token"], "client_id": data["client_id"]}
        request = urllib.request.Request(TOKEN_URLS[self.family], data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with open_request(request, 30) as response:
                new = json.load(response)
            if not isinstance(new.get("access_token"), str) or not new["access_token"]:
                raise ValueError
            updated = {**data, "access_token": new["access_token"],
                       "refresh_token": new.get("refresh_token") or data["refresh_token"],
                       "expires_at": time.time() + float(new["expires_in"])}
            fd, name = tempfile.mkstemp(prefix=".oauth-", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(updated, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, self.path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
            return updated
        except urllib.error.HTTPError as exc:
            raise OAuthError("OAuth refresh rejected", exc.code) from None
        except OAuthError:
            raise
        except (OSError, ValueError, KeyError, TypeError):
            raise OAuthError("OAuth refresh or credential persistence failed") from None


def content_parts(content):
    return [{"type": "text", "text": content}] if isinstance(content, str) else content


def image_data(part):
    uri = part["image_url"]["url"]
    if not uri.startswith("data:") or ";base64," not in uri:
        raise ValueError("OAuth experiments accept inline base64 images only")
    mime, encoded = uri[5:].split(";base64,", 1)
    if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        raise ValueError("Unsupported image media type")
    base64.b64decode(encoded, validate=True)
    return uri, mime, encoded


def request_body(body, backend):
    if backend not in ENDPOINTS:
        raise ValueError("Unknown OAuth backend")
    unknown = set(body) - {"model", "messages", "max_tokens", "temperature", "reasoning_effort"}
    if unknown:
        raise ValueError(f"Unsupported OAuth request fields: {sorted(unknown)}")
    anthropic = backend == "anthropic-oauth"
    model = body["model"].removeprefix("anthropic/" if anthropic else "openai/")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model):
        raise ValueError("Pass an exact model ID matching the OAuth backend")
    messages, system = [], []
    for message in body["messages"]:
        role = message["role"]
        if role not in ("system", "developer", "user", "assistant"):
            raise ValueError("Unsupported OAuth message role")
        parts = []
        for part in content_parts(message["content"]):
            if part["type"] == "text":
                kind = "text" if anthropic else "output_text" if role == "assistant" else "input_text"
                parts.append({"type": kind, "text": part["text"]})
            elif part["type"] == "image_url" and role == "user":
                uri, mime, encoded = image_data(part)
                parts.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": encoded}}
                             if anthropic else {"type": "input_image", "image_url": uri,
                                               "detail": part["image_url"].get("detail", "auto")})
            else:
                raise ValueError("Unsupported OAuth content part or image role")
        if anthropic and role in ("system", "developer"):
            if messages:
                raise ValueError("Anthropic system messages must precede the conversation")
            system.extend(parts)
        else:
            messages.append({"role": "developer" if role == "system" else role, "content": parts})
    if anthropic:
        limit = body.get("max_tokens", DEFAULT_CLAUDE_MAX_TOKENS)
        if type(limit) is not int or limit <= 0:
            raise ValueError("Claude max_tokens must be a positive integer")
        out = {"model": model, "messages": messages, "stream": True, "max_tokens": limit}
        out["system"] = [{"type": "text", "text": claude_billing_marker(messages)}, *system]
        if body.get("reasoning_effort"):
            out.update(thinking={"type": "adaptive"}, output_config={"effort": body["reasoning_effort"]})
        elif body.get("temperature") is not None:
            out["temperature"] = body["temperature"]
    else:
        out = {"model": model, "input": messages,
               "store": False, "stream": True, "tools": []}
        if backend == "openai-codex-oauth":
            out["instructions"] = ""
        if body.get("reasoning_effort"):
            out["reasoning"] = {"effort": body["reasoning_effort"]}
    return out


def sse_response(response):
    """Only a completed terminal event is success; never accept partial text."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            joined, data = "\n".join(data), []
            if joined == "[DONE]":
                break
            event = json.loads(joined)
            if not isinstance(event, dict):
                raise OAuthError("Malformed OAuth stream event")
            if event.get("type") == "response.completed":
                return event["response"]
            if event.get("type") in ("response.failed", "response.incomplete", "error"):
                error = event.get("error") or (event.get("response") or {}).get("error") or {}
                code = error.get("code") if isinstance(error, dict) else None
                status = 429 if code in {"rate_limit_exceeded", "usage_limit_reached",
                                        "subscription_sharing_usage_limit_exceeded"} else 422
                raise OAuthError("OAuth stream ended with " + event["type"], status)
    raise OAuthError("OAuth stream interrupted before response.completed")


def normalized(out, backend):
    anthropic = backend == "anthropic-oauth"
    if not isinstance(out, dict) or out.get("error") or not isinstance(out.get("model"), str):
        raise OAuthError("OAuth response is missing model metadata or contains an error")
    if not isinstance(out.get("content" if anthropic else "output"), list):
        raise OAuthError("OAuth response is missing output content")
    blocks = out.get("content", []) if anthropic else [
        p for item in out.get("output", []) if item.get("type") == "message" for p in item.get("content", [])]
    text = "".join(p.get("text", "") for p in blocks if p.get("type") in ("text", "output_text"))
    usage = out.get("usage") or {}
    prompt = usage.get("input_tokens", 0)
    if anthropic:
        prompt += usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
    completion = usage.get("output_tokens", 0)
    result_usage = {"prompt_tokens": prompt, "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                    "prompt_tokens_details": {"cached_tokens": usage.get("cache_read_input_tokens", 0)}
                    if anthropic else usage.get("input_tokens_details", {}),
                    "completion_tokens_details": usage.get("output_tokens_details", {}),
                    "provider_usage": usage}
    # Missing reasoning usage stays unknown, not a fabricated zero.
    return {"id": out.get("id"), "model": out.get("model"), "usage": result_usage,
            "choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": out.get("stop_reason") if anthropic else out.get("status")} ]}


def anthropic_response(response):
    """Consume a streamed Messages reply; ignore thinking, preserve final usage."""
    message, text, data = None, {}, []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            event, data = json.loads("\n".join(data)), []
            kind = event.get("type")
            if kind == "error":
                error_type = (event.get("error") or {}).get("type")
                status = {"rate_limit_error": 429, "overloaded_error": 529}.get(error_type, 422)
                raise OAuthError("Claude stream returned an error", status)
            if kind == "message_start":
                message = event["message"]
            elif kind == "content_block_start" and event["content_block"].get("type") == "text":
                text[event["index"]] = event["content_block"].get("text", "")
            elif kind == "content_block_delta" and event["delta"].get("type") == "text_delta":
                index = event["index"]
                text[index] = text.get(index, "") + event["delta"]["text"]
            elif kind == "message_delta" and message is not None:
                message.update(event.get("delta") or {})
                message.setdefault("usage", {}).update(event.get("usage") or {})
            elif kind == "message_stop" and message is not None:
                message["content"] = [{"type": "text", "text": text[k]} for k in sorted(text)]
                return message
    raise OAuthError("Claude stream interrupted before message_stop")


class OAuthTransport:
    def __init__(self, backend, credentials=None, timeout=300):
        self.backend = backend
        self.credentials = credentials or Credentials(backend)
        self.timeout = timeout

    def send(self, body):
        # Validate before reading credentials or making network requests.
        native = request_body(body, self.backend)
        rejected = None
        for attempt in range(2):
            auth = self.credentials.get(rejected)
            headers = {"Authorization": "Bearer " + auth["access_token"], "Content-Type": "application/json",
                       "User-Agent": "branchlab-oauth/1"}
            if self.backend == "anthropic-oauth":
                headers.update({"anthropic-version": "2023-06-01", "anthropic-beta": CLAUDE_BETAS,
                                "anthropic-dangerous-direct-browser-access": "true"})
            else:
                headers["Accept"] = "text/event-stream"
                if self.backend == "openai-codex-oauth":
                    headers["ChatGPT-Account-ID"] = auth["account_id"]
            try:
                if self.backend == "anthropic-oauth":
                    headers["Accept"] = "text/event-stream, application/json"
                request = urllib.request.Request(ENDPOINTS[self.backend], data=json.dumps(native).encode(), headers=headers)
                with open_request(request, self.timeout) as response:
                    out = anthropic_response(response) if self.backend == "anthropic-oauth" else sse_response(response)
                result = normalized(out, self.backend)
                result["output_token_limit"] = {"policy": "configured" if self.backend == "anthropic-oauth" else "omitted",
                                                "max_tokens": native.get("max_tokens")}
                if self.backend == "anthropic-oauth":
                    result["request_profile"] = CLAUDE_COMPAT_VERSION
                return result
            except urllib.error.HTTPError as exc:
                if exc.code == 401 and attempt == 0:
                    rejected = auth["access_token"]
                    continue
                raise OAuthError("OAuth inference request rejected", exc.code) from None
            except OAuthError:
                raise
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                raise OAuthError("OAuth network or response parsing failure") from None


def oauth_client(base, backend, credential_path=None, reasoning_effort=None, max_tokens=DEFAULT_CLAUDE_MAX_TOKENS):
    """Adapt RoboDawn's existing retry/logging loop without changing its prompts."""
    class OAuthClient(base):
        def __init__(self, *args, **kwargs):
            kwargs.update(api_key="oauth-managed", base_url=ENDPOINTS[backend], temperature=None)
            if backend == "anthropic-oauth":
                kwargs["max_tokens"] = max_tokens
            super().__init__(*args, **kwargs)
            # Do not let upstream insert family-dependent reasoning defaults.
            self.extra_body = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
            self._oauth = OAuthTransport(backend, Credentials(backend, credential_path), self.timeout_s)

        def _body(self, messages):
            body = super()._body(messages)
            if backend != "anthropic-oauth":
                body.pop("max_tokens", None)
            return body

        def _relax_reasoning(self, error_text, body):
            return False  # an experiment never silently changes effort

        def _post(self, body, tag=""):
            start = time.monotonic()
            try:
                payload, status = self._oauth.send(body), 200
            except OAuthError as exc:
                payload, status = str(exc), exc.status
            except ValueError:
                payload, status = "Invalid OAuth request shape", 400
            if self.log_path:
                raw = payload if isinstance(payload, dict) else {}
                entry = {"tag": tag, "status": status, "seconds": time.monotonic() - start,
                         "backend": backend, "model": body["model"], "response_model": raw.get("model"),
                         "response_id": raw.get("id"), "usage": raw.get("usage"),
                         "output_token_limit": raw.get("output_token_limit"),
                         "request_profile": raw.get("request_profile"),
                         "reasoning_fields": {"reasoning_effort": reasoning_effort} if reasoning_effort else {},
                         "finish_reason": (raw.get("choices") or [{}])[0].get("finish_reason")}
                if status != 200:
                    entry["error"] = payload
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with (self.log_path.parent / "transport.jsonl").open("a") as stream:
                    stream.write(json.dumps(entry) + "\n")
            return status, payload
    return OAuthClient
