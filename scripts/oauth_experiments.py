"""Shared, secret-free settings for the optional direct OAuth experiment path."""
import re
from pathlib import Path

BACKENDS = ("vertex", "openai-oauth", "openai-codex-oauth", "anthropic-oauth")
ENDPOINTS = {"openai-oauth": "https://api.openai.com/v1/responses",
             "openai-codex-oauth": "https://chatgpt.com/backend-api/codex/responses",
             "anthropic-oauth": "https://api.anthropic.com/v1/messages?beta=true"}


def add_arguments(parser):
    parser.add_argument("--backend", choices=BACKENDS, default="vertex")
    parser.add_argument("--model", help="Exact model ID; required for OAuth")
    parser.add_argument("--oauth-credentials", type=Path,
                        help="Private OAuth JSON; otherwise read *_OAUTH_ACCESS_TOKEN environment variable")
    parser.add_argument("--reasoning-effort", help="Explicit OAuth effort; omission uses provider default")


def settings(args):
    backend = getattr(args, "backend", "vertex")
    if backend == "vertex":
        if getattr(args, "model", None) or getattr(args, "reasoning_effort", None):
            raise ValueError("--model/--reasoning-effort are for OAuth experiments; Vertex baseline is pinned")
        return {}
    model = getattr(args, "model", None)
    if not model or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", model):
        raise ValueError("OAuth experiments require --model with an exact model ID")
    prefix = "anthropic/" if backend == "anthropic-oauth" else "openai/"
    model = model.removeprefix(prefix)
    if "/" in model:
        raise ValueError("Model prefix does not match OAuth backend")
    effort = getattr(args, "reasoning_effort", None)
    allowed = {"low", "medium", "high", "xhigh", "max"} if backend == "anthropic-oauth" else {
        "none", "minimal", "low", "medium", "high", "xhigh", "max"}
    if effort and effort not in allowed:
        raise ValueError("Invalid reasoning effort for this backend")
    return {"backend": backend, "model": model, "endpoint": ENDPOINTS[backend], "tier": "oauth",
            "reasoning": effort or "provider default", "temperature": "not sent",
            "timeout_s": 300, "stall_timeout_s": 900,
            "max_tokens": (getattr(args, "max_tokens", None) or 64000) if backend == "anthropic-oauth" else None,
            "oauth": {"transport_version": 4, "reasoning_effort": effort,
                      "request_profile": "risu-0.1.7-cc-2.1.295" if backend == "anthropic-oauth" else None,
                      "system_preamble": "dynamic Claude billing marker" if backend == "anthropic-oauth" else None,
                      "output_token_limit_policy": "configured" if backend == "anthropic-oauth" else "omitted",
                      "max_output_tokens_sent": backend == "anthropic-oauth",
                      "credential_source": "file" if getattr(args, "oauth_credentials", None) else "environment",
                      "compatibility": "official Sign in with ChatGPT" if backend == "openai-oauth" else
                                       "experimental direct token transport; live compatibility unverified"},
            "limitations": ["OAuth quota and latency differ from Vertex. No API price estimate is implied.",
                            "Reasoning tokens may be unreported; missing usage is not proof of disabled reasoning."] +
                           (["GPT OAuth sends no output token cap; --max-tokens is ignored."] if backend != "anthropic-oauth"
                            else ["Claude max_tokens defaults to 64000; --max-tokens overrides it.",
                                  "Claude prepends the reference provider's dynamic billing marker to system content."])}


def model_matches(requested, returned):
    # Permit dated snapshots of an exact alias, not arbitrary matching prefixes.
    return isinstance(returned, str) and (returned == requested or
        bool(re.fullmatch(re.escape(requested) + r"-(?:\d{8}|\d{4}-\d{2}-\d{2})", returned)))


def condition_errors(config, transport):
    answered = [row for row in transport if row["status"] == 200]
    errors = []
    if not answered or any(not model_matches(config["model"], row.get("response_model")) for row in answered):
        errors.append("OAuth response model is missing or differs from requested model")
    expected = config["oauth"]["reasoning_effort"]
    if any(row.get("backend") != config["backend"] or
           row.get("reasoning_fields", {}).get("reasoning_effort") != expected for row in transport):
        errors.append("OAuth transport or reasoning setting differs from experiment configuration")
    return errors
