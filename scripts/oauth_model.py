#!/usr/bin/env python3
"""Preview or send ONE text/image request over direct OAuth HTTP.

No inference CLI is launched. Default: dry run, with no credential access.
--execute sends a real request; --status only checks local token metadata.

Examples:
  python scripts/oauth_model.py --backend openai-oauth --model gpt-6.1-sol
  python scripts/oauth_model.py --backend anthropic-oauth --model MODEL_ID --image frame.png
  python scripts/oauth_model.py --backend openai-codex-oauth --model MODEL_ID \
      --oauth-credentials ~/.config/branchlab/codex-oauth.json --execute

Authentication:
  Set OPENAI_OAUTH_ACCESS_TOKEN or ANTHROPIC_OAUTH_ACCESS_TOKEN (OAuth access
  tokens, not API keys). Codex also requires OPENAI_OAUTH_ACCOUNT_ID.
  Or pass --oauth-credentials with a private JSON file (mode 0600) OUTSIDE git:
    {"backend":"openai-oauth", "access_token":"...", "refresh_token":"...",
     "client_id":"...", "expires_at":1790000000}
  expires_at is Unix seconds. Refresh uses the client_id that issued the token;
  Codex credentials also require account_id. Optional refresh fields can be
  omitted for access-token-only use. Errors then require reauthentication.
  All workers should share ONE file so rotating refresh tokens stay synchronized.
  Do not copy refresh tokens that another CLI/process is actively refreshing.

Routes:
  openai-oauth: official Sign in with ChatGPT token -> api.openai.com/v1/responses
  openai-codex-oauth: Codex token -> chatgpt.com/backend-api/codex/responses
  anthropic-oauth: token -> api.anthropic.com/v1/messages?beta=true.
  Tokens are specific to their issuing client/scopes; these are not interchangeable.
  Codex and Claude subscription direct transports are experimental. A Bearer
  request alone does not establish support for every subscription token.
  Claude follows the user's claude-oauth-provider.js v0.1.7 request profile,
  including its dynamic billing system block and beta headers, with the
  compatibility version pinned to locally verified Claude Code 2.1.295. Experiment
  instructions follow that block unchanged. No agent identity block is added.
  GPT omits max_output_tokens. Claude uses max_tokens=64000 by default;
  --max-tokens overrides it. Claude replies stream for large output budgets.
  Effort is explicit or left at the provider's default.

References checked 2026-10-10:
  https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference
  https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations
  https://platform.claude.com/docs/en/api/overview
"""
import argparse
import base64
import json
import mimetypes
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from branchlab.oauth import Credentials, ENDPOINTS, OAuthError, OAuthTransport, request_body


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=ENDPOINTS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--oauth-credentials", type=Path)
    parser.add_argument("--prompt", default="Reply with exactly OK.")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--max-tokens", type=int, default=64000, help="Claude output budget; ignored for GPT")
    parser.add_argument("--timeout", type=float, default=300)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args()
    credentials = Credentials(args.backend, args.oauth_credentials)
    if args.status:
        # No refresh or network access, and no account ID/token output.
        data = credentials._read()
        print(json.dumps({"backend": args.backend, "configured": True,
                          "refresh_configured": bool(data.get("refresh_token") and data.get("client_id")),
                          "expired": float(data["expires_at"]) <= time.time() if data.get("expires_at") else None}))
        return
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    parts = [{"type": "text", "text": args.prompt}]
    for path in args.image:
        mime = mimetypes.guess_type(path.name)[0]
        uri = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    body = {"model": args.model, "messages": [{"role": "user", "content": parts}]}
    if args.backend == "anthropic-oauth":
        body["max_tokens"] = args.max_tokens
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    native = request_body(body, args.backend)
    if not args.execute:
        print(json.dumps({"dry_run": True, "backend": args.backend, "endpoint": ENDPOINTS[args.backend],
                          "model": native["model"], "images": len(args.image),
                          "request_fields": sorted(native), "reasoning_effort": args.reasoning_effort,
                          "output_token_limit_policy": "configured" if args.backend == "anthropic-oauth" else "omitted",
                          "max_tokens": native.get("max_tokens"),
                          "max_output_tokens_sent": args.backend == "anthropic-oauth"}, indent=2))
        return
    start = time.monotonic()
    result = OAuthTransport(args.backend, credentials, args.timeout).send(body)
    print(json.dumps({"seconds": time.monotonic() - start, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except OAuthError as exc:
        print(f"{exc} (status={exc.status})", file=sys.stderr)
        raise SystemExit(1)
