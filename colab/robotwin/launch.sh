#!/usr/bin/env bash
# Start a Colab VM and install RoboTwin on it, from this machine.
#
#   bash colab/robotwin/launch.sh [--textures] [--session NAME] [--gpu L4]
#
# --textures also downloads the 11 GB of background textures, needed only for
# the domain-randomized scenes of official evaluations comparable to RoboDawn.
# The Hugging Face token is taken from HUGGINGFACE_API_KEY in the project .env
# and uploaded as a file (/content/.hf_token), so it never appears in executed
# code, which the Colab CLI records in its session history. Nothing else from
# .env leaves this machine.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
colab=$(command -v colab || echo "$HOME/.local/bin/colab")
session=robotwin; gpu=L4; textures=0
# A `colab` call occasionally hangs or loses its connection (2026-10-08), so each call is bounded
# (perl's alarm: macOS has no `timeout`) and retried; everything sent with vm is safe to run twice.
limit() { perl -e 'alarm shift; exec @ARGV' "$@"; }
retry() { for try in 1 2 3; do "$@" && return 0; echo "failed (try $try/3): $*" >&2; sleep 15; done; return 1; }
vm_once() { printf '%s\n' "$1" | limit 180 "$colab" exec -s "$session" --timeout 120 > /dev/null; }
vm() { local code; code=$(cat); retry vm_once "$code"; }
while [ $# -gt 0 ]; do
  case $1 in
    --textures) textures=1 ;;
    --session) session=$2; shift ;;
    --gpu) gpu=$2; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if ! limit 60 "$colab" sessions 2>&1 | grep -qw "$session"; then
  limit 900 "$colab" new -s "$session" --gpu "$gpu"
fi
# Detached work leaves the kernel idle, and Colab reclaims an idle VM in about 15 minutes.
bash "$here/keepalive.sh" start "$session"
for f in bootstrap download_assets setup_env install_rt run_expert; do
  retry limit 300 "$colab" upload -s "$session" "$here/$f.sh" "/content/$f.sh"
done

token=$(grep -E '^HUGGINGFACE_API_KEY=' "$root/.env" 2>/dev/null | cut -d= -f2- | tr -d "\"' \r" || true)
if [ -n "$token" ]; then
  tmp=$(mktemp); chmod 600 "$tmp"; printf '%s' "$token" > "$tmp"
  retry limit 300 "$colab" upload -s "$session" "$tmp" /content/.hf_token
  rm -f "$tmp"
  echo 'import os; os.chmod("/content/.hf_token", 0o600)' | vm
else
  echo "no HUGGINGFACE_API_KEY in .env; downloading anonymously"
fi

# bootstrap.log exists as soon as the bootstrap starts, so a retried exec starts it only once.
vm <<EOF
import os, subprocess
if not os.path.exists("/content/bootstrap.log"):
    subprocess.Popen('WITH_TEXTURES=$textures bash /content/bootstrap.sh > /content/bootstrap.log 2>&1', shell=True, executable='/bin/bash')
EOF
echo "bootstrap started on '$session' (textures: $textures). Follow it with:"
echo "  echo 'print(open(\"/content/bootstrap.log\").read())' | colab exec -s $session"
echo "Stop the VM when done: bash colab/robotwin/subset.sh --session $session stop (also stops the keepalive)"
