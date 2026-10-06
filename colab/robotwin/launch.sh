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
while [ $# -gt 0 ]; do
  case $1 in
    --textures) textures=1 ;;
    --session) session=$2; shift ;;
    --gpu) gpu=$2; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if ! "$colab" sessions 2>&1 | grep -qw "$session"; then
  "$colab" new -s "$session" --gpu "$gpu"
fi
for f in bootstrap download_assets setup_env install_rt run_expert; do
  "$colab" upload -s "$session" "$here/$f.sh" "/content/$f.sh"
done

token=$(grep -E '^HUGGINGFACE_API_KEY=' "$root/.env" 2>/dev/null | cut -d= -f2- | tr -d "\"' \r" || true)
if [ -n "$token" ]; then
  tmp=$(mktemp); chmod 600 "$tmp"; printf '%s' "$token" > "$tmp"
  "$colab" upload -s "$session" "$tmp" /content/.hf_token
  rm -f "$tmp"
  echo 'import os; os.chmod("/content/.hf_token", 0o600)' | "$colab" exec -s "$session" > /dev/null
else
  echo "no HUGGINGFACE_API_KEY in .env; downloading anonymously"
fi

echo "import subprocess; subprocess.Popen('WITH_TEXTURES=$textures bash /content/bootstrap.sh > /content/bootstrap.log 2>&1', shell=True, executable='/bin/bash')" \
  | "$colab" exec -s "$session" > /dev/null
echo "bootstrap started on '$session' (textures: $textures). Follow it with:"
echo "  echo 'print(open(\"/content/bootstrap.log\").read())' | colab exec -s $session"
echo "Stop the VM when done: colab stop -s $session"
