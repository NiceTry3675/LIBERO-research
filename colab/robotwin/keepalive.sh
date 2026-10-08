#!/usr/bin/env bash
# From this machine: keep a Colab session from being reclaimed while its work runs detached.
#
#   bash colab/robotwin/keepalive.sh start SESSION   start the pinger in the background (no-op if running)
#   bash colab/robotwin/keepalive.sh stop  SESSION   stop it
#
# The Colab CLI keeps a VM only while its kernel is active, and launch.sh / subset.sh start their
# work detached and return, so an idle kernel was reclaimed about 15 minutes after the last
# `colab exec` (2026-10-08). The pinger runs a trivial exec every 240 s and exits by itself once the
# session has been missing from `colab sessions` three times in a row. launch.sh and every
# subset.sh command start it; subset.sh stop stops it. Pid and log: outputs/colab_keepalive/.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
colab=$(command -v colab || echo "$HOME/.local/bin/colab")
cmd=${1:-}; session=${2:-}
[ -n "$session" ] || { sed -n '2,5p' "$0"; exit 2; }
dir="$root/outputs/colab_keepalive"; pid="$dir/$session.pid"; log="$dir/$session.log"
mkdir -p "$dir"
# A `colab exec` occasionally never returns; perl's alarm bounds it (macOS has no `timeout`).
limit() { perl -e 'alarm shift; exec @ARGV' "$@"; }
running() { [ -f "$pid" ] && kill -0 "$(cat "$pid")" 2>/dev/null; }

case $cmd in
  start)
    running && exit 0
    nohup bash "$0" loop "$session" >> "$log" 2>&1 &
    echo $! > "$pid"
    echo "keepalive started for '$session' (log: ${log#"$root"/})"
    ;;
  stop)
    if running; then kill "$(cat "$pid")" && echo "keepalive stopped for '$session'"; fi
    rm -f "$pid"
    ;;
  loop)
    missing=0
    while true; do
      if echo 'print("alive")' | limit 90 "$colab" exec -s "$session" --timeout 60 2>&1 | grep -q alive; then
        missing=0; echo "$(date '+%F %T') ok"
      elif limit 60 "$colab" sessions 2>&1 | grep -q "^\[$session\]"; then
        echo "$(date '+%F %T') ping failed; session still listed"
      else
        missing=$((missing + 1)); echo "$(date '+%F %T') session missing ($missing/3)"
        [ $missing -ge 3 ] && { rm -f "$pid"; exit 0; }
      fi
      sleep 240
    done
    ;;
  *) sed -n '2,5p' "$0"; exit 2 ;;
esac
