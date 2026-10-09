#!/usr/bin/env bash
# On the VM: replay the recorded episodes of outputs/replay/manifest.json and capture depth (scripts/replay_capture.py).
#
#   bash colab/robotwin/run_replay.sh [--jobs 3] [--shard-size 4] [--part K/N] [--manifest PATH]
#
# Episodes are grouped into shards of one task (the environment is built per task), one process per shard,
# at most --jobs at once, the shards with the most recorded turns first. --part K/N gives this VM every N-th
# shard of that order, starting at the K-th, so N VMs share the work evenly. A shard whose process dies is
# restarted while each attempt finishes at least one more episode (finished episodes are skipped), as in
# run_subset.sh. Progress: outputs/replay/status.tsv; one log per shard next to it.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
py=${ROBODAWN_PYTHON:-/content/mamba/envs/rt/bin/python}
jobs=3; size=4; part=1/1; manifest=$root/outputs/replay/manifest.json
while [ $# -gt 0 ]; do
  case $1 in
    --jobs) jobs=$2; shift ;;
    --shard-size) size=$2; shift ;;
    --part) part=$2; shift ;;
    --manifest) manifest=$2; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
logs="$root/outputs/replay/logs"; mkdir -p "$logs"
status="$root/outputs/replay/status.tsv"
export root py status logs manifest

finished() {   # finished episodes of a shard: those with a meta.json
  "$py" - "$root" "$@" <<'PY'
import json, sys
from pathlib import Path
root, keys = Path(sys.argv[1]), sys.argv[2:]
print(sum((root/"outputs/replay"/k.split("/")[0]/k.split("/")[1]/f"episode_{int(k.split('/')[2]):03d}"/"meta.json").is_file() for k in keys))
PY
}
run_shard() {
  name=$1; shift
  printf '%s\t%s\tstart\n' "$(date '+%F %T')" "$name" >> "$status"
  while true; do
    before=$(finished "$@")
    "$py" "$root/scripts/replay_capture.py" replay --manifest "$manifest" --keys "$@" >> "$logs/$name.log" 2>&1
    code=$?
    after=$(finished "$@")
    [ $code -eq 0 ] || [ "$after" -le "$before" ] || [ "$after" -ge $# ] && break
    printf '%s\t%s\trestart (exit %s, %s/%s episodes)\n' "$(date '+%F %T')" "$name" "$code" "$after" "$#" >> "$status"
  done
  printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$([ $code -eq 0 ] && echo done || echo "failed $code")" >> "$status"
}
export -f finished run_shard

# "name key key ..." per shard of this part
shards=$("$py" - "$manifest" "$size" "$part" <<'PY'
import json, sys
from collections import defaultdict
manifest, size = json.load(open(sys.argv[1])), int(sys.argv[2])
k, n = map(int, sys.argv[3].split("/"))
by = defaultdict(list)
for e in manifest:
    by[(e["run"], e["task"])].append(e)
shards = []
for (run, task), eps in sorted(by.items()):
    eps.sort(key=lambda e: e["episode"])
    for i in range(0, len(eps), size):
        chunk = eps[i:i + size]
        shards.append((sum(len(e["turns"]) for e in chunk), f"{run}__{task}__{chunk[0]['episode']}", [e["key"] for e in chunk]))
shards.sort(key=lambda s: -s[0])
for turns, name, keys in shards[k - 1::n]:
    print(name, *keys)
PY
) || exit 1
echo "$(date '+%F %T') replay start: jobs=$jobs shard-size=$size part=$part shards=$(printf '%s\n' "$shards" | wc -l)"
printf '%s\n' "$shards" | xargs -P "$jobs" -L1 bash -c 'run_shard "$@"' _
echo "$(date '+%F %T') replay end"
echo REPLAY_DONE
