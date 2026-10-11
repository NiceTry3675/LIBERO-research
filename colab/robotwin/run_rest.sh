#!/usr/bin/env bash
# On the VM: finish a task subset on another tier, one process per episode, skipping episodes already finished
# on the earlier tier. Used on 2026-10-10 when flex slowed down (recent median call over 200 s) mid-run.
#
#   bash colab/robotwin/run_rest.sh --from flex --tier standard [--jobs 3] task ...
#
# Episodes finished under --from stay there; the rest run under --tier in shard_<episode> directories
# (scripts/compare_robodawn.py merges shards). One episode per process also avoids the per-process GPU memory
# growth that made third episodes fail on an L4. Progress: outputs/robodawn/rest_<tier>/status.tsv.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
py=${ROBODAWN_PYTHON:-/content/mamba/envs/rt/bin/python}
export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-/content/vertex_key.json}
from=flex; tier=standard; jobs=3; tasks=()
while [ $# -gt 0 ]; do
  case $1 in
    --from) from=$2; shift ;;
    --tier) tier=$2; shift ;;
    --jobs) jobs=$2; shift ;;
    *) tasks+=("$1") ;;
  esac
  shift
done
logs=$root/outputs/robodawn/rest_$tier; mkdir -p "$logs"
export root py tier logs
todo=$("$py" - "$root" "$from" "$tier" "${tasks[@]}" <<'PY'
import json, sys
from pathlib import Path
root, old, new, tasks = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
manifest = {t["task"]: t["episodes"] for t in json.loads((root/"robodawn_site/reproduction_manifest.json").read_text())["tasks"]}
def done(tier, task):
    eps = set()
    for r in (root/f"outputs/robodawn/gemini_flash_{tier}/{task}").glob("shard_*/results.json"):
        eps |= {e["episode_index"] for e in json.loads(r.read_text())["episodes"]}
    return eps
rows = []
for t in tasks:
    skip = done(old, t) | done(new, t)
    rows += [(e["site_turns"], t, e["episode"]) for e in manifest[t] if e["episode"] not in skip]
for _, t, e in sorted(rows, reverse=True):
    print(t, e)
PY
)
echo "$(date '+%F %T') rest start: from=$from tier=$tier jobs=$jobs episodes=$(printf '%s\n' "$todo" | grep -c .)"
run_one() {
  printf '%s\t%s %s\tstart\n' "$(date '+%F %T')" "$1" "$2" >> "$logs/status.tsv"
  "$py" "$root/scripts/run_robodawn_baseline.py" --task "$1" --start-episode "$2" --episodes 1 --tier "$tier" \
    >> "$logs/${1}__$2.log" 2>&1
  code=$?
  printf '%s\t%s %s\t%s\n' "$(date '+%F %T')" "$1" "$2" "$([ $code -eq 0 ] && echo done || echo "failed $code")" >> "$logs/status.tsv"
}
export -f run_one
printf '%s\n' "$todo" | grep . | xargs -P "$jobs" -L1 bash -c 'run_one "$@"' _
echo "$(date '+%F %T') rest end"
echo REST_DONE
