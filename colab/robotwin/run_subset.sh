#!/usr/bin/env bash
# On the VM: run RoboDawn's baseline on a subset of tasks, a few tasks at once.
#
#   bash colab/robotwin/run_subset.sh [--tier flex] [--jobs 3] [--episodes 10] [--shard-size 5]
#                                     [--part K/N] [--label NAME] [--force] [task ...]
#
# Without tasks it runs the ten-task subset below, chosen so that the site's
# success counts spread from 0 to 10 and both demonstration entries appear.
# Each task's episodes are split into shards of --shard-size episodes, one
# process per shard, at most --jobs at once (about 8 GB of GPU memory each:
# 3 on a 24 GB L4, about 9 on an 80 GB A100). Splitting matters because the
# slowest task would otherwise run all its episodes in a row; the shards with
# the most expected turns (the site's) start first. Runs resume: finished
# episodes are skipped, so the script can be started again after a disconnect.
# It refuses to start until the two smoke episodes have passed their condition
# check on the same tier (--force skips that guard; subset.sh passes it on a
# second VM when the smoke checks passed on another VM).
#
# --part K/N splits the tasks over N VMs: tasks go whole to a part, balanced
# by the site's turns (longest first), and the two smoke tasks stay in part 1,
# the VM that ran them, so no episode runs on two VMs.
#
# Progress: outputs/robodawn/subset_<tier>[_<label>]/status.tsv (one line per
# start and end), one log per shard next to it.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
py=${ROBODAWN_PYTHON:-/content/mamba/envs/rt/bin/python}
export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-/content/vertex_key.json}

tier=flex; jobs=3; episodes=10; shard=5; part=1/1; label=; force=0; tasks=()
while [ $# -gt 0 ]; do
  case $1 in
    --tier) tier=$2; shift ;;
    --jobs) jobs=$2; shift ;;
    --episodes) episodes=$2; shift ;;
    --shard-size) shard=$2; shift ;;
    --part) part=$2; shift ;;
    --label) label=$2; shift ;;
    --force) force=1 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) tasks+=("$1") ;;
  esac
  shift
done
# site successes: 10 9 8 7 7 6 4 3 2 0 (56 / 100)
[ ${#tasks[@]} -gt 0 ] || tasks=(click_bell place_a2b_right move_can_pot adjust_bottle place_empty_cup
                                 place_object_stand place_fan open_laptop handover_block lift_pot)

[ -f "$GOOGLE_APPLICATION_CREDENTIALS" ] || { echo "no service account key at $GOOGLE_APPLICATION_CREDENTIALS" >&2; exit 1; }
if [ $force -eq 0 ]; then
  for smoke in adjust_bottle place_empty_cup; do
    check="$root/outputs/robodawn/gemini_flash_$tier/$smoke/shard_0/condition_check.json"
    "$py" -c "import json,sys; sys.exit(bool(json.load(open('$check'))['errors']))" 2>/dev/null \
      || { echo "smoke condition check missing or failed: $check (run the smoke episode or pass --force)" >&2; exit 1; }
  done
fi

logs="$root/outputs/robodawn/subset_$tier${label:+_$label}"
mkdir -p "$logs"
status="$logs/status.tsv"
export root py tier episodes status logs

run_shard() {
  task=$1; start=$2; count=$3; name="$task episodes $start-$((start + count - 1))"
  printf '%s\t%s\tstart\n' "$(date '+%F %T')" "$name" >> "$status"
  "$py" "$root/scripts/run_robodawn_baseline.py" --task "$task" --start-episode "$start" --episodes "$count" \
    --tier "$tier" >> "$logs/${task}__$start.log" 2>&1
  code=$?
  printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$([ $code -eq 0 ] && echo done || echo "failed $code")" >> "$status"
}
export -f run_shard

# "task start count" per shard of this part, the most site turns first (longest-first keeps the last shard short)
shards=$("$py" - "$root/robodawn_site/reproduction_manifest.json" "$episodes" "$shard" "$part" "${tasks[@]}" <<'PY'
import json, sys
manifest, episodes, size, tasks = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[5:]
k, n = map(int, sys.argv[4].split("/"))
if not 1 <= k <= n:
    sys.exit(f"bad --part {k}/{n}")
turns = {t["task"]: [e["site_turns"] for e in t["episodes"]] for t in json.load(open(manifest))["tasks"]}
missing = [t for t in tasks if t not in turns]
if missing:
    sys.exit(f"unknown tasks: {missing}")
load = lambda t: sum(turns[t][:episodes])
parts = [[] for _ in range(n)]
smoke = [t for t in ("adjust_bottle", "place_empty_cup") if t in tasks]   # their episode 0 ran on part 1
parts[0] += smoke
for t in sorted((t for t in tasks if t not in smoke), key=load, reverse=True):
    min(parts, key=lambda p: sum(map(load, p))).append(t)
jobs = [(sum(turns[t][s:min(s + size, episodes)]), t, s, min(size, episodes - s))
        for t in parts[k - 1] for s in range(0, episodes, size)]
for _, t, s, c in sorted(jobs, key=lambda j: -j[0]):
    print(t, s, c)
PY
) || exit 1
[ -n "$shards" ] || { echo "part $part has no tasks" >&2; exit 1; }
mine=($(printf '%s\n' "$shards" | cut -d' ' -f1 | sort -u))

echo "$(date '+%F %T') subset start: tier=$tier jobs=$jobs episodes=$episodes shard=$shard part=$part tasks=${mine[*]}"
printf '%s\n' "$shards" | xargs -P "$jobs" -L1 bash -c 'run_shard "$@"' _
echo "$(date '+%F %T') subset end"
"$py" "$root/scripts/compare_robodawn.py" --tier "$tier" "${mine[@]}" | tee "$logs/compare.txt"
echo SUBSET_DONE
