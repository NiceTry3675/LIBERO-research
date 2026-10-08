#!/usr/bin/env bash
# On the VM: run RoboDawn's baseline on a subset of tasks, a few tasks at once.
#
#   bash colab/robotwin/run_subset.sh [--tier flex] [--jobs 3] [--episodes 10] [--shard-size 5]
#                                     [--part K/N] [--label NAME] [--force] [task ...]
#   bash colab/robotwin/run_subset.sh --variant V [--checker C] [--start 24] [--max-tokens N] [same options] [task ...]
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
# --variant runs scripts/run_robodawn_relay.py instead (baseline|prompt|open|checked; checked needs
# --checker clef|lite) on episodes --start .. --start+episodes-1. Episodes 24-49 are development
# episodes, written under outputs/robodawn_dev (12-23 hold the demonstrations' source seeds, which the
# runner refuses); the smoke guard applies to the baseline runner only.
#
# Progress: outputs/robodawn/subset_<tier>[_<label>]/status.tsv (relay: outputs/robodawn[_dev]/
# subset_<run>[_<label>]/), one line per start, restart and end, one log per shard next to it.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
py=${ROBODAWN_PYTHON:-/content/mamba/envs/rt/bin/python}
export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-/content/vertex_key.json}

tier=flex; jobs=3; episodes=10; shard=5; part=1/1; label=; force=0; tasks=(); variant=; checker=; start=0; maxtok=
while [ $# -gt 0 ]; do
  case $1 in
    --tier) tier=$2; shift ;;
    --jobs) jobs=$2; shift ;;
    --episodes) episodes=$2; shift ;;
    --shard-size) shard=$2; shift ;;
    --part) part=$2; shift ;;
    --label) label=$2; shift ;;
    --force) force=1 ;;
    --variant) variant=$2; shift ;;
    --checker) checker=$2; shift ;;
    --start) start=$2; shift ;;
    --max-tokens) maxtok=$2; shift ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) tasks+=("$1") ;;
  esac
  shift
done
# site successes: 10 9 8 7 7 6 4 3 2 0 (56 / 100)
[ ${#tasks[@]} -gt 0 ] || tasks=(click_bell place_a2b_right move_can_pot adjust_bottle place_empty_cup
                                 place_object_stand place_fan open_laptop handover_block lift_pot)

[ -f "$GOOGLE_APPLICATION_CREDENTIALS" ] || { echo "no service account key at $GOOGLE_APPLICATION_CREDENTIALS" >&2; exit 1; }
if [ -n "$variant" ]; then
  [ "$start" -ge 24 ] || [ $((start + episodes)) -le 10 ] \
    || { echo "run evaluation episodes (0-9) or development episodes (24-49); 10-23 overlap the demonstrations" >&2; exit 2; }
  out=$root/outputs/$([ "$start" -ge 10 ] && echo robodawn_dev || echo robodawn)
  runner="$root/scripts/run_robodawn_relay.py --variant $variant${checker:+ --checker $checker}${maxtok:+ --max-tokens $maxtok}"
  # the runner names the result directory (settings that differ from the defaults are part of the name)
  run=$("$py" $runner --task "${tasks[0]}" --start-episode "$start" --episodes 1 --tier "$tier" --dry-run \
        | "$py" -c 'import json,sys,pathlib; print(pathlib.Path(json.load(sys.stdin)["output"]).parents[1].name)') \
    || { echo "the relay runner refused these settings (see above)" >&2; exit 2; }
  [ "$checker" != clef ] || [ -s /content/.openrouter_key ] || { echo "no OpenRouter key at /content/.openrouter_key (subset.sh setup)" >&2; exit 1; }
else
  [ "$start" -eq 0 ] || { echo "the baseline runner covers episodes 0-9 only; use --variant baseline for others" >&2; exit 2; }
  [ -z "$maxtok" ] || { echo "--max-tokens applies to relay variants only (the baseline keeps 8000)" >&2; exit 2; }
  run=gemini_flash_$tier; out=$root/outputs/robodawn; runner="$root/scripts/run_robodawn_baseline.py"
fi
if [ -z "$variant" ] && [ $force -eq 0 ]; then
  for smoke in adjust_bottle place_empty_cup; do
    check="$root/outputs/robodawn/gemini_flash_$tier/$smoke/shard_0/condition_check.json"
    "$py" -c "import json,sys; sys.exit(bool(json.load(open('$check'))['errors']))" 2>/dev/null \
      || { echo "smoke condition check missing or failed: $check (run the smoke episode or pass --force)" >&2; exit 1; }
  done
fi

logs=$out/subset_$([ -n "$variant" ] && echo "$run" || echo "$tier")${label:+_$label}
mkdir -p "$logs"
status="$logs/status.tsv"
export root py tier episodes status logs run out runner

# A shard whose process dies is restarted while each attempt finishes at least one more episode:
# finished episodes are skipped, so a restart only continues. On a 24 GB L4 with three processes,
# each process's GPU memory grows with every episode and the third episode's reset failed with
# "cannot create buffer" (2026-10-08); a fresh process starts from about 4.6 GB again. An attempt
# that finishes nothing new (a condition error, a missing key) is not repeated.
finished() {
  # the runner's own output directory for this shard (its --dry-run prints the configuration)
  local dir
  dir=$("$py" $runner --task "$1" --start-episode "$2" --episodes 1 --tier "$tier" --dry-run 2>/dev/null \
        | "$py" -c 'import json,sys; print(json.load(sys.stdin)["output"])' 2>/dev/null) || { echo 0; return; }
  "$py" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["episodes"]))' "$dir/results.json" 2>/dev/null || echo 0
}
run_shard() {
  task=$1; start=$2; count=$3; name="$task episodes $start-$((start + count - 1))"
  printf '%s\t%s\tstart\n' "$(date '+%F %T')" "$name" >> "$status"
  while true; do
    before=$(finished "$task" "$start")
    "$py" $runner --task "$task" --start-episode "$start" --episodes "$count" \
      --tier "$tier" >> "$logs/${task}__$start.log" 2>&1
    code=$?
    after=$(finished "$task" "$start")
    [ $code -eq 0 ] || [ "$after" -le "$before" ] || [ "$after" -ge "$count" ] && break
    printf '%s\t%s\trestart (exit %s, %s/%s episodes)\n' "$(date '+%F %T')" "$name" "$code" "$after" "$count" >> "$status"
  done
  printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$([ $code -eq 0 ] && echo done || echo "failed $code")" >> "$status"
}
export -f finished run_shard

# "task start count" per shard of this part, the most site turns first (longest-first keeps the last shard short);
# development episodes have no site turns, so each counts as its task's mean
shards=$("$py" - "$root/robodawn_site/reproduction_manifest.json" "$episodes" "$shard" "$part" "$start" "${tasks[@]}" <<'PY'
import json, sys
manifest, episodes, size, first, tasks = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[5]), sys.argv[6:]
k, n = map(int, sys.argv[4].split("/"))
if not 1 <= k <= n:
    sys.exit(f"bad --part {k}/{n}")
turns = {t["task"]: [e["site_turns"] for e in t["episodes"]] for t in json.load(open(manifest))["tasks"]}
missing = [t for t in tasks if t not in turns]
if missing:
    sys.exit(f"unknown tasks: {missing}")
turns = {t: v if first == 0 else [sum(v) / len(v)] * episodes for t, v in turns.items()}
load = lambda t: sum(turns[t][:episodes])
parts = [[] for _ in range(n)]
smoke = [t for t in ("adjust_bottle", "place_empty_cup") if t in tasks and first == 0]   # their episode 0 ran on part 1
parts[0] += smoke
for t in sorted((t for t in tasks if t not in smoke), key=load, reverse=True):
    min(parts, key=lambda p: sum(map(load, p))).append(t)
jobs = [(sum(turns[t][s:min(s + size, episodes)]), t, first + s, min(size, episodes - s))
        for t in parts[k - 1] for s in range(0, episodes, size)]
for _, t, s, c in sorted(jobs, key=lambda j: -j[0]):
    print(t, s, c)
PY
) || exit 1
[ -n "$shards" ] || { echo "part $part has no tasks" >&2; exit 1; }
mine=($(printf '%s\n' "$shards" | cut -d' ' -f1 | sort -u))

echo "$(date '+%F %T') subset start: run=$run tier=$tier jobs=$jobs episodes=$start+$episodes shard=$shard part=$part tasks=${mine[*]}"
printf '%s\n' "$shards" | xargs -P "$jobs" -L1 bash -c 'run_shard "$@"' _
echo "$(date '+%F %T') subset end"
if [ -n "$variant" ]; then
  "$py" "$root/scripts/compare_relay.py" "${mine[@]}" --tier "$tier" $([ "$start" -ge 10 ] && echo --dev) --runs "$root/outputs" \
    | tee "$logs/compare.txt"
else
  "$py" "$root/scripts/compare_robodawn.py" --tier "$tier" "${mine[@]}" | tee "$logs/compare.txt"
fi
echo SUBSET_DONE
