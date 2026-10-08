#!/usr/bin/env bash
# From this machine: drive the RoboDawn baseline reproduction on a Colab VM
# that launch.sh --textures has installed.
#
#   bash colab/robotwin/subset.sh setup  [--key PATH]  upload the bundle and the Vertex key, prepare RoboDawn
#   bash colab/robotwin/subset.sh smoke  [--tier T]    episode 0 of adjust_bottle and place_empty_cup
#   bash colab/robotwin/subset.sh run    [run_subset.sh options]   the ten-task subset, in the background
#   bash colab/robotwin/subset.sh status               logs, progress and the comparison so far
#   bash colab/robotwin/subset.sh fetch  [--full]      results to outputs/robodawn_colab/<session> here
#                                                    (--full: with images and videos), then compare all sessions
#   bash colab/robotwin/subset.sh stop                 delete the key on the VM, then stop the VM
#
# --session NAME (default robotwin) goes before the command. Two L4 VMs:
#   subset.sh setup / smoke / fetch                      on robotwin (smoke runs only here)
#   subset.sh --session robotwin2 setup                  (after launch.sh --textures --session robotwin2)
#   subset.sh run --part 1/2;  subset.sh --session robotwin2 run --part 2/2
# A VM without its own smoke results starts only if the smoke checks of another
# session, fetched here, passed; run then passes --force to run_subset.sh. The key is the
# Vertex AI service account key ($GOOGLE_APPLICATION_CREDENTIALS by default). It is
# uploaded as a file, never placed in executed code, which the Colab CLI records
# in its session history.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
colab=$(command -v colab || echo "$HOME/.local/bin/colab")
session=robotwin
if [ "${1:-}" = "--session" ]; then session=$2; shift 2; fi
cmd=${1:-}; shift || true
PY=/content/mamba/envs/rt/bin/python
VM=/content/branchlab

# A `colab` call occasionally hangs or loses its connection (2026-10-08), so each call is bounded
# (perl's alarm: macOS has no `timeout`) and retried. Everything sent with vm must be safe to run twice.
limit() { perl -e 'alarm shift; exec @ARGV' "$@"; }
retry() { for try in 1 2 3; do "$@" && return 0; echo "failed (try $try/3): $*" >&2; sleep 15; done; return 1; }
# vm [--timeout SECONDS]: run Python from stdin on the VM and print its output.
vm() {
  local t=120 code out
  if [ "${1:-}" = "--timeout" ]; then t=$2; fi
  code=$(cat)
  for try in 1 2 3; do
    if out=$(printf '%s\n' "$code" | limit $((t + 60)) "$colab" exec -s "$session" --timeout "$t" 2>&1); then
      printf '%s\n' "$out"; return 0
    fi
    echo "colab exec failed (try $try/3): $(printf '%s\n' "$out" | tail -1)" >&2; sleep 15
  done
  return 1
}

# Start a shell command on the VM in the background, logging to $2. A marker named after this call
# makes a retried exec start it only once.
background() {
  local command=$1 log=$2 marker="/content/.started-$(date +%s)-$$-$RANDOM"
  printf 'import os, subprocess\nif not os.path.exists("%s"):\n    open("%s", "w").close()\n    subprocess.Popen(%s, shell=True, executable="/bin/bash", start_new_session=True)\n' \
    "$marker" "$marker" "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "{ $command; } > $log 2>&1")" | vm > /dev/null
  echo "started on '$session'; log: $log"
}

# Detached work leaves the kernel idle, and Colab reclaims an idle VM in about 15 minutes.
case $cmd in setup|smoke|run|status|fetch) bash "$here/keepalive.sh" start "$session" ;; esac

case $cmd in
  setup)
    key=${GOOGLE_APPLICATION_CREDENTIALS:-}
    if [ "${1:-}" = "--key" ]; then key=$2; fi
    [ -f "$key" ] || { echo "pass --key PATH or set GOOGLE_APPLICATION_CREDENTIALS" >&2; exit 2; }
    echo 'print("BOOTSTRAP_DONE" in open("/content/bootstrap.log").read())' | vm | grep -q True \
      || { echo "bootstrap is not done on '$session' (launch.sh --textures first)" >&2; exit 1; }
    python3 "$root/scripts/package_robodawn.py"
    retry limit 300 "$colab" upload -s "$session" "$root/outputs/robodawn_reproduction.zip" /content/robodawn_reproduction.zip
    retry limit 300 "$colab" upload -s "$session" "$key" /content/vertex_key.json
    vm > /dev/null <<'EOF'
import hashlib, json, os, zipfile
from pathlib import Path
os.chmod("/content/vertex_key.json", 0o600)
with zipfile.ZipFile("/content/robodawn_reproduction.zip") as archive:
    assert all(n.startswith("branchlab/") and ".." not in n for n in archive.namelist())
    archive.extractall("/content")
root = Path("/content/branchlab")
for name, digest in json.loads((root / "bundle_checksums.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
EOF
    background "cd $VM && bash colab/robotwin/prepare_robodawn.sh" /content/prepare.log
    echo "wait for ROBODAWN_READY in /content/prepare.log (subset.sh status)"
    ;;
  smoke)
    tier=flex
    if [ "${1:-}" = "--tier" ]; then tier=$2; fi
    run="GOOGLE_APPLICATION_CREDENTIALS=/content/vertex_key.json $PY scripts/run_robodawn_baseline.py --episodes 1 --tier $tier"
    background "cd $VM && $run --task adjust_bottle; $run --task place_empty_cup; echo SMOKE_DONE" /content/smoke_$tier.log
    ;;
  run)
    tier=flex; args=("$@")
    for ((i = 0; i < ${#args[@]}; i++)); do [ "${args[$i]}" = "--tier" ] && tier=${args[$((i + 1))]}; done
    own=$(vm <<PY
import json
from pathlib import Path
checks = [Path("$VM/outputs/robodawn/gemini_flash_$tier") / t / "shard_0/condition_check.json"
          for t in ("adjust_bottle", "place_empty_cup")]
print("OWN_SMOKE_OK" if all(c.exists() and not json.loads(c.read_text())["errors"] for c in checks) else "NO_OWN_SMOKE")
PY
)
    extra=""
    if [[ $own != *OWN_SMOKE_OK* ]]; then
      other=$(python3 - "$root/outputs/robodawn_colab" "$tier" "$session" <<'PY'
import json, sys
from pathlib import Path
root, tier, session = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for s in sorted(p.name for p in root.iterdir()) if root.is_dir() else []:
    checks = [root / s / f"robodawn/gemini_flash_{tier}/{t}/shard_0/condition_check.json"
              for t in ("adjust_bottle", "place_empty_cup")]
    if s != session and all(c.exists() and not json.loads(c.read_text())["errors"] for c in checks):
        print(s)
        break
PY
)
      [ -n "$other" ] || { echo "no passed $tier smoke checks on '$session' or fetched from another session" >&2; exit 1; }
      echo "smoke checks passed on '$other' (fetched); starting '$session' with --force"
      extra="--force"
    fi
    background "cd $VM && bash colab/robotwin/run_subset.sh $* --label $session $extra" /content/subset.log
    ;;
  status)
    vm --timeout 120 <<EOF
import glob, json, subprocess
from pathlib import Path
for log in ["/content/prepare.log", *sorted(glob.glob("/content/smoke_*.log")), "/content/subset.log"]:
    p = Path(log)
    if p.exists():
        lines = p.read_text(errors="replace").splitlines()
        print(f"== {log} (last 5 of {len(lines)} lines)")
        print("\n".join(lines[-5:]))
for check in sorted(glob.glob("$VM/outputs/robodawn/gemini_flash_*/*/shard_*/condition_check.json")):
    report = json.load(open(check))
    print("check", *check.split("/")[-4:-1], "ok" if not report["errors"] else report["errors"])
for status in sorted(glob.glob("$VM/outputs/robodawn/subset_*/status.tsv")):
    print("==", status); print(open(status).read().rstrip())
for tier in sorted(p.name.removeprefix("gemini_flash_") for p in Path("$VM/outputs/robodawn").glob("gemini_flash_*")):
    print(f"== comparison ({tier})")
    print(subprocess.run(["$PY", "$VM/scripts/compare_robodawn.py", "--tier", tier],
                         capture_output=True, text=True, cwd="$VM").stdout)
EOF
    ;;
  fetch)
    exclude="--exclude=*.png --exclude=*.mp4 --exclude=*.jpg"
    if [ "${1:-}" = "--full" ]; then exclude=""; fi
    vm --timeout 1800 <<EOF > /dev/null
import subprocess
subprocess.run("tar czf /content/robodawn_results.tgz $exclude -C $VM/outputs robodawn", shell=True, check=True)
EOF
    dest="$root/outputs/robodawn_colab/$session"
    mkdir -p "$dest"
    tmp=$(mktemp -d)
    retry limit 3600 "$colab" download -s "$session" /content/robodawn_results.tgz "$tmp/robodawn_results.tgz"
    tar xzf "$tmp/robodawn_results.tgz" -C "$dest"
    rm -rf "$tmp"
    echo "fetched to ${dest#"$root"/}/robodawn"
    runs=("$root"/outputs/robodawn_colab/*/robodawn)
    for tier in $(for r in "${runs[@]}"; do ls -d "$r"/gemini_flash_* 2>/dev/null; done | sed 's#.*/gemini_flash_##' | sort -u); do
      echo "== $tier, sessions: $(ls "$root/outputs/robodawn_colab" | tr '\n' ' ')"
      python3 "$root/scripts/compare_robodawn.py" --tier "$tier" --runs "${runs[@]}"
    done
    ;;
  stop)
    bash "$here/keepalive.sh" stop "$session"
    echo 'import os; [os.remove(p) for p in ["/content/vertex_key.json"] if os.path.exists(p)]; print("key removed")' | vm \
      || echo "could not reach '$session' to delete the key" >&2
    retry limit 300 "$colab" stop -s "$session"
    ;;
  *)
    sed -n '2,16p' "$0"; exit 2 ;;
esac
