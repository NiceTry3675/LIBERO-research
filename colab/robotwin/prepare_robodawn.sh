#!/usr/bin/env bash
# Fetch the audited RoboDawn harness and demo bank into an existing project.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
repo="$root/third_party/robodawn"
commit=9247f366cd31f278e10f2fbe5fe8469b5f1b5b94
if [ ! -d "$repo/.git" ]; then
  mkdir -p "$root/third_party"
  git clone --filter=blob:none --sparse https://github.com/Hugo-AGI/RoboDawn.git "$repo"
  git -C "$repo" fetch origin "$commit" --depth 1
  git -C "$repo" checkout --detach "$commit"
fi
if [ "$(git -C "$repo" rev-parse HEAD)" != "$commit" ]; then
  echo "RoboDawn checkout differs from $commit" >&2
  exit 1
fi
git -C "$repo" sparse-checkout set --no-cone '/harness/' '/demos/robotwin2/' '/README.md' '/.gitmodules' '/LICENSE'
py=${ROBODAWN_PYTHON:-/content/mamba/envs/rt/bin/python}
"$py" -m uv pip install --python "$py" 'pillow>=10' 'PyYAML>=6' numpy==1.26.4
"$py" "$root/scripts/audit_robodawn.py"
echo ROBODAWN_READY
