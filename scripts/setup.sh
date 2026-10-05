#!/usr/bin/env bash
# One-time environment setup: clone LIBERO at a pinned commit, write its
# project-local config, and install everything into .venv with uv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBERO_REPO="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_DIR="$ROOT/third_party/LIBERO"

if [ ! -d "$LIBERO_DIR/.git" ]; then
    mkdir -p "$ROOT/third_party"
    git clone "$LIBERO_REPO" "$LIBERO_DIR"
fi
git -C "$LIBERO_DIR" checkout --quiet "$LIBERO_COMMIT"

# LIBERO prompts on stdin when this file is missing, so write it up front.
BENCH="$LIBERO_DIR/libero/libero"
mkdir -p "$ROOT/.libero" "$ROOT/third_party/datasets"
cat > "$ROOT/.libero/config.yaml" <<EOF
benchmark_root: $BENCH
bddl_files: $BENCH/bddl_files
init_states: $BENCH/init_files
assets: $BENCH/assets
datasets: $ROOT/third_party/datasets
EOF

cd "$ROOT"
uv sync
echo "Done. Verify with: uv run python scripts/smoke_test.py"
