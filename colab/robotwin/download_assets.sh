# Download and unpack the RoboTwin 2.0 assets into a staging directory.
#
# Runs on its own, independent of the RoboTwin environment, so it can start
# at the same time as the installation. It uses a recent huggingface_hub in a
# throwaway uv environment: chunked parallel transfer (hf_xet, or hf_transfer
# for non-Xet files) instead of the single connection of the 0.25.0 RoboTwin
# pins. The three archives are unpacked in parallel.
#
# Environment variables:
#   STAGE          staging directory (default /content/assets_stage)
#   WITH_TEXTURES  1 = also fetch background_texture.zip (11 GB). Only the
#                  domain-randomized scenes use it, i.e. official evaluations
#                  comparable to RoboDawn; development runs skip it.
#   ASSET_FILES    override the archive list, e.g. "embodiments" for a quick test
#   HF_TOKEN       optional; raises request limits, does not change bandwidth.
#                  If unset, read from TOKEN_FILE (default /content/.hf_token).
#
# Writes $STAGE/ASSETS_READY on success and $STAGE/ASSETS_FAILED on failure.

STAGE=${STAGE:-/content/assets_stage}
if [ -n "$ASSET_FILES" ]; then
  FILES=$ASSET_FILES
elif [ "${WITH_TEXTURES:-0}" = "1" ]; then
  FILES="background_texture embodiments objects"
else
  FILES="embodiments objects"
fi
TOKEN_FILE=${TOKEN_FILE:-/content/.hf_token}
if [ -z "$HF_TOKEN" ] && [ -s "$TOKEN_FILE" ]; then
  export HF_TOKEN=$(cat "$TOKEN_FILE")
fi
mkdir -p "$STAGE"
rm -f "$STAGE/ASSETS_READY" "$STAGE/ASSETS_FAILED"
fail() { echo "ASSETS_FAILED: $1"; touch "$STAGE/ASSETS_FAILED"; exit 1; }

echo "files: $FILES | token: $([ -n "$HF_TOKEN" ] && echo yes || echo no)"
start=$SECONDS
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
patterns=$(for f in $FILES; do printf '"%s.zip",' "$f"; done)
for attempt in 1 2; do
  uv run --quiet --no-project --with "huggingface_hub[hf_xet,hf_transfer]>=0.34" python - <<EOF && break
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    repo_type="dataset",
    allow_patterns=[$patterns],
    local_dir="$STAGE",
    max_workers=8,
)
EOF
  [ $attempt = 2 ] && fail "download"
  echo "download failed, retrying"
done
echo "TIME download=$((SECONDS - start))s"

start=$SECONDS
pids=()
for f in $FILES; do
  (cd "$STAGE" && unzip -q -o "$f.zip" && rm -f "$f.zip") &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p || fail "unzip"; done
echo "TIME unzip=$((SECONDS - start))s"

du -sh "$STAGE"
touch "$STAGE/ASSETS_READY"
echo ASSETS_READY
