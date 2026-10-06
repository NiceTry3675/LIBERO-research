# One entry point for a fresh Colab VM: start the asset download in the
# background, install the environment and RoboTwin meanwhile, and report how
# long each part took. Logs: /content/{download_assets,setup_env,install_rt}.log
#
#   bash /content/bootstrap.sh                   # clean scenes only (development)
#   WITH_TEXTURES=1 bash /content/bootstrap.sh   # + 11 GB of textures for randomized scenes
cd /content
t0=$SECONDS
bash /content/download_assets.sh > /content/download_assets.log 2>&1 &
s=$SECONDS; bash /content/setup_env.sh > /content/setup_env.log 2>&1; echo "TIME setup_env=$((SECONDS - s))s"
grep -q SETUP_DONE /content/setup_env.log || { echo "BOOTSTRAP_FAILED: setup_env"; exit 1; }
s=$SECONDS; bash /content/install_rt.sh > /content/install_rt.log 2>&1; echo "TIME install_rt=$((SECONDS - s))s"
grep -a "^TIME" /content/download_assets.log /content/install_rt.log
grep -q INSTALL_DONE /content/install_rt.log && grep -q STEP_ASSETS_OK /content/install_rt.log \
  || { echo "BOOTSTRAP_FAILED: install_rt"; exit 1; }
echo "TIME total=$((SECONDS - t0))s"
echo BOOTSTRAP_DONE
