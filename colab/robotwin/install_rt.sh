set -euo pipefail
set -x
export MAMBA_ROOT_PREFIX=/content/mamba
ENV=/content/mamba/envs/rt; PY=$ENV/bin/python
cd /content
/content/bin/micromamba install -y -q -n rt -c conda-forge gcc_linux-64=12 gxx_linux-64=12 && echo STEP_GCC_OK
robotwin_commit=${ROBOTWIN_COMMIT:-96c1feab536306b50c26af200044fcdf126e8904}
git clone -q --no-checkout https://github.com/RoboTwin-Platform/RoboTwin.git
cd /content/RoboTwin
git checkout -q --detach "$robotwin_commit"
git submodule update --init --recursive
echo "STEP_CLONE_OK $(git rev-parse HEAD)"
grep -v -E "^torch==|^torchvision" scripts/requirements.txt > /tmp/req.txt
$PY -m uv pip install -q --python $PY -r /tmp/req.txt toppra && echo STEP_REQ_OK
# Keep XPolicyLab at the benchmark's gitlink; update_xpolicylab.sh follows
# origin/main and would silently change the pinned benchmark dependencies.
$PY -m uv pip install -q --python $PY -e XPolicyLab && echo STEP_XPL_OK
SAPIEN_LOCATION=$($PY -m pip show sapien | grep Location | awk '{print $2}')/sapien
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $SAPIEN_LOCATION/wrapper/urdf_loader.py
MPLIB_LOCATION=$($PY -m pip show mplib | grep Location | awk '{print $2}')/mplib
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $MPLIB_LOCATION/planner.py
echo STEP_PATCH_OK
# cuRobo with the env's CUDA 12.1 and gcc 12
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH TORCH_CUDA_ARCH_LIST="8.9"
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ CUDAHOSTCXX=$ENV/bin/x86_64-conda-linux-gnu-g++
cd envs && git clone -q --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git && cd curobo
$PY -m pip install -q -e . --no-build-isolation && $PY -m pip install -q warp-lang==1.12.0 setuptools==69.5.1 && echo STEP_CUROBO_OK
cd /content/RoboTwin
$PY -c "import curobo, sapien, mplib; from curobo.wrap.reacher.motion_gen import MotionGen; print('imports ok')" && echo STEP_IMPORT_OK
# The assets are downloaded in parallel by download_assets.sh (started by
# bootstrap.sh); wait for them, then move them into place.
STAGE=${STAGE:-/content/assets_stage}
w=$SECONDS
while [ ! -e $STAGE/ASSETS_READY ] && [ ! -e $STAGE/ASSETS_FAILED ]; do sleep 10; done
echo "TIME wait_assets=$((SECONDS - w))s"
[ -e $STAGE/ASSETS_FAILED ] && { echo "assets failed, see download_assets.log"; exit 1; }
for d in background_texture embodiments objects; do [ -d $STAGE/$d ] && mv $STAGE/$d assets/; done
$PY ./scripts/update_embodiment_config_path.py < /dev/null && echo STEP_ASSETS_OK
du -sh /content/RoboTwin/assets; df -h /content | tail -1
echo INSTALL_DONE
