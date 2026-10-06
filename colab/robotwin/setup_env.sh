set -x
cd /content
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
export MAMBA_ROOT_PREFIX=/content/mamba
./bin/micromamba create -y -q -n rt -c nvidia/label/cuda-12.1.1 -c conda-forge python=3.10 cuda-nvcc cuda-cudart-dev cuda-libraries-dev ffmpeg
PY=/content/mamba/envs/rt/bin/python
$PY -m pip install -q uv
$PY -m uv pip install -q --python $PY torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
$PY -m uv pip install -q --python $PY sapien==3.0.0b1 numpy==1.26.4 setuptools==69.5.1 imageio
$PY -c "import torch, sapien; print('torch', torch.__version__, torch.cuda.is_available(), 'sapien', sapien.__version__)"
echo SETUP_DONE
