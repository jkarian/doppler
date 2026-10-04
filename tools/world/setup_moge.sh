#!/usr/bin/env bash
# MoGe (Microsoft, MIT) in its own venv inside WSL: it needs numpy 2, Stable Virtual Camera needs numpy 1.
# Linux has nvcc, so flex-gemm (needed by MoGe-3) builds here, unlike on Windows.
#   wsl -d Ubuntu-24.04 -- bash /mnt/d/Projects/doppler-canyon/tools/world/setup_moge.sh
set -e
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9
export HF_HOME=/mnt/c/AI_Models/huggingface
cd /opt/world
[ -d venv-moge ] || python3 -m venv venv-moge
. venv-moge/bin/activate
pip install -q -U pip wheel "setuptools<82"
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q --no-build-isolation "git+https://github.com/microsoft/MoGe.git"
python -c "from moge.model.v2 import MoGeModel; print('moge ok')"
