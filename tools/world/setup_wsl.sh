#!/usr/bin/env bash
# One-time setup of the "dream the views, then splat" tools inside WSL2 Ubuntu 24.04 (run as root).
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/d/Projects/doppler-canyon/tools/world/setup_wsl.sh
# Needs: the CUDA toolkit 12.8 (apt: cuda-toolkit-12-8 from NVIDIA's wsl-ubuntu repo), build-essential, git, ffmpeg.
# Installs into /opt/world: a Python venv with PyTorch (CUDA 12.8), Stable Virtual Camera's code
# (Stability AI, non-commercial licence) and gsplat (Apache).
set -e
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9

mkdir -p /opt/world && cd /opt/world
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -q -U pip wheel "setuptools<82"
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128

[ -d stable-virtual-camera ] || git clone -q --recursive https://github.com/Stability-AI/stable-virtual-camera
cd stable-virtual-camera
# It pins numpy 1.24.4, which has no Python 3.12 build; any numpy 1.x works.
sed -i 's/"numpy==1.24.4"/"numpy<2"/' pyproject.toml
pip install -q -e .
cd ..

pip install -q --no-build-isolation gsplat

python - <<'EOF'
import torch, gsplat, numpy, seva
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("gsplat", gsplat.__version__, "numpy", numpy.__version__)
EOF
