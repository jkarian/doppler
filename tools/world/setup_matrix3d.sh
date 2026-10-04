#!/usr/bin/env bash
# Matrix-3D (Skywork, MIT) in its own Python 3.10 venv inside WSL (it pins old packages that need 3.10).
#   wsl -d Ubuntu-24.04 -- bash /mnt/d/Projects/doppler-canyon/tools/world/setup_matrix3d.sh
# Builds nvdiffrast, simple-knn, two Gaussian rasterizers, pytorch3d and flash-attn from source (slow, ~1 h).
# Models are fetched separately (some are gated: Skywork/Matrix-3D, black-forest-labs/FLUX.1-Fill-dev).
set -e
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=6
export HF_HOME=/mnt/c/AI_Models/huggingface
cd /opt/world
/opt/world/venv/bin/pip install -q uv
[ -d venv-matrix ] || /opt/world/venv/bin/uv venv -q --python 3.10 venv-matrix
. venv-matrix/bin/activate
/opt/world/venv/bin/uv pip install -q pip "setuptools<70" wheel ninja packaging
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
cd Matrix-3D
# The upstream script expects a conda env; run it with this venv's pip, answering yes to the uninstall,
# and without build isolation so the CUDA extensions see torch.
sed -e 's/^pip uninstall basicsr/pip uninstall -y basicsr/' \
    -e 's/^pip install \.$/pip install --no-build-isolation ./' \
    -e 's#^pip install git+https://github.com/rmurai0610#pip install --no-build-isolation git+https://github.com/rmurai0610#' \
    -e 's#^pip install submodules/odgs#pip install --no-build-isolation submodules/odgs#' \
    -e 's#^pip install "git+https://github.com/facebookresearch/pytorch3d#pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d#' \
    install.sh > /tmp/install_matrix3d.sh
bash -e /tmp/install_matrix3d.sh
python -c "import torch, diffsynth, nvdiffrast.torch, simple_knn; print('matrix-3d env ok', torch.__version__, torch.cuda.is_available())"
