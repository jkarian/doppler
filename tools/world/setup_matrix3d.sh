#!/usr/bin/env bash
# Matrix-3D (Skywork, MIT) in its own Python 3.10 venv inside WSL (it pins old packages that need 3.10).
#   wsl -d Ubuntu-24.04 -- bash /mnt/d/Projects/doppler-canyon/tools/world/setup_matrix3d.sh
# Builds nvdiffrast, simple-knn, two Gaussian rasterizers, pytorch3d and flash-attn from source (slow, ~1 h).
# Models are fetched separately (some are gated: Skywork/Matrix-3D, black-forest-labs/FLUX.1-Fill-dev).
set -e
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=6
export HF_HOME=/mnt/c/AI_Models/huggingface
# Several of the 3DGS rasterizers use uint32_t / FLT_MAX without including their headers (fails on GCC 13):
# include them for every CUDA and C++ compile instead of patching each fresh clone.
export NVCC_PREPEND_FLAGS="-include cstdint -include cfloat" CXXFLAGS="-include cstdint -include cfloat"
cd /opt/world
/opt/world/venv/bin/pip install -q uv
[ -d venv-matrix ] || /opt/world/venv/bin/uv venv -q --python 3.10 venv-matrix
. venv-matrix/bin/activate
/opt/world/venv/bin/uv pip install -q pip "setuptools<70" wheel ninja packaging
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
cd Matrix-3D
# simple-knn uses FLT_MAX without including <cfloat> (fails on newer compilers).
grep -q '<cfloat>' submodules/simple-knn/simple_knn.cu || sed -i '1i #include <cfloat>' submodules/simple-knn/simple_knn.cu
# The upstream script expects a conda env; run it with this venv's pip, answering yes to the uninstall,
# and without build isolation so the CUDA extensions see torch.
sed -e 's/^pip uninstall basicsr/pip uninstall -y basicsr/' \
    -e 's/^pip install \.$/pip install --no-build-isolation ./' \
    -e 's#^pip install git+https://github.com/rmurai0610#pip install --no-build-isolation git+https://github.com/rmurai0610#' \
    -e 's#^pip install submodules/odgs#pip install --no-build-isolation submodules/odgs#' \
    -e 's#^pip install "git+https://github.com/facebookresearch/pytorch3d#pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d#' \
    -e 's/^pip install -e \.$/pip install --no-build-isolation -e ./' \
    install.sh > /tmp/install_matrix3d.sh
# DiffSynth's setup.py needs pkg_resources (old setuptools): no build isolation above, our setuptools<70.
# On a rerun, skip the CUDA extensions that already built.
if python -c "import nvdiffrast.torch, simple_knn, diff_gaussian_rasterization, odgs_gaussian_rasterization" 2>/dev/null; then
  sed -n '/^cd code/,$p' /tmp/install_matrix3d.sh > /tmp/install_matrix3d_rest.sh
  mv /tmp/install_matrix3d_rest.sh /tmp/install_matrix3d.sh
fi
bash -e /tmp/install_matrix3d.sh
# The upstream script installs the newest utils3d, but Matrix-3D's bundled MoGe v1 needs the old API (image_uv...):
# use the copy it ships.
pip uninstall -q -y utils3d
pip install -q --no-build-isolation ./code/pano_init/utils3d
python -c "import utils3d.torch as u; assert hasattr(u, 'image_uv'), 'utils3d too new'"
# OpenCV 5's wheels can't write .exr (the panorama depth step does); MoGe v1 was written against 4.10.
pip install -q "opencv-python==4.10.0.84" "opencv-python-headless==4.10.0.84" "numpy<2"
# The reconstruction's super-resolution step imports tensorboard (needs protobuf 6+); the upstream list pins protobuf
# 3.20 for streamlit, which this pipeline doesn't use.
pip install -q protobuf==6.31.1
python -c "import torch, diffsynth, nvdiffrast.torch, simple_knn; print('matrix-3d env ok', torch.__version__, torch.cuda.is_available())"
