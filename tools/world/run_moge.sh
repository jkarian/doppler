#!/usr/bin/env bash
# Like run.sh, but with the MoGe venv (numpy 2).
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 HF_HOME=/mnt/c/AI_Models/huggingface
cd /mnt/d/Projects/doppler-canyon
exec /opt/world/venv-moge/bin/python "$@"
