#!/usr/bin/env bash
# Run a Python tool inside WSL with the world environment (CUDA, the venv, models from C:\AI_Models), from the repo.
#   wsl -d Ubuntu-24.04 -- bash /mnt/d/Projects/doppler-canyon/tools/world/run.sh tools/world/splat_train.py scenes/canyon spiral
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 HF_HOME=/mnt/c/AI_Models/huggingface
cd /mnt/d/Projects/doppler-canyon
exec /opt/world/venv/bin/python "$@"
