#!/usr/bin/env bash
# Sharper Matrix-3D splat from an existing run: upscale its panoramic video 2x with SeedVR2 (one clip, so detail stays
# consistent from frame to frame), then rebuild the 3D at that size with longer splat training.
#   wsl -d Ubuntu-24.04 -- bash /mnt/d/Projects/doppler-canyon/tools/world/matrix3d_hires.sh scenes/canyon/world/matrix_vista [iterations]
# Writes <run>_hr/ (generated_3dgs_opt.ply etc.). The original run is left as it is.
set -e
REPO=/mnt/d/Projects/doppler-canyon
SRC=$REPO/$1
DST=${SRC}_hr
ITERS=${2:-15000}
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 HF_HOME=/mnt/c/AI_Models/huggingface
export OPENCV_IO_ENABLE_OPENEXR=1

mkdir -p "$DST/generated"
cp -r "$SRC/condition" "$DST/"
cp "$SRC/prompt.txt" "$DST/"

if [ ! -f "$DST/generated/generated.mp4" ]; then
  echo "== 1/2 SeedVR2 2x: 1440x720 -> 2880x1440"
  . /opt/world/venv/bin/activate
  TMP=$(mktemp -d)
  cd /opt/world/SeedVR2
  python inference_cli.py "$SRC/generated/generated.mp4" --output "$TMP/up.mp4" --output_format mp4 \
    --model_dir /mnt/c/AI_Models/models/SEEDVR2 --dit_model seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors \
    --resolution 1440 --batch_size ${BATCH:-13} --temporal_overlap 4 --blocks_to_swap 32 --dit_offload_device cpu \
    --vae_encode_tiled --vae_decode_tiled
  # Keep the frame rate and count the 3D step expects.
  ffmpeg -y -loglevel error -i "$(ls $TMP/*.mp4 | head -1)" -c:v libx264 -crf 12 -pix_fmt yuv420p "$DST/generated/generated.mp4"
  rm -rf "$TMP"
  deactivate
fi

echo "== 2/2 3D at 2880x1440, $ITERS iterations"
. /opt/world/venv-matrix/bin/activate
cd /opt/world/Matrix-3D
python "$REPO/tools/world/matrix3d_recon.py" --inout_dir "$DST" --width 2880 --height 1440 --iterations "$ITERS"
ls -la "$DST"
