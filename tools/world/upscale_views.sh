#!/usr/bin/env bash
# Upscale a dreamed camera move with SeedVR2 (ByteDance, Apache licence), as one clip so detail stays consistent
# from frame to frame (a splat trained on independently upscaled frames turns their disagreements into mush).
#   wsl -d Ubuntu-24.04 -- bash /mnt/d/Projects/doppler-canyon/tools/world/upscale_views.sh scenes/canyon spiral [short side px]
# Reads <scene>/world/<move>/samples-rgb/*.png, writes <scene>/world/<move>/upscaled/*.png.
# Weights live in C:\AI_Models\models\SEEDVR2 (the folder ComfyUI's SeedVR2 node uses too).
set -e
REPO=/mnt/d/Projects/doppler-canyon
DIR=$REPO/$1/world/$2
SHORT=${3:-1152}
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 HF_HOME=/mnt/c/AI_Models/huggingface
. /opt/world/venv/bin/activate

TMP=$(mktemp -d)
# A folder of images would be upscaled one by one; a video is treated as one clip. Near-lossless encode.
ffmpeg -loglevel error -framerate 20 -i "$DIR/samples-rgb/%03d.png" -c:v libx264 -crf 4 -pix_fmt yuv444p "$TMP/in.mp4" \
  || ffmpeg -loglevel error -framerate 20 -pattern_type glob -i "$DIR/samples-rgb/*.png" -c:v libx264 -crf 4 -pix_fmt yuv444p "$TMP/in.mp4"

cd /opt/world/SeedVR2
python inference_cli.py "$TMP/in.mp4" --output "$TMP/out" --output_format png \
  --model_dir /mnt/c/AI_Models/models/SEEDVR2 \
  --dit_model ${DIT:-seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors} \
  --resolution $SHORT --batch_size ${BATCH:-21} --temporal_overlap 4 \
  --blocks_to_swap ${SWAP:-16} --dit_offload_device cpu --vae_encode_tiled --vae_decode_tiled

rm -rf "$DIR/upscaled" && mkdir -p "$DIR/upscaled"
find "$TMP/out" -name '*.png' | sort | awk '{printf "cp \"%s\" \"'"$DIR"'/upscaled/%03d.png\"\n", $0, NR-1}' | sh
echo "upscaled $(ls "$DIR/upscaled" | wc -l) frames to $(python -c "from PIL import Image; print(Image.open('$DIR/upscaled/000.png').size)")"
rm -rf "$TMP"
