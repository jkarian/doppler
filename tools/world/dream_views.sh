#!/usr/bin/env bash
# Dream camera moves around a scene's photo with Stable Virtual Camera (inside WSL, as the user with the
# Hugging Face login).
#   wsl -d Ubuntu-24.04 -- bash /mnt/d/Projects/doppler-canyon/tools/world/dream_views.sh scenes/canyon spiral [move-left ...]
# Moves: orbit spiral move-left move-right move-up move-down move-forward move-backward zoom-in zoom-out.
# Writes <scene>/world/<move>/ with samples-rgb/*.png, transforms.json (each frame's camera) and a video.
set -e
REPO=/mnt/d/Projects/doppler-canyon
SCENE=$REPO/$1; shift
MOVES=${@:-spiral}
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 HF_HOME=/mnt/c/AI_Models/huggingface
. /opt/world/venv/bin/activate

IMAGE=$(python -c "import json,sys; print(json.load(open('$SCENE/scene.json'))['image'])")
IN=$SCENE/world/input && mkdir -p "$IN"
cp "$SCENE/$IMAGE" "$IN/photo.png"

cd /opt/world/stable-virtual-camera
for MOVE in $MOVES; do
  SCALE=2.0
  case $MOVE in move-*|dolly*) SCALE=${CAMERA_SCALE:-10.0} ;; esac   # the model's advice for panning moves
  echo "== $MOVE (camera_scale $SCALE)"
  python demo.py --data_path "$IN" --task img2trajvid_s-prob --replace_or_include_input True \
    --traj_prior "$MOVE" --cfg 4.0,2.0 --guider 1,2 --num_targets ${NUM_TARGETS:-80} --L_short 576 \
    --use_traj_prior True --chunk_strategy interp --camera_scale $SCALE --video_save_fps 20
  rm -rf "$SCENE/world/$MOVE"
  mkdir -p "$SCENE/world/$MOVE"
  cp -r work_dirs/demo/img2trajvid_s-prob/photo/$MOVE/* "$SCENE/world/$MOVE/" 2>/dev/null || \
    cp -r $(ls -dt work_dirs/demo/img2trajvid_s-prob/*/ | head -1)* "$SCENE/world/$MOVE/"
done
