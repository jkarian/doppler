#!/usr/bin/env bash
# Run Matrix-3D (Skywork, MIT) on a scene's photo: photo -> 360 panorama -> panoramic video -> 3D Gaussian splat.
#   wsl -d Ubuntu-24.04 -- bash /mnt/d/Projects/doppler-canyon/tools/world/matrix3d_scene.sh scenes/canyon "prompt"
# Writes <scene>/world/matrix/: pano_img.jpg, pano_video.mp4, generated_3dgs_opt.ply (and Matrix-3D's other files).
# Uses the 5B video model (fits 24 GB). Models: C:\AI_Models\matrix3d (fetch_matrix3d.py) and the HF cache there.
set -e
REPO=/mnt/d/Projects/doppler-canyon
SCENE=$REPO/$1
PROMPT=$2
OUT=$SCENE/world/matrix
export PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 HF_HOME=/mnt/c/AI_Models/huggingface
. /opt/world/venv-matrix/bin/activate
cd /opt/world/Matrix-3D

# The panorama step loads a 7B captioning model up front even when a prompt is given: load it only if needed.
sed -i 's/^\(\s*\)self.Lamma_Video = Lamma_Video(self.device)/\1self.Lamma_Video = None/' code/pano_init/i2p_model.py
sed -i 's/prompt = self.Lamma_Video.extract_prompt(/prompt = Lamma_Video(self.device).extract_prompt(/' code/pano_init/i2p_model.py
python "$REPO/tools/world/matrix3d_patches.py"

mkdir -p "$OUT"
IMAGE=$(python -c "import json; print(json.load(open('$SCENE/scene.json'))['image'])")
python -c "from PIL import Image; Image.open('$SCENE/$IMAGE').convert('RGB').resize((1920, 1080), Image.LANCZOS).save('$OUT/input.jpg', quality=95)"

if [ ! -f "$OUT/pano_img.jpg" ]; then
  echo "== 1/3 photo -> panorama"
  python code/panoramic_image_generation.py --mode=i2p --input_image_path="$OUT/input.jpg" --prompt="$PROMPT" --output_path="$OUT"
fi
if [ ! -f "$OUT/pano_video.mp4" ]; then
  echo "== 2/3 panorama -> panoramic video (5B)"
  torchrun --nproc_per_node 1 code/panoramic_image_to_video.py --inout_dir="$OUT" --resolution=720 --use_5b_model ${VIDEO_ARGS:-}
fi
echo "== 3/3 panoramic video -> 3D Gaussian splat"
python code/panoramic_video_to_3DScene.py --inout_dir="$OUT" --resolution=720
ls -la "$OUT"
