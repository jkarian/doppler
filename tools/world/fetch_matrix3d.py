"""Fetch the models Matrix-3D's 5B route needs into C:\\AI_Models\\matrix3d\\checkpoints (Matrix-3D's ./checkpoints
links there). Gated ones (Skywork/Matrix-3D LoRAs, FLUX.1-Fill-dev) need access granted on the Hugging Face account
first; they are skipped with a message until then.

    bash tools/world/run.sh tools/world/fetch_matrix3d.py
"""

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

CK = Path("/mnt/c/AI_Models/matrix3d/checkpoints")
CK.mkdir(parents=True, exist_ok=True)
repo_ck = Path("/opt/world/Matrix-3D/checkpoints")
if repo_ck.is_dir() and not repo_ck.is_symlink():
    for f in repo_ck.iterdir():
        f.rename(CK / f.name) if not (CK / f.name).exists() else None
    repo_ck.rmdir()
if not repo_ck.exists():
    repo_ck.symlink_to(CK)


def get(what, fn):
    try:
        fn()
        print(f"ok       {what}")
    except (GatedRepoError, HfHubHTTPError) as e:
        print(f"BLOCKED  {what}: {type(e).__name__} (request access on huggingface.co)")


get("Wan2.2-TI2V-5B (video base)", lambda: snapshot_download(
    "Wan-AI/Wan2.2-TI2V-5B", local_dir=CK / "Wan-AI/Wan2.2-TI2V-5B",
    allow_patterns=["models_t5_umt5-xxl-enc-bf16.pth", "diffusion_pytorch_model*.safetensors*", "Wan2.2_VAE.pth", "config.json", "google/*"]))
get("Wan2.2 text encoder", lambda: hf_hub_download(
    "Wan-AI/Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth", local_dir=CK / "Wan-AI/Wan2.2-TI2V-5B"))
get("MoGe v1", lambda: hf_hub_download("Ruicheng/moge-vitl", "model.pt", local_dir=CK / "moge"))
# The reconstruction step super-resolves its rendered views with StableSR.
for f in ("stablesr_turbo.ckpt", "vqgan_cfw_00011.ckpt"):
    get(f"StableSR {f}", lambda f=f: hf_hub_download("Iceclear/StableSR", f, local_dir=CK / "StableSR"))
get("Matrix-3D 5B pano-video LoRA", lambda: hf_hub_download(
    "Skywork/Matrix-3D", "checkpoints/pano_video_gen_720p_5b.safetensors", local_dir=CK / "_skywork"))
get("Matrix-3D pano-image LoRA", lambda: hf_hub_download(
    "Skywork/Matrix-3D", "checkpoints/text2panoimage_lora.safetensors", local_dir=CK / "_skywork"))
get("FLUX.1-Fill-dev (photo -> panorama)", lambda: snapshot_download(
    "black-forest-labs/FLUX.1-Fill-dev", allow_patterns=["*.json", "*.txt", "transformer/*", "text_encoder/*", "text_encoder_2/*",
                                                          "tokenizer/*", "tokenizer_2/*", "vae/*", "scheduler/*"]))

# Put the LoRAs where Matrix-3D's scripts look for them.
sk = CK / "_skywork" / "checkpoints"
for name, dst in (("pano_video_gen_720p_5b.safetensors", "Wan-AI/wan-lora"), ("text2panoimage_lora.safetensors", "flux_lora")):
    if (sk / name).exists():
        (CK / dst).mkdir(parents=True, exist_ok=True)
        link = CK / dst / (name if "video" in name else "pano_image_lora.safetensors")
        if not link.exists():
            link.symlink_to(sk / name)
