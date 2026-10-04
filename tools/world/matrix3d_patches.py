"""Small fixes to Matrix-3D's code for a 24 GB card, applied by matrix3d_scene.sh (idempotent).

- Photo -> panorama: FLUX.1-Fill-dev's transformer in bf16 is 24 GB. Whole-model offload overflowed the card into
  shared system memory; sequential offload re-read it from C: every step (it doesn't fit Linux's page cache).
  Store its weights in fp8 (12 GB, computed in bf16) so it sits on the card. (bitsandbytes 8-bit hit a diffusers bug.)
- Panoramic video (5B): load the video model straight onto the card instead of staging everything in system RAM.
"""

from pathlib import Path

ROOT = Path("/opt/world/Matrix-3D/code")

f = ROOT / "pano_init/src/worldgen/pano_gen.py"
src = f.read_text()
# Undo the earlier 8-bit attempt, if present.
bnb = '''    from diffusers import BitsAndBytesConfig, FluxTransformer2DModel
    # 8-bit transformer (~12 GB): in bf16 (24 GB) it spills into shared memory on a 24 GB card and crawls.
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev", subfolder="transformer",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True), torch_dtype=torch.bfloat16)
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", transformer=transformer, torch_dtype=torch.bfloat16)'''
src = src.replace(bnb, '    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, device=device)')
fill = src[src.index("def build_pano_fill_model"):src.index("def gen_pano_image")]
# Weights stored in fp8 (12 GB), computed in bf16: loaded onto the card once and kept there for all steps.
# (Sequential offload re-read all 24 GB from C:\AI_Models every step: Linux can't cache that much.)
cast = ("pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)\n"
        "    pipe.enable_model_cpu_offload(gpu_id=gpu_id)  # fp8 storage: 12 GB, fits")
patched = fill
for old in ("pipe.enable_model_cpu_offload(gpu_id=gpu_id ) # Save VRAM",
            "pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)  # whole-model offload overflows 24 GB"):
    patched = patched.replace(old, cast)
src = src.replace(fill, patched)
f.write_text(src)
print("fp8 storage" if "enable_layerwise_casting" in src else "WARNING: offload line not found", f)

# Panoramic video (5B): every model was staged in system RAM first (T5 10.6 GB + the video model's 20 GB of files +
# VAE), which the out-of-memory killer stopped at 24 GB. The video model fits the card: load it straight there.
f = ROOT / "panoramic_image_to_video.py"
src = f.read_text()
old = 'model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu")'
if old in src:
    f.write_text(src.replace(old, old.replace('offload_device="cpu"', "offload_device=None")))
    print("video model straight to the card", f)
elif 'diffusion_pytorch_model*.safetensors", offload_device=None' in src:
    print("already patched", f)
else:
    raise SystemExit(f"unexpected contents in {f}: update matrix3d_patches.py")

# Splat training: torch.quantile refuses inputs over ~16M values, which a higher-resolution run exceeds. Estimate the
# threshold from a random sample instead (same answer for practical purposes).
f = ROOT / "Pano_GS_Opt/scene/gaussian_model.py"
src = f.read_text()
old = "        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)"
new = ("        flat = grads_abs.reshape(-1)\n"
       "        if flat.numel() > 10_000_000:\n"
       "            flat = flat[torch.randint(0, flat.numel(), (10_000_000,), device=flat.device)]\n"
       "        Q = torch.quantile(flat, 1 - ratio)")
if old in src:
    f.write_text(src.replace(old, new))
    print("quantile on a sample", f)
elif "flat = grads_abs.reshape(-1)" in src:
    print("already patched", f)
else:
    raise SystemExit(f"unexpected contents in {f}: update matrix3d_patches.py")
