"""Small fixes to Matrix-3D's code for a 24 GB card, applied by matrix3d_scene.sh (idempotent).

- Photo -> panorama: FLUX.1-Fill-dev's transformer in bf16 is 24 GB. Whole-model offload overflowed the card into
  shared system memory; sequential offload re-read it from C: every step (it doesn't fit Linux's page cache).
  Store its weights in fp8 (12 GB, computed in bf16) so it sits on the card. (bitsandbytes 8-bit hit a diffusers bug.)
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
