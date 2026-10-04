"""Small fixes to Matrix-3D's code for a 24 GB card, applied by matrix3d_scene.sh (idempotent).

- Photo -> panorama: FLUX.1-Fill-dev's transformer in bf16 is 24 GB; on a 24 GB card under Windows the driver spills
  the overflow into shared system memory and every step takes minutes. Load it with 8-bit weights (~12 GB).
"""

from pathlib import Path

ROOT = Path("/opt/world/Matrix-3D/code")

f = ROOT / "pano_init/src/worldgen/pano_gen.py"
src = f.read_text()
old = '    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, device=device)'
new = '''    from diffusers import BitsAndBytesConfig, FluxTransformer2DModel
    # 8-bit transformer (~12 GB): in bf16 (24 GB) it spills into shared memory on a 24 GB card and crawls.
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev", subfolder="transformer",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True), torch_dtype=torch.bfloat16)
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", transformer=transformer, torch_dtype=torch.bfloat16)'''
if old in src:
    f.write_text(src.replace(old, new))
    print("patched", f)
elif "BitsAndBytesConfig" in src:
    print("already patched", f)
else:
    raise SystemExit(f"unexpected contents in {f}: update matrix3d_patches.py")
