"""Encode a recorded clip (captures/<dir>/0000.jpg ...) to captures/<dir>.mp4 and remove the frames.
If the clip was recorded with a track, its audio is muxed in from the same start time.

    python tools/encode_clip.py              # newest clip folder
    python tools/encode_clip.py captures/canyon-clip-123
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

root = Path(__file__).resolve().parent.parent
captures = root / "captures"
if len(sys.argv) > 1:
    folder = Path(sys.argv[1])
else:
    folders = [p for p in captures.iterdir() if p.is_dir() and any(p.glob("*.jpg"))]
    if not folders:
        raise SystemExit("no clip folders in captures/")
    folder = max(folders, key=lambda p: p.stat().st_mtime)

meta_path = folder / "clip.json"
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
fps = meta.get("fps", 30)
frames = len(list(folder.glob("*.jpg")))

cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(folder / "%04d.jpg")]
if "audio" in meta:
    cmd += ["-ss", f"{meta['start']:.3f}", "-i", str(root / meta["audio"]), "-map", "0:v", "-map", "1:a",
            "-c:a", "aac", "-b:a", "192k", "-t", f"{frames / fps:.3f}"]
out = folder.with_suffix(".mp4")
cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
subprocess.run(cmd, check=True)
shutil.rmtree(folder)
print(f"wrote {out}")
