"""Minimal stand-in for torchaudio, which has no build for our torch version (2.14, CUDA 13).

Beat This! only needs `torchaudio.transforms.MelSpectrogram` (its file loading isn't used: we pass it
arrays). This reimplements that one transform with torch.stft and librosa's mel filterbank, which
uses the same Slaney formula and triangle construction as torchaudio's `melscale_fbanks`.
Put `tools/shims` on sys.path before importing beat_this.
"""

from . import transforms  # noqa: F401
