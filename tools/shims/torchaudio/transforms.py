import librosa
import torch


class MelSpectrogram(torch.nn.Module):
    """torchaudio.transforms.MelSpectrogram for the arguments Beat This! uses: Hann window, centred
    frames with reflect padding, optional "frame_length" normalisation, filterbank without area norm."""

    def __init__(self, sample_rate=16000, n_fft=400, win_length=None, hop_length=None, f_min=0.0, f_max=None,
                 n_mels=128, power=2.0, normalized=False, mel_scale="htk", norm=None, center=True, pad_mode="reflect"):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or self.win_length // 2
        self.power = power
        self.frame_norm = normalized is True or normalized == "frame_length"
        self.window_norm = normalized == "window"
        self.center, self.pad_mode = center, pad_mode
        self.register_buffer("window", torch.hann_window(self.win_length))
        fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max or sample_rate / 2,
                                 htk=mel_scale == "htk", norm=norm)
        self.register_buffer("fb", torch.tensor(fb, dtype=torch.float32))  # (n_mels, n_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(x, self.n_fft, self.hop_length, self.win_length, self.window, center=self.center,
                          pad_mode=self.pad_mode, normalized=self.frame_norm, onesided=True, return_complex=True)
        if self.window_norm:
            spec = spec / self.window.pow(2).sum().sqrt()
        spec = spec.abs() if self.power == 1 else spec.abs().pow(self.power)
        return self.fb @ spec  # (..., n_mels, frames), like torchaudio
