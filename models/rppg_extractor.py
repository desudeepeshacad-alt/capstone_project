"""
POS (Plane-Orthogonal-to-Skin) rPPG extractor, torch implementation.

Wang et al., "Algorithmic Principles of Remote PPG", IEEE Trans. Biomedical
Engineering, 2017.

Pipeline on a face-ROI clip (B, T, 3, H, W), ImageNet-normalized:
  1. Per-frame mean RGB over the face crop          -> (B, T, 3)
  2. Undo ImageNet normalization (affine, applied to the means only)
  3. Windowed POS (1.6 s windows, 50% overlap, overlap-add)
  4. FFT bandpass 0.7-4 Hz over the clip
Returns a (B, T) pulse signal. No scipy and no CPU round trips.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# POS projection matrix (rows orthogonal to the (1,1,1) intensity axis).
_POS_P = torch.tensor([[0.0, 1.0, -1.0],
                       [-2.0, 1.0, 1.0]], dtype=torch.float32)


def pos_windowed(rgb: torch.Tensor, window: int, eps: float = 1e-8) -> torch.Tensor:
    """Windowed POS.

    Args:
        rgb: (B, T, 3) per-frame mean RGB in raw (un-normalized) scale.
        window: POS window length in frames.

    Returns:
        (B, T) pulse signal (not band-limited).
    """
    if rgb.dim() != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected (B, T, 3), got {tuple(rgb.shape)}")
    B, T, _ = rgb.shape
    if T < window:
        raise ValueError(f"clip has {T} frames but POS window needs {window}")

    stride = max(window // 2, 1)
    # Pad the end by repeating the last frame so windows tile the clip exactly.
    n_extra = (stride - (T - window) % stride) % stride
    if n_extra > 0:
        rgb = torch.cat([rgb, rgb[:, -1:, :].expand(B, n_extra, 3)], dim=1)
    Tp = rgb.shape[1]

    P = _POS_P.to(device=rgb.device, dtype=rgb.dtype)
    out = rgb.new_zeros(B, Tp)

    for s in range(0, Tp - window + 1, stride):
        c = rgb[:, s:s + window, :]                                 # (B, L, 3)
        # Temporal normalization within the window.
        cn = c / (c.mean(dim=1, keepdim=True) + eps) - 1.0
        S = torch.matmul(cn, P.t())                                 # (B, L, 2)
        s1, s2 = S[..., 0], S[..., 1]
        # Adaptive combination that cancels specular/intensity variation.
        alpha = s1.std(dim=1, keepdim=True) / (s2.std(dim=1, keepdim=True) + eps)
        h = s1 + alpha * s2
        h = h - h.mean(dim=1, keepdim=True)
        out[:, s:s + window] += h

    return out[:, :T]


def fft_bandpass(x: torch.Tensor, low: float, high: float, fps: float) -> torch.Tensor:
    """Ideal FFT bandpass over the last dimension. x: (B, T) -> (B, T)."""
    T = x.shape[-1]
    X = torch.fft.rfft(x.float(), dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps, device=x.device)
    mask = ((freqs >= low) & (freqs <= high)).float()
    return torch.fft.irfft(X * mask, n=T, dim=-1)


class POSExtractor(nn.Module):
    """POS rPPG extractor as a torch module (no learned parameters).

    Input:  (B, T, 3, H, W) ImageNet-normalized face clip.
    Output: (B, T) pulse signal.

    Gradients are not propagated into the input; the pulse is a fixed feature.
    """

    def __init__(
        self,
        fps: float = 30.0,
        bandpass_low: float = 0.7,
        bandpass_high: float = 4.0,
        window_seconds: float = 1.6,
    ):
        super().__init__()
        self.fps = float(fps)
        self.bandpass_low = float(bandpass_low)
        self.bandpass_high = float(bandpass_high)
        self.window = max(2, int(round(window_seconds * self.fps)))
        # Non-persistent buffers: not added to state_dict, so checkpoints are unaffected.
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3), persistent=False
        )

    @torch.no_grad()
    def forward(self, face_clip: torch.Tensor) -> torch.Tensor:
        if face_clip.dim() != 5 or face_clip.shape[2] != 3:
            raise ValueError(f"expected (B, T, 3, H, W), got {tuple(face_clip.shape)}")
        # Mean over H, W first (cheap), then undo normalization. The mapping is
        # affine, so this equals un-normalizing every pixel and then averaging.
        rgb_n = face_clip.mean(dim=(-1, -2)).float()     # (B, T, 3)
        rgb = rgb_n * self.std + self.mean               # raw-ish RGB in [0, 1]
        pulse = pos_windowed(rgb, self.window)
        return fft_bandpass(pulse, self.bandpass_low, self.bandpass_high, self.fps)

    @torch.no_grad()
    def from_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """(B, T, 3) per-frame mean RGB in [0, 1] -> (B, T) pulse signal."""
        pulse = pos_windowed(rgb.float(), self.window)
        return fft_bandpass(pulse, self.bandpass_low, self.bandpass_high, self.fps)

    def extra_repr(self) -> str:
        return (f"fps={self.fps}, bandpass=[{self.bandpass_low}, {self.bandpass_high}] Hz, "
                f"window={self.window} frames")


# ---------------------------------------------------------------------------
# Self-tests: python rppg_extractor.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    fps, T = 30.0, 300

    # 1. Brightness invariance: POS normalizes each window, so scaling is a no-op.
    rgb = torch.rand(2, T, 3) + 0.5
    a = pos_windowed(rgb, window=48)
    b = pos_windowed(2.0 * rgb, window=48)
    assert torch.allclose(a, b, atol=1e-4), "POS is not brightness invariant"

    # 2. Frequency recovery: a 1.2 Hz chrominance tone should peak near 1.2 Hz.
    t = torch.arange(T, dtype=torch.float32) / fps
    v = torch.tensor([0.2, 0.5, 0.1])
    rgb_tone = (100.0 + 2.0 * torch.sin(2 * math.pi * 1.2 * t)[:, None] * v[None, :])
    sig = fft_bandpass(pos_windowed(rgb_tone.unsqueeze(0), window=48), 0.7, 4.0, fps)[0]
    spec = torch.fft.rfft(sig).abs()
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps)
    peak = float(freqs[spec.argmax()])
    assert abs(peak - 1.2) < 0.15, f"expected peak near 1.2 Hz, got {peak:.2f} Hz"

    # 3. Shape check through the module.
    ext = POSExtractor(fps=fps)
    clip = torch.randn(2, 64, 3, 32, 32)
    assert ext(clip).shape == (2, 64)