"""Neck-region pulse correlation head.

Compares the face and neck rPPG pulses with explicit, scale-invariant
features (correlation statistics and amplitude ratio), then maps them to a
logit with a small MLP. The features do not depend on clip length, so the
head works for any T.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-clip standardization along the last dimension (population std)."""
    x = x - x.mean(dim=-1, keepdim=True)
    return x / (x.std(dim=-1, keepdim=True, unbiased=False) + eps)


class NeckCorrelationHead(nn.Module):
    """Explicit correlation features -> small MLP -> logit.

    Inputs:  face_pulse (B, T), neck_pulse (B, T)
    Output:  logit (B, 1)
    """

    NUM_FEATURES = 5

    def __init__(
        self,
        clip_length: int = 96,   # kept for API compatibility; not used in the shape
        hidden_dim: int = 32,
        max_lag: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_lag = max_lag
        self.mlp = nn.Sequential(
            nn.Linear(self.NUM_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _max_lag_xcorr(self, f: torch.Tensor, n: torch.Tensor):
        """Max and min normalized cross-correlation over lags in [-max_lag, max_lag]."""
        T = f.shape[-1]
        vals = []
        for k in range(-self.max_lag, self.max_lag + 1):
            if k >= 0:
                a, b = f[:, k:], n[:, :T - k]
            else:
                a, b = f[:, :T + k], n[:, -k:]
            vals.append((a * b).mean(dim=-1))
        xc = torch.stack(vals, dim=-1)                      # (B, 2L+1)
        return xc.max(dim=-1, keepdim=True).values, xc.min(dim=-1, keepdim=True).values

    def forward(self, face_pulse: torch.Tensor, neck_pulse: torch.Tensor) -> torch.Tensor:
        face_pulse = face_pulse.float()
        neck_pulse = neck_pulse.float()

        f = _zscore(face_pulse)
        n = _zscore(neck_pulse)

        corr0 = (f * n).mean(dim=-1, keepdim=True)          # lag-0 Pearson r
        xc_max, xc_min = self._max_lag_xcorr(f, n)
        mse = ((f - n) ** 2).mean(dim=-1, keepdim=True)     # shape disagreement
        log_amp = (torch.log(face_pulse.std(dim=-1, keepdim=True) + 1e-6)
                   - torch.log(neck_pulse.std(dim=-1, keepdim=True) + 1e-6))

        feats = torch.cat([corr0, xc_max, xc_min, mse, log_amp], dim=-1)  # (B, 5)
        return self.mlp(feats)


if __name__ == "__main__":
    torch.manual_seed(0)
    head = NeckCorrelationHead(clip_length=96)
    face = torch.randn(4, 96)
    # Identical pulses should give correlation ~1 and mse ~0.
    f, n = _zscore(face), _zscore(face)
    assert torch.allclose((f * n).mean(-1), torch.ones(4), atol=1e-4)
    # Scale invariance of the z-scored features: scaling the neck should not change corr0.
    c1 = (_zscore(face) * _zscore(face * 3.0)).mean(-1)
    assert torch.allclose(c1, torch.ones(4), atol=1e-4)
    assert head(face, face * 3.0).shape == (4, 1)
    assert head(torch.randn(2, 40), torch.randn(2, 40)).shape == (2, 1)  # any T works