"""Cross-Attention Fusion module.

Implements the fusion of a physiological signal (query) with spatial ViT tokens (keys/values).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Fusion module that attends to spatial tokens using the rPPG signal as a query.

    Architecture:
    - Input Q: (B, 1, D) - The extracted rPPG face signal.
    - Input K, V: (B, N, D) - Spatial tokens from the ViT backbone.
    - Output: (B, 1, D) - Attended spatial representation.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # standard MultiheadAttention
        # batch_first=True means inputs are (B, L, D)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # LayerNorm and FeedForward for stability (Transformer-style)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln_out = nn.LayerNorm(embed_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Forward pass of the fusion module.

        Args:
            q: The query tensor (B, 1, D).
            kv: The key/value tensor (B, N, D).

        Returns:
            Attended tensor (B, 1, D).
        """
        # Normalize inputs
        q = self.ln_q(q)
        kv = self.ln_kv(kv)

        # Cross-attention
        # q = query, kv = key, kv = value
        attn_out, _ = self.mha(q, kv, kv)

        # Residual connection + FFN
        out = q + attn_out
        out = out + self.ffn(self.ln_out(out))

        return out
