"""ViT-B/16 backbone wrapper via timm."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import timm


class ViTBackbone(nn.Module):
    """ViT-B/16 backbone wrapper with access to spatial and CLS tokens."""

    def __init__(self, pretrained: bool = True, freeze_first_n_blocks: int = 8):
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # remove the classification head
        )
        self.freeze_first_n_blocks(freeze_first_n_blocks)

    def freeze_first_n_blocks(self, n: int) -> None:
        """Freeze patch embedding, position/CLS embeddings, and the first n blocks."""
        if n <= 0:
            return
        self.model.patch_embed.requires_grad_(False)
        self.model.cls_token.requires_grad_(False)
        self.model.pos_embed.requires_grad_(False)
        for i in range(min(n, len(self.model.blocks))):
            self.model.blocks[i].requires_grad_(False)

    def forward_both(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone once and return both outputs.

        Input:  (B, C, H, W)
        Output: spatial tokens (B, N, D) and CLS token (B, D)
        """
        tokens = self.model.forward_features(x)           # (B, N+1, D), computed once
        return tokens[:, 1:, :], tokens[:, 0, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial tokens only, (B, N, D). Prefer forward_both to avoid recomputation."""
        return self.forward_both(x)[0]

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """CLS token only, (B, D). Prefer forward_both to avoid recomputation."""
        return self.forward_both(x)[1]