"""Deepfake Shield: ViT spatial stream + POS rPPG stream, fused by cross-attention.

forward takes a precomputed face image and per-frame RGB traces, so the
rPPG branches never touch the image pipeline.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

try:
    # Package use (models/deepfake_shield.py imported as models.deepfake_shield)
    from .vit_backbone import ViTBackbone
    from .rppg_extractor import POSExtractor
    from .cross_attention_fusion import CrossAttentionFusion
    from .neck_anchor import NeckCorrelationHead
except ImportError:
    # Notebook use: these classes are already defined in earlier cells.
    pass


class DeepfakeShield(nn.Module):
    """
    Inputs:
        face_img: (B, 3, H, W)  one face image for the ViT (normalized)
        face_rgb: (B, T, 3)     per-frame mean face RGB in [0, 1]
        neck_rgb: (B, T, 3)     per-frame mean neck RGB in [0, 1]

    Outputs (dict, each (B, 1)):
        logits:         spatial_logits + neck_logits
        spatial_logits: spatial classifier over fused and CLS features
        neck_logits:    neck-correlation head over face and neck pulses
    """

    def __init__(
        self,
        clip_length: int = 96,
        face_size: int = 224,
        fps: float = 30.0,
        vit_freeze_blocks: int = 6,
        embed_dim: int = 768,
        pretrained: bool = True,
    ):
        super().__init__()
        self.clip_length = clip_length

        self.vit = ViTBackbone(pretrained=pretrained,
                               freeze_first_n_blocks=vit_freeze_blocks)
        self.rppg = POSExtractor(fps=fps)
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim)

        self.pulse_proj = nn.Sequential(
            nn.Linear(clip_length, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.spatial_classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        self.neck_head = NeckCorrelationHead(clip_length=clip_length)

    def _project_pulse(self, pulse: torch.Tensor) -> torch.Tensor:
        """(B, T) pulse -> (B, 1, D) query, using a per-clip z-scored pulse."""
        mu = pulse.mean(dim=-1, keepdim=True)
        sd = pulse.std(dim=-1, keepdim=True)
        pulse = (pulse - mu) / (sd + 1e-6)
        return self.pulse_proj(pulse).unsqueeze(1)

    def forward(self, face_img: torch.Tensor, face_rgb: torch.Tensor,
                neck_rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        spatial_tokens, cls_token = self.vit.forward_both(face_img)

        with torch.autocast(device_type=face_img.device.type, enabled=False):
            face_pulse = self.rppg.from_rgb(face_rgb.float())    # (B, T)
            neck_pulse = self.rppg.from_rgb(neck_rgb.float())    # (B, T)

        q = self._project_pulse(face_pulse)                       # (B, 1, D)
        fused = self.fusion(q, spatial_tokens).squeeze(1)         # (B, D)

        spatial_logits = self.spatial_classifier(torch.cat([fused, cls_token], dim=-1))
        neck_logits = self.neck_head(face_pulse, neck_pulse)

        return {
            "logits": spatial_logits + neck_logits,
            "spatial_logits": spatial_logits,
            "neck_logits": neck_logits,
        }