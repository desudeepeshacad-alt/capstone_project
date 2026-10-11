"""Face and neck ROI transforms + JPEG re-encoding helper.

The transforms module provides three things:

1. ``face_crop``: given an image and a face bounding box, crop and resize to
   224x224 with ImageNet normalization.
2. ``neck_crop_from_face_bbox``: compute a neck ROI as a rectangle below the
   chin, anchored to the face bounding box. This is the simplification agreed
   on in the finalised plan — no custom ViT patch embedding, just a fixed
   rectangle relative to the face.
3. ``jpeg_re_encode``: re-encode a PIL image at a configurable JPEG quality
   factor. Used both at training time (to build the heavy-compression view
   for contrastive pairs) and at test time (to build the compression-sweep
   evaluation views).

The augmentation set is intentionally light — random crop, color jitter,
horizontal flip — enough to regularise ViT-B/16 on ~1000 FF++ identities
without destroying the rPPG signal.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

# ImageNet normalization — standard for timm ViT-B/16 pretrained weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FaceBBox:
    """A face bounding box in (x1, y1, x2, y2) pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def get_face_crop_base(
    image: Image.Image,
    bbox: FaceBBox,
) -> Image.Image:
    """Perform the raw crop of the face region.

    Returns the crop as a PIL image without resizing or normalization.
    """
    w, h = image.size
    x1 = max(0, min(bbox.x1, w - 1))
    y1 = max(0, min(bbox.y1, h - 1))
    x2 = max(0, min(bbox.x2, w))
    y2 = max(0, min(bbox.y2, h))
    return image.crop((x1, y1, x2, y2))


def apply_face_transforms(
    cropped_image: Image.Image,
    size: int = 224,
    train: bool = True,
    aug_cfg: dict | None = None,
) -> torch.Tensor:
    """Resize, augment, and normalize a base face crop.
    """
    aug_cfg = aug_cfg or {}
    if train:
        scale_lo = aug_cfg.get("random_crop_scale", [1.0, 1.0])[0]
        scale_hi = aug_cfg.get("random_crop_scale", [1.0, 1.0])[1]
        upscale = 1.0 / max(scale_lo, 1e-3)
        big_size = int(size * upscale) + 1
        cropped = cropped_image.resize((big_size, big_size), Image.BILINEAR)
        cropped = transforms.RandomCrop(size)(cropped)
        if aug_cfg.get("horizontal_flip", False):
            cropped = transforms.RandomHorizontalFlip(p=0.5)(cropped)
        b = aug_cfg.get("color_jitter_brightness", 0.0)
        c = aug_cfg.get("color_jitter_contrast", 0.0)
        s = aug_cfg.get("color_jitter_saturation", 0.0)
        if b or c or s:
            cropped = transforms.ColorJitter(brightness=b, contrast=c, saturation=s)(cropped)
    else:
        cropped = cropped_image.resize((size, size), Image.BILINEAR)

    tensor = TF.to_tensor(cropped)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor


def get_neck_crop_base(
    image: Image.Image,
    bbox: FaceBBox,
) -> Image.Image:
    """Perform the raw crop of the neck region.

    Returns the crop as a PIL image without resizing or normalization.
    """
    w, h = image.size
    face_w = max(bbox.width, 1)
    face_h = max(bbox.height, 1)

    nw = int(face_w * 0.9)
    cx = (bbox.x1 + bbox.x2) // 2
    nx1 = max(0, cx - nw // 2)
    nx2 = min(w, cx + nw // 2)

    ny1 = max(0, bbox.y2 + int(face_h * (1.05 - 1.0)))
    ny2 = min(h, ny1 + face_h)

    if ny2 <= ny1 or nx2 <= nx1:
        ny1 = min(h - 2, bbox.y2)
        ny2 = min(h, ny1 + 2)
        nx1 = max(0, cx - 1)
        nx2 = min(w, cx + 1)

    return image.crop((nx1, ny1, nx2, ny2))


def apply_neck_transforms(
    cropped_image: Image.Image,
    neck_size: Tuple[int, int] = (64, 128),
) -> torch.Tensor:
    """Resize and normalize a base neck crop.
    """
    cropped = cropped_image.resize((neck_size[1], neck_size[0]), Image.BILINEAR)
    tensor = TF.to_tensor(cropped)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor


def jpeg_re_encode(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode a PIL image at the given JPEG quality factor.

    Used to simulate heavy compression artifacts. JPEG QF=20 is the
    project's proxy for FF++ c40 H.264 artifacts.

    Parameters
    ----------
    image : PIL.Image
        Source image (any mode; will be converted to RGB).
    quality : int
        JPEG quality factor, 1-95. 20 = heavy, 40 = light, 100 = lossless.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# Convenience: a numpy -> tensor helper for synthetic unit tests.
# ---------------------------------------------------------------------------
def numpy_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a (H, W, 3) uint8 numpy array to a normalized tensor.

    Used by the unit test that synthesises a face video with a planted sine
    wave at 1.2 Hz to verify the POS extractor.
    """
    
    pil = Image.fromarray(image)
    tensor = TF.to_tensor(pil)  # (3, H, W) in [0, 1]
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor
def face_crop(
    image: Image.Image,
    bbox: FaceBBox,
    size: int = 224,
    train: bool = True,
    aug_cfg: dict | None = None,
) -> torch.Tensor:
    """Wrapper to crop, resize, and normalize the face."""
    base_crop = get_face_crop_base(image, bbox)
    return apply_face_transforms(base_crop, size=size, train=train, aug_cfg=aug_cfg)


def neck_crop_from_face_bbox(
    image: Image.Image,
    bbox: FaceBBox,
    neck_size: Tuple[int, int] = (64, 128),
) -> torch.Tensor:
    """Wrapper to compute the neck ROI, resize, and normalize."""
    base_crop = get_neck_crop_base(image, bbox)
    return apply_neck_transforms(base_crop, neck_size=neck_size)