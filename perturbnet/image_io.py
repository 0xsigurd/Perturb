from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image


def decode_image_b64(image_b64: str) -> torch.Tensor:
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def encode_image_b64(image_chw: torch.Tensor) -> str:
    clipped = image_chw.detach().cpu().clamp(0.0, 1.0)
    arr = (clipped.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

