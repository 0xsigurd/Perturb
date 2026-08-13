from __future__ import annotations

import base64
import hashlib
import io

import numpy as np
import requests
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


def image_url_to_b64(image_url: str, *, timeout_seconds: float = 10.0) -> str:
    response = requests.get(image_url, timeout=timeout_seconds)
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(f"image download failed HTTP {response.status_code}: {response.text[:120]}")
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_pixel_hash(image_b64: str) -> str:
    """sha256 over the decoded RGB pixel buffer (shape-prefixed).

    Hashing pixels instead of file bytes keeps the hash stable across the
    lossless PNG re-encodes in the upload/download pipeline, while still
    changing if the image content at a URL is modified after submission.
    """
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(f"{arr.shape[0]}x{arr.shape[1]}".encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def quantize_image_uint8_grid(image_chw: torch.Tensor) -> torch.Tensor:
    return (image_chw.detach().clamp(0.0, 1.0) * 255.0).round().div(255.0)


def changed_pixel_count(x_clean: torch.Tensor, x_adv: torch.Tensor) -> int:
    if x_clean.shape != x_adv.shape or x_clean.ndim != 3:
        return 0
    return int((x_clean != x_adv).any(dim=0).sum().item())

