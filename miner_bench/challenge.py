"""Reproduce the validator's challenge generation flow against a real Pexels source.

Faithful to `PerturbValidator.generate_challenge`:
- Pick a prompt from `perturbnet.constants.PROMPTS` (validator picks via
  `system_random.choice`; the harness lets the caller pick or randomise).
- Hit the public Pexels search endpoint with the same headers and params.
- Sample one photo from the response, download its `medium` variant, base64
  the raw bytes.
- Run the validator's EfficientNetV2-M on the clean image to obtain the
  ground-truth label (this is exactly what the validator does — the prompt
  is just a Pexels query; the model's own prediction is what miners attack).
- Sample epsilon deterministically from `seed` using the validator's formula.

The validator additionally runs an LLM endpoint check that the model
prediction matches the prompt label. The harness skips it: if the
prediction is reasonable, the challenge is usable. If you want strict
parity, add your own LLM check on top of `fetch_challenge`.
"""

from __future__ import annotations

import base64
import hashlib
import random
import time
from dataclasses import dataclass
from typing import Optional

import requests
import torch

from perturbnet import constants as C
from perturbnet.image_io import decode_image_b64
from perturbnet.model import normalize_prediction_label, predict_label


@dataclass
class Challenge:
    task_id: str
    prompt: str
    clean_image_b64: str
    true_label: str
    epsilon: float
    norm_type: str = "Linf"
    timeout_seconds: int = C.TIMEOUT_SECONDS


def sample_epsilon(seed: int) -> float:
    """Validator's deterministic epsilon sampler. Range: [0.06, 0.2)."""
    return 0.06 + (seed % 1400) / 10000.0


def derive_seed(namespace: str, netuid: int, block: int) -> int:
    """Validator's seed derivation: sha256("perturb:netuid:block")[:16] as int."""
    digest = hashlib.sha256(f"{namespace}:{netuid}:{block}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def fetch_pexels_image(
    *,
    prompt: str,
    api_key: str,
    endpoint: str = C.IMAGE_ENDPOINT,
    per_page: int = C.PEXELS_PER_PAGE,
    page_span: int = C.PEXELS_PAGE_SPAN,
    image_variant: str = C.PEXELS_IMAGE_VARIANT,
    rng: Optional[random.Random] = None,
    request_timeout: float = 12.0,
) -> str:
    """Fetch one image for `prompt` from Pexels; return base64 of the raw bytes."""
    if not api_key:
        raise ValueError("Pexels API key required (set PEXELS_API_KEY).")
    chooser = rng or random.SystemRandom()
    params = {
        "query": prompt,
        "page": chooser.randint(1, max(1, page_span)),
        "per_page": max(1, min(80, int(per_page))),
    }
    response = requests.get(
        endpoint,
        params=params,
        headers={"Authorization": api_key},
        timeout=request_timeout,
    )
    response.raise_for_status()
    data = response.json()
    photos = data.get("photos") if isinstance(data, dict) else None
    if not isinstance(photos, list) or not photos:
        raise ValueError(f"Pexels returned no photos for prompt={prompt!r}")
    photo = photos[chooser.randrange(len(photos))]
    src = photo.get("src", {}) if isinstance(photo, dict) else {}
    if not isinstance(src, dict):
        src = {}
    image_url = (
        src.get(image_variant)
        or src.get("medium")
        or src.get("large")
        or src.get("large2x")
        or src.get("original")
    )
    if not isinstance(image_url, str) or not image_url.strip():
        raise ValueError("Pexels photo src missing usable image URL")

    image_response = requests.get(image_url, timeout=request_timeout)
    image_response.raise_for_status()
    image_bytes = image_response.content
    if not image_bytes:
        raise ValueError("Downloaded Pexels image is empty")
    return base64.b64encode(image_bytes).decode("utf-8")


def fetch_challenge(
    *,
    api_key: str,
    model: torch.nn.Module,
    device: torch.device,
    prompt: Optional[str] = None,
    seed: Optional[int] = None,
    epsilon: Optional[float] = None,
    timeout_seconds: int = C.TIMEOUT_SECONDS,
    max_attempts: int = 5,
    rng: Optional[random.Random] = None,
    endpoint: str = C.IMAGE_ENDPOINT,
    image_variant: str = C.PEXELS_IMAGE_VARIANT,
) -> Challenge:
    """Build one Challenge by sampling a prompt, fetching an image, predicting a label.

    On Pexels / decode / inference failures, retries up to `max_attempts` with
    fresh samples. Raises `RuntimeError` if no attempt succeeds.
    """
    chooser = rng or random.SystemRandom()
    if seed is None:
        seed = chooser.randrange(2**63)
    eps = float(epsilon) if epsilon is not None else sample_epsilon(seed)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        chosen_prompt = prompt or chooser.choice(list(C.PROMPTS))
        try:
            image_b64 = fetch_pexels_image(
                prompt=chosen_prompt,
                api_key=api_key,
                endpoint=endpoint,
                image_variant=image_variant,
                rng=chooser,
            )
            image = decode_image_b64(image_b64).to(device)
            predicted = predict_label(model, image)
            true_label = normalize_prediction_label(predicted)
        except Exception as exc:
            last_err = exc
            continue

        task_id = f"bench-{int(time.time() * 1000)}-{attempt}"
        return Challenge(
            task_id=task_id,
            prompt=chosen_prompt,
            clean_image_b64=image_b64,
            true_label=true_label,
            epsilon=eps,
            timeout_seconds=timeout_seconds,
        )

    raise RuntimeError(f"fetch_challenge: exhausted {max_attempts} attempts: {last_err}")
