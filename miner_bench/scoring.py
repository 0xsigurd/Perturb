"""Faithful, standalone port of the mainnet validator's `verify_and_score`.

This module reproduces the scoring path defined in `neurons/validator.py` so
miners can evaluate a candidate perturbation locally with the same gates,
weights, and reasons the validator applies. It deliberately avoids any
dependency on the `PerturbValidator` class state — every knob is exposed
on `BenchConfig` with the same defaults as `perturbnet.constants`.

Reviewers should be able to diff `score_response` against
`PerturbValidator.verify_and_score` and confirm the two compute identical
results for any (clean_image, perturbed_image, response_time) triple.

If the validator's scoring rules change upstream, this module is the only
file that needs to be updated to keep the harness faithful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F

from perturbnet import constants as C
from perturbnet.image_io import decode_image_b64
from perturbnet.model import (
    load_efficientnet_v2_m,
    normalize_prediction_label,
    predict_label,
)


class ScoreReason(str, Enum):
    SUCCESS = "success"
    DECODE_FAILED = "decode_failed"
    SHAPE_MISMATCH = "shape_mismatch"
    VALUE_OUT_OF_RANGE = "value_out_of_range"
    MODEL_INFERENCE_FAILED = "model_inference_failed"
    BELOW_MIN_DELTA = "below_min_delta"
    ABOVE_MAX_DELTA = "above_max_delta"
    LABEL_MATCH_WITH_ORIGINAL = "label_match_with_original"
    BELOW_MIN_SSIM = "below_min_ssim"
    BELOW_MIN_PSNR_DB = "below_min_psnr_db"


@dataclass
class BenchConfig:
    """Scoring knobs. Defaults mirror perturbnet.constants for the mainnet validator."""

    min_linf_delta: float = C.MIN_LINF_DELTA
    max_linf_delta: float = C.MAX_LINF_DELTA
    min_ssim: float = C.MIN_SSIM
    min_psnr_db: float = C.MIN_PSNR_DB
    linf_component_weight: float = C.LINF_COMPONENT_WEIGHT
    rmse_component_weight: float = C.RMSE_COMPONENT_WEIGHT
    speed_weight: float = C.SPEED_WEIGHT
    perturbation_weight: float = C.PERTURBATION_WEIGHT


@dataclass
class EvaluationResult:
    score: float
    reason: str
    model_prediction: str = ""
    response_time_ms: int = 0
    norm: float = 0.0
    rmse: float = 0.0
    epsilon: float = 0.0
    ssim: float = 0.0
    psnr_db: float = 0.0


def _compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
    if x_clean.ndim != 3 or x_adv.ndim != 3:
        return 0.0
    if x_clean.shape != x_adv.shape:
        return 0.0
    padding = kernel_size // 2
    x = x_clean.unsqueeze(0)
    y = x_adv.unsqueeze(0)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_y

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())


def _compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def score_response(
    *,
    clean_image_b64: str,
    perturbed_image_b64: str,
    true_label: str,
    epsilon: float,
    response_time_ms: int,
    timeout_seconds: int,
    model: torch.nn.Module,
    device: torch.device,
    config: Optional[BenchConfig] = None,
    norm_type: str = "Linf",
) -> EvaluationResult:
    """Reproduce `PerturbValidator.verify_and_score` exactly.

    `model` must be the same EfficientNetV2-M weights the validator uses
    (see `perturbnet.model.load_efficientnet_v2_m`).
    """
    cfg = config or BenchConfig()
    try:
        x_clean = decode_image_b64(clean_image_b64).to(device)
        x_adv = decode_image_b64(perturbed_image_b64).to(device)
    except Exception as exc:
        return EvaluationResult(
            score=0.0,
            reason=f"{ScoreReason.DECODE_FAILED.value}:{exc}",
            response_time_ms=response_time_ms,
        )

    if x_adv.shape != x_clean.shape:
        return EvaluationResult(
            score=0.0,
            reason=ScoreReason.SHAPE_MISMATCH.value,
            response_time_ms=response_time_ms,
        )
    if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
        return EvaluationResult(
            score=0.0,
            reason=ScoreReason.VALUE_OUT_OF_RANGE.value,
            response_time_ms=response_time_ms,
        )

    prediction = ""
    try:
        prediction = predict_label(model, x_adv)
    except Exception as exc:
        return EvaluationResult(
            score=0.0,
            reason=f"{ScoreReason.MODEL_INFERENCE_FAILED.value}:{exc}",
            response_time_ms=response_time_ms,
        )

    if norm_type == "Linf":
        norm = float((x_adv - x_clean).abs().max().item())
    elif norm_type == "L2":
        norm = float((x_adv - x_clean).norm(2).item())
    else:
        norm = float((x_adv - x_clean).ne(0).sum().item())

    if norm < cfg.min_linf_delta:
        return EvaluationResult(
            score=0.0,
            reason=ScoreReason.BELOW_MIN_DELTA.value,
            model_prediction=prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            epsilon=float(epsilon),
        )
    effective_max_delta = min(float(epsilon), float(cfg.max_linf_delta))
    if norm > effective_max_delta:
        return EvaluationResult(
            score=0.0,
            reason=ScoreReason.ABOVE_MAX_DELTA.value,
            model_prediction=prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
            epsilon=float(epsilon),
        )

    normalized_prediction = normalize_prediction_label(prediction)
    if normalized_prediction == true_label:
        return EvaluationResult(
            score=0.0,
            reason=ScoreReason.LABEL_MATCH_WITH_ORIGINAL.value,
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
            epsilon=float(epsilon),
        )

    rmse = float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item())

    ssim = _compute_ssim(x_clean=x_clean, x_adv=x_adv)
    if ssim < cfg.min_ssim:
        return EvaluationResult(
            score=0.0,
            reason=ScoreReason.BELOW_MIN_SSIM.value,
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            rmse=rmse,
            epsilon=float(epsilon),
            ssim=ssim,
        )

    psnr_db = _compute_psnr_db(x_clean=x_clean, x_adv=x_adv)
    if cfg.min_psnr_db > 0.0 and psnr_db < cfg.min_psnr_db:
        return EvaluationResult(
            score=0.0,
            reason=ScoreReason.BELOW_MIN_PSNR_DB.value,
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            rmse=rmse,
            epsilon=float(epsilon),
            ssim=ssim,
            psnr_db=psnr_db,
        )

    denom = max(1e-12, effective_max_delta - cfg.min_linf_delta)
    linf_ratio = (norm - cfg.min_linf_delta) / denom
    linf_ratio = min(max(linf_ratio, 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2

    rmse_ratio = rmse / max(1e-12, effective_max_delta)
    rmse_ratio = min(max(rmse_ratio, 0.0), 1.0)
    rmse_score = (1.0 - rmse_ratio) ** 2

    total_weight = max(1e-12, cfg.linf_component_weight + cfg.rmse_component_weight)
    perturbation_score = (
        (cfg.linf_component_weight * linf_score) + (cfg.rmse_component_weight * rmse_score)
    ) / total_weight

    time_ratio = response_time_ms / (timeout_seconds * 1000.0)
    speed_score = 1.0 - min(time_ratio, 1.0)

    score = cfg.perturbation_weight * perturbation_score + cfg.speed_weight * speed_score
    return EvaluationResult(
        score=float(score),
        reason=ScoreReason.SUCCESS.value,
        model_prediction=normalized_prediction,
        response_time_ms=response_time_ms,
        norm=norm,
        rmse=rmse,
        epsilon=float(epsilon),
        ssim=ssim,
        psnr_db=psnr_db,
    )


def load_model(device: torch.device) -> torch.nn.Module:
    """Convenience re-export of the validator's model loader."""
    return load_efficientnet_v2_m(device)
