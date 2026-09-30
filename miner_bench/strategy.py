"""Pluggable miner-attack interface plus a textbook PGD baseline.

The `MinerStrategy` ABC is the only contract the harness expects: given a
clean image (CHW, [0,1] float tensor), a true label, and an epsilon budget,
return a perturbed image in the same shape that the validator's scorer can
re-encode.

`PGDStrategy` is a deliberately weak, vanilla baseline so the harness has a
working example out of the box. It is not competitive on mainnet — it
exists to demonstrate the interface and give you a calibration point. Drop
in your own subclass to benchmark a real attack.

References for PGD: Madry et al., "Towards Deep Learning Models Resistant
to Adversarial Attacks" (2018), arXiv:1706.06083.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from perturbnet import constants as C
from perturbnet.image_io import encode_image_b64
from perturbnet.model import (
    LABEL_TO_INDEX,
    logits_for_images,
    resolve_target_index,
)


@dataclass
class StrategyContext:
    """Inputs the harness hands to every strategy call."""

    clean_image_b64: str
    true_label: str
    epsilon: float
    timeout_seconds: int
    model: torch.nn.Module
    device: torch.device
    max_linf_delta: float = C.MAX_LINF_DELTA
    min_linf_delta: float = C.MIN_LINF_DELTA

    @property
    def effective_max_delta(self) -> float:
        """The L∞ cap the validator will actually enforce: min(epsilon, max_linf_delta)."""
        return min(float(self.epsilon), float(self.max_linf_delta))


class MinerStrategy(ABC):
    """Subclass and implement `attack` to plug your own attack into the harness."""

    name: str = "abstract"

    @abstractmethod
    def attack(self, ctx: StrategyContext) -> str:
        """Return a PNG-encoded base64 string of the perturbed image."""


def _decode_clean_to_tensor(clean_image_b64: str, device: torch.device) -> torch.Tensor:
    from perturbnet.image_io import decode_image_b64

    return decode_image_b64(clean_image_b64).to(device)


def _quantise_to_uint8_then_float(x: torch.Tensor) -> torch.Tensor:
    """Match the round-trip the validator sees: PNG encode quantises to uint8,
    decode divides by 255. We simulate that quantisation in-loop so SSIM/RMSE
    computed on the encoded image matches what we predict here."""
    return (x.clamp(0.0, 1.0) * 255.0).round() / 255.0


class PGDStrategy(MinerStrategy):
    """Untargeted PGD in the L-inf ball, with epsilon as the radius.

    Parameters
    ----------
    steps        : number of gradient steps (default 20)
    step_ratio   : per-step magnitude as a fraction of epsilon (default 0.15)
    margin       : early-stop when adversarial logit beats the runner-up by this
                   margin (in logits). 0 disables early stop.
    """

    name = "pgd"

    def __init__(self, steps: int = 20, step_ratio: float = 0.15, margin: float = 2.0):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = steps
        self.step_ratio = step_ratio
        self.margin = margin

    def attack(self, ctx: StrategyContext) -> str:
        model = ctx.model
        device = ctx.device
        x_clean = _decode_clean_to_tensor(ctx.clean_image_b64, device)

        true_idx = resolve_target_index(ctx.true_label)
        if true_idx is None:
            # Unknown label — return the clean image; validator will reject it.
            return encode_image_b64(x_clean)

        # The validator caps the effective L∞ at min(epsilon, max_linf_delta).
        # A naive PGD that uses raw epsilon will reliably hit `above_max_delta`.
        eps = float(ctx.effective_max_delta)
        # Per-step magnitude in pixel space.
        alpha = max(1.0 / 255.0, eps * self.step_ratio)

        # PGD operates on the perturbation delta so we can clip it cleanly.
        delta = torch.zeros_like(x_clean, requires_grad=True)
        true_idx_t = torch.tensor([true_idx], device=device, dtype=torch.long)

        for _ in range(self.steps):
            x_adv = (x_clean + delta).clamp(0.0, 1.0)
            logits = logits_for_images(model, x_adv.unsqueeze(0))
            loss = F.cross_entropy(logits, true_idx_t)
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                delta.data = (delta.data + alpha * grad.sign()).clamp(-eps, eps)
                # Keep x_adv inside [0,1] given the clean image and current delta.
                delta.data = (x_clean + delta.data).clamp(0.0, 1.0) - x_clean

                if self.margin > 0.0:
                    # Evaluate the early-stop check on the quantised tensor that
                    # the validator will actually see (PNG -> uint8 -> /255).
                    adv_quant = _quantise_to_uint8_then_float(x_clean + delta.data)
                    logits_q = logits_for_images(model, adv_quant.unsqueeze(0)).squeeze(0)
                    top2 = torch.topk(logits_q, k=2)
                    top_idx = int(top2.indices[0].item())
                    if top_idx != true_idx:
                        true_logit = float(logits_q[true_idx].item())
                        winner_logit = float(top2.values[0].item())
                        if (winner_logit - true_logit) >= self.margin:
                            delta = delta.detach().requires_grad_(False)
                            break

            delta.requires_grad_(True)

        x_adv_final = _quantise_to_uint8_then_float(x_clean + delta.detach())
        return encode_image_b64(x_adv_final)


def list_known_labels() -> list[str]:
    """Helper for callers that want to introspect the model's label space."""
    return sorted(LABEL_TO_INDEX.keys())
