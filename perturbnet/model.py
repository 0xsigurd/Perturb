from __future__ import annotations

import torch
from torchvision.models import EfficientNet_B5_Weights, efficientnet_b5

WEIGHTS = EfficientNet_B5_Weights.IMAGENET1K_V1
LABELS = [label.lower() for label in WEIGHTS.meta.get("categories", [])]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}

def load_efficientnet_b5(device: torch.device) -> torch.nn.Module:
    try:
        model = efficientnet_b5(weights=WEIGHTS)
    except Exception:
        # Keep model family stable even if pretrained weights are unavailable.
        model = efficientnet_b5(weights=None)
    return model.to(device).eval()


def resolve_target_index(target_label: str) -> int | None:
    return LABEL_TO_INDEX.get(target_label.strip().lower())


def normalize_prediction_label(raw_label: str) -> str:
    return raw_label.strip().lower().replace("_", " ")


def predict_label(model: torch.nn.Module, image_chw: torch.Tensor) -> str:
    with torch.no_grad():
        logits = model(image_chw.unsqueeze(0))
        idx = int(logits.argmax(dim=1).item())
    if 0 <= idx < len(LABELS):
        return LABELS[idx]
    return str(idx)

