"""Shared pieces for train.py, evaluate.py and submit.py."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l
from torchvision.transforms import InterpolationMode

# These scripts are run from <repo>/training or from the repo root; make the
# perturbnet package importable either way.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perturbnet.model_commit import ARCHITECTURE, CONFIG_FILE, MODEL_FILE, NUM_CLASSES  # noqa: E402

BASE_WEIGHTS = EfficientNet_V2_L_Weights.IMAGENET1K_V1

_REF = BASE_WEIGHTS.transforms()
EVAL_RESIZE = int(_REF.resize_size[0])
EVAL_CROP = int(_REF.crop_size[0])
MEAN = tuple(float(v) for v in _REF.mean)
STD = tuple(float(v) for v in _REF.std)
INTERPOLATION: InterpolationMode = _REF.interpolation

DEFAULT_ADV_DATASET = "perturb-ai/efficientnet-v2-l-adv-dataset"
DEFAULT_IMAGENET_DATASET = "ILSVRC/imagenet-1k"


# --------------------------------------------------------------------------- model


def create_model(pretrained: bool = True) -> nn.Module:
    return efficientnet_v2_l(weights=BASE_WEIGHTS if pretrained else None)


def model_config(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    config = {
        "architecture": ARCHITECTURE,
        "framework": "torchvision",
        "num_classes": NUM_CLASSES,
        "base_weights": f"torchvision/{BASE_WEIGHTS}",
        "weights_file": MODEL_FILE,
        "preprocess": {
            "resize": EVAL_RESIZE,
            "crop": EVAL_CROP,
            "interpolation": INTERPOLATION.value,
            "antialias": True,
            "mean": list(MEAN),
            "std": list(STD),
        },
    }
    if extra:
        config.update(extra)
    return config


def save_model(model: nn.Module, out_dir: str | os.PathLike[str], extra: dict[str, Any] | None = None) -> Path:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
    tmp = out / (MODEL_FILE + ".tmp")
    save_file(state, str(tmp), metadata={"format": "pt", "architecture": ARCHITECTURE})
    os.replace(tmp, out / MODEL_FILE)
    tmp_cfg = out / (CONFIG_FILE + ".tmp")
    tmp_cfg.write_text(json.dumps(model_config(extra), indent=2), encoding="utf-8")
    os.replace(tmp_cfg, out / CONFIG_FILE)
    return out


def resolve_weights_file(source: str, revision: str | None = None, token: str | None = None) -> Path:
    path = Path(source)
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / MODEL_FILE
        if not candidate.exists():
            raise FileNotFoundError(f"{candidate} not found")
        return candidate
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=source, filename=MODEL_FILE, revision=revision, token=token))


def load_model(source: str | None, *, device: torch.device, revision: str | None = None, token: str | None = None) -> nn.Module:
    if source is None or source == "pretrained":
        model = create_model(pretrained=True)
    else:
        from safetensors.torch import load_file

        model = create_model(pretrained=False)
        state = load_file(str(resolve_weights_file(source, revision=revision, token=token)))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"State dict mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
    return model.to(device)


# --------------------------------------------------------------------------- transforms


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(EVAL_RESIZE, interpolation=INTERPOLATION, antialias=True),
            transforms.CenterCrop(EVAL_CROP),
            transforms.Normalize(MEAN, STD),
        ]
    )


def train_transform(resolution: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(resolution, scale=(0.35, 1.0), interpolation=INTERPOLATION, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


# --------------------------------------------------------------------------- labels

IMAGENET1K_CATEGORIES: list[str] = list(BASE_WEIGHTS.meta["categories"])


def _norm(label: str) -> str:
    return label.strip().lower().replace("_", " ")


_CATEGORY_INDEX: dict[str, list[int]] = {}
for _i, _c in enumerate(IMAGENET1K_CATEGORIES):
    _CATEGORY_INDEX.setdefault(_norm(_c), []).append(_i)


def imagenet1k_index_for_name(label_name: str) -> int | None:
    aliases = [_norm(label_name)] + [_norm(p) for p in label_name.split(",")]
    for alias in aliases:
        if not alias:
            continue
        hits = _CATEGORY_INDEX.get(alias, [])
        if len(hits) == 1:
            return hits[0]
    return None


class LabelMapper:

    def __init__(self, overrides: dict[str, int] | None = None) -> None:
        self.overrides = {k: int(v) for k, v in (overrides or {}).items()}
        self.cache: dict[str, int | None] = {}
        self.unmapped: set[str] = set()

    @classmethod
    def from_file(cls, path: str | None) -> "LabelMapper":
        if not path:
            return cls()
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __call__(self, label_name: str | None) -> int | None:
        if not label_name:
            return None
        if label_name in self.overrides:
            return self.overrides[label_name]
        if label_name not in self.cache:
            index = imagenet1k_index_for_name(label_name)
            self.cache[label_name] = index
            if index is None:
                self.unmapped.add(label_name)
        return self.cache[label_name]


# --------------------------------------------------------------------------- chain commitment

from perturbnet.model_commit import (  # noqa: E402,F401
    CHAIN_COMMIT_MAX_BYTES,
    CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS,
    MODEL_HASH_CHARS,
    MinerChainCommit,
    parse_chain_commit,
    serialize_chain_commit,
    sha256_model_and_hotkey,
    validate_chain_commit_payload,
)
