from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_first(names: tuple[str, ...], default: str) -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default

# Shared subnet identity/constants.
SUBNET_NAMESPACE = "perturb"
MODEL_NAME = "EfficientNet-B5"
PROMPTS = (
    "dog",
    "cat",
    "bird",
    "fish",
    "reptile",
    "amphibian",
    "insect",
    "arachnid",
    "crustacean",
    "mollusk",
    "primate",
    "rodent",
    "rabbit",
    "equine",
    "bovine",
    "ovine",
    "caprine",
    "porcine",
    "ungulate",
    "marine_mammal",
)

# Validator runtime state files.
VALIDATOR_STATE_FILENAME = "perturb_validator_state.json"
FALLBACK_IMAGE_RELATIVE_PATH = "assets/dog_1.jpg"
FALLBACK_LABEL = "dog"

# Validator runtime constants.
IMAGE_ENDPOINT = os.getenv("PERTURB_IMAGE_ENDPOINT", "http://api.picflux.io/v1/search")
IMAGE_SIZE = _env_int("PERTURB_IMAGE_SIZE", 64)
TIMEOUT_SECONDS = _env_int("PERTURB_TIMEOUT_SECONDS", 60)
QUERY_INTERVAL_SECONDS = _env_int("PERTURB_QUERY_INTERVAL_SECONDS", 15)
K_MINERS = _env_int("PERTURB_K_MINERS", 8)
HISTORY_SIZE = _env_int("PERTURB_HISTORY_SIZE", 100)
MIN_PROCESSED_COUNT = _env_int("PERTURB_MIN_PROCESSED_COUNT", 100)
LLM_ENDPOINT_URL = _env_first(
    ("PERTURB_LLM_ENDPOINT_URL", "PERTURB_LABEL_MATCH_ENDPOINT", "PERTURB_LLM_VERIFY_ENDPOINT"),
    "http://127.0.0.1:8081/verify-label",
)
LLM_ENDPOINT_MODEL = _env_first(
    ("PERTURB_LLM_ENDPOINT_MODEL", "PERTURB_LABEL_MATCH_MODEL", "PERTURB_LLM_VERIFY_MODEL"),
    "Qwen2.5-1.5B-Instruct",
)
MIN_LINF_DELTA = _env_float("PERTURB_MIN_LINF_DELTA", 0.002)
MAX_LINF_DELTA = _env_float("PERTURB_MAX_LINF_DELTA", 0.12)
MAX_CHALLENGE_ATTEMPTS = _env_int("PERTURB_MAX_CHALLENGE_ATTEMPTS", 12)

VALIDATOR_CONFIG = {
    "image_endpoint": IMAGE_ENDPOINT,
    "image_size": IMAGE_SIZE,
    "timeout_seconds": TIMEOUT_SECONDS,
    "query_interval_seconds": QUERY_INTERVAL_SECONDS,
    "k_miners": K_MINERS,
    "history_size": HISTORY_SIZE,
    "min_processed_count": MIN_PROCESSED_COUNT,
    "llm_endpoint_url": LLM_ENDPOINT_URL,
    "llm_endpoint_model": LLM_ENDPOINT_MODEL,
    "min_linf_delta": MIN_LINF_DELTA,
    "max_linf_delta": MAX_LINF_DELTA,
    "max_challenge_attempts": MAX_CHALLENGE_ATTEMPTS,
}

# Validator scoring defaults.
SPEED_WEIGHT = _env_float("PERTURB_SPEED_WEIGHT", 0.35)
PERTURBATION_WEIGHT = _env_float("PERTURB_PERTURBATION_WEIGHT", 0.65)
GAMMA_HISTORY_WEIGHT = _env_float("PERTURB_GAMMA_HISTORY_WEIGHT", 0.7)

