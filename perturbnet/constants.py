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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_first(names: tuple[str, ...], default: str) -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default

# Shared subnet identity/constants.
SUBNET_NAMESPACE = "perturb"
MODEL_NAME = "EfficientNetV2-L"

# Validator runtime state files.
VALIDATOR_STATE_FILENAME = "perturb_validator_state.json"

# Validator runtime constants.
# Challenge dataset is fixed: full ImageNet-100 train split from Hugging Face,
# auto-downloaded once into the local cache.
IMAGENET100_REPO_ID = "clane9/imagenet-100"
IMAGENET100_SPLIT = "train"
# If challenge generation fails (e.g. transient dataset issue), sleep this
# long and try again; there is no fallback image.
CHALLENGE_RETRY_DELAY_SECONDS = _env_int("PERTURB_CHALLENGE_RETRY_DELAY_SECONDS", 180)
IMAGE_SIZE = _env_int("PERTURB_IMAGE_SIZE", 64)
K_MINERS = _env_int("PERTURB_K_MINERS", 150)
HISTORY_SIZE = _env_int("PERTURB_HISTORY_SIZE", 300)
MIN_WEIGHT_HISTORY_SIZE = _env_int("PERTURB_MIN_WEIGHT_HISTORY_SIZE", 50)
MIN_LINF_DELTA = _env_float("PERTURB_MIN_LINF_DELTA", 0.003)
MAX_LINF_DELTA = _env_float("PERTURB_MAX_LINF_DELTA", 0.03)
MIN_SSIM = _env_float("PERTURB_MIN_SSIM", 0.98)
MIN_PSNR_DB = _env_float("PERTURB_MIN_PSNR_DB", 38.0)
LINF_COMPONENT_WEIGHT = _env_float("PERTURB_LINF_COMPONENT_WEIGHT", 0.7)
RMSE_COMPONENT_WEIGHT = _env_float("PERTURB_RMSE_COMPONENT_WEIGHT", 0.3)
MAX_CHALLENGE_ATTEMPTS = _env_int("PERTURB_MAX_CHALLENGE_ATTEMPTS", 12)
MINER_EXPLORATION_RATIO = _env_float("PERTURB_MINER_EXPLORATION_RATIO", 0.20)
ANALYZE_BUCKET_MARGIN_WEIGHT = _env_float("ANALYZE_BUCKET_MARGIN_WEIGHT", 0.03)
ANALYZE_BUCKET_NOVELTY_WEIGHT = _env_float("ANALYZE_BUCKET_NOVELTY_WEIGHT", 0.01)
ANALYZE_BUCKET_NOVELTY_TARGET_PIXELS = _env_int("ANALYZE_BUCKET_NOVELTY_TARGET_PIXELS", 8)
STORAGE_BACKEND = os.getenv("PERTURB_STORAGE_BACKEND", "hippius").strip().lower() or "hippius"
BURN_RATE_ENDPOINT = "https://api.perturbai.io/api/v1/burn-rate"
DEFAULT_BURN_RATE = 0.0
BURN_RATE_FETCH_TIMEOUT_SECONDS = 5.0
BURN_UID = 0
LEADERBOARD_REPORTING_ENABLED = _env_bool("PERTURB_LEADERBOARD_ENABLED", False)
LEADERBOARD_API_URL = "https://api.perturbai.io/api/v1/report"
LEADERBOARD_LAST_WEIGHT_UPDATE_API_URL = "https://api.perturbai.io/api/v1/last-weight-update"
LEADERBOARD_REPORT_TIMEOUT_SECONDS = 10.0
LEADERBOARD_NO_IMAGE_URL = ""
PERTURB_API_BASE_URL = os.getenv("PERTURB_API_BASE_URL", "https://api.perturbai.io/api/v1").strip()
PERTURB_API_KEY = os.getenv("PERTURB_API_KEY", "").strip()
PERTURB_API_TIMEOUT_SECONDS = _env_float("PERTURB_API_TIMEOUT_SECONDS", 10.0)
TASK_POLL_TIME = _env_float("PERTURB_TASK_POLL_TIME", 2.0)
TASK_CADENCE_SECONDS = _env_int("PERTURB_TASK_CADENCE_SECONDS", 120)
TASK_PRE_BOUNDARY_FETCH_SECONDS = _env_float("PERTURB_TASK_PRE_BOUNDARY_FETCH_SECONDS", 10.0)
TASK_FETCH_RETRIES = _env_int("PERTURB_TASK_FETCH_RETRIES", 10)
TASK_FETCH_RETRY_SECONDS = _env_float("PERTURB_TASK_FETCH_RETRY_SECONDS", 2.0)
VALIDATOR_EVALUATION_DELAY_SECONDS = _env_float("PERTURB_VALIDATOR_EVALUATION_DELAY_SECONDS", 40.0)
VALIDATOR_EVALUATION_POLL_SECONDS = _env_float("PERTURB_VALIDATOR_EVALUATION_POLL_SECONDS", 5.0)
VALIDATOR_EVALUATION_POLL_RETRIES = _env_int("PERTURB_VALIDATOR_EVALUATION_POLL_RETRIES", 14)
STORAGE_BUCKET = os.getenv("PERTURB_STORAGE_BUCKET", "").strip()
STORAGE_ENDPOINT_URL = os.getenv("PERTURB_STORAGE_ENDPOINT_URL", "").strip()
STORAGE_ACCESS_KEY_ID = os.getenv("PERTURB_STORAGE_ACCESS_KEY_ID", "").strip()
STORAGE_SECRET_ACCESS_KEY = os.getenv("PERTURB_STORAGE_SECRET_ACCESS_KEY", "").strip()
STORAGE_REGION = os.getenv("PERTURB_STORAGE_REGION", "").strip()
STORAGE_PREFIX = "perturb"
STORAGE_PRESIGNED_URL_EXPIRES_SECONDS = 604800

VALIDATOR_CONFIG = {
    "imagenet100_repo_id": IMAGENET100_REPO_ID,
    "imagenet100_split": IMAGENET100_SPLIT,
    "image_size": IMAGE_SIZE,
    "k_miners": K_MINERS,
    "history_size": HISTORY_SIZE,
    "min_weight_history_size": MIN_WEIGHT_HISTORY_SIZE,
    "min_linf_delta": MIN_LINF_DELTA,
    "max_linf_delta": MAX_LINF_DELTA,
    "min_ssim": MIN_SSIM,
    "min_psnr_db": MIN_PSNR_DB,
    "linf_component_weight": LINF_COMPONENT_WEIGHT,
    "rmse_component_weight": RMSE_COMPONENT_WEIGHT,
    "max_challenge_attempts": MAX_CHALLENGE_ATTEMPTS,
    "miner_exploration_ratio": MINER_EXPLORATION_RATIO,
    "analyze_bucket_margin_weight": ANALYZE_BUCKET_MARGIN_WEIGHT,
    "analyze_bucket_novelty_weight": ANALYZE_BUCKET_NOVELTY_WEIGHT,
    "analyze_bucket_novelty_target_pixels": ANALYZE_BUCKET_NOVELTY_TARGET_PIXELS,
    "storage_backend": STORAGE_BACKEND,
    "burn_rate_endpoint": BURN_RATE_ENDPOINT,
    "default_burn_rate": DEFAULT_BURN_RATE,
    "burn_rate_fetch_timeout_seconds": BURN_RATE_FETCH_TIMEOUT_SECONDS,
    "burn_uid": BURN_UID,
    "leaderboard_reporting_enabled": LEADERBOARD_REPORTING_ENABLED,
    "leaderboard_api_url": LEADERBOARD_API_URL,
    "leaderboard_last_weight_update_api_url": LEADERBOARD_LAST_WEIGHT_UPDATE_API_URL,
    "leaderboard_report_timeout_seconds": LEADERBOARD_REPORT_TIMEOUT_SECONDS,
    "leaderboard_no_image_url": LEADERBOARD_NO_IMAGE_URL,
    "api_base_url": PERTURB_API_BASE_URL,
    "api_key": PERTURB_API_KEY,
    "api_timeout_seconds": PERTURB_API_TIMEOUT_SECONDS,
    "task_poll_time": TASK_POLL_TIME,
    "task_cadence_seconds": TASK_CADENCE_SECONDS,
    "task_pre_boundary_fetch_seconds": TASK_PRE_BOUNDARY_FETCH_SECONDS,
    "task_fetch_retries": TASK_FETCH_RETRIES,
    "task_fetch_retry_seconds": TASK_FETCH_RETRY_SECONDS,
    "validator_evaluation_delay_seconds": VALIDATOR_EVALUATION_DELAY_SECONDS,
    "validator_evaluation_poll_seconds": VALIDATOR_EVALUATION_POLL_SECONDS,
    "validator_evaluation_poll_retries": VALIDATOR_EVALUATION_POLL_RETRIES,
    "storage_bucket": STORAGE_BUCKET,
    "storage_endpoint_url": STORAGE_ENDPOINT_URL,
    "storage_access_key_id": STORAGE_ACCESS_KEY_ID,
    "storage_secret_access_key": STORAGE_SECRET_ACCESS_KEY,
    "storage_region": STORAGE_REGION,
    "storage_prefix": STORAGE_PREFIX,
    "storage_presigned_url_expires_seconds": STORAGE_PRESIGNED_URL_EXPIRES_SECONDS,
}

# Validator scoring defaults.
PERTURBATION_WEIGHT = _env_float("PERTURB_PERTURBATION_WEIGHT", 1)
GAMMA_HISTORY_WEIGHT = _env_float("PERTURB_GAMMA_HISTORY_WEIGHT", 0.7)

