from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging as pylogging
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import bittensor as bt
import numpy as np
import requests
import torch
import torch.nn.functional as F
try:
    import wandb  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional dependency
    wandb = None

from perturbnet import constants as C
from perturbnet.image_io import decode_image_b64
from perturbnet.model import load_efficientnet_v2_m, normalize_prediction_label, predict_label
from perturbnet.protocol import AttackChallenge

logger = pylogging.getLogger(__name__)


@dataclass
class ChallengeSpec:
    task_id: str
    model_name: str
    prompt: str
    clean_image_b64: str
    true_label: str
    epsilon: float
    norm_type: str
    timeout_seconds: int
    fallback_used: bool = False
    verified_by_llm: bool = True


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


def _make_wallet(config):
    wallet_name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    wallet_hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))
    if hasattr(bt, "wallet"):
        try:
            return bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        except Exception:
            return bt.wallet(config=config)
    wallet_cls = getattr(bt, "Wallet", None)
    if wallet_cls is None:
        raise RuntimeError("No wallet constructor found in bittensor.")
    try:
        return wallet_cls(name=wallet_name, hotkey=wallet_hotkey)
    except TypeError:
        return wallet_cls(config=config)


def _make_subtensor(config):
    network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    if hasattr(bt, "subtensor"):
        try:
            return bt.subtensor(network=network)
        except Exception:
            return bt.subtensor(config=config)
    subtensor_cls = getattr(bt, "Subtensor", None)
    if subtensor_cls is None:
        raise RuntimeError("No subtensor constructor found in bittensor.")
    try:
        return subtensor_cls(network=network)
    except Exception:
        return subtensor_cls(config=config)


def _make_dendrite(wallet):
    if hasattr(bt, "dendrite"):
        return bt.dendrite(wallet=wallet)
    dendrite_cls = getattr(bt, "Dendrite", None)
    if dendrite_cls is None:
        raise RuntimeError("No dendrite constructor found in bittensor.")
    return dendrite_cls(wallet=wallet)


def _configure_log_level(level_raw: str) -> None:
    level_name = (level_raw or "DEBUG").upper()
    requested_level = getattr(pylogging, level_name, pylogging.INFO)
    level = max(int(pylogging.INFO), int(requested_level))
    pylogging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    pylogging.getLogger().setLevel(level)


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


class _WandbConsoleHandler(pylogging.Handler):
    def __init__(self, owner: "PerturbValidator") -> None:
        super().__init__()
        self.owner = owner

    def emit(self, record: pylogging.LogRecord) -> None:
        run = self.owner.wandb_run
        if run is None:
            return
        if record.name.startswith("wandb"):
            return
        try:
            run.log(
                {
                    "validator/console_level": record.levelname,
                    "validator/console_logger": record.name,
                    "validator/console_message": self.format(record),
                    "validator/console_ts": float(record.created),
                },
                step=int(self.owner.step),
            )
        except Exception:
            # Never let console-forwarding failures affect validator runtime.
            pass


class PerturbValidator:
    def __init__(self, config: bt.config) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = _make_subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.dendrite = _make_dendrite(wallet=self.wallet)
        self._query_loop = asyncio.new_event_loop()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.system_random = random.SystemRandom()

        self.model = load_efficientnet_v2_m(self.device)
        self.step = 0
        self.last_weight_block = 0
        self.state_path = os.path.join(self.config.logging.logging_dir, C.VALIDATOR_STATE_FILENAME)
        self.wandb_run: Any | None = None
        self._wandb_console_handler: pylogging.Handler | None = None
        hotkey = getattr(self.wallet.hotkey, "ss58_address", "unknown")
        self.run_id = f"{str(hotkey)[:8]}-n{self.config.netuid}-p{os.getpid()}"
        self.reason_counts_total: Counter[str] = Counter()
        self.miner_emission_share = 1

        self.processed_counts = np.zeros(int(self.metagraph.n), dtype=np.int32)
        self.score_histories: list[list[float]] = [[] for _ in range(int(self.metagraph.n))]
        self.uid_hotkeys: list[str] = list(self.metagraph.hotkeys[: int(self.metagraph.n)])

        self._load_state()
        self._init_wandb()

    def _log_step_start(self, step_name: str, **context: Any) -> None:
        if context:
            rendered = " ".join([f"{k}={v}" for k, v in context.items()])
            logger.debug(f"{step_name} {rendered}")
        else:
            logger.debug(step_name)

    def _log_summary(self, event: str, **context: Any) -> None:
        if context:
            rendered = " ".join([f"{k}={context[k]}" for k in sorted(context.keys())])
            logger.info(f"[run_id={self.run_id}] {event} {rendered}")
        else:
            logger.info(f"[run_id={self.run_id}] {event}")

    def sync(self) -> None:
        old_n = int(self.metagraph.n)
        self.metagraph.sync(subtensor=self.subtensor)
        new_n = int(self.metagraph.n)
        if new_n != old_n:
            resized_counts = np.zeros(new_n, dtype=np.int32)
            copied = min(len(self.processed_counts), new_n)
            resized_counts[:copied] = self.processed_counts[:copied]
            self.processed_counts = resized_counts
            if new_n > len(self.score_histories):
                self.score_histories.extend([[] for _ in range(new_n - len(self.score_histories))])
            else:
                self.score_histories = self.score_histories[:new_n]
            if new_n > len(self.uid_hotkeys):
                self.uid_hotkeys.extend([""] * (new_n - len(self.uid_hotkeys)))
            else:
                self.uid_hotkeys = self.uid_hotkeys[:new_n]
        self._reconcile_uid_identities()

    def _reset_uid_stats(self, uid: int, reason: str) -> None:
        self.processed_counts[uid] = 0
        self.score_histories[uid] = []
        logger.info(f"Reset uid={uid} stats due to {reason}.")

    def _reconcile_uid_identities(self) -> None:
        n = int(self.metagraph.n)
        if len(self.uid_hotkeys) < n:
            self.uid_hotkeys.extend([""] * (n - len(self.uid_hotkeys)))
        elif len(self.uid_hotkeys) > n:
            self.uid_hotkeys = self.uid_hotkeys[:n]

        for uid in range(n):
            current_hotkey = str(self.metagraph.hotkeys[uid])
            previous_hotkey = self.uid_hotkeys[uid]
            if previous_hotkey and previous_hotkey != current_hotkey:
                self._reset_uid_stats(uid, reason="hotkey_changed")
            self.uid_hotkeys[uid] = current_hotkey

    def _load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        with open(self.state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        self.step = int(state.get("step", 0))
        self.last_weight_block = int(state.get("last_weight_block", 0))

        saved_counts = state.get("processed_counts", [])
        copied = min(len(saved_counts), len(self.processed_counts))
        for idx in range(copied):
            self.processed_counts[idx] = int(saved_counts[idx])

        saved_histories = state.get("score_histories", [])
        copied_h = min(len(saved_histories), len(self.score_histories))
        for idx in range(copied_h):
            raw = saved_histories[idx]
            if isinstance(raw, list):
                self.score_histories[idx] = [float(x) for x in raw[-self.config.perturb.history_size :]]

        saved_hotkeys = state.get("uid_hotkeys", [])
        if isinstance(saved_hotkeys, list):
            copied_keys = min(len(saved_hotkeys), len(self.uid_hotkeys))
            for idx in range(copied_keys):
                value = saved_hotkeys[idx]
                if isinstance(value, str):
                    self.uid_hotkeys[idx] = value
        self._reconcile_uid_identities()

    def _save_state(self) -> None:
        directory = os.path.dirname(self.state_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "step": int(self.step),
            "last_weight_block": int(self.last_weight_block),
            "processed_counts": self.processed_counts.tolist(),
            "score_histories": [history[-self.config.perturb.history_size :] for history in self.score_histories],
            "uid_hotkeys": self.uid_hotkeys,
        }
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def _init_wandb(self) -> None:
        if not bool(getattr(self.config.perturb, "wandb_enabled", False)):
            return
        if wandb is None:
            logger.warning("PERTURB_WANDB_ENABLED is true, but `wandb` is not installed.")
            return
        try:
            init_kwargs: dict[str, Any] = {
                "project": str(getattr(self.config.perturb, "wandb_project", "perturb-validator")),
                "config": {
                    "netuid": int(self.config.netuid),
                    "network": str(getattr(self.config.subtensor, "network", "unknown")),
                    "k_miners": int(self.config.perturb.k_miners),
                    "history_size": int(self.config.perturb.history_size),
                    "min_processed_count": int(self.config.perturb.min_processed_count),
                },
            }
            entity = str(getattr(self.config.perturb, "wandb_entity", "")).strip()
            run_name = str(getattr(self.config.perturb, "wandb_run_name", "")).strip()
            mode = str(getattr(self.config.perturb, "wandb_mode", "online")).strip() or "online"
            if not run_name:
                uid_suffix = self._resolve_validator_uid_for_run_name()
                run_name = f"{time.strftime('%Y%m%d-%H%M%S')}-uid{uid_suffix}"
            if entity:
                init_kwargs["entity"] = entity
            init_kwargs["name"] = run_name
            init_kwargs["mode"] = mode
            self.wandb_run = wandb.init(**init_kwargs)
            self._attach_wandb_console_handler()
            logger.info("W&B logging initialized for validator.")
        except Exception as exc:
            logger.warning(f"Failed to initialize W&B logging: {exc}")
            self.wandb_run = None

    def _resolve_validator_uid_for_run_name(self) -> str:
        hotkey = str(getattr(self.wallet.hotkey, "ss58_address", "") or "")
        try:
            if hotkey and hotkey in self.metagraph.hotkeys:
                return str(self.metagraph.hotkeys.index(hotkey))
        except Exception:
            pass
        return "unknown"

    def _attach_wandb_console_handler(self) -> None:
        if not bool(getattr(self.config.perturb, "wandb_log_console", True)):
            return
        if self._wandb_console_handler is not None:
            return
        handler = _WandbConsoleHandler(self)
        handler.setLevel(pylogging.NOTSET)
        handler.setFormatter(
            pylogging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            )
        )
        pylogging.getLogger().addHandler(handler)
        self._wandb_console_handler = handler

    def _detach_wandb_console_handler(self) -> None:
        if self._wandb_console_handler is None:
            return
        root_logger = pylogging.getLogger()
        root_logger.removeHandler(self._wandb_console_handler)
        self._wandb_console_handler = None

    def _wandb_log(self, payload: dict[str, Any]) -> None:
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log(payload, step=int(self.step))
        except Exception as exc:
            logger.warning(f"W&B log failed: {exc}")

    def _finish_wandb(self) -> None:
        self._detach_wandb_console_handler()
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.finish()
        except Exception as exc:
            logger.warning(f"W&B finish failed: {exc}")
        finally:
            self.wandb_run = None

    def _seed_from_block(self, block: int) -> int:
        digest = hashlib.sha256(f"{C.SUBNET_NAMESPACE}:{self.config.netuid}:{block}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _sample_epsilon(self, seed: int) -> float:
        # Deterministic epsilon in [0.06, 0.2]
        return 0.06 + (seed % 1400) / 10000.0

    def _choose_prompt(self, seed: int) -> str:
        _ = seed  # Keep signature stable; prompt selection is intentionally non-deterministic.
        return self.system_random.choice(list(C.PROMPTS))

    def _parse_llm_endpoint_result(self, payload: Any) -> bool | None:
        if isinstance(payload, bool):
            return payload
        if not isinstance(payload, dict):
            return None

        for key in ("is_match", "match", "ok", "valid"):
            value = payload.get(key)
            if isinstance(value, bool):
                return value
        return None

    def _llm_endpoint_check(self, predicted_label: str, expected_label: str) -> bool:
        endpoint = str(
            getattr(
                self.config.perturb,
                "llm_endpoint_url",
                getattr(self.config.perturb, "label_match_endpoint", ""),
            )
            or ""
        ).strip()
        normalized_prediction = normalize_prediction_label(predicted_label)
        if not endpoint:
            logger.error("LLM endpoint url is empty; rejecting verification check.")
            return False

        payload = {
            "prediction": normalized_prediction,
            "target_label": expected_label,
            "llm_model": str(
                getattr(
                    self.config.perturb,
                    "llm_endpoint_model",
                    getattr(self.config.perturb, "label_match_model", C.LLM_ENDPOINT_MODEL),
                )
            ),
        }
        timeout_seconds = float(
            getattr(self.config.perturb, "llm_endpoint_timeout_seconds", 20)
        )
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            parsed = self._parse_llm_endpoint_result(response.json())
            if parsed is None:
                logger.error("LLM endpoint returned unrecognized payload shape; rejecting check.")
                return False
            return bool(parsed)
        except Exception as exc:
            logger.error(
                f"LLM endpoint request failed ({exc}); timeout={timeout_seconds}s; rejecting check."
            )
            return False

    def _fetch_image_for_prompt(self, prompt: str, seed: int) -> str:
        endpoint = str(self.config.perturb.image_endpoint).strip()
        api_key = str(getattr(self.config.perturb, "pexels_api_key", "")).strip()
        if not api_key:
            raise ValueError("Missing Pexels API key. Set PERTURB_PEXELS_API_KEY in validator env.")
        per_page = max(1, min(80, int(getattr(self.config.perturb, "pexels_per_page", 40))))
        page_span = max(1, int(getattr(self.config.perturb, "pexels_page_span", 10)))
        image_variant = str(getattr(self.config.perturb, "pexels_image_variant", "medium")).strip().lower()
        _ = seed  # Keep signature stable; page/photo sampling is intentionally non-deterministic.
        params = {
            "query": prompt,
            "page": self.system_random.randint(1, page_span),
            "per_page": per_page,
        }
        response = requests.get(
            endpoint,
            params=params,
            headers={"Authorization": api_key},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        photos = data.get("photos") if isinstance(data, dict) else None
        if not isinstance(photos, list) or not photos:
            raise ValueError("Pexels response has no photos for the requested prompt")
        photo = photos[self.system_random.randrange(len(photos))]
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
            raise ValueError("Pexels photo src is missing usable image URL")

        image_response = requests.get(image_url, timeout=12)
        image_response.raise_for_status()
        image_bytes = image_response.content
        if not image_bytes:
            raise ValueError("Downloaded Pexels image is empty")
        return base64.b64encode(image_bytes).decode("utf-8")

    def _load_fallback_image_b64(self) -> str:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        fallback_path = os.path.join(project_root, C.FALLBACK_IMAGE_RELATIVE_PATH)
        with open(fallback_path, "rb") as handle:
            raw = handle.read()
        if not raw:
            raise ValueError(f"fallback image is empty: {fallback_path}")
        return base64.b64encode(raw).decode("utf-8")

    def generate_challenge(self, block: int) -> ChallengeSpec:
        model_name = C.MODEL_NAME
        base_seed = self._seed_from_block(block)
        self._log_step_start(
            "generate_challenge",
            block=block,
            base_seed=base_seed,
            max_attempts=self.config.perturb.max_challenge_attempts,
        )
        for attempt in range(self.config.perturb.max_challenge_attempts):
            seed = base_seed + attempt
            chosen_prompt = self._choose_prompt(seed)
            self._log_summary(
                "challenge_attempt",
                attempt=attempt + 1,
                max_attempts=self.config.perturb.max_challenge_attempts,
                prompt=chosen_prompt,
                seed=seed,
            )
            self._log_step_start(
                "challenge_attempt_internal",
                attempt=attempt + 1,
            )
            try:
                self._log_step_start("challenge_fetch_image", prompt=chosen_prompt, seed=seed)
                image_b64 = self._fetch_image_for_prompt(prompt=chosen_prompt, seed=seed)
                effective_prompt = chosen_prompt
                used_fallback = False
            except Exception as exc:
                logger.warning(f"Challenge image fetch failed ({exc}), using fallback dog image.")
                try:
                    self._log_step_start("challenge_load_fallback_image", label=C.FALLBACK_LABEL)
                    image_b64 = self._load_fallback_image_b64()
                    effective_prompt = C.FALLBACK_LABEL
                    used_fallback = True
                except Exception as fallback_exc:
                    logger.warning(f"Fallback image load failed, retrying: {fallback_exc}")
                    continue

            epsilon = self._sample_epsilon(seed)
            task_id = f"{block}-{seed}"
            self._log_step_start("challenge_prepare", task_id=task_id, epsilon=f"{epsilon:.4f}")

            try:
                self._log_step_start("challenge_model_inference", task_id=task_id)
                image = decode_image_b64(image_b64).to(self.device)
                predicted = predict_label(self.model, image)
                predicted_label = normalize_prediction_label(predicted)
                self._log_summary("challenge_model_output", predicted_label=predicted_label)
            except Exception as exc:
                logger.warning(f"Challenge decode/model validation failed, retrying: {exc}")
                continue

            # Verify the candidate by semantically checking model output against the API prompt label.
            verify_ok = self._llm_endpoint_check(predicted_label, effective_prompt)
            self._log_summary("challenge_llm_verify", passed=verify_ok)
            if not verify_ok:
                logger.info("Sleeping 60s after llm verify-label failure before next challenge attempt.")
                time.sleep(60)
                continue

            return ChallengeSpec(
                task_id=task_id,
                model_name=model_name,
                prompt=effective_prompt,
                clean_image_b64=image_b64,
                # Use exact EfficientNet class label for miner targeting and response verification.
                true_label=predicted_label,
                epsilon=epsilon,
                norm_type="Linf",
                timeout_seconds=self.config.perturb.timeout_seconds,
                fallback_used=used_fallback,
                verified_by_llm=True,
            )

        raise RuntimeError("Unable to build a validated challenge after max attempts")

    def _available_miner_uids(self) -> list[int]:
        my_hotkey = self.wallet.hotkey.ss58_address
        candidate_uids: list[int] = []
        for uid in range(int(self.metagraph.n)):
            if self.metagraph.hotkeys[uid] == my_hotkey:
                continue
            if self.metagraph.axons[uid].ip == "0.0.0.0":
                continue
            candidate_uids.append(uid)

        if len(candidate_uids) <= 1:
            return candidate_uids

        # Merge miners that likely belong to the same operator:
        # - same coldkey OR same axon ip
        # Keep one randomly-chosen uid representative from each merged group
        # so every hotkey in a group gets a fair chance of being queried.
        parent: dict[int, int] = {uid: uid for uid in candidate_uids}

        def _find(uid: int) -> int:
            while parent[uid] != uid:
                parent[uid] = parent[parent[uid]]
                uid = parent[uid]
            return uid

        def _union(a: int, b: int) -> None:
            ra = _find(a)
            rb = _find(b)
            if ra == rb:
                return
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        first_uid_by_coldkey: dict[str, int] = {}
        first_uid_by_ip: dict[str, int] = {}
        coldkeys = getattr(self.metagraph, "coldkeys", [])
        for uid in candidate_uids:
            coldkey = ""
            if uid < len(coldkeys):
                coldkey = str(coldkeys[uid] or "").strip()
            if coldkey:
                seen_uid = first_uid_by_coldkey.get(coldkey)
                if seen_uid is None:
                    first_uid_by_coldkey[coldkey] = uid
                else:
                    _union(seen_uid, uid)

            ip = str(getattr(self.metagraph.axons[uid], "ip", "") or "").strip()
            if ip:
                seen_uid = first_uid_by_ip.get(ip)
                if seen_uid is None:
                    first_uid_by_ip[ip] = uid
                else:
                    _union(seen_uid, uid)

        uids_by_group: dict[int, list[int]] = {}
        for uid in candidate_uids:
            root = _find(uid)
            uids_by_group.setdefault(root, []).append(uid)

        selected: list[int] = []
        for group_uids in uids_by_group.values():
            selected.append(self.system_random.choice(group_uids))

        return sorted(selected)

    def _valuable_miner_uids(self, candidate_uids: Sequence[int]) -> list[int]:
        min_processed = int(self.config.perturb.min_processed_count)
        return [uid for uid in candidate_uids if int(self.processed_counts[uid]) >= min_processed]

    def _select_random_miners(self, candidate_uids: Sequence[int], seed: int) -> list[int]:
        if not candidate_uids:
            return []
        valuable = self._valuable_miner_uids(candidate_uids)
        pool = list(candidate_uids)
        k = min(int(self.config.perturb.k_miners), len(pool))
        rng = random.Random(seed)
        if k <= 0:
            return []
        if not valuable:
            return sorted(rng.sample(pool, k=k))

        valuable_set = set(valuable)
        newcomers = [uid for uid in pool if uid not in valuable_set]
        ratio = float(max(0.0, min(1.0, self.config.perturb.miner_exploration_ratio)))
        explore_k = min(len(newcomers), int(round(k * ratio)))
        if newcomers and ratio > 0.0 and explore_k == 0:
            explore_k = 1
        exploit_k = min(len(valuable), k - explore_k)
        if exploit_k + explore_k < k:
            explore_k = min(len(newcomers), explore_k + (k - (exploit_k + explore_k)))

        selected: list[int] = []
        if exploit_k > 0:
            selected.extend(rng.sample(list(valuable), k=exploit_k))
        if explore_k > 0:
            selected.extend(rng.sample(newcomers, k=explore_k))

        if len(selected) < k:
            remaining = [uid for uid in pool if uid not in set(selected)]
            selected.extend(rng.sample(remaining, k=min(k - len(selected), len(remaining))))
        return sorted(selected)

    async def _query_miners(self, uids: Sequence[int], challenge: ChallengeSpec):
        self._log_step_start(
            "query_miners",
            task_id=challenge.task_id,
            miner_count=len(uids),
            timeout=challenge.timeout_seconds,
        )
        axons = [self.metagraph.axons[uid] for uid in uids]
        synapse = AttackChallenge(
            task_id=challenge.task_id,
            model_name=challenge.model_name,
            prompt=challenge.prompt,
            clean_image_b64=challenge.clean_image_b64,
            true_label=challenge.true_label,
            epsilon=challenge.epsilon,
            norm_type=challenge.norm_type,
            min_delta=self.config.perturb.min_linf_delta,
            timeout_seconds=challenge.timeout_seconds,
        )
        responses = await self.dendrite.forward(
            axons=axons,
            synapse=synapse,
            deserialize=False,
            timeout=challenge.timeout_seconds,
        )
        return responses

    def _run_query_miners(self, uids: Sequence[int], challenge: ChallengeSpec):
        # Keep a persistent event loop for dendrite calls; asyncio.run() closes
        # the loop each call and can trigger "Event loop is closed" on reuse.
        if self._query_loop.is_closed():
            self._query_loop = asyncio.new_event_loop()
        return self._query_loop.run_until_complete(self._query_miners(uids, challenge))

    def verify_and_score(
        self,
        challenge: ChallengeSpec,
        perturbed_image_b64: str,
        response_time_ms: int,
    ) -> EvaluationResult:
        try:
            x_clean = decode_image_b64(challenge.clean_image_b64).to(self.device)
            x_adv = decode_image_b64(perturbed_image_b64).to(self.device)
        except Exception as exc:
            return EvaluationResult(score=0.0, reason=f"decode_failed:{exc}", response_time_ms=response_time_ms)

        if x_adv.shape != x_clean.shape:
            return EvaluationResult(score=0.0, reason="shape_mismatch", response_time_ms=response_time_ms)
        if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
            return EvaluationResult(score=0.0, reason="value_out_of_range", response_time_ms=response_time_ms)

        prediction = ""
        try:
            prediction = predict_label(self.model, x_adv)
        except Exception as exc:
            return EvaluationResult(
                score=0.0,
                reason=f"model_inference_failed:{exc}",
                response_time_ms=response_time_ms,
            )

        if challenge.norm_type == "Linf":
            norm = (x_adv - x_clean).abs().max().item()
        elif challenge.norm_type == "L2":
            norm = float((x_adv - x_clean).norm(2).item())
        else:
            norm = float((x_adv - x_clean).ne(0).sum().item())

        if norm < self.config.perturb.min_linf_delta:
            return EvaluationResult(
                score=0.0,
                reason="below_min_delta",
                model_prediction=prediction,
                response_time_ms=response_time_ms,
                norm=float(norm),
                epsilon=float(challenge.epsilon),
            )
        effective_max_delta = min(float(challenge.epsilon), float(self.config.perturb.max_linf_delta))
        if norm > effective_max_delta:
            return EvaluationResult(
                score=0.0,
                reason="above_max_delta",
                model_prediction=prediction,
                response_time_ms=response_time_ms,
                norm=float(norm),
                rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
                epsilon=float(challenge.epsilon),
            )

        normalized_prediction = normalize_prediction_label(prediction)
        # Successful perturbation means the response label changes from original model output.
        if normalized_prediction == challenge.true_label:
            return EvaluationResult(
                score=0.0,
                reason="label_match_with_original",
                model_prediction=normalized_prediction,
                response_time_ms=response_time_ms,
                norm=float(norm),
                rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
                epsilon=float(challenge.epsilon),
            )

        rmse = float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item())

        ssim = _compute_ssim(x_clean=x_clean, x_adv=x_adv)
        min_ssim = float(getattr(self.config.perturb, "min_ssim", 0.98))
        if ssim < min_ssim:
            return EvaluationResult(
                score=0.0,
                reason="below_min_ssim",
                model_prediction=normalized_prediction,
                response_time_ms=response_time_ms,
                norm=float(norm),
                rmse=float(rmse),
                epsilon=float(challenge.epsilon),
                ssim=float(ssim),
            )

        psnr_db = _compute_psnr_db(x_clean=x_clean, x_adv=x_adv)
        min_psnr_db = float(getattr(self.config.perturb, "min_psnr_db", 0.0))
        if min_psnr_db > 0.0 and psnr_db < min_psnr_db:
            return EvaluationResult(
                score=0.0,
                reason="below_min_psnr_db",
                model_prediction=normalized_prediction,
                response_time_ms=response_time_ms,
                norm=float(norm),
                rmse=float(rmse),
                epsilon=float(challenge.epsilon),
                ssim=float(ssim),
                psnr_db=float(psnr_db),
            )

        denom = max(1e-12, effective_max_delta - float(self.config.perturb.min_linf_delta))
        linf_ratio = (norm - float(self.config.perturb.min_linf_delta)) / denom
        linf_ratio = min(max(linf_ratio, 0.0), 1.0)
        linf_score = (1.0 - linf_ratio) ** 2

        rmse_ratio = rmse / max(1e-12, effective_max_delta)
        rmse_ratio = min(max(rmse_ratio, 0.0), 1.0)
        rmse_score = (1.0 - rmse_ratio) ** 2

        linf_weight = float(getattr(self.config.perturb, "linf_component_weight", 0.7))
        rmse_weight = float(getattr(self.config.perturb, "rmse_component_weight", 0.3))
        total_weight = max(1e-12, linf_weight + rmse_weight)
        perturbation_score = ((linf_weight * linf_score) + (rmse_weight * rmse_score)) / total_weight

        time_ratio = response_time_ms / (challenge.timeout_seconds * 1000.0)
        speed_score = 1.0 - min(time_ratio, 1.0)

        score = C.PERTURBATION_WEIGHT * perturbation_score + C.SPEED_WEIGHT * speed_score
        return EvaluationResult(
            score=float(score),
            reason="success",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=float(norm),
            rmse=float(rmse),
            epsilon=float(challenge.epsilon),
            ssim=float(ssim),
            psnr_db=float(psnr_db),
        )

    def _update_histories(self, uids: Sequence[int], rewards: Sequence[float]) -> None:
        for uid, reward in zip(uids, rewards):
            self.processed_counts[uid] += 1
            self.score_histories[uid].append(float(reward))

    def _set_weights(self) -> None:
        self._log_step_start(
            "set_weights",
            min_processed=self.config.perturb.min_processed_count,
            history_size=self.config.perturb.history_size,
        )
        eligible: list[tuple[int, float]] = []
        history_size = int(self.config.perturb.history_size)
        min_processed = int(self.config.perturb.min_processed_count)
        for uid in range(int(self.metagraph.n)):
            if int(self.processed_counts[uid]) < min_processed:
                continue
            history = self.score_histories[uid]
            if len(history) < history_size:
                continue
            tail = history[-history_size:]
            avg_score = float(sum(tail) / history_size)
            eligible.append((uid, avg_score))

        if not eligible:
            logger.warning(f"No eligible miners with processed_count >= {min_processed}.")
            return

        eligible.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        n_eligible = len(eligible)
        emission_raw = np.zeros(int(self.metagraph.n), dtype=np.float32)

        rank_to_uid: dict[int, int] = {}
        for rank0, (uid, avg_score) in enumerate(eligible):
            rank = rank0 + 1
            rank_to_uid[rank] = uid

        # Fixed top-3 emission schedule; ranks 4+ intentionally receive zero.
        top3_shares = (0.70, 0.29, 0.01)
        for rank, share in enumerate(top3_shares, start=1):
            if rank <= n_eligible:
                emission_raw[rank_to_uid[rank]] = float(share)

        # Only miners with positive average score may receive non-zero emissions.
        positive_uids = [uid for uid, avg_score in eligible if avg_score > 0.0]
        if not positive_uids:
            logger.warning("No miners with positive average score; setting all weights to zero.")
            zero_weights = np.zeros(int(self.metagraph.n), dtype=np.float32)
            uids = list(range(len(zero_weights)))
            ok, msg = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uids,
                weights=[float(v) for v in zero_weights.tolist()],
                wait_for_inclusion=False,
                wait_for_finalization=False,
            )
            if ok:
                logger.info("set_weights success (all zero)")
            else:
                logger.error(f"set_weights failed (all zero): {msg}")
            return
        active_emission_total = float(sum(float(emission_raw[uid]) for uid in positive_uids))
        if active_emission_total <= 0.0:
            logger.warning("No positive-score miners in weighted rank buckets; setting all weights to zero.")
            zero_weights = np.zeros(int(self.metagraph.n), dtype=np.float32)
            uids = list(range(len(zero_weights)))
            ok, msg = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uids,
                weights=[float(v) for v in zero_weights.tolist()],
                wait_for_inclusion=False,
                wait_for_finalization=False,
            )
            if ok:
                logger.info("set_weights success (all zero)")
            else:
                logger.error(f"set_weights failed (all zero): {msg}")
            return

        normalized = np.zeros(int(self.metagraph.n), dtype=np.float32)
        for uid in positive_uids:
            normalized[uid] = float(emission_raw[uid]) / active_emission_total
        for rank0, (uid, avg_score) in enumerate(eligible[:10]):
            rank = rank0 + 1
            logger.debug(
                f"rank={rank} uid={uid} avg_score={avg_score:.6f} emission_raw={emission_raw[uid]:.6f} emission={normalized[uid]:.6f}"
            )
        top_weight_items: list[str] = []
        for rank, (uid, avg_score) in enumerate(eligible[:10], start=1):
            top_weight_items.append(f"r{rank}:uid{uid}:avg={avg_score:.4f}:w={normalized[uid]:.4f}")
        self._log_summary(
            "weights_summary",
            eligible=n_eligible,
            distributed=min(10, n_eligible),
            top10="|".join(top_weight_items) if top_weight_items else "none",
        )

        # Scale miner emissions by configured share; route remainder to uid 0.
        miner_share = float(min(max(self.miner_emission_share, 0.0), 1.0))
        scaled = normalized * miner_share
        remainder = 1.0 - miner_share
        if len(scaled) > 0:
            scaled[0] = remainder

        uids = list(range(len(scaled)))
        weights = [float(v) for v in scaled.tolist()]
        ok, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=False,
            wait_for_finalization=False,
        )
        if ok:
            logger.info("set_weights success")
        else:
            logger.error(f"set_weights failed: {msg}")

    def run(self) -> None:
        self._log_step_start("validator_boot")
        self.sync()
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Validator hotkey is not registered on this netuid.")

        tempo = self.subtensor.get_subnet_hyperparameters(self.config.netuid).tempo
        self._log_summary(
            "validator_config",
            timeout=self.config.perturb.timeout_seconds,
            k_miners=self.config.perturb.k_miners,
            history_size=self.config.perturb.history_size,
            min_processed=self.config.perturb.min_processed_count,
            min_linf=self.config.perturb.min_linf_delta,
            max_linf=self.config.perturb.max_linf_delta,
            min_ssim=self.config.perturb.min_ssim,
            min_psnr_db=self.config.perturb.min_psnr_db,
            perturb_weight=C.PERTURBATION_WEIGHT,
            speed_weight=C.SPEED_WEIGHT,
            tempo=tempo,
            run_id=self.run_id,
        )

        while True:
            try:
                self._log_step_start("loop_sync_metagraph")
                self.sync()
                self._log_step_start("loop_get_current_block")
                block = self.subtensor.get_current_block()
                self._log_step_start("loop_generate_challenge", block=block)
                challenge = self.generate_challenge(block=block)
                self._log_summary(
                    "challenge_summary",
                    task_id=challenge.task_id,
                    prompt=challenge.prompt,
                    epsilon=f"{challenge.epsilon:.4f}",
                    true_label=challenge.true_label,
                    llm_verified=challenge.verified_by_llm,
                    fallback_used=challenge.fallback_used,
                )

                self._log_step_start("loop_discover_miners")
                available_uids = self._available_miner_uids()
                if not available_uids:
                    logger.warning("No miners available")
                    time.sleep(self.config.perturb.query_interval_seconds)
                    continue
                self._log_step_start("loop_select_miners", candidate_count=len(available_uids))
                valuable_uids = self._valuable_miner_uids(available_uids)
                miner_uids = self._select_random_miners(available_uids, seed=self._seed_from_block(block))
                if not miner_uids:
                    logger.warning("Miner selection is empty")
                    time.sleep(self.config.perturb.query_interval_seconds)
                    continue
                self._log_summary(
                    "miner_selection",
                    selected=len(miner_uids),
                    valuable_pool=len(valuable_uids),
                    total_pool=len(available_uids),
                )

                self._log_step_start("loop_query_miners", selected_count=len(miner_uids))
                responses = self._run_query_miners(miner_uids, challenge)
                self._log_step_start("loop_score_responses", response_count=len(responses))
                rewards: list[float] = []
                results_by_uid: list[tuple[int, EvaluationResult]] = []
                response_log_lines: list[str] = []
                for uid, response in zip(miner_uids, responses):
                    status_code = getattr(response.dendrite, "status_code", 0) if response.dendrite else 0
                    process_time = getattr(response.dendrite, "process_time", None) if response.dendrite else None
                    response_time_ms = int((process_time or challenge.timeout_seconds) * 1000)

                    if status_code != 200 or not response.perturbed_image_b64:
                        result = EvaluationResult(
                            score=0.0,
                            reason="response_missing_or_status_error",
                            model_prediction="unavailable",
                            response_time_ms=response_time_ms,
                        )
                    else:
                        result = self.verify_and_score(
                            challenge=challenge,
                            perturbed_image_b64=response.perturbed_image_b64,
                            response_time_ms=response_time_ms,
                        )
                    score = float(result.score)
                    rewards.append(score)
                    results_by_uid.append((uid, result))
                    self.reason_counts_total[result.reason] += 1
                    response_log_lines.append(
                        f"uid={uid} status={status_code} score={score:.6f} "
                        f"response_time_ms={result.response_time_ms} "
                        f"processed={int(self.processed_counts[uid]) + 1} "
                        f"reason={result.reason} "
                        f"norm={result.norm:.6f} rmse={result.rmse:.6f} epsilon={result.epsilon:.6f} "
                        f"ssim={result.ssim:.6f} psnr_db={result.psnr_db:.4f}"
                    )

                all_zero_scores = bool(rewards) and all(score <= 0.0 for score in rewards)

                self._log_step_start("loop_update_histories")
                if all_zero_scores:
                    logger.warning(
                        "Skipping history update because all selected miner scores are zero "
                        f"(block={block}, selected={len(miner_uids)})."
                    )
                else:
                    if response_log_lines:
                        logger.info(
                            f"miner_response_evaluations block={block} count={len(response_log_lines)}\n"
                            + "\n".join(response_log_lines)
                        )
                    self._update_histories(miner_uids, rewards)
                    reason_counts = Counter(result.reason for _, result in results_by_uid)
                    success_count = int(reason_counts.get("success", 0))
                    avg_score = float(sum(rewards) / max(1, len(rewards)))
                    max_score = float(max(rewards)) if rewards else 0.0
                    min_score = float(min(rewards)) if rewards else 0.0
                    avg_norm = float(sum(result.norm for _, result in results_by_uid) / max(1, len(results_by_uid)))
                    avg_rmse = float(sum(result.rmse for _, result in results_by_uid) / max(1, len(results_by_uid)))
                    self._log_summary(
                        "loop_summary",
                        block=block,
                        selected=len(miner_uids),
                        success=f"{success_count}/{len(results_by_uid)}",
                        avg_score=f"{avg_score:.4f}",
                        min_score=f"{min_score:.4f}",
                        max_score=f"{max_score:.4f}",
                        avg_norm=f"{avg_norm:.5f}",
                        avg_rmse=f"{avg_rmse:.5f}",
                        reasons=",".join([f"{k}:{v}" for k, v in sorted(reason_counts.items())]),
                    )
                
                self._log_step_start("loop_save_state")
                self._save_state()

                blocks_since_weights = block - self.last_weight_block
                if blocks_since_weights >= tempo:
                    self._log_step_start("loop_maybe_set_weights", blocks_since_weights=blocks_since_weights, tempo=tempo)
                    self._set_weights()
                    self.last_weight_block = block

                self.step += 1
                self._log_step_start("loop_sleep", seconds=self.config.perturb.query_interval_seconds)
                time.sleep(self.config.perturb.query_interval_seconds)
            except KeyboardInterrupt:
                logger.info("Validator stopped by user.")
                break
            except Exception as exc:
                logger.error(f"Validator loop error: {exc}")
                time.sleep(5)
        if not self._query_loop.is_closed():
            self._query_loop.close()
        self._finish_wandb()


def build_config() -> bt.config:
    parser = argparse.ArgumentParser(description="Perturb subnet validator")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument("--wallet.name", dest="wallet_name", type=str, default=os.getenv("WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", type=str, default=os.getenv("HOTKEY_NAME", "default"))
    parser.add_argument("--logging-dir", dest="logging_dir", type=str, default=os.getenv("LOGGING_DIR", "./logs"))
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "DEBUG"))
    if hasattr(bt, "config"):
        config = bt.config(parser)
    else:
        config = parser.parse_args()

    if not hasattr(config, "wallet"):
        config.wallet = type("WalletConfig", (), {})()
    config.wallet.name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    config.wallet.hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))

    if not hasattr(config, "subtensor"):
        config.subtensor = type("SubtensorConfig", (), {})()
    config.subtensor.network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))

    if not hasattr(config, "logging"):
        config.logging = type("LoggingConfig", (), {})()
    config.logging.logging_dir = getattr(config.logging, "logging_dir", getattr(config, "logging_dir", "./logs"))
    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))

    perturb_cfg = type("PerturbConfig", (), {})()
    config.perturb = perturb_cfg
    for key, value in C.VALIDATOR_CONFIG.items():
        setattr(config.perturb, key, value)
    return config


if __name__ == "__main__":
    validator = PerturbValidator(config=build_config())
    validator.run()
    