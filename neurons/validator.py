from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging as pylogging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import bittensor as bt
import numpy as np
import requests
import torch
import torch.nn.functional as F

from perturbnet import constants as C
from perturbnet.api_client import SubmittedResponse, get_current_task, get_submitted_responses
from perturbnet.duplicate_responses import zero_duplicate_responses
from perturbnet.emissions import ranked_emission_shares
from perturbnet.image_io import changed_pixel_count, decode_image_b64, image_url_to_b64, quantize_image_uint8_grid
from perturbnet.leaderboard_payload import build_report, update_score_histories
from perturbnet.leaderboard_reporter import LeaderboardReporter
from perturbnet.metagraph_utils import miner_uids as metagraph_miner_uids
from perturbnet.model import (
    label_for_index,
    load_efficientnet_v2_l,
    logits_for_images,
    normalize_prediction_label,
    predict_label,
    resolve_target_index,
)
from perturbnet.task_timing import sleep_until_next_task_boundary

logger = pylogging.getLogger(__name__)


@dataclass
class ChallengeSpec:
    task_id: str
    image_id: str
    model_name: str
    clean_image_b64: str
    true_label: str
    epsilon: float
    norm_type: str


@dataclass
class EvaluationResult:
    score: float
    reason: str
    model_prediction: str = ""
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


class PerturbValidator:
    def __init__(self, config: bt.config) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = _make_subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_efficientnet_v2_l(self.device)
        self.step = 0
        self.last_weight_block = 0
        self.state_path = os.path.join(self.config.logging.logging_dir, C.VALIDATOR_STATE_FILENAME)
        hotkey = getattr(self.wallet.hotkey, "ss58_address", "unknown")
        self.run_id = f"{str(hotkey)[:8]}-n{self.config.netuid}-p{os.getpid()}"
        self.leaderboard_reporter = LeaderboardReporter(
            enabled=bool(getattr(self.config.perturb, "leaderboard_reporting_enabled", True)),
            api_url=str(getattr(self.config.perturb, "leaderboard_api_url", C.LEADERBOARD_API_URL)),
            last_weight_update_api_url=str(
                getattr(
                    self.config.perturb,
                    "leaderboard_last_weight_update_api_url",
                    C.LEADERBOARD_LAST_WEIGHT_UPDATE_API_URL,
                )
            ),
            api_key=str(getattr(self.config.perturb, "api_key", C.PERTURB_API_KEY)),
            timeout_seconds=float(getattr(self.config.perturb, "leaderboard_report_timeout_seconds", 10.0)),
            wallet=self.wallet,
            validator_hotkey=str(hotkey),
        )
        self.reason_counts_total: Counter[str] = Counter()
        self.miner_emission_share = 1
        self.last_validated_api_task_id = ""

        self.processed_counts = np.zeros(int(self.metagraph.n), dtype=np.int32)
        self.score_histories: list[list[float]] = [[] for _ in range(int(self.metagraph.n))]
        # Separate leaderboard buffer persisted under its own key; its window
        # intentionally follows HISTORY_SIZE so operators have one history knob.
        self.leaderboard_score_histories: list[list[float]] = [[] for _ in range(int(self.metagraph.n))]
        self.uid_hotkeys: list[str] = list(self.metagraph.hotkeys[: int(self.metagraph.n)])

        self._load_state()

    def _api_key(self) -> str:
        return str(getattr(self.config.perturb, "api_key", C.PERTURB_API_KEY)).strip()

    def _ensure_api_key(self) -> None:
        if not self._api_key():
            msg = "PERTURB_API_KEY is not set; validator cannot fetch submitted responses."
            logger.error(msg)
            raise RuntimeError(msg)

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
            if new_n > len(self.leaderboard_score_histories):
                self.leaderboard_score_histories.extend(
                    [[] for _ in range(new_n - len(self.leaderboard_score_histories))]
                )
            else:
                self.leaderboard_score_histories = self.leaderboard_score_histories[:new_n]
            if new_n > len(self.uid_hotkeys):
                self.uid_hotkeys.extend([""] * (new_n - len(self.uid_hotkeys)))
            else:
                self.uid_hotkeys = self.uid_hotkeys[:new_n]
        self._reconcile_uid_identities()

    def _reset_uid_stats(self, uid: int, reason: str) -> None:
        self.processed_counts[uid] = 0
        self.score_histories[uid] = []
        self.leaderboard_score_histories[uid] = []
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
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except Exception as exc:
            # Saves are atomic, so this should never happen; never brick startup on it.
            logger.error(f"Validator state file is unreadable, starting fresh: {self.state_path} ({exc})")
            return
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

        leaderboard_window = int(getattr(self.config.perturb, "history_size", C.HISTORY_SIZE))
        saved_leaderboard_histories = state.get("leaderboard_score_histories", [])
        copied_lh = min(len(saved_leaderboard_histories), len(self.leaderboard_score_histories))
        for idx in range(copied_lh):
            raw = saved_leaderboard_histories[idx]
            if isinstance(raw, list):
                self.leaderboard_score_histories[idx] = [float(x) for x in raw[-leaderboard_window:]]

        saved_hotkeys = state.get("uid_hotkeys", [])
        if isinstance(saved_hotkeys, list):
            copied_keys = min(len(saved_hotkeys), len(self.uid_hotkeys))
            for idx in range(copied_keys):
                value = saved_hotkeys[idx]
                if isinstance(value, str):
                    self.uid_hotkeys[idx] = value
        self.last_validated_api_task_id = str(state.get("last_validated_api_task_id", "") or "")
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
            "leaderboard_score_histories": [
                history[-int(getattr(self.config.perturb, "history_size", C.HISTORY_SIZE)) :]
                for history in self.leaderboard_score_histories
            ],
            "uid_hotkeys": self.uid_hotkeys,
            "last_validated_api_task_id": self.last_validated_api_task_id,
        }
        tmp_path = f"{self.state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.state_path)

    def challenge_from_task_api(self, *, task_id: str, image_url: str) -> ChallengeSpec:
        image_b64 = image_url_to_b64(
            image_url,
            timeout_seconds=float(getattr(self.config.perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS)),
        )
        image = decode_image_b64(image_b64).to(self.device)
        predicted_label = normalize_prediction_label(predict_label(self.model, image))
        return ChallengeSpec(
            task_id=task_id,
            image_id=task_id,
            model_name=C.MODEL_NAME,
            clean_image_b64=image_b64,
            true_label=predicted_label,
            epsilon=float(getattr(self.config.perturb, "max_linf_delta", C.MAX_LINF_DELTA)),
            norm_type="Linf",
        )

    def verify_and_score(
        self,
        challenge: ChallengeSpec,
        perturbed_image_b64: str,
        novelty_pixel_count: int | None = None,
    ) -> EvaluationResult:
        try:
            x_clean = quantize_image_uint8_grid(decode_image_b64(challenge.clean_image_b64).to(self.device))
            x_adv = quantize_image_uint8_grid(decode_image_b64(perturbed_image_b64).to(self.device))
        except Exception as exc:
            return EvaluationResult(score=0.0, reason=f"decode_failed:{exc}")

        if x_adv.shape != x_clean.shape:
            return EvaluationResult(score=0.0, reason="shape_mismatch")
        if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
            return EvaluationResult(score=0.0, reason="value_out_of_range")

        prediction = ""
        prediction_index: int | None = None
        true_index = resolve_target_index(challenge.true_label)
        logits: torch.Tensor | None = None
        try:
            with torch.no_grad():
                logits = logits_for_images(self.model, x_adv.unsqueeze(0))
            prediction_index = int(logits.argmax(dim=1).item())
            prediction = label_for_index(prediction_index)
        except Exception as exc:
            return EvaluationResult(
                score=0.0,
                reason=f"model_inference_failed:{exc}",
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
                norm=float(norm),
                epsilon=float(challenge.epsilon),
            )
        effective_max_delta = min(float(challenge.epsilon), float(self.config.perturb.max_linf_delta))
        if norm > effective_max_delta:
            return EvaluationResult(
                score=0.0,
                reason="above_max_delta",
                model_prediction=prediction,
                norm=float(norm),
                rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
                epsilon=float(challenge.epsilon),
            )

        normalized_prediction = normalize_prediction_label(prediction)
        # Successful perturbation means the response label changes from original model output.
        label_matches_original = (
            prediction_index == true_index
            if prediction_index is not None and true_index is not None
            else normalized_prediction == normalize_prediction_label(challenge.true_label)
        )
        if label_matches_original:
            return EvaluationResult(
                score=0.0,
                reason="label_match_with_original",
                model_prediction=normalized_prediction,
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

        margin_score = 0.0
        if logits is not None and true_index is not None and 0 <= true_index < logits.shape[1]:
            row = logits[0]
            true_logit = float(row[true_index].item())
            masked = row.clone()
            masked[true_index] = -float("inf")
            best_non_true_logit = float(masked.max().item())
            margin = best_non_true_logit - true_logit
            margin_score = min(max(margin / 10.0, 0.0), 1.0)

        if novelty_pixel_count is None:
            novelty_pixel_count = changed_pixel_count(x_clean=x_clean, x_adv=x_adv)
        novelty_target = max(1, int(getattr(self.config.perturb, "analyze_bucket_novelty_target_pixels", 8)))
        novelty_score = min(max(float(novelty_pixel_count) / float(novelty_target), 0.0), 1.0)

        score = (
            C.PERTURBATION_WEIGHT * perturbation_score
            + float(getattr(self.config.perturb, "analyze_bucket_margin_weight", 0.03)) * margin_score
            + float(getattr(self.config.perturb, "analyze_bucket_novelty_weight", 0.01)) * novelty_score
        )
        return EvaluationResult(
            score=float(score),
            reason="success",
            model_prediction=normalized_prediction,
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

    def _update_leaderboard_histories(self, uids: Sequence[int], rewards: Sequence[float]) -> None:
        update_score_histories(
            self.leaderboard_score_histories,
            uids,
            rewards,
            int(getattr(self.config.perturb, "history_size", C.HISTORY_SIZE)),
        )

    def _submit_leaderboard_report(
        self,
        *,
        challenge: ChallengeSpec,
        results_by_uid: Sequence[tuple[int, EvaluationResult]],
        image_url_by_uid: dict[int, str],
    ) -> None:
        self.leaderboard_reporter.submit(
            build_report(
                task_id=challenge.task_id,
                validator_hotkey=str(getattr(self.wallet.hotkey, "ss58_address", "")),
                score_histories=self.leaderboard_score_histories,
                avg_window=int(getattr(self.config.perturb, "history_size", C.HISTORY_SIZE)),
                results_by_uid=results_by_uid,
                image_url_by_uid=image_url_by_uid,
            )
        )

    def _leaderboard_results_for_all_miners(
        self,
        results_by_uid: Sequence[tuple[int, EvaluationResult]],
    ) -> list[tuple[int, EvaluationResult]]:
        selected_results = {uid: result for uid, result in results_by_uid}
        report_rows: list[tuple[int, EvaluationResult]] = []
        for uid in self._leaderboard_miner_uids():
            result = selected_results.get(uid)
            if result is None:
                result = EvaluationResult(
                    score=0.0,
                    reason="leaderboard_unavailable",
                    model_prediction="unavailable",
                )
            report_rows.append((uid, result))
        return report_rows

    def _leaderboard_miner_uids(self) -> list[int]:
        own_hotkey = str(getattr(self.wallet.hotkey, "ss58_address", "") or "")
        own_uid = self.metagraph.hotkeys.index(own_hotkey) if own_hotkey in self.metagraph.hotkeys else None
        return [uid for uid in metagraph_miner_uids(self.metagraph) if own_uid is None or uid != own_uid]

    def _wait_for_next_task_boundary(self) -> None:
        cadence = int(getattr(self.config.perturb, "task_cadence_seconds", C.TASK_CADENCE_SECONDS))
        slept = sleep_until_next_task_boundary(cadence_seconds=cadence)
        logger.info(f"Reached task boundary after waiting {slept:.2f}s")

    def _wait_before_task_fetch(self) -> None:
        delay = float(
            getattr(
                self.config.perturb,
                "validator_evaluation_delay_seconds",
                C.VALIDATOR_EVALUATION_DELAY_SECONDS,
            )
        )
        logger.info(f"Waiting {delay:.2f}s after task boundary before fetching task")
        time.sleep(max(0.0, delay))

    def _fetch_current_task(self):
        retries = int(getattr(self.config.perturb, "task_fetch_retries", C.TASK_FETCH_RETRIES))
        retry_seconds = float(getattr(self.config.perturb, "task_fetch_retry_seconds", C.TASK_FETCH_RETRY_SECONDS))
        for attempt in range(1, retries + 1):
            try:
                task = get_current_task(
                    base_url=str(getattr(self.config.perturb, "api_base_url", C.PERTURB_API_BASE_URL)),
                    timeout_seconds=float(
                        getattr(self.config.perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS)
                    ),
                )
            except Exception as exc:
                logger.warning(f"Task API request failed attempt={attempt}/{retries}: {exc}")
                task = None
            if task is not None:
                return task
            logger.info(f"No task available attempt={attempt}/{retries}")
            if attempt < retries:
                time.sleep(retry_seconds)
        return None

    def _wait_for_submitted_responses(self, *, task_id: str) -> list[SubmittedResponse] | None:
        poll_seconds = float(
            getattr(
                self.config.perturb,
                "validator_evaluation_poll_seconds",
                C.VALIDATOR_EVALUATION_POLL_SECONDS,
            )
        )
        retries = int(
            getattr(
                self.config.perturb,
                "validator_evaluation_poll_retries",
                C.VALIDATOR_EVALUATION_POLL_RETRIES,
            )
        )
        base_url = str(getattr(self.config.perturb, "api_base_url", C.PERTURB_API_BASE_URL))
        api_key = self._api_key()
        timeout_seconds = float(
            getattr(self.config.perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS)
        )

        for attempt in range(1, retries + 1):
            try:
                submitted_responses = get_submitted_responses(
                    base_url=base_url,
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                logger.warning(
                    f"Submitted responses request failed task_id={task_id} attempt={attempt}/{retries}: {exc}"
                )
                submitted_responses = []
            if submitted_responses:
                logger.info(
                    f"Submitted responses available task_id={task_id} count={len(submitted_responses)} "
                    f"attempt={attempt}/{retries}"
                )
                return submitted_responses
            logger.info(f"No submitted responses yet task_id={task_id} attempt={attempt}/{retries}")
            if attempt < retries:
                time.sleep(max(0.1, poll_seconds))
        logger.warning(f"No submitted responses after {retries} attempts task_id={task_id}")
        return None

    def _fallback_burn_rate(self) -> float:
        return min(max(float(getattr(self.config.perturb, "default_burn_rate", 0.0)), 0.0), 1.0)

    def _fetch_burn_rate(self) -> float:
        endpoint = str(getattr(self.config.perturb, "burn_rate_endpoint", C.BURN_RATE_ENDPOINT)).strip()
        timeout = float(getattr(self.config.perturb, "burn_rate_fetch_timeout_seconds", 5.0))
        fallback = self._fallback_burn_rate()
        if not endpoint:
            logger.warning(f"Burn rate endpoint is empty; using fallback burn={fallback:.4f}")
            return fallback
        try:
            response = requests.get(endpoint, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            raw_burn = payload.get("burnRate") if isinstance(payload, dict) else None
            burn = float(raw_burn)
            if not math.isfinite(burn) or burn < 0.0 or burn > 1.0:
                raise ValueError(f"burnRate must be in [0,1], got {raw_burn!r}")
            return min(max(burn, 0.0), 1.0)
        except Exception as exc:
            # Burn is policy-controlled by the API, but weight setting must never
            # fail or use garbage data because the endpoint is slow/unavailable.
            logger.warning(f"Failed to fetch burn rate from {endpoint}; using fallback burn={fallback:.4f}: {exc}")
            return fallback

    def _set_weights(self, *, block: int) -> bool:
        self._log_step_start(
            "set_weights",
            history_size=self.config.perturb.history_size,
        )
        eligible: list[tuple[int, float]] = []
        history_size = int(self.config.perturb.history_size)
        min_history = int(
            getattr(self.config.perturb, "min_weight_history_size", C.MIN_WEIGHT_HISTORY_SIZE)
        )
        burn_uid = int(getattr(self.config.perturb, "burn_uid", 0))
        if burn_uid < 0 or burn_uid >= int(self.metagraph.n):
            logger.warning(f"Configured burn_uid={burn_uid} is outside metagraph; falling back to UID 0.")
            burn_uid = 0
        # Until any miner reaches the full history window, eligibility follows
        # the longest available history so new validators can set weights
        # early without letting short-history miners qualify prematurely.
        longest_history = max(
            (len(self.score_histories[uid]) for uid in range(int(self.metagraph.n)) if uid != burn_uid),
            default=0,
        )
        effective_history_size = min(history_size, longest_history)
        if effective_history_size < min_history:
            logger.warning(
                f"Longest miner history {longest_history} is below the minimum {min_history}; "
                "skipping weight setting."
            )
            return False
        if effective_history_size < history_size:
            logger.info(
                f"Using effective_history_size={effective_history_size} "
                f"(longest available history) instead of history_size={history_size}."
            )
        burn_rate = self._fetch_burn_rate()
        for uid in range(int(self.metagraph.n)):
            if uid == burn_uid:
                continue
            history = self.score_histories[uid]
            if len(history) < effective_history_size:
                continue
            tail = history[-effective_history_size:]
            avg_score = float(sum(tail) / effective_history_size)
            eligible.append((uid, avg_score))

        if not eligible:
            logger.warning(f"No eligible miners with history_size={effective_history_size}.")
            return False

        eligible.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        n_eligible = len(eligible)
        emission_raw = np.zeros(int(self.metagraph.n), dtype=np.float32)

        # Only miners with positive average score may receive non-zero emissions.
        positive_eligible = [(uid, avg_score) for uid, avg_score in eligible if avg_score > 0.0]
        positive_uids = [uid for uid, _ in positive_eligible]
        if not positive_uids:
            logger.warning("No miners with positive average score; routing 100% to burn UID.")
            zero_weights = np.zeros(int(self.metagraph.n), dtype=np.float32)
            if len(zero_weights) > burn_uid:
                zero_weights[burn_uid] = 1.0
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
                self.leaderboard_reporter.submit_last_weight_update(last_weight_update=block)
            else:
                logger.error(f"set_weights failed (all zero): {msg}")
            return bool(ok)

        # Ranks 3-10 split the final 15% by descending rank weight, not evenly.
        for uid, share in ranked_emission_shares(positive_uids).items():
            emission_raw[uid] = float(share)

        normalized = np.zeros(int(self.metagraph.n), dtype=np.float32)
        for uid in positive_uids:
            normalized[uid] = float(emission_raw[uid])
        for rank0, (uid, avg_score) in enumerate(eligible[:10]):
            rank = rank0 + 1
            logger.debug(
                f"rank={rank} uid={uid} avg_score={avg_score:.6f} emission_raw={emission_raw[uid]:.6f} emission={normalized[uid]:.6f}"
            )
        top_weight_items: list[str] = []
        for rank, (uid, avg_score) in enumerate(positive_eligible[:5], start=1):
            top_weight_items.append(f"r{rank}:uid{uid}:avg={avg_score:.4f}:w={normalized[uid]:.4f}")
        self._log_summary(
            "weights_summary",
            burn=f"{burn_rate:.4f}",
            burn_uid=burn_uid,
            eligible=n_eligible,
            distributed=len(positive_eligible),
            top="|".join(top_weight_items) if top_weight_items else "none",
        )

        # Burn is fetched live each cycle and assigned to UID 0 (existing subnet
        # convention for the burn/null destination). Miner weights are scaled by
        # the remaining share so the final vector remains normalized.
        miner_share = float(min(max(self.miner_emission_share, 0.0), 1.0))
        miner_distribution_share = miner_share * (1.0 - burn_rate)
        scaled = normalized * miner_distribution_share
        if len(scaled) > burn_uid:
            scaled[burn_uid] = float(burn_rate + (1.0 - miner_share))

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
            self.leaderboard_reporter.submit_last_weight_update(last_weight_update=block)
        else:
            logger.error(f"set_weights failed: {msg}")
        return bool(ok)

    def run(self) -> None:
        self._log_step_start("validator_boot")
        self.sync()
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Validator hotkey is not registered on this netuid.")
        self._ensure_api_key()

        tempo = self.subtensor.get_subnet_hyperparameters(self.config.netuid).tempo
        self._log_summary(
            "validator_config",
            k_miners=self.config.perturb.k_miners,
            history_size=self.config.perturb.history_size,
            min_linf=self.config.perturb.min_linf_delta,
            max_linf=self.config.perturb.max_linf_delta,
            min_ssim=self.config.perturb.min_ssim,
            min_psnr_db=self.config.perturb.min_psnr_db,
            perturb_weight=C.PERTURBATION_WEIGHT,
            tempo=tempo,
            run_id=self.run_id,
        )

        while True:
            try:
                self._log_step_start("loop_sync_metagraph")
                self.sync()
                self._log_step_start("loop_get_current_block")
                block = self.subtensor.get_current_block()
                self._log_step_start("loop_wait_task_boundary", block=block)
                self._wait_for_next_task_boundary()
                self._log_step_start("loop_wait_before_task_fetch", block=block)
                self._wait_before_task_fetch()
                self._log_step_start("loop_get_api_task", block=block)
                task = self._fetch_current_task()
                if task is None:
                    continue
                if task.task_id == self.last_validated_api_task_id:
                    logger.info(
                        f"Task unchanged task_id={task.task_id}; waiting for next task boundary"
                    )
                    continue
                try:
                    challenge = self.challenge_from_task_api(task_id=task.task_id, image_url=task.image_url)
                except Exception as exc:
                    logger.warning(f"Task API challenge load failed task_id={task.task_id}: {exc}")
                    time.sleep(float(getattr(self.config.perturb, "task_poll_time", C.TASK_POLL_TIME)))
                    continue
                self._log_summary(
                    "challenge_summary",
                    task_id=challenge.task_id,
                    epsilon=f"{challenge.epsilon:.4f}",
                    true_label=challenge.true_label,
                )

                self._log_step_start("loop_wait_for_submitted_responses", task_id=challenge.task_id)
                submitted_responses = self._wait_for_submitted_responses(task_id=challenge.task_id)
                if not submitted_responses:
                    self.last_validated_api_task_id = challenge.task_id
                    self._save_state()
                    continue

                submitted_response_by_uid: dict[int, SubmittedResponse] = {}
                for item in submitted_responses:
                    if 0 <= int(item.miner_uid) < int(self.metagraph.n):
                        submitted_response_by_uid[int(item.miner_uid)] = item
                miner_uids = sorted(submitted_response_by_uid)
                available_uids = list(miner_uids)
                if not miner_uids:
                    logger.warning("Submitted response list had no valid metagraph miner UIDs.")
                    time.sleep(float(getattr(self.config.perturb, "task_poll_time", C.TASK_POLL_TIME)))
                    continue
                self._log_summary("miner_submission_selection", submitted=len(miner_uids))

                self._log_step_start("loop_score_responses", response_count=len(miner_uids))
                rewards: list[float] = []
                results_by_uid: list[tuple[int, EvaluationResult]] = []
                response_log_lines: list[str] = []
                response_hash_by_uid: dict[int, str] = {}
                status_code_by_uid: dict[int, int] = {}
                for uid in miner_uids:
                    submitted = submitted_response_by_uid[uid]
                    status_code = 200
                    try:
                        perturbed_image_b64 = image_url_to_b64(
                            submitted.image_url,
                            timeout_seconds=float(
                                getattr(self.config.perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS)
                            ),
                        )
                    except Exception as exc:
                        status_code = 0
                        perturbed_image_b64 = ""
                        logger.warning(f"Submitted response download failed uid={uid} url={submitted.image_url}: {exc}")
                    status_code_by_uid[uid] = int(status_code)

                    if status_code != 200 or not perturbed_image_b64:
                        result = EvaluationResult(
                            score=0.0,
                            reason="response_missing_or_status_error",
                            model_prediction="unavailable",
                        )
                    else:
                        try:
                            response_hash_by_uid[uid] = hashlib.sha256(base64.b64decode(perturbed_image_b64)).hexdigest()
                        except Exception:
                            pass
                        result = self.verify_and_score(
                            challenge=challenge,
                            perturbed_image_b64=perturbed_image_b64,
                        )
                    results_by_uid.append((uid, result))

                zero_duplicate_responses(results_by_uid=results_by_uid, response_hash_by_uid=response_hash_by_uid)

                image_url_by_uid: dict[int, str] = {}
                for uid, result in results_by_uid:
                    score = float(result.score)
                    rewards.append(score)
                    if result.reason == "success":
                        image_url_by_uid[uid] = submitted_response_by_uid[uid].image_url
                    self.reason_counts_total[result.reason] += 1
                    response_log_lines.append(
                        f"uid={uid} status={status_code_by_uid.get(uid, 0)} score={score:.6f} "
                        f"processed={int(self.processed_counts[uid]) + 1} "
                        f"reason={result.reason} "
                        f"norm={result.norm:.6f} rmse={result.rmse:.6f} epsilon={result.epsilon:.6f} "
                        f"ssim={result.ssim:.6f} psnr_db={result.psnr_db:.4f}"
                    )

                all_zero_scores = bool(rewards) and all(score <= 0.0 for score in rewards)

                self._log_step_start("loop_update_histories")
                leaderboard_results_by_uid = self._leaderboard_results_for_all_miners(results_by_uid)
                leaderboard_uids = [uid for uid, _ in leaderboard_results_by_uid]
                leaderboard_rewards = [float(result.score) for _, result in leaderboard_results_by_uid]
                self._update_leaderboard_histories(leaderboard_uids, leaderboard_rewards)
                self._submit_leaderboard_report(
                    challenge=challenge,
                    results_by_uid=leaderboard_results_by_uid,
                    image_url_by_uid=image_url_by_uid,
                )
                self._log_summary(
                    "leaderboard_report_queued",
                    task_id=challenge.task_id,
                    miners=len(leaderboard_results_by_uid),
                    valid=sum(1 for _, result in leaderboard_results_by_uid if result.reason == "success"),
                    response_urls=len(image_url_by_uid),
                    available=len(available_uids),
                )
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
                    available_uid_set = set(available_uids)
                    unavailable_all_uids = [uid for uid in range(int(self.metagraph.n)) if uid not in available_uid_set]
                    if unavailable_all_uids:
                        self._update_histories(unavailable_all_uids, [0.0] * len(unavailable_all_uids))
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
                
                self.last_validated_api_task_id = challenge.task_id
                self._log_step_start("loop_save_state")
                self._save_state()

                blocks_since_weights = block - self.last_weight_block
                if blocks_since_weights >= tempo:
                    self._log_step_start("loop_maybe_set_weights", blocks_since_weights=blocks_since_weights, tempo=tempo)
                    if self._set_weights(block=block):
                        self.last_weight_block = block

                self.step += 1
            except KeyboardInterrupt:
                logger.info("Validator stopped by user.")
                break
            except Exception as exc:
                logger.error(f"Validator loop error: {exc}")
                time.sleep(5)
        self.leaderboard_reporter.close()


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
    