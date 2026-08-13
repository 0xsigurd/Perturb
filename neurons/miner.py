from __future__ import annotations

import argparse
import hashlib
import logging as pylogging
import os
import time
import typing

import bittensor as bt
import torch
import torch.nn.functional as F

from perturbnet import constants as C
from perturbnet.api_client import get_current_task, submit_miner_response
from perturbnet.image_io import decode_image_b64, encode_image_b64, image_pixel_hash, image_url_to_b64
from perturbnet.model import load_efficientnet_v2_l, logits_for_images, predict_index, predict_label, resolve_target_index
from perturbnet.storage_uploader import ImageStorageUploader
from perturbnet.task_timing import sleep_until_next_task_boundary

logger = pylogging.getLogger(__name__)


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
    chain_endpoint = getattr(config.subtensor, "chain_endpoint", None) or getattr(config, "chain_endpoint", None)
    if hasattr(bt, "subtensor"):
        if chain_endpoint:
            try:
                return bt.subtensor(chain_endpoint=chain_endpoint)
            except Exception:
                pass
        try:
            return bt.subtensor(network=network)
        except Exception:
            return bt.subtensor(config=config)
    subtensor_cls = getattr(bt, "Subtensor", None)
    if subtensor_cls is None:
        raise RuntimeError("No subtensor constructor found in bittensor.")
    if chain_endpoint:
        try:
            return subtensor_cls(chain_endpoint=chain_endpoint)
        except Exception:
            pass
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


class PerturbMiner:
    def __init__(self, config: typing.Any) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = self._init_subtensor_with_retry()
        self.metagraph = self._init_metagraph_with_retry()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_efficientnet_v2_l(self.device)
        hotkey = str(getattr(self.wallet.hotkey, "ss58_address", "unknown"))
        self.response_exporter = ImageStorageUploader(
            run_id=f"miner-{hotkey[:8]}-{os.getpid()}",
            netuid=int(self.config.netuid),
            uploader_hotkey=hotkey,
        )
        self.last_processed_task_id = ""

    def _init_subtensor_with_retry(self):
        max_attempts = int(os.getenv("SUBTENSOR_CONNECT_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("SUBTENSOR_CONNECT_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Connecting subtensor (attempt {attempt}/{max_attempts})")
                return _make_subtensor(config=self.config)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Subtensor connect failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to connect subtensor after {max_attempts} attempts: {last_error}")

    def _init_metagraph_with_retry(self):
        max_attempts = int(os.getenv("METAGRAPH_SYNC_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("METAGRAPH_SYNC_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Loading metagraph netuid={self.config.netuid} (attempt {attempt}/{max_attempts})")
                return self.subtensor.metagraph(netuid=self.config.netuid)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Metagraph load failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to load metagraph after {max_attempts} attempts: {last_error}")

    def sync(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)

    def _miner_uid(self) -> int:
        hotkey = str(getattr(self.wallet.hotkey, "ss58_address", ""))
        if hotkey not in self.metagraph.hotkeys:
            raise RuntimeError("Miner hotkey is not registered on this netuid.")
        return int(self.metagraph.hotkeys.index(hotkey))

    def _attack_image(self, *, task_id: str, clean_image_b64: str) -> tuple[str, str]:
        clean = decode_image_b64(clean_image_b64).to(self.device)
        predicted_label = predict_label(self.model, clean)
        target_index = resolve_target_index(predicted_label)
        if target_index is None:
            raise RuntimeError(f"Unable to resolve predicted label for task={task_id}: {predicted_label}")

        epsilon = float(getattr(self.config.perturb, "max_linf_delta", C.MAX_LINF_DELTA))
        min_delta = float(getattr(self.config.perturb, "min_linf_delta", C.MIN_LINF_DELTA))
        steps = 10
        step_size = max(epsilon / 4.0, 1.0 / 255.0)
        adv = clean.clone().detach()
        best = adv.clone()
        best_delta = 0.0
        final_pred = target_index
        for _ in range(steps):
            adv.requires_grad_(True)
            logits = logits_for_images(model=self.model, image_bchw=adv.unsqueeze(0))
            loss = F.cross_entropy(logits, torch.tensor([target_index], device=self.device))
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + step_size * grad.sign()
            adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)

            pred = predict_index(model=self.model, image_chw=adv)
            final_pred = pred
            delta = float((adv - clean).abs().max().item())
            if delta > best_delta:
                best = adv.clone()
                best_delta = delta
            if pred != target_index and delta >= min_delta:
                best = adv.clone()
                break

        logger.info(
            f"Finished task={task_id} target_idx={target_index} final_pred={final_pred} "
            f"best_delta={best_delta:.6f} min_delta={min_delta:.6f}"
        )
        return encode_image_b64(best), str(predicted_label)

    def _upload_response(self, *, task_id: str, perturbed_image_b64: str) -> str:
        uid = self._miner_uid()
        hotkey = str(getattr(self.wallet.hotkey, "ss58_address", ""))
        miner_storage_key = self.response_exporter.miner_storage_key(miner_uid=uid, miner_hotkey=hotkey)
        safe_task_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)
        image_hash = hashlib.sha256(perturbed_image_b64.encode("utf-8")).hexdigest()[:12]
        key = f"{C.STORAGE_PREFIX.strip().strip('/')}/miner-responses/{safe_task_id}/{miner_storage_key}_{image_hash}.png"
        return self.response_exporter.upload_image_b64(key=key, image_b64=perturbed_image_b64)

    def _submission_succeeded(self, response: typing.Any) -> bool:
        if response is None:
            return True
        if isinstance(response, dict):
            raw = response.get("success", response.get("ok", response.get("status")))
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str):
                return raw.strip().lower() in {"success", "succeeded", "ok", "true"}
            has_miner_uid = any(key in response for key in ("miner_uid", "miner_id", "minerUid"))
            has_image_url = any(response.get(key) for key in ("imageURL", "imageUrl", "image_url"))
            if has_miner_uid and has_image_url:
                return True
        return False

    def _process_task(self, *, task_id: str, image_url: str) -> None:
        # Processing, upload, and API submission should complete within 20 seconds.
        started_at = time.time()
        clean_image_b64 = image_url_to_b64(
            image_url,
            timeout_seconds=float(getattr(self.config.perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS)),
        )
        perturbed_image_b64, _ = self._attack_image(task_id=task_id, clean_image_b64=clean_image_b64)
        response_url = self._upload_response(
            task_id=task_id,
            perturbed_image_b64=perturbed_image_b64,
        )
        # Validators recompute this pixel hash from the downloaded image and
        # zero the submission if the content at the URL changed after submit.
        image_hash = image_pixel_hash(perturbed_image_b64)
        submit_response = submit_miner_response(
            base_url=str(getattr(self.config.perturb, "api_base_url", C.PERTURB_API_BASE_URL)),
            wallet=self.wallet,
            image_url=response_url,
            image_hash=image_hash,
            timeout_seconds=float(getattr(self.config.perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS)),
        )
        if not self._submission_succeeded(submit_response):
            raise RuntimeError(f"Response submission failed task={task_id} api_response={submit_response}")
        elapsed = time.time() - started_at
        logger.info(f"Submitted task={task_id} response_url={response_url} elapsed_seconds={elapsed:.2f}")

    def _wait_for_next_task_boundary(self) -> None:
        cadence = int(getattr(self.config.perturb, "task_cadence_seconds", C.TASK_CADENCE_SECONDS))
        slept = sleep_until_next_task_boundary(cadence_seconds=cadence)
        logger.info(f"Reached task boundary after waiting {slept:.2f}s")

    def _snapshot_current_task_before_boundary(self) -> None:
        cadence = int(getattr(self.config.perturb, "task_cadence_seconds", C.TASK_CADENCE_SECONDS))
        prefetch_seconds = float(
            getattr(self.config.perturb, "task_pre_boundary_fetch_seconds", C.TASK_PRE_BOUNDARY_FETCH_SECONDS)
        )
        now = time.time()
        cadence = max(1, cadence)
        prefetch_seconds = min(max(0.0, prefetch_seconds), float(cadence))
        boundary = (int(now // cadence) + 1) * cadence
        sleep_seconds = max(0.0, boundary - prefetch_seconds - now)
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)
        try:
            task = get_current_task(
                base_url=str(getattr(self.config.perturb, "api_base_url", C.PERTURB_API_BASE_URL)),
                timeout_seconds=float(getattr(self.config.perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS)),
            )
        except Exception as exc:
            logger.warning(f"Pre-boundary task fetch failed: {exc}")
            return
        if task is not None:
            self.last_processed_task_id = task.task_id
            logger.info(f"Pre-boundary baseline task_id={self.last_processed_task_id}")

    def _fetch_new_task_at_boundary(self):
        retries = int(getattr(self.config.perturb, "task_fetch_retries", C.TASK_FETCH_RETRIES))
        retry_seconds = float(getattr(self.config.perturb, "task_fetch_retry_seconds", C.TASK_FETCH_RETRY_SECONDS))
        for attempt in range(1, retries + 1):
            task = get_current_task(
                base_url=str(getattr(self.config.perturb, "api_base_url", C.PERTURB_API_BASE_URL)),
                timeout_seconds=float(getattr(self.config.perturb, "api_timeout_seconds", C.PERTURB_API_TIMEOUT_SECONDS)),
            )
            if task is not None and task.task_id != self.last_processed_task_id:
                return task
            logger.info(f"No new task at boundary attempt={attempt}/{retries}")
            if attempt < retries:
                time.sleep(retry_seconds)
        return None

    def run(self) -> None:
        self.sync()
        self._miner_uid()
        logger.info("Miner started. Waiting for task boundaries.")
        while True:
            try:
                self._snapshot_current_task_before_boundary()
                self._wait_for_next_task_boundary()
                task = self._fetch_new_task_at_boundary()
                if task is None:
                    continue

                logger.info(f"New task found task_id={task.task_id} image_url={task.image_url}")

                self._process_task(task_id=task.task_id, image_url=task.image_url)
            except Exception as exc:
                logger.warning(f"Miner task loop failed: {exc}")
                time.sleep(float(getattr(self.config.perturb, "task_poll_time", C.TASK_POLL_TIME)))


def build_config() -> typing.Any:
    parser = argparse.ArgumentParser(description="Perturb subnet miner")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument(
        "--subtensor.chain_endpoint",
        dest="chain_endpoint",
        type=str,
        default=os.getenv("SUBTENSOR_CHAIN_ENDPOINT", os.getenv("CHAIN_ENDPOINT", "")),
    )
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
    config.subtensor.chain_endpoint = getattr(
        config.subtensor, "chain_endpoint", getattr(config, "chain_endpoint", "")
    )

    if not hasattr(config, "logging"):
        config.logging = type("LoggingConfig", (), {})()
    config.logging.logging_dir = getattr(config.logging, "logging_dir", getattr(config, "logging_dir", "./logs"))

    if not hasattr(config, "perturb"):
        config.perturb = type("PerturbConfig", (), {})()
    for key, value in C.VALIDATOR_CONFIG.items():
        setattr(config.perturb, key, getattr(config.perturb, key, value))

    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))
    return config


if __name__ == "__main__":
    miner = PerturbMiner(config=build_config())
    miner.run()
