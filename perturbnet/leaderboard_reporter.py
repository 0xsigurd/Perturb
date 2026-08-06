from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import requests

from perturbnet import constants as C

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeaderboardMinerResult:
    uid: int
    avg_score: float
    last_score: float
    graph: list[int | float]
    rmse: float
    norm: float
    margin: float
    result: str
    image_url: str


@dataclass(frozen=True)
class LeaderboardNetworkMetrics:
    avg_score: float
    avg_rmse: float
    avg_norm: float
    avg_margin: float
    success_count: int


@dataclass(frozen=True)
class LeaderboardReport:
    task_id: str
    validator_hotkey: str
    network: LeaderboardNetworkMetrics
    miners: list[LeaderboardMinerResult]


@dataclass(frozen=True)
class LeaderboardLastWeightUpdate:
    validator_hotkey: str
    last_weight_update: int


class LeaderboardReporter:
    """Non-blocking signed reports to the Perturb leaderboard backend."""

    def __init__(
        self,
        *,
        enabled: bool,
        api_url: str,
        last_weight_update_api_url: str | None = None,
        api_key: str = "",
        timeout_seconds: float,
        wallet: Any,
        validator_hotkey: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.api_url = str(api_url).strip()
        self.api_key = str(api_key).strip()
        self.last_weight_update_api_url = (
            str(last_weight_update_api_url).strip()
            if last_weight_update_api_url
            else self._default_last_weight_update_url(self.api_url)
        )
        self.timeout_seconds = float(timeout_seconds)
        self.wallet = wallet
        self.validator_hotkey = validator_hotkey
        self.queue: queue.Queue[LeaderboardReport | LeaderboardLastWeightUpdate | None] = queue.Queue(maxsize=128)
        self.thread: threading.Thread | None = None
        if not self.enabled:
            return
        if not self.api_url:
            logger.warning("Leaderboard reporting enabled but API URL is empty; reports disabled.")
            self.enabled = False
            return
        self.thread = threading.Thread(target=self._worker, name="leaderboard-reporter", daemon=True)
        self.thread.start()
        logger.info(f"Leaderboard reporting enabled url={self.api_url}")

    def submit(self, report: LeaderboardReport) -> None:
        if not self.enabled:
            return
        try:
            self.queue.put_nowait(report)
        except queue.Full:
            logger.warning("Leaderboard report queue is full; dropping report.")

    def submit_last_weight_update(self, *, last_weight_update: int) -> None:
        if not self.enabled:
            return
        update = LeaderboardLastWeightUpdate(
            validator_hotkey=self.validator_hotkey,
            last_weight_update=int(last_weight_update),
        )
        try:
            self.queue.put_nowait(update)
        except queue.Full:
            logger.warning("Leaderboard report queue is full; dropping last-weight update.")

    def close(self, timeout_seconds: float = 10.0) -> None:
        if not self.enabled or self.thread is None:
            return
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=timeout_seconds)

    def _worker(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            try:
                if isinstance(item, LeaderboardLastWeightUpdate):
                    self._post_last_weight_update_with_retry(update=item)
                    logger.info(f"Leaderboard last-weight update succeeded block={item.last_weight_update}")
                else:
                    self._post_with_retry(report=item)
                    logger.info(
                        f"Leaderboard report succeeded task_id={item.task_id} "
                        f"miners={len(item.miners)} valid={item.network.success_count}"
                    )
            except Exception as exc:
                if isinstance(item, LeaderboardLastWeightUpdate):
                    logger.warning(f"Leaderboard last-weight update failed block={item.last_weight_update}: {exc}")
                else:
                    logger.warning(f"Leaderboard report failed task_id={item.task_id}: {exc}")
            finally:
                self.queue.task_done()

    def _post_with_retry(self, *, report: LeaderboardReport) -> None:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                self._post(report=report)
                return
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    logger.warning(f"Leaderboard report attempt failed task_id={report.task_id}, retrying once: {exc}")
        if last_error is not None:
            raise last_error

    def _post_last_weight_update_with_retry(self, *, update: LeaderboardLastWeightUpdate) -> None:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                self._post_last_weight_update(update=update)
                return
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    logger.warning(
                        f"Leaderboard last-weight update attempt failed block={update.last_weight_update}, "
                        f"retrying once: {exc}"
                    )
        if last_error is not None:
            raise last_error

    def _post(self, *, report: LeaderboardReport) -> None:
        # Timestamp must be current at send time for backend replay protection.
        payload = {
            "task_id": report.task_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "validator_hotkey": report.validator_hotkey,
            "network": asdict(report.network),
            "miners": [asdict(miner) for miner in report.miners],
        }
        # Serialize once, sign these exact bytes, and POST the same bytes with
        # data=body. Do not use requests.post(json=...), which re-serializes.
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        signature = self._sign_payload(body)
        headers = {
            "Content-Type": "application/json",
            "X-Validator-Hotkey": self.validator_hotkey,
            "X-Signature": signature,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.api_url,
            data=body,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]} body_bytes={len(body)}")

    def _post_last_weight_update(self, *, update: LeaderboardLastWeightUpdate) -> None:
        payload = {
            "validator_hotkey": update.validator_hotkey,
            "last_weight_update": int(update.last_weight_update),
        }
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        signature = self._sign_payload(body)
        headers = {
            "Content-Type": "application/json",
            "X-Validator-Hotkey": self.validator_hotkey,
            "X-Signature": signature,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.last_weight_update_api_url,
            data=body,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

    @staticmethod
    def _default_last_weight_update_url(api_url: str) -> str:
        normalized = str(api_url).strip()
        if normalized.endswith("/report"):
            return normalized[: -len("/report")] + "/last-weight-update"
        return normalized.rstrip("/") + "/last-weight-update"

    def _sign_payload(self, body: bytes) -> str:
        hotkey = getattr(self.wallet, "hotkey", None)
        if hotkey is None or not hasattr(hotkey, "sign"):
            raise RuntimeError("Wallet hotkey does not support signing leaderboard reports.")
        signature = hotkey.sign(body)
        if isinstance(signature, bytes):
            return "0x" + signature.hex()
        if isinstance(signature, str):
            return signature if signature.startswith("0x") else "0x" + signature
        if hasattr(signature, "hex"):
            return "0x" + signature.hex()
        raise RuntimeError(f"Unsupported signature type: {type(signature).__name__}")
