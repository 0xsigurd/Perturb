from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests


@dataclass(frozen=True)
class CurrentTask:
    task_id: str
    image_url: str


@dataclass(frozen=True)
class SubmittedResponse:
    miner_uid: int
    image_url: str


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _json_response(response: requests.Response) -> Any:
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
    if not response.text.strip():
        return None
    return response.json()


def _image_url_from_payload(payload: dict[str, Any]) -> str:
    return str(payload.get("imageURL") or payload.get("imageUrl") or payload.get("image_url") or "").strip()


def sign_body(wallet: Any, body: bytes) -> str:
    hotkey = getattr(wallet, "hotkey", None)
    if hotkey is None or not hasattr(hotkey, "sign"):
        raise RuntimeError("Wallet hotkey does not support request signing.")
    signature = hotkey.sign(body)
    if isinstance(signature, bytes):
        return "0x" + signature.hex()
    if isinstance(signature, str):
        return signature if signature.startswith("0x") else "0x" + signature
    if hasattr(signature, "hex"):
        return "0x" + signature.hex()
    raise RuntimeError(f"Unsupported signature type: {type(signature).__name__}")


def _authorization_headers(api_key: str) -> dict[str, str]:
    api_key = str(api_key).strip()
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def get_current_task(*, base_url: str, timeout_seconds: float) -> CurrentTask | None:
    response = requests.get(_url(base_url, "/task"), timeout=timeout_seconds)
    payload = _json_response(response)
    if not isinstance(payload, dict):
        return None
    task_id = str(payload.get("task_id") or payload.get("taskId") or "").strip()
    image_url = _image_url_from_payload(payload)
    if not task_id or not image_url:
        return None
    return CurrentTask(task_id=task_id, image_url=image_url)


def post_task(
    *,
    base_url: str,
    api_key: str,
    task_id: str,
    image_url: str,
    status: str,
    hotkeys: list[str],
    timeout_seconds: float,
) -> Any:
    payload = {"task_id": task_id, "imageURL": image_url, "status": status, "hotkeys": hotkeys}
    response = requests.post(
        _url(base_url, "/task"),
        json=payload,
        headers=_authorization_headers(api_key),
        timeout=timeout_seconds,
    )
    return _json_response(response)


def submit_miner_response(
    *,
    base_url: str,
    wallet: Any,
    image_url: str,
    timeout_seconds: float,
) -> Any:
    hotkey = str(getattr(getattr(wallet, "hotkey", None), "ss58_address", ""))
    payload = {
        "miner_hotkey": hotkey,
        "timestamp": datetime.now(UTC).isoformat(),
        "imageURL": image_url,
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    response = requests.post(
        _url(base_url, "/submits"),
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Miner-Hotkey": hotkey,
            "X-Signature": sign_body(wallet, body),
        },
        timeout=timeout_seconds,
    )
    return _json_response(response)


def get_leaderboard_avg_scores(
    *,
    base_url: str,
    validator_hotkey: str,
    timeout_seconds: float,
) -> dict[int, float]:
    """Return miner uid -> avgScore from one validator's leaderboard report."""
    response = requests.get(
        _url(base_url, f"/leaderboard/{validator_hotkey}"),
        timeout=timeout_seconds,
    )
    payload = _json_response(response)
    if not isinstance(payload, dict):
        return {}
    raw_miners = payload.get("miners")
    if not isinstance(raw_miners, list):
        return {}
    scores: dict[int, float] = {}
    for item in raw_miners:
        if not isinstance(item, dict):
            continue
        try:
            uid = int(item.get("uid"))
            avg_score = float(item.get("avgScore", item.get("avg_score")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(avg_score):
            scores[uid] = avg_score
    return scores


def get_submitted_responses(*, base_url: str, api_key: str, timeout_seconds: float) -> list[SubmittedResponse]:
    response = requests.get(
        _url(base_url, "/submits"),
        headers=_authorization_headers(api_key),
        timeout=timeout_seconds,
    )
    payload = _json_response(response)
    if not isinstance(payload, list):
        return []

    responses: list[SubmittedResponse] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_uid = item.get("miner_uid", item.get("miner_id", item.get("minerUid")))
        image_url = _image_url_from_payload(item)
        try:
            miner_uid = int(raw_uid)
        except (TypeError, ValueError):
            continue
        if image_url:
            responses.append(SubmittedResponse(miner_uid=miner_uid, image_url=image_url))
    return responses

