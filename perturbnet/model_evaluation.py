"""Daily evaluation of miners' committed models."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image as PILImage

from perturbnet.api_client import ApiCommitment
from perturbnet.model import PREPROCESS
from perturbnet.model_commit import (
    ARCHITECTURE,
    CONFIG_FILE,
    MODEL_FILE,
    MODEL_HASH_CHARS,
    NUM_CLASSES,
    parse_chain_commit,
    parse_repo_at_revision,
    sha256_model_and_hotkey,
)

logger = logging.getLogger(__name__)

BASELINE_UID = -1
UNPINNED_REVISION = "main"
NOISY_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "datasets", "filelock", "fsspec", "urllib3")


# --------------------------------------------------------------------------- results


@dataclass
class ModelEvalResult:
    uid: int
    hotkey: str
    repo_id: str
    revision: str
    block: int
    valid: bool = False
    reason: str = "pending"
    imagenet_top1: float = 0.0
    adv_top1: float = 0.0
    adv_clean_top1: float = 0.0
    overall: float = 0.0
    eval_seconds: float = 0.0

    @property
    def commitment(self) -> str:
        return f"{self.repo_id}@{self.revision}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "uid": int(self.uid),
            "hotkey": self.hotkey,
            "repoId": self.repo_id or None,
            "revision": self.revision or None,
            "commitBlock": int(self.block),
            "score": round(float(self.overall), 6),
            "robustAccuracy": round(float(self.adv_top1), 6),
            "cleanAccuracy": round(float(self.imagenet_top1), 6),
            "advCleanAccuracy": round(float(self.adv_clean_top1), 6),
            "valid": bool(self.valid),
            "reason": self.reason,
            "evalSeconds": round(float(self.eval_seconds), 1),
        }


@dataclass
class ModelEvaluationOutcome:
    date: str
    block: int
    winner_uid: int | None
    baseline: ModelEvalResult | None
    results: list[ModelEvalResult]
    imagenet_samples: int
    adv_rows: int
    adv_images: int
    adv_dataset_revision: str
    group_epsilon: float
    duration_seconds: float
    top_group_uids: list[int] = field(default_factory=list)
    imagenet_floor: float = -1.0

    def model_scores(self) -> dict[int, float]:
        return {int(self.winner_uid): 1.0} if self.winner_uid is not None else {}

    @property
    def valid_results(self) -> list[ModelEvalResult]:
        return [r for r in self.results if r.valid]

    def network_payload(self) -> dict[str, Any]:
        valid = self.valid_results
        count = len(valid)
        mean = lambda key: round(sum(getattr(r, key) for r in valid) / count, 6) if count else 0.0
        return {
            "success_count": count,
            "candidate_count": len(self.results),
            "avg_score": mean("overall"),
            "avg_imagenet_top1": mean("imagenet_top1"),
            "avg_adv_top1": mean("adv_top1"),
        }

    def table_data(self) -> list[dict[str, Any]]:
        """One record per miner that committed a model, ordered by uid."""
        return [result.to_payload() for result in sorted(self.results, key=lambda r: int(r.uid))]

    def to_payload(self) -> dict[str, Any]:
        network = self.network_payload()
        return {
            "success_count": int(network["success_count"]),
            "last_evaluation_block": int(self.block),
            "avg_score": float(network["avg_score"]),
            "table_data": self.table_data(),
            "date": self.date,
            "block": int(self.block),
            "winner_uid": None if self.winner_uid is None else int(self.winner_uid),
            "top_group_uids": [int(uid) for uid in self.top_group_uids],
            "group_epsilon": float(self.group_epsilon),
            "imagenet_floor": float(self.imagenet_floor),
            "baseline": None if self.baseline is None else self.baseline.to_payload(),
            "network": self.network_payload(),
            "imagenet_samples": int(self.imagenet_samples),
            "adv_rows": int(self.adv_rows),
            "adv_images": int(self.adv_images),
            "adv_dataset_revision": self.adv_dataset_revision,
            "duration_seconds": round(float(self.duration_seconds), 1),
        }


# --------------------------------------------------------------------------- selection


def resolve_miner_uid(miner_id: str, uid_by_hotkey: dict[str, int]) -> int | None:
    text = str(miner_id or "").strip()
    if text.isdigit():
        return int(text)
    return uid_by_hotkey.get(text)


def meets_imagenet_floor(imagenet_top1: float, *, baseline_imagenet_top1: float, imagenet_floor: float) -> bool:
    if imagenet_floor < 0.0:
        return True
    return float(imagenet_top1) >= float(baseline_imagenet_top1) - float(imagenet_floor)


def select_winner(
    results: Sequence[ModelEvalResult],
    *,
    epsilon: float,
    baseline: ModelEvalResult | None,
    must_beat_baseline: bool,
    imagenet_floor: float = -1.0,
) -> tuple[int | None, list[int]]:
    eligible = [r for r in results if r.valid]
    if baseline is not None:
        eligible = [
            r for r in eligible
            if meets_imagenet_floor(r.imagenet_top1, baseline_imagenet_top1=baseline.imagenet_top1, imagenet_floor=imagenet_floor)
        ]
    if must_beat_baseline and baseline is not None:
        eligible = [r for r in eligible if r.overall > baseline.overall + epsilon]
    if not eligible:
        return None, []
    best = max(r.overall for r in eligible)
    group = [r for r in eligible if r.overall >= best - epsilon]
    winner = min(group, key=lambda r: (int(r.block), int(r.uid)))
    return int(winner.uid), sorted(int(r.uid) for r in group)


@dataclass(frozen=True)
class ValidatorEvaluationReport:
    validator_hotkey: str
    date: str
    block: int
    baseline_overall: float
    scores: dict[int, float]
    blocks: dict[int, int]
    baseline_imagenet: float = 0.0
    imagenet: dict[int, float] = field(default_factory=dict)

    @staticmethod
    def _field(item: dict[str, Any], *names: str) -> Any:
        for name in names:
            if name in item:
                return item[name]
        return None

    @classmethod
    def from_payload(cls, payload: Any, *, validator_hotkey: str = "") -> "ValidatorEvaluationReport | None":
        if not isinstance(payload, dict):
            return None
        miners = cls._field(payload, "table_data", "tableData", "miners", "models")
        if not isinstance(miners, list):
            return None
        baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
        try:
            baseline_overall = float(cls._field(baseline, "score", "overall") or 0.0)
            baseline_imagenet = float(cls._field(baseline, "cleanAccuracy", "clean_accuracy", "imagenet_top1") or 0.0)
        except (TypeError, ValueError):
            baseline_overall = 0.0
            baseline_imagenet = 0.0
        scores: dict[int, float] = {}
        blocks: dict[int, int] = {}
        imagenet: dict[int, float] = {}
        for item in miners:
            if not isinstance(item, dict):
                continue
            try:
                uid = int(cls._field(item, "uid", "miner_uid", "miner_id"))
            except (TypeError, ValueError):
                continue
            if not bool(item.get("valid", True)):
                continue
            try:
                overall = float(cls._field(item, "score", "overall") or 0.0)
                block = int(cls._field(item, "commitBlock", "commit_block", "block") or 0)
                imagenet_top1 = float(cls._field(item, "cleanAccuracy", "clean_accuracy", "imagenet_top1") or 0.0)
            except (TypeError, ValueError):
                continue
            scores[uid] = overall
            blocks[uid] = block
            imagenet[uid] = imagenet_top1
        try:
            block = int(cls._field(payload, "last_evaluation_block", "lastEvaluationBlock", "block") or 0)
        except (TypeError, ValueError):
            block = 0
        return cls(
            validator_hotkey=str(cls._field(payload, "validator_hotkey", "validatorHotkey") or validator_hotkey),
            date=str(payload.get("date") or ""),
            block=block,
            baseline_overall=baseline_overall,
            scores=scores,
            blocks=blocks,
            baseline_imagenet=baseline_imagenet,
            imagenet=imagenet,
        )


def consensus_model_winner(
    reports: Sequence[tuple[float, ValidatorEvaluationReport]],
    *,
    epsilon: float,
    must_beat_baseline: bool,
    imagenet_floor: float = -1.0,
) -> tuple[int | None, list[int], dict[int, float]]:
    """Stake-weighted consensus over validators' evaluation reports.

    Returns (winner uid, top-group uids, consensus overall per uid). A miner
    absent from a validator's valid set counts as 0 for that validator's stake.
    """
    weighted = [(float(stake), report) for stake, report in reports if stake > 0.0]
    total_stake = sum(stake for stake, _ in weighted)
    if total_stake <= 0.0:
        return None, [], {}
    uids = {uid for _, report in weighted for uid in report.scores}
    consensus = {
        uid: sum(stake * report.scores.get(uid, 0.0) for stake, report in weighted) / total_stake
        for uid in uids
    }
    baseline = sum(stake * report.baseline_overall for stake, report in weighted) / total_stake
    baseline_imagenet = sum(stake * report.baseline_imagenet for stake, report in weighted) / total_stake
    imagenet = {
        uid: sum(stake * report.imagenet.get(uid, 0.0) for stake, report in weighted) / total_stake
        for uid in uids
    }
    blocks = {
        uid: min(report.blocks[uid] for _, report in weighted if uid in report.blocks)
        for uid in uids
    }
    eligible = {
        uid: score
        for uid, score in consensus.items()
        if meets_imagenet_floor(imagenet[uid], baseline_imagenet_top1=baseline_imagenet, imagenet_floor=imagenet_floor)
        and (not must_beat_baseline or score > baseline + epsilon)
    }
    if not eligible:
        return None, [], consensus
    best = max(eligible.values())
    group = [uid for uid, score in eligible.items() if score >= best - epsilon]
    winner = min(group, key=lambda uid: (blocks[uid], uid))
    return int(winner), sorted(group), consensus


def overall_score(*, imagenet_top1: float, adv_top1: float, imagenet_weight: float, adv_weight: float) -> float:
    total = float(imagenet_weight) + float(adv_weight)
    if total <= 0.0:
        return 0.0
    return (float(imagenet_weight) * float(imagenet_top1) + float(adv_weight) * float(adv_top1)) / total


# --------------------------------------------------------------------------- evaluation data


@dataclass
class AdversarialRow:
    clean: bytes
    adversarial: list[bytes]
    reference_label: int


@dataclass
class EvalData:
    imagenet: list[tuple[bytes, int]]
    adversarial: list[AdversarialRow]
    adv_dataset_revision: str

    @property
    def adv_images(self) -> int:
        return sum(len(row.adversarial) for row in self.adversarial)


def day_seed(date: str) -> int:
    return int(hashlib.sha256(date.encode("utf-8")).hexdigest()[:8], 16)


def _decode(image_bytes: bytes) -> PILImage.Image:
    return PILImage.open(io.BytesIO(image_bytes)).convert("RGB")


def _image_bytes(value: Any) -> bytes | None:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw:
            return bytes(raw)
        path = value.get("path")
        if path:
            return Path(path).read_bytes()
        return None
    if isinstance(value, PILImage.Image):
        buffer = io.BytesIO()
        value.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()
    return None


class _BytesDataset(torch.utils.data.Dataset):
    def __init__(self, items: Sequence[tuple[bytes, int]]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        raw, label = self.items[index]
        return PREPROCESS(_decode(raw)), int(label)


@torch.no_grad()
def predict_labels(
    model: torch.nn.Module,
    items: Sequence[tuple[bytes, int]],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int = 2,
) -> list[int]:
    if not items:
        return []
    loader = torch.utils.data.DataLoader(
        _BytesDataset(items),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=device.type == "cuda",
    )
    predictions: list[int] = []
    use_amp = device.type == "cuda"
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        predictions.extend(int(v) for v in logits.argmax(dim=1).tolist())
    return predictions


def pinned_dataset_revision(repo_id: str, *, before: datetime, token: str | None) -> str | None:
    from huggingface_hub import HfApi

    commits = HfApi(token=token or None).list_repo_commits(repo_id, repo_type="dataset")
    for commit in commits:
        created = commit.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if created < before:
            return str(commit.commit_id)
    return None


RawAdvRow = tuple[bytes, list[bytes], str]

SHARD_RE = re.compile(r"^data/(?P<split>[^/]+)/(?P<stamp>\d{8}T\d{6}Z)-\d+\.parquet$")
SHARD_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


class EvaluationDataUnavailable(RuntimeError):
    """The adversarial evaluation data for the day cannot be obtained (repo unreachable or no rows)."""


def _parse_created_at(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


ADV_COLUMNS = ("image_id", "created_at", "image", "adversarial")


def _rows_from_parquet(
    path: Path, rows: list[RawAdvRow], *, since: datetime | None = None, until: datetime | None = None
) -> int:
    """Appends (clean, adversarial[], image_id) rows from one shard; returns how many rows the
    created_at window skipped. Reads the parquet directly so shards with different column sets
    (e.g. written before/after a schema change) can be mixed."""
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(path).names)
    columns = [name for name in ADV_COLUMNS if name in schema_names]
    table = pq.read_table(path, columns=columns)
    skipped = 0
    for batch in table.to_batches():
        for example in batch.to_pylist():
            if since is not None or until is not None:
                created = _parse_created_at(example.get("created_at"))
                if created is None or (since is not None and created < since) or (until is not None and created >= until):
                    skipped += 1
                    continue
            clean = _image_bytes(example.get("image"))
            adversarial = [b for b in (_image_bytes(v) for v in (example.get("adversarial") or [])) if b]
            if clean and adversarial:
                rows.append((clean, adversarial, str(example.get("image_id") or len(rows))))
    return skipped


def _subsample(rows: list[RawAdvRow], *, max_rows: int, seed: int) -> list[RawAdvRow]:
    rows.sort(key=lambda item: item[2])
    if 0 < int(max_rows) < len(rows):
        keep = sorted(random.Random(seed).sample(range(len(rows)), int(max_rows)))
        rows = [rows[i] for i in keep]
    return rows


def window_shards(files: Sequence[str], *, since: datetime, until: datetime) -> list[str]:
    """Shards whose publish stamp falls in [since, until). A shard is stamped when it is
    uploaded, after every row in it was created, so shards stamped before `since` cannot
    hold rows from the window; rows are then filtered exactly by `created_at`."""
    selected: list[str] = []
    for path in files:
        match = SHARD_RE.match(path)
        if not match:
            continue
        stamp = datetime.strptime(match.group("stamp"), SHARD_STAMP_FORMAT).replace(tzinfo=timezone.utc)
        if since <= stamp < until:
            selected.append(path)
    return sorted(selected)


def load_adversarial_window(
    repo_id: str,
    *,
    revision: str | None,
    since: datetime,
    until: datetime,
    max_rows: int,
    seed: int,
    token: str | None,
    cache_dir: str | None = None,
) -> list[RawAdvRow]:
    """Rows of `repo_id` (at `revision`) whose task was created in [since, until)."""
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi(token=token or None).list_repo_files(repo_id, repo_type="dataset", revision=revision)
    # Shards published after `until` are still candidates: a row created at 23:50 may be
    # uploaded at 00:10. The pinned revision bounds what is visible, created_at bounds the rows.
    shards = window_shards(files, since=since, until=until + timedelta(days=1))
    logger.info(
        f"Adversarial window repo={repo_id} created {since:%Y-%m-%dT%H:%MZ} -> {until:%Y-%m-%dT%H:%MZ}: "
        f"reading {len(shards)} shard(s)"
    )
    rows: list[RawAdvRow] = []
    skipped = 0
    for index, shard in enumerate(shards, start=1):
        local = hf_hub_download(
            repo_id, shard, repo_type="dataset", revision=revision, token=token or None, cache_dir=cache_dir
        )
        skipped += _rows_from_parquet(Path(local), rows, since=since, until=until)
        if index % 10 == 0 or index == len(shards):
            logger.info(f"  shard {index}/{len(shards)}: {len(rows)} rows in window")
    logger.info(f"Adversarial window rows={len(rows)} outside_window={skipped}")
    return _subsample(rows, max_rows=max_rows, seed=seed)


def load_imagenet_samples(
    repo_id: str,
    *,
    split: str,
    samples: int,
    seed: int,
    token: str | None,
) -> list[tuple[bytes, int]]:
    from datasets import Image, load_dataset

    if int(samples) <= 0:
        return []
    stream = load_dataset(repo_id, split=split, streaming=True, token=token or None)
    stream = stream.cast_column("image", Image(decode=False))
    stream = stream.shuffle(seed=int(seed), buffer_size=max(1000, int(samples))).take(int(samples))
    items: list[tuple[bytes, int]] = []
    started = time.time()
    for example in stream:
        raw = _image_bytes(example.get("image"))
        label = example.get("label")
        if raw is not None and label is not None and int(label) >= 0:
            items.append((raw, int(label)))
        if not items:
            continue
        if len(items) == 1:
            logger.info(f"  first image after {time.time() - started:.0f}s")
        elif len(items) % 250 == 0:
            logger.info(f"  {len(items)}/{int(samples)} images")
    return items


def load_eval_data(
    *,
    date: str,
    reference_model: torch.nn.Module,
    device: torch.device,
    adv_repo_id: str,
    adv_max_rows: int,
    imagenet_repo_id: str,
    imagenet_split: str,
    imagenet_samples: int,
    batch_size: int,
    token: str | None,
) -> EvalData:
    seed = day_seed(date)
    started = time.time()
    day_start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    window_start = day_start - timedelta(days=1)

    # Snapshot of the dataset as of 00:00 UTC today: every validator, whenever it runs
    # during the day, reads the same commit and therefore the same rows.
    try:
        adv_revision = pinned_dataset_revision(adv_repo_id, before=day_start, token=token)
    except Exception as exc:
        raise EvaluationDataUnavailable(f"adversarial dataset {adv_repo_id} is not accessible: {exc}") from exc
    if adv_revision is None:
        raise EvaluationDataUnavailable(
            f"adversarial dataset {adv_repo_id} has no commit before {day_start:%Y-%m-%dT%H:%MZ}"
        )

    # Rows are copied into memory, so the datasets cache for the pinned revision is
    # disposable: keep it in a temp dir that is removed here instead of growing
    # ~/.cache/huggingface/datasets by one revision per day.
    with tempfile.TemporaryDirectory(prefix="perturb-adv-data-") as adv_cache_dir:
        try:
            raw_rows = load_adversarial_window(
                adv_repo_id,
                revision=adv_revision,
                since=window_start,
                until=day_start,
                max_rows=adv_max_rows,
                seed=seed,
                token=token,
                cache_dir=adv_cache_dir,
            )
        except Exception as exc:
            raise EvaluationDataUnavailable(f"adversarial dataset {adv_repo_id}@{adv_revision[:8]} could not be read: {exc}") from exc
    if not raw_rows:
        raise EvaluationDataUnavailable(
            f"adversarial dataset {adv_repo_id} has no rows created between "
            f"{window_start:%Y-%m-%dT%H:%MZ} and {day_start:%Y-%m-%dT%H:%MZ}"
        )
    logger.info(f"Labelling {len(raw_rows)} clean images with the reference model")
    reference_labels = predict_labels(
        reference_model, [(clean, 0) for clean, _, _ in raw_rows], device=device, batch_size=batch_size
    )
    adversarial = [
        AdversarialRow(clean=clean, adversarial=adv, reference_label=label)
        for (clean, adv, _), label in zip(raw_rows, reference_labels)
    ]
    logger.info(f"Streaming {imagenet_samples} {imagenet_repo_id}:{imagenet_split} images (first batch can take minutes)")
    imagenet = load_imagenet_samples(
        imagenet_repo_id, split=imagenet_split, samples=imagenet_samples, seed=seed, token=token
    )
    if imagenet_samples > 0 and not imagenet:
        raise RuntimeError(f"{imagenet_repo_id}:{imagenet_split} yielded no samples (HF_TOKEN / gated access?)")
    logger.info(
        f"Evaluation data ready in {time.time() - started:.0f}s: imagenet={len(imagenet)} "
        f"adv_rows={len(adversarial)} adv_images={sum(len(r.adversarial) for r in adversarial)} "
        f"adv_revision={adv_revision[:8]}"
    )
    return EvalData(imagenet=imagenet, adversarial=adversarial, adv_dataset_revision=adv_revision)


# --------------------------------------------------------------------------- per-model evaluation


def evaluate_model(
    model: torch.nn.Module,
    data: EvalData,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    metrics = {"imagenet_top1": 0.0, "adv_top1": 0.0, "adv_clean_top1": 0.0}
    if data.imagenet:
        predictions = predict_labels(model, data.imagenet, device=device, batch_size=batch_size)
        metrics["imagenet_top1"] = sum(int(p == y) for p, (_, y) in zip(predictions, data.imagenet)) / len(data.imagenet)
    if data.adversarial:
        clean_items = [(row.clean, row.reference_label) for row in data.adversarial]
        adv_items = [(image, row.reference_label) for row in data.adversarial for image in row.adversarial]
        clean_predictions = predict_labels(model, clean_items, device=device, batch_size=batch_size)
        adv_predictions = predict_labels(model, adv_items, device=device, batch_size=batch_size)
        metrics["adv_clean_top1"] = sum(int(p == y) for p, (_, y) in zip(clean_predictions, clean_items)) / len(clean_items)
        if adv_items:
            metrics["adv_top1"] = sum(int(p == y) for p, (_, y) in zip(adv_predictions, adv_items)) / len(adv_items)
    return metrics


def load_committed_model(weights_path: Path, config_path: Path | None, *, device: torch.device) -> torch.nn.Module:
    from safetensors.torch import load_file
    from torchvision.models import efficientnet_v2_l

    if config_path is not None and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("architecture", ARCHITECTURE) != ARCHITECTURE or int(config.get("num_classes", NUM_CLASSES)) != NUM_CLASSES:
            raise ValueError(
                f"unsupported config architecture={config.get('architecture')} num_classes={config.get('num_classes')}"
            )
    model = efficientnet_v2_l(weights=None)
    state = load_file(str(weights_path))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# --------------------------------------------------------------------------- orchestration


@dataclass(frozen=True)
class ModelEvaluatorConfig:
    netuid: int
    commitments_api_url: str
    adv_dataset: str
    adv_max_rows: int
    imagenet_repo_id: str
    imagenet_split: str
    imagenet_samples: int
    batch_size: int
    imagenet_weight: float
    adv_weight: float
    group_epsilon: float
    must_beat_baseline: bool
    hf_token: str
    api_timeout_seconds: float
    max_model_bytes: int
    imagenet_floor: float = -1.0


class ModelEvaluator:
    def __init__(
        self,
        config: ModelEvaluatorConfig,
        *,
        reference_model: torch.nn.Module,
        device: torch.device,
        fetch_api_commitments: Callable[[], list[ApiCommitment]],
        fetch_chain_commitments: Callable[[], dict[str, str]],
    ) -> None:
        self.config = config
        self.reference_model = reference_model
        self.device = device
        self.fetch_api_commitments = fetch_api_commitments
        self.fetch_chain_commitments = fetch_chain_commitments

    # ---- candidates -------------------------------------------------------

    def candidates(self, hotkeys: Sequence[str]) -> list[ModelEvalResult]:
        api_rows = self.fetch_api_commitments()
        if not api_rows:
            # An empty snapshot is an upstream problem, not "every miner failed": treat it as
            # missing data so the caller keeps the previous winner instead of clearing it.
            raise EvaluationDataUnavailable(f"{self.config.commitments_api_url} returned no commitments")
        chain = self.fetch_chain_commitments()
        uid_by_hotkey = {str(hotkey): uid for uid, hotkey in enumerate(hotkeys)}
        by_uid: dict[int, ModelEvalResult] = {}
        for row in api_rows:
            uid = resolve_miner_uid(row.miner_id, uid_by_hotkey)
            if uid is None or not (0 <= uid < len(hotkeys)):
                logger.debug(f"Commitment row skipped: miner_id={row.miner_id!r} is not a registered uid/hotkey")
                continue
            parsed = parse_repo_at_revision(row.commitment)
            result = ModelEvalResult(
                uid=uid,
                hotkey=str(hotkeys[uid]),
                repo_id=parsed.repo_id if parsed else row.commitment,
                revision=parsed.revision if parsed else "",
                block=int(row.block),
            )
            if parsed is None:
                result.reason = "commitment_unparseable"
            elif parsed.revision.strip().lower() == UNPINNED_REVISION:
                result.reason = "revision_is_main"
            else:
                result.reason = self._chain_check(result, chain)
            if uid not in by_uid or result.block < by_uid[uid].block:
                by_uid[uid] = result
        return [by_uid[uid] for uid in sorted(by_uid)]

    @staticmethod
    def _chain_check(result: ModelEvalResult, chain: dict[str, str]) -> str:
        raw = chain.get(result.hotkey)
        if not raw:
            return "chain_commitment_missing"
        commit = parse_chain_commit(raw)
        if commit is None:
            return "chain_commitment_unparseable"
        if commit.hf_repo_id != result.repo_id or commit.hf_revision != result.revision:
            return "chain_commitment_changed"
        return f"pending:{commit.model_hash}"

    # ---- download + hash --------------------------------------------------

    def download(self, result: ModelEvalResult, cache_dir: Path) -> tuple[Path, Path | None]:
        from huggingface_hub import HfApi, hf_hub_download

        token = self.config.hf_token or None
        info = HfApi(token=token).model_info(result.repo_id, revision=result.revision, files_metadata=True)
        sizes = {sibling.rfilename: sibling.size for sibling in info.siblings or []}
        if MODEL_FILE not in sizes:
            raise FileNotFoundError(f"{MODEL_FILE} not in {result.commitment}")
        size = sizes.get(MODEL_FILE) or 0
        if size > int(self.config.max_model_bytes):
            raise ValueError(f"{MODEL_FILE} is {size} bytes > limit {self.config.max_model_bytes}")
        weights = Path(
            hf_hub_download(result.repo_id, MODEL_FILE, revision=result.revision, token=token, cache_dir=str(cache_dir))
        )
        config_path: Path | None = None
        if CONFIG_FILE in sizes:
            config_path = Path(
                hf_hub_download(result.repo_id, CONFIG_FILE, revision=result.revision, token=token, cache_dir=str(cache_dir))
            )
        return weights, config_path

    # ---- run --------------------------------------------------------------

    def run(self, *, date: str, block: int, hotkeys: Sequence[str]) -> ModelEvaluationOutcome:
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
        started = time.time()
        cfg = self.config
        results = self.candidates(hotkeys)
        pending = [r for r in results if r.reason.startswith("pending:")]
        logger.info(f"Valid commitments: {len(pending)}")
        for result in pending:
            logger.info(
                f"  miner_uid={result.uid} repo_id={result.repo_id} "
                f"revision={result.revision[:5]} block={result.block}"
            )

        if not pending:
            logger.info(f"No verifiable model commitments among {len(results)} submission(s); no model winner")
            return ModelEvaluationOutcome(
                date=date,
                block=int(block),
                winner_uid=None,
                baseline=None,
                results=results,
                imagenet_samples=0,
                adv_rows=0,
                adv_images=0,
                adv_dataset_revision="",
                group_epsilon=cfg.group_epsilon,
                duration_seconds=time.time() - started,
                imagenet_floor=cfg.imagenet_floor,
            )

        logger.info("Preparing evaluation data")
        data = load_eval_data(
            date=date,
            reference_model=self.reference_model,
            device=self.device,
            adv_repo_id=cfg.adv_dataset,
            adv_max_rows=cfg.adv_max_rows,
            imagenet_repo_id=cfg.imagenet_repo_id,
            imagenet_split=cfg.imagenet_split,
            imagenet_samples=cfg.imagenet_samples,
            batch_size=cfg.batch_size,
            token=cfg.hf_token or None,
        )

        baseline = ModelEvalResult(
            uid=BASELINE_UID, hotkey="", repo_id="torchvision", revision="IMAGENET1K_V1", block=0, valid=True, reason="baseline"
        )
        self._score(baseline, self.reference_model, data)

        with tempfile.TemporaryDirectory(prefix="perturb-model-eval-") as tmp:
            cache_dir = Path(tmp)
            for result in pending:
                logger.info(f"Evaluating miner_uid={result.uid}")
                self._evaluate_candidate(result, data, cache_dir)

        for result in results:
            if result.valid and not meets_imagenet_floor(
                result.imagenet_top1, baseline_imagenet_top1=baseline.imagenet_top1, imagenet_floor=cfg.imagenet_floor
            ):
                result.reason = "below_imagenet_floor"
        winner, group = select_winner(
            results,
            epsilon=cfg.group_epsilon,
            baseline=baseline,
            must_beat_baseline=cfg.must_beat_baseline,
            imagenet_floor=cfg.imagenet_floor,
        )
        outcome = ModelEvaluationOutcome(
            date=date,
            block=int(block),
            winner_uid=winner,
            baseline=baseline,
            results=results,
            imagenet_samples=len(data.imagenet),
            adv_rows=len(data.adversarial),
            adv_images=data.adv_images,
            adv_dataset_revision=data.adv_dataset_revision,
            group_epsilon=cfg.group_epsilon,
            duration_seconds=time.time() - started,
            top_group_uids=group,
            imagenet_floor=cfg.imagenet_floor,
        )
        scored = [r for r in results if r.valid]
        logger.info(
            f"Scores: {len(scored)} candidates (baseline imagenet={baseline.imagenet_top1:.4f} "
            f"adv={baseline.adv_top1:.4f} overall={baseline.overall:.4f}; imagenet floor={baseline.imagenet_top1 - cfg.imagenet_floor:.4f})"
        )
        for result in scored:
            flag = "" if result.reason == "ok" else f" [{result.reason}]"
            logger.info(
                f"  miner_uid={result.uid} imagenet={result.imagenet_top1:.4f} "
                f"adv={result.adv_top1:.4f} overall={result.overall:.4f}{flag}"
            )
        logger.info(f"Top group miner_uids={group or 'none'}")
        logger.info(f"Winner miner_uid={winner if winner is not None else 'none'}")
        return outcome

    def _evaluate_candidate(self, result: ModelEvalResult, data: EvalData, cache_dir: Path) -> None:
        expected_hash = result.reason.split(":", 1)[1][:MODEL_HASH_CHARS]
        started = time.time()
        weights: Path | None = None
        config_path: Path | None = None
        model: torch.nn.Module | None = None
        try:
            weights, config_path = self.download(result, cache_dir)
            actual = sha256_model_and_hotkey(weights, result.hotkey)[:MODEL_HASH_CHARS]
            if actual != expected_hash:
                result.reason = "hash_mismatch"
                logger.warning(f"Model hash mismatch uid={result.uid} {result.commitment} chain={expected_hash} file={actual}")
                return
            try:
                model = load_committed_model(weights, config_path, device=self.device)
            except Exception as exc:
                result.reason = f"load_failed:{type(exc).__name__}"
                logger.warning(f"Model load failed uid={result.uid} {result.commitment}: {exc}")
                return
            try:
                self._score(result, model, data)
            except Exception as exc:
                result.reason = f"eval_failed:{type(exc).__name__}"
                logger.warning(f"Model evaluation failed uid={result.uid} {result.commitment}: {exc}")
                return
            result.valid = True
            result.reason = "ok"
            logger.info(
                f"  uid={result.uid} imagenet_top1={result.imagenet_top1:.4f} adv_top1={result.adv_top1:.4f} "
                f"overall={result.overall:.4f} ({time.time() - started:.0f}s)"
            )
        except Exception as exc:
            result.reason = f"download_failed:{type(exc).__name__}"
            logger.warning(f"Model download failed uid={result.uid} {result.commitment}: {exc}")
        finally:
            result.eval_seconds = time.time() - started
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            for path in (weights, config_path):
                if path is not None:
                    Path(path).unlink(missing_ok=True)

    @staticmethod
    def _summary(outcome: ModelEvaluationOutcome, data: EvalData) -> str:
        rows = [("uid", "commitment", "block", "imagenet", "adv", "overall", "status")]
        entries = ([outcome.baseline] if outcome.baseline else []) + list(outcome.results)
        for r in entries:
            uid = "baseline" if r.uid == BASELINE_UID else str(r.uid)
            commitment = f"{r.repo_id}@{r.revision[:5]}"
            block = "-" if r.uid == BASELINE_UID else str(r.block)
            scored = r.valid
            rows.append((
                uid, commitment, block,
                f"{r.imagenet_top1:.4f}" if scored else "-",
                f"{r.adv_top1:.4f}" if scored else "-",
                f"{r.overall:.4f}" if scored else "-",
                r.reason,
            ))
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        table = "\n".join("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows)
        return (
            f"Model evaluation results date={outcome.date} imagenet_samples={len(data.imagenet)} "
            f"adv_rows={len(data.adversarial)} adv_images={data.adv_images} in {outcome.duration_seconds:.0f}s\n"
            f"{table}\n"
            f"  winner={outcome.winner_uid if outcome.winner_uid is not None else 'none'} top_group={outcome.top_group_uids}"
        )

    def _score(self, result: ModelEvalResult, model: torch.nn.Module, data: EvalData) -> None:
        metrics = evaluate_model(model, data, device=self.device, batch_size=self.config.batch_size)
        result.imagenet_top1 = metrics["imagenet_top1"]
        result.adv_top1 = metrics["adv_top1"]
        result.adv_clean_top1 = metrics["adv_clean_top1"]
        result.overall = overall_score(
            imagenet_top1=result.imagenet_top1,
            adv_top1=result.adv_top1,
            imagenet_weight=self.config.imagenet_weight,
            adv_weight=self.config.adv_weight,
        )


def evaluation_date(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")


def evaluation_due(*, last_date: str, hour_utc: int, now: datetime | None = None) -> bool:
    current = now or datetime.now(timezone.utc)
    if current.hour < int(hour_utc):
        return False
    return evaluation_date(current) != str(last_date or "")
