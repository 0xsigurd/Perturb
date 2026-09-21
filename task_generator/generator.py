from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from perturbnet import constants as C
from perturbnet.api_client import post_task
from perturbnet.image_io import decode_image_b64
from perturbnet.imagenet1k import ImageNet1kRows
from perturbnet.model import load_efficientnet_v2_l, normalize_prediction_label, predict_label
from perturbnet.storage_uploader import ImageStorageUploader


@dataclass(frozen=True)
class GeneratedTask:
    task_id: str
    image_url: str
    image_id: str
    true_label: str


class TaskGenerator:
    """Walks the ImageNet-1k train split in a persisted random order.

    Rows are fetched one at a time through the datasets-server API
    (`ImageNet1kRows`), so no local dataset download is needed. The traversal
    order is a seeded permutation of row indices; seed/cursor live in the state
    file so restarts continue where they left off and no image repeats within
    an epoch of the permutation.
    """

    def __init__(
        self,
        *,
        state_path: str | os.PathLike[str] = "task_generator_state.json",
        rows: ImageNet1kRows | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.system_random = random.SystemRandom()
        self.order_seed = 0
        self.order_cursor = 0
        self.order_fingerprint = ""
        self.order_epoch = 0
        self._order_cache: list[int] = []
        self._order_cache_key: tuple[str, int] = ("", 0)
        self.rows = rows or ImageNet1kRows(timeout_seconds=C.PERTURB_API_TIMEOUT_SECONDS * 3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_efficientnet_v2_l(self.device)
        self._load_state()

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.order_seed = max(0, int(state.get("order_seed", 0)))
        self.order_cursor = max(0, int(state.get("order_cursor", 0)))
        self.order_fingerprint = str(state.get("order_fingerprint", "") or "")
        self.order_epoch = max(0, int(state.get("order_epoch", 0)))

    def _save_state(self) -> None:
        payload = {
            "order_seed": int(self.order_seed),
            "order_cursor": int(self.order_cursor),
            "order_fingerprint": self.order_fingerprint,
            "order_epoch": int(self.order_epoch),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_path, self.state_path)

    def _fingerprint(self) -> str:
        """Identifies the dataset snapshot the persisted order belongs to."""
        digest = hashlib.sha256()
        digest.update(f"{self.rows.repo_id}:{self.rows.split}:{self.rows.num_rows}".encode("utf-8"))
        return digest.hexdigest()

    def _reset_order(self, *, fingerprint: str, epoch: int) -> None:
        self.order_seed = self.system_random.randrange(1, 2**63)
        self.order_cursor = 0
        self.order_fingerprint = fingerprint
        self.order_epoch = epoch
        self._order_cache = []
        self._order_cache_key = ("", 0)

    def _ensure_order(self) -> None:
        total_rows = int(self.rows.num_rows)
        if total_rows <= 0:
            raise RuntimeError(f"{self.rows.repo_id}:{self.rows.split} is empty.")
        fingerprint = self._fingerprint()
        if self.order_fingerprint != fingerprint or self.order_seed <= 0:
            self._reset_order(fingerprint=fingerprint, epoch=0)
        elif self.order_cursor >= total_rows:
            self._reset_order(fingerprint=fingerprint, epoch=self.order_epoch + 1)

        cache_key = (self.order_fingerprint, int(self.order_seed))
        if self._order_cache_key != cache_key or not self._order_cache:
            order = list(range(total_rows))
            random.Random(int(self.order_seed)).shuffle(order)
            self._order_cache = order
            self._order_cache_key = cache_key

    def _sample_image(self) -> tuple[str, str]:
        self._ensure_order()
        row = self._order_cache[self.order_cursor]
        fetched = self.rows.fetch(row)
        image_id = f"hf-{self.rows.version()}-{row:07d}"
        self.order_cursor += 1
        self._save_state()
        return image_id, base64.b64encode(fetched.image_bytes).decode("utf-8")

    def generate(self) -> tuple[str, str, str]:
        for _ in range(C.MAX_CHALLENGE_ATTEMPTS):
            image_id, image_b64 = self._sample_image()
            image = decode_image_b64(image_b64).to(self.device)
            label = normalize_prediction_label(predict_label(self.model, image))
            if label:
                return image_id, image_b64, label
        raise RuntimeError("Unable to generate task after max attempts.")


def generate_and_publish_task(
    *,
    state_path: str | os.PathLike[str] = "task_generator_state.json",
    status: str = "open",
    hotkeys: Sequence[str],
) -> GeneratedTask:
    netuid = int(os.getenv("NETUID", "26"))
    generator = TaskGenerator(state_path=state_path)
    image_id, image_b64, label = generator.generate()
    task_id = f"{int(time.time())}-{image_id}"
    exporter = ImageStorageUploader(
        run_id="task-generator",
        netuid=netuid,
        uploader_hotkey="task-generator",
    )
    image_key = f"{C.STORAGE_PREFIX.strip().strip('/')}/tasks/current.png"
    image_url = exporter.upload_image_b64(key=image_key, image_b64=image_b64)
    post_task(
        base_url=C.PERTURB_API_BASE_URL,
        api_key=C.PERTURB_API_KEY,
        task_id=task_id,
        image_url=image_url,
        status=status,
        hotkeys=[str(hotkey) for hotkey in hotkeys],
        timeout_seconds=C.PERTURB_API_TIMEOUT_SECONDS,
    )
    return GeneratedTask(task_id=task_id, image_url=image_url, image_id=image_id, true_label=label)
