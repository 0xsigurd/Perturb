"""Row-level access to ImageNet-1k on Hugging Face without downloading the split.

The task generator needs one random training image every couple of minutes.
The full train split is ~150 GB, so instead of `load_dataset` we ask the
datasets-server API for single rows by offset (`/rows?offset=N&length=1`) and
download the returned image asset. ImageNet-1k is gated: requests carry
`HF_TOKEN`, which must belong to an account that accepted the dataset terms.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any

import requests
from PIL import Image

from perturbnet.constants import HF_TOKEN, IMAGENET1K_REPO_ID, IMAGENET1K_SPLIT

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co"


@dataclass(frozen=True)
class ImageNetRow:
    row: int
    image_bytes: bytes  # RGB JPEG
    label: int
    label_name: str


class ImageNet1kRows:
    def __init__(
        self,
        *,
        repo_id: str = IMAGENET1K_REPO_ID,
        split: str = IMAGENET1K_SPLIT,
        token: str = HF_TOKEN,
        timeout_seconds: float = 30.0,
        jpeg_quality: int = 95,
    ) -> None:
        self.repo_id = repo_id
        self.split = split
        self.token = str(token or "").strip()
        self.timeout_seconds = float(timeout_seconds)
        self.jpeg_quality = int(jpeg_quality)
        self._num_rows: int | None = None
        self._label_names: list[str] = []

    # ---- HTTP ---------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _rows_page(self, offset: int, length: int = 1) -> dict[str, Any]:
        response = requests.get(
            f"{DATASETS_SERVER_URL}/rows",
            params={"dataset": self.repo_id, "config": "default", "split": self.split, "offset": int(offset), "length": int(length)},
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        if response.status_code < 200 or response.status_code >= 300:
            hint = " (gated dataset: set HF_TOKEN for an account that accepted the ImageNet terms)" if response.status_code in (401, 403, 404) else ""
            raise RuntimeError(f"datasets-server HTTP {response.status_code}{hint}: {response.text[:200]}")
        payload = response.json()
        if not self._label_names:
            for feature in payload.get("features", []):
                if feature.get("name") == "label":
                    self._label_names = [str(name) for name in feature.get("type", {}).get("names", [])]
        if self._num_rows is None and payload.get("num_rows_total") is not None:
            self._num_rows = int(payload["num_rows_total"])
        return payload

    # ---- public -------------------------------------------------------------

    @property
    def num_rows(self) -> int:
        if self._num_rows is None:
            self._rows_page(0)
            if self._num_rows is None:
                raise RuntimeError(f"datasets-server did not report num_rows_total for {self.repo_id}:{self.split}")
        return int(self._num_rows)

    @property
    def label_names(self) -> list[str]:
        if not self._label_names:
            self._rows_page(0)
        return list(self._label_names)

    def version(self) -> str:
        """Stable short identifier for this dataset snapshot (repo/split/row-count only,
        so a persisted traversal survives restarts and library upgrades)."""
        base = f"{self.repo_id}:{self.split}:{self.num_rows}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

    def fetch(self, row: int) -> ImageNetRow:
        payload = self._rows_page(int(row))
        rows = payload.get("rows") or []
        if not rows:
            raise ValueError(f"{self.repo_id}:{self.split} has no row {row}")
        record = rows[0].get("row", {})
        image = record.get("image") or {}
        src = str(image.get("src") or "").strip()
        if not src:
            raise ValueError(f"{self.repo_id}:{self.split} row {row} has no image asset")
        asset = requests.get(src, headers=self._headers(), timeout=self.timeout_seconds)
        if asset.status_code < 200 or asset.status_code >= 300:
            raise RuntimeError(f"image asset HTTP {asset.status_code} for row {row}")
        # Normalize to RGB JPEG the same way the generator always has.
        buffer = io.BytesIO()
        Image.open(io.BytesIO(asset.content)).convert("RGB").save(buffer, format="JPEG", quality=self.jpeg_quality)
        label = record.get("label")
        label_index = int(label) if label is not None else -1
        names = self.label_names
        label_name = names[label_index] if 0 <= label_index < len(names) else ""
        return ImageNetRow(row=int(row), image_bytes=buffer.getvalue(), label=label_index, label_name=label_name)
