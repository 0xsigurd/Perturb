"""Verify ImageNet-1k access for the task generator / validator.

Nothing is downloaded beyond a single row: the check confirms that HF_TOKEN can
read the gated dataset through the datasets-server API and that the image
asset is retrievable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturbnet.constants import HF_TOKEN, IMAGENET1K_REPO_ID, IMAGENET1K_SPLIT
from perturbnet.imagenet1k import ImageNet1kRows


def main() -> int:
    if not HF_TOKEN:
        print("HF_TOKEN is not set; ImageNet-1k is gated and requires a token for an account that accepted its terms.", file=sys.stderr)
        return 1
    rows = ImageNet1kRows(repo_id=IMAGENET1K_REPO_ID, split=IMAGENET1K_SPLIT)
    fetched = rows.fetch(0)
    print(
        f"ImageNet-1k ready: repo={IMAGENET1K_REPO_ID} split={IMAGENET1K_SPLIT} rows={rows.num_rows} "
        f"classes={len(rows.label_names)} sample_label={fetched.label_name!r} sample_bytes={len(fetched.image_bytes)}"
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"ImageNet-1k check failed: {exc}", file=sys.stderr)
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
