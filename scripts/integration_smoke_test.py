from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    print("[1/3] Fetch an ImageNet-1k challenge candidate (datasets-server, needs HF_TOKEN)")
    from perturbnet.imagenet1k import ImageNet1kRows

    rows = ImageNet1kRows()
    fetched = rows.fetch(0)
    print(f"  dataset rows={rows.num_rows} sample=row 0 label={fetched.label_name!r}")
    image_b64 = base64.b64encode(fetched.image_bytes).decode("utf-8")

    print("[2/3] Run EfficientNetV2-L inference")
    import torch

    from perturbnet.image_io import decode_image_b64
    from perturbnet.model import load_efficientnet_v2_l, predict_label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet_v2_l(device=device)
    image = decode_image_b64(image_b64).to(device)
    prediction = predict_label(model=model, image_chw=image)
    print(f"  model_prediction={prediction}")

    print("[3/3] Challenge target label will use model prediction directly")
    if not prediction:
        raise RuntimeError("Model prediction is empty")
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        exit_code = 1
    # Hard-exit: pyarrow/datasets can leave non-joinable IO threads that
    # deadlock normal interpreter shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
