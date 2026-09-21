"""Upload a trained model to Hugging Face and commit it on-chain. Configured by .env only."""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

from common import CONFIG_FILE, MODEL_FILE, MODEL_HASH_CHARS, MinerChainCommit, sha256_model_and_hotkey, validate_chain_commit_payload


def env(key: str, default: str | None = None) -> str:
    value = (os.getenv(key) or "").strip()
    if value:
        return value
    if default is None:
        sys.exit(f"{key} is not set in .env (see .env.example)")
    return default


def model_card(repo_id: str, hotkey: str, netuid: int, model_hash: str) -> str:
    return f"""---
license: apache-2.0
library_name: torchvision
tags: [image-classification, adversarial-training, efficientnet, perturb, bittensor]
---

# {repo_id}

EfficientNetV2-L (torchvision `efficientnet_v2_l`, 1000 ImageNet classes) fine-tuned with adversarial
training for the [Perturb](https://perturbai.io) subnet (netuid {netuid}).

- miner hotkey: `{hotkey}`
- on-chain hash: sha256(model.safetensors || hotkey) = `{model_hash}`
- preprocessing: `EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()` (bicubic resize 480, center crop 480, mean = std = 0.5)

```python
from safetensors.torch import load_file
from torchvision.models import efficientnet_v2_l
model = efficientnet_v2_l(weights=None)
model.load_state_dict(load_file("{MODEL_FILE}"))
```
"""


def main() -> None:
    load_dotenv()
    hf_token, repo_id = env("HF_TOKEN"), env("HF_REPO_ID")
    model_dir = Path(env("MODEL_DIR", "runs/v1/last"))
    netuid, network = int(env("NETUID", "26")), env("SUBTENSOR_NETWORK", "finney")
    for name in (MODEL_FILE, CONFIG_FILE):
        if not (model_dir / name).exists():
            sys.exit(f"{model_dir / name} not found (run train.py first, or set MODEL_DIR)")

    import bittensor as bt

    wallet = bt.Wallet(name=env("WALLET_NAME", "miner"), hotkey=env("WALLET_HOTKEY", "default"), path=env("WALLET_PATH", "~/.bittensor/wallets"))
    hotkey = wallet.hotkey.ss58_address
    model_hash = sha256_model_and_hotkey(model_dir / MODEL_FILE, hotkey)
    (model_dir / "README.md").write_text(model_card(repo_id, hotkey, netuid, model_hash), encoding="utf-8")

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    if not api.repo_exists(repo_id, repo_type="model"):
        api.create_repo(repo_id, repo_type="model", private=True)
    api.update_repo_settings(repo_id, repo_type="model", private=True)
    print(f"[1/2] uploading {model_dir} to https://huggingface.co/{repo_id} (private)")
    revision = api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(model_dir),
        allow_patterns=[MODEL_FILE, CONFIG_FILE, "README.md"], commit_message="Perturb submission",
    ).oid
    print(f"      revision {revision}")

    subtensor = bt.Subtensor(network=network)
    uid = subtensor.get_uid_for_hotkey_on_subnet(hotkey, netuid)
    if uid is None:
        sys.exit(f"hotkey {hotkey} is not registered on netuid {netuid}; repo stays private")
    commit = MinerChainCommit(hf_repo_id=repo_id, hf_revision=revision, model_hash=model_hash[:MODEL_HASH_CHARS])
    _, data = validate_chain_commit_payload(commit)
    print(f"[2/2] committing {data} as uid {uid} ({hotkey})")
    response = subtensor.set_commitment(wallet=wallet, netuid=netuid, data=data, raise_error=False)
    if not response.success:
        sys.exit(f"chain commit failed: {response.message}; repo stays private, re-run to retry")

    block = subtensor.get_current_block()
    while subtensor.get_current_block() <= block:
        time.sleep(3)
    api.update_repo_settings(repo_id, repo_type="model", private=False)
    print(f"done: https://huggingface.co/{repo_id}/tree/{revision} is public (block {block + 1})")


if __name__ == "__main__":
    try:
        main()
        code = 0
    except SystemExit as exc:
        code = 0 if exc.code in (None, 0) else 1
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
    except Exception:
        traceback.print_exc()
        code = 1
    # Hard-exit: the Subtensor websocket leaves a non-daemon thread that blocks
    # normal interpreter shutdown, so the process would hang after the last print.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
