# Perturb Subnet

Perturb is a Bittensor subnet where validators issue adversarial attack challenges and miners return perturbed images that fool a classifier under norm constraints.

This implementation follows Axon/Dendrite communication from the Bittensor subnet template:

- Validator builds a verified `AttackChallenge` synapse from external challenge sources.
- Validator sends synapse to miner axons via dendrite.
- Miner runs a default iterative PGD-style baseline and returns only the perturbed image.
- Validator deterministically verifies and scores responses.

## Hardware Requirements

Basic operator guidance for stable uptime:

- Miner (minimum): 4 CPU cores, 16 GB RAM, 50 GB SSD, stable 20+ Mbps network
- Miner (recommended): 8 CPU cores, 32 GB RAM, NVIDIA GPU with 8+ GB VRAM, 100+ GB SSD
- Validator (minimum): 8 CPU cores, 32 GB RAM, NVIDIA GPU with 12+ GB VRAM, 100+ GB SSD
- Validator (recommended): 16 CPU cores, 64 GB RAM, NVIDIA GPU with 24+ GB VRAM, 200+ GB SSD
- Validator llm_endpoint: run locally and expose `POST /verify-label` on the configured endpoint
- llm_endpoint service (minimum): 2 CPU cores, 4 GB RAM (when Ollama model already loaded)

## Task Definition (v1)

- **Input**: image generated from prompt labels (example: `dog`, `cat`, `bird`, `fish`, `reptile`, `amphibian`, ...)
- **Model name**: fixed to `EfficientNet-B5`
- **Constraint**: `Linf` perturbation budget (`epsilon`) and minimum perturbation delta
- **Success**: prediction on perturbed image differs from the original EfficientNet class label

Validator challenge construction:

1. choose random prompt from constants
2. pull image from image endpoint using prompt input (`prompt`, `seed`, `image_size`)
   - if image API pull fails, validator falls back to local `assets/dog_1.jpg` and uses prompt label `dog`
3. use fixed model `EfficientNet-B5`
4. verify challenge is consistent:
   - classifier prediction semantically matches prompt target label via `llm_endpoint`
   - validator stores exact EfficientNet class label as challenge `true_label`
   - validator-hosted LLM endpoint confirms parsed label match (strict mode by default)
5. retry until a valid challenge is formed

## Scoring

For successful attacks:

- perturbation score = `1 - (norm / epsilon)` (weight 0.65)
- speed score = `1 - min(response_time / timeout, 1.0)` (weight 0.35)
- final challenge score = `0.65 * perturbation + 0.35 * speed`

Constraint violations, failed attacks, too-small perturbations, and excessive deltas score `0.0`.

Miner selection and weighting:

- validator randomly samples `K` miners per round (prefers miners with `processed_count > 100`)
- all queried miners update `processed_count` and rolling score history
- only miners with `processed_count > 100` are eligible for emissions weighting
- eligible miners are ranked by average score over last 100 responses
- rank points: `50, 30, 10, 5 (next 7), 3 (remaining)`
- final on-chain weights combine normalized last-100 average and normalized rank bonus

## API Contract

Image endpoint must return image only:

- query params expected by validator: `prompt`, `seed`, `image_size`, `random_mode`
- JSON response expected: `{ "image_base64": "<base64 image>" }`

llm_endpoint (validator-hosted, local network recommended):

- default endpoint: `http://127.0.0.1:8081/verify-label`
- default model hint in request payload: `Qwen2.5-1.5B-Instruct`
- validator verification is LLM-only (no local fallback matching)
- accepted response formats:
  - `{ "is_match": true|false }`
  - `{ "label": "<normalized_label>" }`
  - raw boolean or raw label string
- additional ops endpoints:
  - `GET /health`
  - `GET /metrics`

## Step-by-Step: Miner Node

1. Install system dependencies:
   - Python 3.10+
   - build tools required by `pip` packages (if your OS needs them)
2. Create miner runtime config:
   - `cp scripts/miner.env.example scripts/miner.env`
   - edit `WALLET_NAME`, `WALLET_HOTKEY`, `NETUID`, `NETWORK`
3. Start miner with one command:
   - `./scripts/run_miner_node.sh`
4. Confirm miner is serving:
   - log should show `Serving miner axon...` and `Miner started. Waiting for validator queries.`

## Step-by-Step: Validator Node

1. Install system dependencies:
   - Python 3.10+
   - GPU drivers/CUDA stack appropriate for your PyTorch install
2. Start your local llm_endpoint service:
   - `cp scripts/llm_endpoint.env.example scripts/llm_endpoint.env`
   - edit `OLLAMA_URL`/model values as needed
   - run `./scripts/run_llm_endpoint.sh`
   - verify with `curl http://127.0.0.1:8081/health`
3. Create validator runtime config:
   - `cp scripts/validator.env.example scripts/validator.env`
   - edit `WALLET_NAME`, `WALLET_HOTKEY`, `NETUID`, `NETWORK`
   - edit `PERTURB_LLM_ENDPOINT_URL` and optionally `PERTURB_LLM_ENDPOINT_MODEL`
4. Start validator with one command:
   - `./scripts/run_validator_node.sh`
5. Confirm validator loop:
   - log should show challenge creation, miner selection, response scoring, and periodic `set_weights` calls

## One-Command Launchers

The scripts below are the recommended way to run nodes:

```bash
./scripts/run_llm_endpoint.sh
./scripts/run_miner_node.sh
./scripts/run_validator_node.sh
```

They perform:

- load `scripts/*.env` configuration
- create `.venv` if needed
- install dependencies from `requirements.txt`
- run the correct node entrypoint

If script execution is blocked, run once:

```bash
chmod +x scripts/run_llm_endpoint.sh scripts/run_miner_node.sh scripts/run_validator_node.sh
```

## Integration Smoke Test

After llm_endpoint is running, execute:

```bash
python scripts/integration_smoke_test.py
```

This checks:

- llm_endpoint health endpoint
- semantic match behavior (`irish terrier` vs `dog`)
- image fetch from `PERTURB_IMAGE_ENDPOINT`
- local EfficientNet-B5 inference
- challenge semantic verification through llm_endpoint

Manual run (advanced):

Install:

```bash
pip install -r requirements.txt
```

Run miner:

```bash
python neurons/miner.py --netuid 1 --network local --wallet.name miner --wallet.hotkey default
```

Run validator:

```bash
python neurons/validator.py \
  --netuid 1 \
  --network local \
  --wallet.name validator \
  --wallet.hotkey default
```

## Troubleshooting

- If validator cannot reach llm_endpoint, challenge verification fails by design until endpoint recovers.
- If no miners are selected, confirm miner hotkeys are registered and axons are reachable.
- If challenge generation fails repeatedly, verify `PERTURB_IMAGE_ENDPOINT` returns `image_base64`.
- If `torch` install fails, install the CUDA/CPU build that matches your host before rerunning scripts.
- Ensure fallback image exists at `assets/dog_1.jpg` for API outage handling.

## Readiness Checklist

Use `docs/READINESS_CHECKLIST.md` before long-running tests or deployment.

## Files

- `perturbnet/protocol.py`: Synapse schema
- `perturbnet/model.py`: Classifier + label helpers
- `perturbnet/image_io.py`: Base64 image encoding/decoding
- `neurons/miner.py`: iterative PGD-style miner
- `neurons/validator.py`: challenge generation, verification, scoring, and weight setting
- `tools/llm_endpoint_service.py`: validator-hosted llm_endpoint service (`/verify-label`, `/health`, `/metrics`)
- `scripts/integration_smoke_test.py`: local end-to-end challenge verification test

