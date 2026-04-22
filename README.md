# Perturb Subnet

Perturb is a Bittensor subnet where validators create adversarial-image challenges and miners return perturbed images under bounded distortion constraints.

This repository provides:

- validator node implementation (`neurons/validator.py`)
- baseline miner implementation (`neurons/miner.py`)
- validator-side local LLM semantic verification service (`tools/llm_endpoint_service.py`)
- one-command launchers for validator, miner, and llm endpoint

## Architecture

### Validator responsibilities

- Pull challenge images from external API (`PERTURB_IMAGE_ENDPOINT`)
- Run fixed classifier (`EfficientNet-B5`) on pulled image
- Verify semantic consistency of model output vs prompt label through local `llm_endpoint`
- Build and broadcast `AttackChallenge` synapse to selected miners
- Verify miner responses and compute rewards
- Maintain rolling histories and set on-chain weights periodically

### Miner responsibilities

- Receive `AttackChallenge` over Axon
- Run baseline PGD-style attack
- Return only `perturbed_image_b64`
- Let validator handle all authoritative verification and scoring

### Challenge lifecycle

1. Validator samples a prompt from `perturbnet/constants.py` (`PROMPTS`)
2. Validator fetches image from API with `prompt`, `seed`, `image_size`, `random_mode=true`
3. If API pull fails, validator falls back to `assets/dog_1.jpg` and sets prompt to `dog`
4. Validator runs `EfficientNet-B5` and gets exact model label string
5. Validator calls local `llm_endpoint` (`POST /verify-label`) to confirm semantic match between model label and prompt
6. On success, validator creates challenge where `true_label` is the exact EfficientNet label
7. Validator sends challenge to sampled miners and scores returned perturbations

## Hardware and System Requirements

### Miner

- Minimum: 4 vCPU, 16 GB RAM, 50 GB SSD, stable 20+ Mbps network
- Recommended: 8 vCPU, 32 GB RAM, NVIDIA GPU with 8+ GB VRAM, 100+ GB SSD

### Validator

- Minimum: 8 vCPU, 32 GB RAM, NVIDIA GPU with 12+ GB VRAM, 100 GB SSD
- Recommended: 16 vCPU, 64 GB RAM, NVIDIA GPU with 24+ GB VRAM, 200 GB SSD

### Validator-side llm_endpoint

- Minimum: 2 vCPU, 4 GB RAM (assuming model already served by Ollama)
- Recommended: run on same private network/host as validator for low latency

### Common software prerequisites

- Python 3.10+
- `pip` and virtualenv support (`python -m venv`)
- OS build tools needed by Python wheels
- For GPU usage: correct NVIDIA driver + CUDA stack compatible with installed PyTorch

## Installation and Setup (Validator Side)

This section is specifically for validator operators.

### 1) Clone and enter repository

```bash
git clone https://github.com/0xsigurd/Perturb
cd Perturb
```

### 2) Configure and run local llm_endpoint

Create endpoint config:

```bash
cp scripts/llm_endpoint.env.example scripts/llm_endpoint.env
```

Edit `scripts/llm_endpoint.env`:

- `LLM_ENDPOINT_HOST` (default `127.0.0.1`)
- `LLM_ENDPOINT_PORT` (default `8081`)
- `OLLAMA_URL` (default `http://127.0.0.1:11434`)
- `PERTURB_LLM_ENDPOINT_MODEL` (default `qwen2.5:1.5b-instruct`)

Start llm_endpoint:

```bash
./scripts/run_llm_endpoint.sh
```

Health check:

```bash
curl "http://127.0.0.1:8081/health"
```

### 3) Configure validator runtime

Create validator env:

```bash
cp scripts/validator.env.example scripts/validator.env
```

Edit required fields in `scripts/validator.env`:

- `WALLET_NAME`
- `WALLET_HOTKEY`
- `NETUID`
- `NETWORK`

Important validator-specific fields:

- `PERTURB_IMAGE_ENDPOINT`
- `PERTURB_LLM_ENDPOINT_URL` (must point to your running llm endpoint, e.g. `http://127.0.0.1:8081/verify-label`)
- `PERTURB_LLM_ENDPOINT_MODEL`
- `PERTURB_K_MINERS`
- `PERTURB_HISTORY_SIZE`
- `PERTURB_MIN_PROCESSED_COUNT`
- `PERTURB_MIN_LINF_DELTA`
- `PERTURB_MAX_LINF_DELTA`

### 4) Start validator

```bash
./scripts/run_validator_node.sh
```

Expected log behavior:

- challenge generation messages
- miner selection messages
- per-miner score logs
- periodic `set_weights` attempts

### 5) Validator-side notes

- Verification is LLM-only by design; if llm_endpoint is down, challenge verification fails.
- Keep fallback image `assets/dog_1.jpg` present for external image API outage handling.

## Installation and Setup (Miner Side)

This section is specifically for miner operators.

### 1) Clone and enter repository

```bash
git clone https://github.com/0xsigurd/Perturb
cd Perturb
```

### 2) Configure miner runtime

Create miner env:

```bash
cp scripts/miner.env.example scripts/miner.env
```

Edit required fields in `scripts/miner.env`:

- `WALLET_NAME`
- `WALLET_HOTKEY`
- `NETUID`
- `NETWORK`

Optional:

- `PYTHON_BIN`
- `MINER_EXTRA_ARGS`

### 3) Start miner

```bash
./scripts/run_miner_node.sh
```

Expected log behavior:

- `Serving miner axon...`
- `Miner started. Waiting for validator queries.`

### 4) Miner-side notes

- Baseline miner is intentionally simple; competitive miners should optimize attack logic.
- Miner does not run llm_endpoint; semantic verification is validator-side only.

## API and Protocol Contracts

### Image API contract (validator input source)

- Request params: `prompt`, `seed`, `image_size`, `random_mode`
- Response JSON must include:

```json
{"image_base64":"<base64 image bytes>"}
```

### llm_endpoint contract (validator verification)

- Endpoint: `POST /verify-label`
- Request JSON:

```json
{
  "prediction": "<efficientnet_label>",
  "target_label": "<prompt_label>",
  "llm_model": "<optional model hint>"
}
```

- Response JSON must contain a boolean verdict key, typically:

```json
{
  "is_match": true,
  "reason": "short explanation",
  "method": "ollama"
}
```

Operations endpoints:

- `GET /health`
- `GET /metrics`

### Synapse contract (`AttackChallenge`)

Key fields sent to miners:

- `task_id`
- `model_name` (fixed `EfficientNet-B5`)
- `prompt` (broad label)
- `clean_image_b64`
- `true_label` (exact EfficientNet class label)
- `epsilon`, `norm_type`, `min_delta`, `timeout_seconds`

Miner response field:

- `perturbed_image_b64`

## Scoring and Weighting

Per-response score (if verification passes):

- `perturbation_score = 1 - min(norm / epsilon, 1)`
- `speed_score = 1 - min(response_time / timeout, 1)`
- `final = 0.65 * perturbation_score + 0.35 * speed_score`

Any verification or constraint failure gets `0.0`.

Weight setting:

- Only miners with `processed_count > 100` are weight-eligible
- Rank bonuses: `50, 30, 10, 5 (ranks 4-10), 3 (remaining)`
- Final weights combine normalized rolling average and normalized rank bonus, then normalize to sum 1

## One-Command Launchers

Recommended launch order:

```bash
./scripts/run_llm_endpoint.sh
./scripts/run_miner_node.sh
./scripts/run_validator_node.sh
```

If scripts are not executable:

```bash
chmod +x scripts/run_llm_endpoint.sh scripts/run_miner_node.sh scripts/run_validator_node.sh
```

## Integration Smoke Test

Run after llm_endpoint is up:

```bash
python scripts/integration_smoke_test.py
```

The smoke test validates:

- llm_endpoint health and semantic sanity checks
- image fetch from configured image endpoint
- local EfficientNet-B5 inference path
- challenge semantic verification through llm endpoint

## Troubleshooting

- Validator fails verification loop: check `PERTURB_LLM_ENDPOINT_URL` and llm_endpoint health.
- Frequent API failures: verify `PERTURB_IMAGE_ENDPOINT`; fallback should load `assets/dog_1.jpg`.
- No miner scoring activity: ensure miner hotkeys are registered and publicly reachable.
- Dependency install issues: install CUDA/CPU-specific PyTorch build compatible with your host.
- Slow verifier responses: reduce model size or place llm_endpoint closer to validator process.

## Readiness

Use `docs/READINESS_CHECKLIST.md` before long-run validation or deployment.

## Repository Map

- `neurons/validator.py`: validator loop, challenge build, verification, scoring, set_weights
- `neurons/miner.py`: baseline miner logic and Axon serving
- `perturbnet/protocol.py`: `AttackChallenge` synapse schema
- `perturbnet/model.py`: EfficientNet model load and label prediction helpers
- `perturbnet/image_io.py`: base64 image encode/decode helpers
- `tools/llm_endpoint_service.py`: validator-side semantic verification service
- `scripts/run_llm_endpoint.sh`: one-command llm_endpoint launcher
- `scripts/run_validator_node.sh`: one-command validator launcher
- `scripts/run_miner_node.sh`: one-command miner launcher
- `scripts/integration_smoke_test.py`: local integration test

