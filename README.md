# Perturb Subnet

Perturb is a decentralized adversarial robustness network built on Bittensor. Miners compete to find adversarial examples — imperceptible input perturbations that cause state-of-the-art image classifiers to fail — while validators construct challenges from real images, verify every response with mathematical precision, and reward the best attackers with on-chain emissions.

Modern AI models achieve remarkable accuracy on clean data yet remain catastrophically brittle: a perturbation invisible to any human observer can make a production classifier misclassify a tumor scan, a stop sign, or a fraudulent transaction. The tooling to systematically discover these vulnerabilities is fragmented, expensive, and static. Perturb replaces it with a financially incentivized, continuously improving adversarial testing network — every day miners compete, attacks get stronger and the network's outputs get more valuable.

The network produces two commercially valuable outputs:

- **Adversarial training dataset** — a continuously growing corpus of verified adversarial examples, the raw material for adversarial training (the most effective known defense)
- **Model robustness certificates** — on-chain, auditable proof of adversarial evaluation, relevant to EU AI Act conformity and enterprise AI procurement

Why Bittensor: finding an adversarial example is computationally hard, but verifying one is trivially cheap — run the model, compare the prediction, measure the perturbation norm. This verification asymmetry makes the incentive mechanism clean, objective, and manipulation-resistant, while TAO emissions drive a level of continuous attack research no salaried red team can match.

Read the full vision and roadmap in the [Perturb whitepaper](https://www.perturbai.io/whitepaper).

This repository provides:

- validator node implementation (`neurons/validator.py`)
- baseline miner implementation (`neurons/miner.py`)
- one-command launchers for validator and miner

## Architecture

### Validator responsibilities

- Sample challenge images from the full ImageNet-100 train split (~126k images, auto-downloaded)
- Run fixed classifier (`EfficientNetV2-L`) on pulled image
- Build and broadcast `AttackChallenge` synapse to selected miners
- Verify miner responses and compute rewards
- Maintain rolling histories and set on-chain weights periodically

### Miner responsibilities

- Poll the task API for the current task
- Run baseline PGD-style attack
- Upload the perturbed image and submit its URL
- Let validator handle all authoritative verification and scoring

### Challenge lifecycle

1. The team task generator samples an ImageNet-100 image and publishes one API task
2. Miners poll the task API, perturb the task image, upload the result, and submit the image URL
3. Validators read submitted miner images from the API
4. Validators score responses and report the full miner results

## Hardware and System Requirements

### Miner

- Minimum: 4 vCPU, 16 GB RAM, 50 GB SSD, stable 20+ Mbps network
- Recommended: 8 vCPU, 32 GB RAM, NVIDIA GPU with 8+ GB VRAM, 100+ GB SSD

### Validator

- Minimum: 8 vCPU, 32 GB RAM, NVIDIA GPU with 12+ GB VRAM, 100 GB SSD
- Recommended: 16 vCPU, 64 GB RAM, NVIDIA GPU with 24+ GB VRAM, 200 GB SSD

### Common software prerequisites

- Python 3.10+
- Node.js 18+ (includes `npm`) for PM2 installation
- `pip` and virtualenv support (`python -m venv`)
- OS build tools needed by Python wheels
- For GPU usage: correct NVIDIA driver + CUDA stack compatible with installed PyTorch

## Common Installation (Do Once)

Run role-specific setup once before starting nodes:

```bash
git clone https://github.com/0xsigurd/Perturb
cd Perturb
```

For miner setup:

```bash
bash ./scripts/setup_common.sh miner
```

For validator setup:

```bash
bash ./scripts/setup_common.sh validator
```

`setup_common.sh` behavior by role:

- both roles: install PM2, create `.venv`, install Python/Bittensor dependencies

If `npm: command not found`, install Node.js first, then rerun:

macOS (Homebrew):

```bash
brew install node
node --version
npm --version
bash ./scripts/setup_common.sh validator
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y nodejs npm
node --version
npm --version
bash ./scripts/setup_common.sh validator
```

## Installation and Setup (Validator Side)

This section is specifically for validator operators.

### 1) Configure validator runtime

Create validator env:

```bash
cp scripts/validator.env.example scripts/validator.env
```

Edit required fields in `scripts/validator.env`:

- `WALLET_NAME`
- `WALLET_HOTKEY`
- `PERTURB_API_KEY`

Optional:

- `PERTURB_API_BASE_URL`
- `LOG_LEVEL` (`DEBUG` default, set `INFO`/`WARNING`/`ERROR` for quieter logs)

### 2) Start validator

```bash
bash ./scripts/run_validator.sh
```

Expected log behavior:

- API task polling messages
- submitted response scoring logs
- per-miner score logs
- periodic `set_weights` attempts

### 3) Validator-side notes

- Validators fetch the task image and submitted miner image URLs from the API, then run local verification and scoring.
- Validators use miner-submitted response URLs in leaderboard reports.

## Installation and Setup (Miner Side)

This section is specifically for miner operators.

### 1) Configure miner runtime

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
- `LOG_LEVEL` (`DEBUG` default, set `INFO`/`WARNING`/`ERROR` if you want quieter logs)
- `PERTURB_API_BASE_URL`
- Storage credentials (`PERTURB_STORAGE_BACKEND`, `PERTURB_STORAGE_BUCKET`, `PERTURB_STORAGE_ACCESS_KEY_ID`, `PERTURB_STORAGE_SECRET_ACCESS_KEY`; Hippius is default, R2 is supported)
- `MINER_EXTRA_ARGS`

### 2) Start miner

```bash
bash ./scripts/run_miner.sh
```

Expected log behavior:

- `Miner started. Polling task API.`
- task upload/submission messages

### 3) Miner-side notes

- Baseline miner is intentionally simple; competitive miners should optimize attack logic.
- Miners don't serve an axon for challenge handling.
- Validators handle all challenge verification and scoring.

## Task Generation

Task generation is run separately by the team through `generate_and_publish_task(...)`.

The generator samples ImageNet-100, uploads the clean task image with the configured storage settings, and publishes the current API task with the provided hotkeys. Hippius is the default storage backend; set `PERTURB_STORAGE_BACKEND="r2"` to use R2.

## API and Protocol Contracts

### Task API contract

- Task generator publishes the current task with `task_id` and `imageURL`.
- Miners read the current task from `GET /task`.
- Miners submit response image URLs to `POST /submits`.
- Validators read submitted response image URLs from `GET /submits` (Bearer auth, available while task status is `validating`).

### Task generator

Task generation is separated from validator runtime under `task_generator/`. It samples ImageNet-100, uploads the clean task image, and overwrites the current task row through the API.

### Leaderboard reporting

After each scoring round, validators submit a leaderboard report to the API configured in `perturbnet/constants.py`. Reports are queued in a background thread; leaderboard API failures, non-2xx responses, or timeouts are logged and skipped without affecting validator scoring.

Reports include network metrics and full miner details for every registered non-validator UID. Successful responses include presigned response-storage image URLs for UI display; miners without an exported image use the configured placeholder image.


## Scoring and Weighting

Per-response score (if verification passes):

- Hard gates:
  - `min_linf_delta <= norm <= min(epsilon, max_linf_delta)`
  - `ssim(clean, adv) >= min_ssim`
  - `psnr_db(clean, adv) >= min_psnr_db`
  - predicted label must differ from the original label
- `linf_ratio = clamp((norm - min_linf_delta) / (min(epsilon, max_linf_delta) - min_linf_delta), 0, 1)`
- `rmse_ratio = clamp(rmse / min(epsilon, max_linf_delta), 0, 1)`
- `linf_score = (1 - linf_ratio)^2`
- `rmse_score = (1 - rmse_ratio)^2`
- `perturbation_score = weighted_avg(linf_score, rmse_score)` using `PERTURB_LINF_COMPONENT_WEIGHT` and `PERTURB_RMSE_COMPONENT_WEIGHT`
- `margin = best_non_true_logit - true_class_logit`
- `margin_score = clamp(margin / 10, 0, 1)` using `ANALYZE_BUCKET_MARGIN_WEIGHT` (default `0.03`)
- `novelty_score = clamp(changed_pixel_count / ANALYZE_BUCKET_NOVELTY_TARGET_PIXELS, 0, 1)` using `ANALYZE_BUCKET_NOVELTY_WEIGHT` (default `0.01`)
- `final = PERTURB_PERTURBATION_WEIGHT * perturbation_score + margin_weight * margin_score + novelty_weight * novelty_score`

Any verification or constraint failure gets `0.0`.

Labels are normalized with `strip -> lowercase -> replace "_" with " "`. When possible, the validator resolves the true label to an EfficientNet class index and compares class indices instead of only strings, including comma-separated ImageNet label aliases. Miner responses are quantized onto the same uint8 PNG grid used by base64 image submissions before norms are measured, so nonzero `Linf` values are effectively multiples of `1/255`.

Weight setting:

- Miners are weight-eligible once their score history reaches the effective window: `min(PERTURB_HISTORY_SIZE, longest miner history)`. Weight setting starts once the longest history reaches `PERTURB_MIN_WEIGHT_HISTORY_SIZE` (default `50`), so new validators don't wait for the full window, while miners still need as many records as the current longest history (capped at `PERTURB_HISTORY_SIZE`).
- Emission schedule: rank 1 receives `70%`, rank 2 receives `15%`, and the remaining `15%` is split by descending rank weight among all other positive-score eligible miners
- At each weight-setting cycle, the validator fetches `burnRate` from the burn endpoint configured in `perturbnet/constants.py` and assigns that share to the configured burn UID. Miner weights are scaled by `1 - burnRate`, keeping the submitted vector normalized. If the API is unavailable or invalid, the default burn rate from `constants.py` is used instead.

## Integration Smoke Test

Run after setup:

```bash
python scripts/integration_smoke_test.py
```

The smoke test validates:

- local EfficientNetV2-L inference path
- scoring dependencies

## Troubleshooting

- Validator cannot generate challenges: verify internet access to Hugging Face for the first dataset download.
- No miner scoring activity: ensure miner hotkeys are registered and publicly reachable.
- Dependency install issues: install CUDA/CPU-specific PyTorch build compatible with your host.

## Readiness

Use `docs/READINESS_CHECKLIST.md` before long-run validation or deployment.

## Repository Map

- `neurons/validator.py`: validator loop, challenge build, verification, scoring, set_weights
- `neurons/miner.py`: baseline miner logic and Axon serving
- `perturbnet/protocol.py`: `AttackChallenge` synapse schema
- `perturbnet/model.py`: EfficientNet model load and label prediction helpers
- `perturbnet/image_io.py`: base64 image encode/decode helpers
- `perturbnet/imagenet100_bootstrap.py`: ImageNet-100 full-split download/open helpers
- `scripts/run_validator.sh`: start/restart validator with PM2
- `scripts/run_miner.sh`: start/restart miner with PM2
- `scripts/setup_common.sh`: role-aware bootstrap (PM2 + Python deps; `validator` also pre-downloads ImageNet-100)
- `scripts/bootstrap_imagenet100.py`: manual ImageNet-100 pre-download CLI
- `scripts/integration_smoke_test.py`: local integration test

