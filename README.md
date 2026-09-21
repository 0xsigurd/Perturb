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
- miner model-track scripts: fine-tune, evaluate and submit an adversarially trained EfficientNetV2-L (`training/`)
- one-command launchers for validator and miner

Miners are rewarded on two tracks, both scored by validators:

- **Adversarial scanning (80% of miner emissions)**: every two minutes, find an imperceptible perturbation of the current task image that flips EfficientNetV2-L's prediction.
- **Adversarial training (20% of miner emissions)**: commit a fine-tuned EfficientNetV2-L on-chain; once a day validators evaluate every committed model on ImageNet-1k and the latest adversarial dataset, and the best model of the day takes the whole model share.

## Architecture

### Validator responsibilities

- Fetch the current task image (an ImageNet-1k training image published by the task generator) and run the fixed classifier (`EfficientNetV2-L`) on it
- Verify every submitted miner image and compute scanning rewards
- Once a day (00:00 UTC), evaluate all committed miner models and pick the day's model winner
- Report scanning results and model evaluations to the API; set on-chain weights once per epoch from the cross-validator consensus

### Miner responsibilities

- Poll the task API for the current task
- Run baseline PGD-style attack
- Upload the perturbed image and submit its URL
- Optionally, fine-tune EfficientNetV2-L with `training/train.py` and publish it with `training/submit.py`
- Let validator handle all authoritative verification and scoring

### Challenge lifecycle

1. The team task generator samples an ImageNet-1k train image and publishes one API task
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
bash ./scripts/setup_common.sh
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y nodejs npm
node --version
npm --version
bash ./scripts/setup_common.sh
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
- `HF_TOKEN`: a Hugging Face token whose account has accepted the terms of the gated [`ILSVRC/imagenet-1k`](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset. The daily model evaluation streams ImageNet-1k validation samples with it; `python scripts/check_imagenet1k.py` verifies access.

Optional:

- `PERTURB_API_BASE_URL`
- `LOG_LEVEL` (`DEBUG` default, set `INFO`/`WARNING`/`ERROR` for quieter logs)
- `PERTURB_MODEL_EVAL_*` (see *Adversarial training track* below)

### 2) Start validator

```bash
bash ./scripts/run_validator.sh
```

Expected log behavior:

- API task polling messages
- submitted response scoring logs
- per-miner score logs
- once a day, `model_evaluation date=...` followed by one `Model evaluated uid=...` line per committed model and a `model_evaluation_summary`
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

The generator samples the ImageNet-1k train split (1.28M images), uploads the clean task image with the configured storage settings, and publishes the current API task with the provided hotkeys. Rows are fetched one at a time through the Hugging Face datasets-server API, so the 150 GB split is never downloaded; `HF_TOKEN` in `task_generator/task_generator.env` must have accepted the ImageNet-1k terms. The traversal is a persisted random permutation of row indices (`task_generator_state.json`), so no image repeats until the whole split has been used. Hippius is the default storage backend; set `PERTURB_STORAGE_BACKEND="r2"` to use R2.

## API and Protocol Contracts

### Task API contract

- Task generator publishes the current task with `task_id` and `imageURL`.
- Miners read the current task from `GET /task`.
- Miners submit their response image URL plus `imageHash` to `POST /submits`. The hash is sha256 over the decoded RGB pixel buffer of the perturbed image (see `image_pixel_hash` in `perturbnet/image_io.py`), so it is stable across lossless PNG re-encodes.
- Validators read submitted response image URLs and hashes from `GET /submits` (Bearer auth, available while task status is `validating`). At evaluation time the validator recomputes the pixel hash from the downloaded image; a missing hash (`image_hash_missing`) or a mismatch (`image_hash_mismatch`) zeroes the submission. This prevents miners from submitting a URL early and swapping the image content behind it afterwards.

### Task generator

Task generation is separated from validator runtime under `task_generator/`. It samples ImageNet-1k, uploads the clean task image, and overwrites the current task row through the API.

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

- Weights are set once per epoch, inside the final `PERTURB_WEIGHT_WINDOW_BLOCKS` (default `100`) blocks before the next epoch boundary, so all validators submit at roughly the same moment. The boundary is computed from the chain's own counters (`last_step + tempo`); `scripts/epoch_countdown.py` prints the live countdown.
- At weight-setting time, the validator fetches every validator's leaderboard report from `GET /leaderboard/<validator_hotkey>` and computes a stake-weighted average of each miner's `avgScore` across all reporting validators (a validator's report counts proportionally to its stake). These consensus averages are the only input to weight setting; validators whose leaderboard fetch fails are skipped. If no consensus data is available at all, weight setting is skipped for that cycle.
- Validators are identified by validator permit plus a minimum stake of `10,000`.
- History gating happens at reporting time, not weight-setting time: a miner's reported `avgScore` averages over `min(PERTURB_HISTORY_SIZE, longest miner history)` records and is `0` until the miner reaches that window (or until any miner reaches `PERTURB_MIN_WEIGHT_HISTORY_SIZE`, default `50`).
- Scanning emission schedule: rank 1 receives `75%`, rank 2 receives `20%`, rank 3 receives `5%`; every other miner receives no scanning share (with one or two positive miners the missing ranks roll up to the last one present).
- Track blend: `miner weights = PERTURB_SCANNING_EMISSION_SHARE (0.8) x scanning shares + PERTURB_MODEL_EMISSION_SHARE (0.2) x {model winner: 1}`. Both inputs are stake-weighted consensus values across validators (scanning: leaderboard `avg_score`s; model: `overall` scores from the evaluation reports, then the same epsilon-group / earliest-block rule). A uid that leads both tracks receives `0.8 x 0.75 + 0.2 = 0.8` of the miner allocation. Without a model winner the model share is not awarded and scanning receives the whole miner allocation; without any positive scanning miner the model winner receives it.
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

- `datasets-server HTTP 401/403/404 ... gated dataset`: `HF_TOKEN` is missing or its account has not accepted the ImageNet-1k terms; run `python scripts/check_imagenet1k.py`.
- `Model evaluation failed ...`: the commitments API, the adversarial dataset or ImageNet-1k was unreachable; the previous winner stays in place until the next daily run (no retry).
- No miner scoring activity: ensure miner hotkeys are registered and publicly reachable.
- Dependency install issues: install CUDA/CPU-specific PyTorch build compatible with your host.

## Readiness

Use `docs/READINESS_CHECKLIST.md` before long-run validation or deployment.

## Repository Map

- `neurons/validator.py`: validator loop, challenge build, verification, scoring, set_weights
- `neurons/miner.py`: baseline miner logic and Axon serving
- `perturbnet/protocol.py`: `AttackChallenge` synapse schema
- `perturbnet/model.py`: EfficientNet model load and label prediction helpers
- `perturbnet/model_commit.py`: on-chain model commitment schema and `sha256(model || hotkey)` (shared by validator and `training/submit.py`)
- `perturbnet/model_evaluation.py`: daily model evaluation (commitment verification, evaluation data, scoring, winner selection)
- `perturbnet/emissions.py`: 75/20/5 scanning schedule and the scanning/model track blend
- `perturbnet/imagenet1k.py`: row-level ImageNet-1k access through the datasets-server API (task generator)
- `perturbnet/image_io.py`: base64 image encode/decode helpers
- `training/`: miner scripts for the adversarial training track (`train.py`, `evaluate.py`, `submit.py`)
- `scripts/run_validator.sh`: start/restart validator with PM2
- `scripts/run_miner.sh`: start/restart miner with PM2
- `scripts/setup_common.sh`: role-aware bootstrap (PM2 + Python deps)
- `scripts/check_imagenet1k.py`: verify `HF_TOKEN` can read ImageNet-1k
- `scripts/integration_smoke_test.py`: local integration test

