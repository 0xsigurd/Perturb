# Perturb Subnet Readiness Checklist

Use this before long validator uptime tests or mainnet deployment.

## 1) Environment

- [ ] Wallet hotkeys are registered for validator and miners on target `NETUID`
- [ ] `scripts/validator.env` and `scripts/miner.env` are configured
- [ ] `HF_TOKEN` belongs to a Hugging Face account that accepted the `ILSVRC/imagenet-1k` terms (the task generator samples ImageNet-1k rows)
- [ ] `python scripts/check_imagenet1k.py` prints `ImageNet-1k ready: ...`
- [ ] GPU drivers/CUDA stack matches installed PyTorch build

## 2) Validator + Miner Launch

- [ ] Miner starts with `bash ./scripts/run_miner.sh`
- [ ] Validator starts with `bash ./scripts/run_validator.sh`
- [ ] Validator boot log shows `validator_config ...`
- [ ] Validator logs task fetch and miner scoring each loop
- [ ] Validator logs periodic `set_weights` attempts

## 3) Integration Smoke Test

- [ ] Run:
  - `python scripts/integration_smoke_test.py`
- [ ] Check output reports:
  - ImageNet-1k row fetch succeeds
  - EfficientNetV2-L prediction succeeds
  - challenge target label is selected from model prediction
  - validator logs include `ssim` and `psnr_db` for scored responses

## 4) Long-Run Reliability

- [ ] Run validator/miner together for 6-24 hours
- [ ] No repeated validator crash loops
- [ ] No persistent Hugging Face access failures (ImageNet-1k)

## 5) Operational Guardrails

- [ ] Log rotation policy configured for long-running nodes
- [ ] Alerting in place for validator exceptions
- [ ] Backups enabled for validator state/log artifacts if required
