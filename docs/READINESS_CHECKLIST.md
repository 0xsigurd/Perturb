# Perturb Subnet Readiness Checklist

Use this before long validator uptime tests or mainnet deployment.

## 1) Environment

- [ ] Wallet hotkeys are registered for validator and miners on target `NETUID`
- [ ] `scripts/validator.env`, `scripts/miner.env`, and `scripts/llm_endpoint.env` are configured
- [ ] GPU drivers/CUDA stack matches installed PyTorch build

## 2) LLM Endpoint Service

- [ ] Local llm_endpoint starts with `./scripts/run_llm_endpoint.sh`
- [ ] `GET /health` returns `status=ok`
- [ ] `POST /verify-label` returns deterministic JSON with `is_match`
- [ ] `GET /metrics` increments `total_requests`

## 3) Validator + Miner Launch

- [ ] Miner starts with `bash ./scripts/run_miner.sh`
- [ ] Validator starts with `bash ./scripts/run_validator.sh`
- [ ] Validator logs challenge creation and miner selection each loop
- [ ] Validator logs periodic `set_weights` attempts

## 4) Integration Smoke Test

- [ ] Run:
  - `python scripts/integration_smoke_test.py`
- [ ] Check output reports:
  - llm_endpoint health success
  - semantic sanity checks pass
  - EfficientNet-B5 prediction succeeds
  - challenge semantic verification passes

## 5) Long-Run Reliability

- [ ] Run validator/miner/llm_endpoint together for 6-24 hours
- [ ] `llm_failures` in llm_endpoint metrics stay low and explainable
- [ ] No repeated validator crash loops
- [ ] No persistent image endpoint failures

## 6) Operational Guardrails

- [ ] Log rotation policy configured for long-running nodes
- [ ] Alerting in place for llm_endpoint downtime and validator exceptions
- [ ] Backups enabled for validator state/log artifacts if required
