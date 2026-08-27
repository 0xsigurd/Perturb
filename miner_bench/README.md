# miner-bench: local benchmark for SN26 Perturb miners

A standalone harness that lets miners evaluate attack strategies against
the same scoring rules the mainnet validator applies, without consuming
validator resources or polluting their on-chain history.

The scoring path in [`scoring.py`](./scoring.py) is a faithful, dependency-light
port of `PerturbValidator.verify_and_score` in [`neurons/validator.py`](../neurons/validator.py).
Diff the two if you want to confirm parity.

## What you get

- **Realistic challenges.** Pulls real images from Pexels with the same
  endpoint, headers, and `medium`-variant download the validator uses, then
  uses EfficientNetV2-M's prediction as the ground-truth label (just like
  the validator does).
- **Validator-identical scoring.** Same SSIM/PSNR/L∞/RMSE gates, same
  perturbation+speed score combination, same reason taxonomy.
- **Pluggable strategies.** Implement `MinerStrategy.attack(ctx)` and drop
  it into `run_bench(...)`. A textbook PGD baseline ships in the box.
- **Per-run JSON dumps.** Pipe to your dashboard of choice.

## Install

The harness reuses the repo's `perturbnet` package and `setup.py`
dependencies — no extra installs needed beyond the standard validator/miner
environment.

```bash
pip install -e .                # from repo root
export PEXELS_API_KEY=...       # get one at https://www.pexels.com/api/
```

## CLI

```bash
python scripts/bench_miner.py --n-challenges 20
```

Useful flags:

```
--n-challenges N        how many challenges to run (default 10)
--prompt LABEL          pin to one Pexels query (default: rotate through PROMPTS)
--seed N                deterministic RNG for prompt/photo choice
--timeout-seconds N     validator's challenge timeout, controls speed_score (default 10)
--device cpu|cuda|mps   torch device (auto-detected by default)
--out results.json      write per-challenge results to JSON
--pgd-steps N           PGD iterations (default 20)
--pgd-step-ratio R      per-step magnitude as fraction of epsilon (default 0.15)
--pgd-margin M          early-stop margin in logits (default 2.0)
```

## Library use

```python
from miner_bench import run_bench, PGDStrategy

report = run_bench(
    api_key="YOUR_PEXELS_KEY",
    strategy=PGDStrategy(steps=30),
    n_challenges=20,
    seed=42,                     # reproducible
)
print(report.summary())
print(report.success_rate, report.mean_score)
```

## Plugging in your own attack

```python
from miner_bench import MinerStrategy, run_bench
from miner_bench.strategy import StrategyContext

class MyAttack(MinerStrategy):
    name = "my-attack"
    def attack(self, ctx: StrategyContext) -> str:
        # ctx.clean_image_b64, ctx.true_label, ctx.epsilon, ctx.timeout_seconds
        # ctx.model (EfficientNetV2-M), ctx.device
        # Return PNG-encoded base64.
        ...

report = run_bench(api_key="...", strategy=MyAttack(), n_challenges=20)
```

## Output shape

```
strategy        : pgd
challenges      : 10 / requested 10
success rate    : 70.0% (7/10)
mean score      : 0.4123  (all)
mean score ok   : 0.5890  (successes only)
mean attack_ms  : 412
reasons         :
  success                        7
  below_min_ssim                 2
  label_match_with_original      1
```

Each `--out` JSON entry contains the full `EvaluationResult` (score, reason,
norm, rmse, ssim, psnr_db, model_prediction).

## Calibrating against mainnet

The harness intentionally diverges from the live validator in two safe
ways, so your offline numbers are a *lower bound* on mainnet performance,
not an upper bound:

1. **Attack time vs response time.** `attack_ms` here measures only your
   strategy's wall-clock; the validator's `response_time_ms` additionally
   includes dendrite serialization and network round-trip (usually
   50–150 ms extra). Your real `speed_score` on mainnet will be slightly
   lower than the harness reports for the same strategy.
2. **No LLM prompt-label check.** The validator runs an additional LLM
   verification that the model's prediction matches the Pexels prompt.
   The harness skips this — it accepts whatever EfficientNet predicts on
   the clean image as the ground-truth label. If you want strict parity,
   wrap `fetch_challenge` and filter on your own LLM endpoint.

## Reproducibility note

The validator picks prompt and Pexels photo via `system_random` (non-deterministic
by design — see `_choose_prompt` and `_fetch_image_for_prompt`). To get
reproducible benchmark runs across strategy comparisons, pass `--seed N` —
the harness then derives prompt/photo choice from a seeded RNG. This makes
A/B comparisons meaningful: same images, same epsilons, different attacks.

## Layout

```
miner_bench/
  __init__.py        public exports
  scoring.py         port of verify_and_score (faithful)
  challenge.py       Pexels fetch + Challenge dataclass
  strategy.py        MinerStrategy ABC + PGDStrategy baseline
  runner.py          run_bench() + CLI main()
  README.md          this file
scripts/
  bench_miner.py     thin CLI entrypoint
```
