"""Benchmark runner: fetch challenges, run a strategy, score with validator rules.

Usage as a library:

    from miner_bench import run_bench, PGDStrategy
    report = run_bench(api_key="...", strategy=PGDStrategy(), n_challenges=10)
    print(report.summary())

Usage as a CLI: see `scripts/bench_miner.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from miner_bench.challenge import Challenge, fetch_challenge
from miner_bench.scoring import (
    BenchConfig,
    EvaluationResult,
    ScoreReason,
    load_model,
    score_response,
)
from miner_bench.strategy import MinerStrategy, PGDStrategy, StrategyContext


@dataclass
class ChallengeResult:
    task_id: str
    prompt: str
    true_label: str
    epsilon: float
    attack_ms: int
    evaluation: EvaluationResult


@dataclass
class BenchReport:
    strategy_name: str
    n_challenges: int
    results: list[ChallengeResult] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.results if r.evaluation.reason == ScoreReason.SUCCESS.value)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.n_success / len(self.results)

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.evaluation.score for r in self.results) / len(self.results)

    @property
    def mean_success_score(self) -> float:
        ok = [r.evaluation.score for r in self.results if r.evaluation.reason == ScoreReason.SUCCESS.value]
        return sum(ok) / len(ok) if ok else 0.0

    @property
    def mean_attack_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.attack_ms for r in self.results) / len(self.results)

    def reason_counts(self) -> dict[str, int]:
        return dict(Counter(r.evaluation.reason for r in self.results))

    def summary(self) -> str:
        lines = [
            f"strategy        : {self.strategy_name}",
            f"challenges      : {len(self.results)} / requested {self.n_challenges}",
            f"success rate    : {self.success_rate:.1%} ({self.n_success}/{len(self.results)})",
            f"mean score      : {self.mean_score:.4f}  (all)",
            f"mean score ok   : {self.mean_success_score:.4f}  (successes only)",
            f"mean attack_ms  : {self.mean_attack_ms:.0f}",
            "reasons         :",
        ]
        for reason, count in sorted(self.reason_counts().items(), key=lambda kv: -kv[1]):
            lines.append(f"  {reason:<30s} {count}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "n_challenges": self.n_challenges,
            "success_rate": self.success_rate,
            "mean_score": self.mean_score,
            "mean_success_score": self.mean_success_score,
            "mean_attack_ms": self.mean_attack_ms,
            "reason_counts": self.reason_counts(),
            "results": [
                {
                    "task_id": r.task_id,
                    "prompt": r.prompt,
                    "true_label": r.true_label,
                    "epsilon": r.epsilon,
                    "attack_ms": r.attack_ms,
                    "evaluation": asdict(r.evaluation),
                }
                for r in self.results
            ],
        }


def _pick_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_bench(
    *,
    api_key: str,
    strategy: Optional[MinerStrategy] = None,
    n_challenges: int = 10,
    config: Optional[BenchConfig] = None,
    device: Optional[str] = None,
    prompt: Optional[str] = None,
    seed: Optional[int] = None,
    timeout_seconds: int = 10,
    verbose: bool = True,
    out_path: Optional[str] = None,
) -> BenchReport:
    """Run `n_challenges` rounds against Pexels, scoring each with validator rules.

    If `seed` is given, the per-challenge RNG (prompt/photo choice, epsilon
    sampling) is deterministic across runs — useful for comparing strategies.
    """
    if not api_key:
        raise ValueError("api_key required")
    strat = strategy or PGDStrategy()
    cfg = config or BenchConfig()
    torch_device = _pick_device(device)

    if verbose:
        print(f"[bench] loading EfficientNetV2-M on {torch_device}...", flush=True)
    model = load_model(torch_device)

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    report = BenchReport(strategy_name=strat.name, n_challenges=n_challenges)

    for idx in range(1, n_challenges + 1):
        if verbose:
            print(f"[bench] challenge {idx}/{n_challenges} ...", flush=True)
        try:
            challenge: Challenge = fetch_challenge(
                api_key=api_key,
                model=model,
                device=torch_device,
                prompt=prompt,
                seed=rng.randrange(2**63) if seed is not None else None,
                timeout_seconds=timeout_seconds,
                rng=rng,
            )
        except Exception as exc:
            if verbose:
                print(f"[bench] challenge fetch failed: {exc}", flush=True)
            continue

        ctx = StrategyContext(
            clean_image_b64=challenge.clean_image_b64,
            true_label=challenge.true_label,
            epsilon=challenge.epsilon,
            timeout_seconds=challenge.timeout_seconds,
            model=model,
            device=torch_device,
            max_linf_delta=cfg.max_linf_delta,
            min_linf_delta=cfg.min_linf_delta,
        )
        t0 = time.perf_counter()
        try:
            perturbed_b64 = strat.attack(ctx)
        except Exception as exc:
            if verbose:
                print(f"[bench] strategy raised: {exc}", flush=True)
            continue
        attack_ms = int((time.perf_counter() - t0) * 1000)

        evaluation = score_response(
            clean_image_b64=challenge.clean_image_b64,
            perturbed_image_b64=perturbed_b64,
            true_label=challenge.true_label,
            epsilon=challenge.epsilon,
            response_time_ms=attack_ms,
            timeout_seconds=challenge.timeout_seconds,
            model=model,
            device=torch_device,
            config=cfg,
            norm_type=challenge.norm_type,
        )

        result = ChallengeResult(
            task_id=challenge.task_id,
            prompt=challenge.prompt,
            true_label=challenge.true_label,
            epsilon=challenge.epsilon,
            attack_ms=attack_ms,
            evaluation=evaluation,
        )
        report.results.append(result)

        if verbose:
            print(
                f"  prompt={challenge.prompt:<12s} true={challenge.true_label:<25s} "
                f"eps={challenge.epsilon:.4f} attack_ms={attack_ms:>5d} "
                f"score={evaluation.score:.4f} reason={evaluation.reason}",
                flush=True,
            )

    if out_path:
        with open(out_path, "w") as fh:
            json.dump(report.to_jsonable(), fh, indent=2)
        if verbose:
            print(f"[bench] wrote {out_path}", flush=True)

    if verbose:
        print("\n" + report.summary(), flush=True)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench_miner",
        description="Local benchmark for SN26 Perturb miner attacks against the live scoring rules.",
    )
    p.add_argument("--api-key", default=os.getenv("PEXELS_API_KEY", ""),
                   help="Pexels API key (or set PEXELS_API_KEY env var).")
    p.add_argument("--n-challenges", type=int, default=10,
                   help="Number of challenges to run.")
    p.add_argument("--strategy", default="pgd", choices=["pgd"],
                   help="Built-in strategy to run. Add your own by importing run_bench from Python.")
    p.add_argument("--prompt", default=None,
                   help="Pin to one Pexels query (default: rotate through PROMPTS).")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed the per-run RNG for prompt/photo choice; reproducible runs.")
    p.add_argument("--timeout-seconds", type=int, default=10,
                   help="Validator's challenge timeout (controls speed_score denominator).")
    p.add_argument("--device", default=None,
                   help="torch device (cpu|cuda|mps); auto-detected by default.")
    p.add_argument("--out", dest="out_path", default=None,
                   help="Write per-challenge JSON results to this path.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-challenge logging.")
    # PGD knobs
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--pgd-step-ratio", type=float, default=0.15)
    p.add_argument("--pgd-margin", type=float, default=2.0)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if not args.api_key:
        print("error: Pexels API key required (--api-key or PEXELS_API_KEY).", file=sys.stderr)
        return 2
    if args.strategy == "pgd":
        strategy: MinerStrategy = PGDStrategy(
            steps=args.pgd_steps,
            step_ratio=args.pgd_step_ratio,
            margin=args.pgd_margin,
        )
    else:
        print(f"error: unknown strategy {args.strategy!r}", file=sys.stderr)
        return 2

    try:
        run_bench(
            api_key=args.api_key,
            strategy=strategy,
            n_challenges=args.n_challenges,
            prompt=args.prompt,
            seed=args.seed,
            timeout_seconds=args.timeout_seconds,
            device=args.device,
            verbose=not args.quiet,
            out_path=args.out_path,
        )
    except KeyboardInterrupt:
        print("\n[bench] interrupted by user", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
