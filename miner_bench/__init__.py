"""Local benchmarking harness for SN26 Perturb miners.

A standalone, reproducible test bed that lets miners measure their attack
strategies against the same scoring rules the mainnet validator applies,
without consuming validator resources or polluting their on-chain history.

Public surface:
    BenchConfig, score_response, EvaluationResult, ScoreReason  -- scoring
    Challenge, fetch_challenge                                   -- challenge gen
    MinerStrategy, PGDStrategy                                   -- attack plug-in
    run_bench, BenchReport                                       -- runner
"""

from miner_bench.scoring import (
    BenchConfig,
    EvaluationResult,
    ScoreReason,
    score_response,
)
from miner_bench.challenge import Challenge, fetch_challenge
from miner_bench.strategy import MinerStrategy, PGDStrategy
from miner_bench.runner import BenchReport, run_bench

__all__ = [
    "BenchConfig",
    "EvaluationResult",
    "ScoreReason",
    "score_response",
    "Challenge",
    "fetch_challenge",
    "MinerStrategy",
    "PGDStrategy",
    "BenchReport",
    "run_bench",
]
