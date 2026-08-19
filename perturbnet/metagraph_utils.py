from __future__ import annotations

from typing import Any


VALIDATOR_STAKE_THRESHOLD = 10_000.0


def _sequence_value(values: Any, uid: int, default: Any = None) -> Any:
    try:
        return values[uid]
    except Exception:
        return default


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "tao"):
        return float(value.tao)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _as_bool(value: Any) -> bool:
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def neuron_stake(metagraph: Any, uid: int) -> float:
    stakes = getattr(metagraph, "stake", getattr(metagraph, "S", []))
    return _as_float(_sequence_value(stakes, uid, 0.0))


def is_validator_neuron(metagraph: Any, uid: int) -> bool:
    permits = getattr(metagraph, "validator_permit", getattr(metagraph, "validator_permits", []))
    has_permit = _as_bool(_sequence_value(permits, uid, False))
    return has_permit and neuron_stake(metagraph, uid) > VALIDATOR_STAKE_THRESHOLD


def count_miners(metagraph: Any) -> int:
    return len(miner_uids(metagraph))


def miner_uids(metagraph: Any) -> list[int]:
    total_neurons = int(getattr(metagraph, "n", 0))
    return [uid for uid in range(total_neurons) if not is_validator_neuron(metagraph, uid)]


def miner_incentive(metagraph: Any, uid: int) -> float:
    incentives = getattr(metagraph, "I", getattr(metagraph, "incentive", []))
    return _as_float(_sequence_value(incentives, uid, 0.0))
