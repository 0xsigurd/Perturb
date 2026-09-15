from __future__ import annotations

from typing import Sequence

SCANNING_RANK_SHARES = (0.75, 0.20, 0.05)
MAX_RANKED_MINER = len(SCANNING_RANK_SHARES)


def ranked_emission_shares(ranked_uids: Sequence[int]) -> dict[int, float]:
    uids = [int(uid) for uid in ranked_uids]
    if not uids:
        return {}
    if len(uids) == 1:
        return {uids[0]: 1.0}
    if len(uids) == 2:
        return {uids[0]: SCANNING_RANK_SHARES[0], uids[1]: 1.0 - SCANNING_RANK_SHARES[0]}
    return {uid: share for uid, share in zip(uids[:MAX_RANKED_MINER], SCANNING_RANK_SHARES)}


def blend_track_weights(
    *,
    scanning_shares: dict[int, float],
    model_winner_uid: int | None,
    scanning_share: float,
    model_share: float,
) -> dict[int, float]:
    total_shares = float(scanning_share) + float(model_share)
    if total_shares <= 0.0:
        return {}
    weights: dict[int, float] = {}
    if model_winner_uid is None:
        for uid, share in scanning_shares.items():
            weights[uid] = weights.get(uid, 0.0) + float(share)
    else:
        for uid, share in scanning_shares.items():
            weights[uid] = weights.get(uid, 0.0) + float(share) * float(scanning_share) / total_shares
        weights[int(model_winner_uid)] = weights.get(int(model_winner_uid), 0.0) + float(model_share) / total_shares
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    return {uid: value / total for uid, value in weights.items()}
