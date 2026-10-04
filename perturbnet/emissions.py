from __future__ import annotations

from typing import Mapping, Sequence

SCANNING_RANK_SHARES = (0.75, 0.20, 0.05)
MAX_RANKED_MINER = len(SCANNING_RANK_SHARES)


def fractional_ranks(scores: Mapping[int, float]) -> dict[int, float]:
    """Rank 1 is the highest score; tied scores share the mean of the positions they span."""
    ordered = sorted(scores, key=lambda uid: -scores[uid])
    ranks: dict[int, float] = {}
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and scores[ordered[end + 1]] == scores[ordered[start]]:
            end += 1
        for uid in ordered[start : end + 1]:
            ranks[uid] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def stake_weighted_ranks(reports: Sequence[tuple[float, Mapping[int, float]]]) -> dict[int, float]:
    """uid -> stake-weighted mean of the rank each validator gives it by score (1 = best).

    Only miners with a positive score on at least one validator are ranked. A validator
    with no positive score for one of them ranks it tied last, so every miner is averaged
    over the same validators and the same total stake.
    """
    weighted = [(float(stake), scores) for stake, scores in reports if float(stake) > 0.0]
    uids = {uid for _, scores in weighted for uid, score in scores.items() if float(score) > 0.0}
    total_stake = sum(stake for stake, _ in weighted)
    if not uids or total_stake <= 0.0:
        return {}
    totals = dict.fromkeys(uids, 0.0)
    for stake, scores in weighted:
        ranks = fractional_ranks({uid: max(0.0, float(scores.get(uid, 0.0))) for uid in uids})
        for uid, rank in ranks.items():
            totals[uid] += stake * rank
    return {uid: total / total_stake for uid, total in totals.items()}


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
