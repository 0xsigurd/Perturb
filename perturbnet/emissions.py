from __future__ import annotations

from typing import Sequence

# Ranks 1-3 receive fixed shares; ranks 4 through MAX_RANKED_MINER split the tail 5%.
MAX_RANKED_MINER = 10


def ranked_emission_shares(ranked_uids: Sequence[int]) -> dict[int, float]:
    """70 / 15 / 10 / 5 emission schedule with rank-weighted tail for ranks 4-10.

    Rank 1 receives 70%, rank 2 receives 15%, rank 3 receives 10%, and ranks
    4 through MAX_RANKED_MINER split the final 5% by descending rank weight.
    Miners ranked below MAX_RANKED_MINER receive no emission share.
    """
    uids = [int(uid) for uid in ranked_uids]
    if not uids:
        return {}
    if len(uids) == 1:
        return {uids[0]: 1.0}
    if len(uids) == 2:
        return {uids[0]: 0.70, uids[1]: 0.30}
    if len(uids) == 3:
        return {uids[0]: 0.70, uids[1]: 0.15, uids[2]: 0.15}

    shares = {uids[0]: 0.70, uids[1]: 0.15, uids[2]: 0.10}
    tail = uids[3:MAX_RANKED_MINER]

    rank_weights = list(range(len(tail), 0, -1))
    total_rank_weight = float(sum(rank_weights))
    shares.update({uid: 0.05 * weight / total_rank_weight for uid, weight in zip(tail, rank_weights)})
    return shares
