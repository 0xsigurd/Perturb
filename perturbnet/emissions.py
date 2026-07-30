from __future__ import annotations

from typing import Sequence

# Ranks 1-2 receive fixed shares; ranks 3 through MAX_RANKED_MINER split the tail 15%.
MAX_RANKED_MINER = 10


def ranked_emission_shares(ranked_uids: Sequence[int]) -> dict[int, float]:
    """70 / 15 / 15 emission schedule with rank-weighted tail for ranks 3-10.

    Rank 1 receives 70%, rank 2 receives 15%, and ranks 3 through
    MAX_RANKED_MINER split the final 15% by descending rank weight.
    Miners ranked below MAX_RANKED_MINER receive no emission share.
    """
    uids = [int(uid) for uid in ranked_uids]
    if not uids:
        return {}
    if len(uids) == 1:
        return {uids[0]: 1.0}
    if len(uids) == 2:
        return {uids[0]: 0.70, uids[1]: 0.30}

    shares = {uids[0]: 0.70, uids[1]: 0.15}
    tail = uids[2:MAX_RANKED_MINER]
    if not tail:
        return shares

    rank_weights = list(range(len(tail), 0, -1))
    total_rank_weight = float(sum(rank_weights))
    shares.update({uid: 0.15 * weight / total_rank_weight for uid, weight in zip(tail, rank_weights)})
    return shares
