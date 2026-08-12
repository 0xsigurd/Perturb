from __future__ import annotations

from typing import Any, Sequence

from perturbnet import constants as C
from perturbnet.leaderboard_reporter import LeaderboardMinerResult, LeaderboardNetworkMetrics, LeaderboardReport


def update_score_histories(histories: list[list[float]], uids: Sequence[int], rewards: Sequence[float], window: int) -> None:
    for uid, reward in zip(uids, rewards):
        histories[uid].append(float(reward))
        histories[uid] = histories[uid][-int(window):]


def effective_avg_window(
    histories: list[list[float]],
    uids: Sequence[int],
    window: int,
    min_window: int,
) -> int:
    """Averaging window used for reported avg_scores.

    Follows the longest miner history (capped at `window`) so short-history
    miners can't report a competitive average. Returns 0 when no miner has
    reached `min_window` records yet, which zeroes every reported avg_score.
    """
    longest_history = max(
        (len(histories[uid]) for uid in uids if 0 <= uid < len(histories)),
        default=0,
    )
    if longest_history < int(min_window):
        return 0
    return min(int(window), longest_history)


def avg_score(histories: list[list[float]], uid: int, window: int) -> float:
    if int(window) <= 0 or uid >= len(histories):
        return 0.0
    history = histories[uid]
    if len(history) < int(window):
        return 0.0
    tail = history[-int(window):]
    return float(sum(tail) / len(tail))


GRAPH_SCORE_LIMIT = 50


def compact_graph_score(score: float) -> int | float:
    rounded = round(float(score), 4)
    if rounded == 0.0:
        return 0
    return rounded


def score_graph(histories: list[list[float]], uid: int) -> list[int | float]:
    if uid >= len(histories):
        return []
    return [compact_graph_score(score) for score in histories[uid][-GRAPH_SCORE_LIMIT:]]


def result_status(result: Any) -> str:
    if result.reason == "success":
        return "Valid"
    if result.reason in {"duplicate_response"}:
        return "Duplicate"
    if result.reason == "response_missing_or_status_error":
        return "Inactive"
    if result.reason == "leaderboard_unavailable":
        return "Unavailable"
    return "Invalid"


def network_metrics(
    *,
    results_by_uid: Sequence[tuple[int, Any]],
) -> LeaderboardNetworkMetrics:
    successful_results = [result for _, result in results_by_uid if result_status(result) == "Valid"]
    success_count = len(successful_results)
    if success_count == 0:
        return LeaderboardNetworkMetrics(
            avg_score=0.0,
            avg_rmse=0.0,
            avg_norm=0.0,
            avg_margin=0.0,
            success_count=0,
        )
    return LeaderboardNetworkMetrics(
        avg_score=float(sum(result.score for result in successful_results) / success_count),
        avg_rmse=float(sum(result.rmse for result in successful_results) / success_count),
        avg_norm=float(sum(result.norm for result in successful_results) / success_count),
        avg_margin=float(sum(result.margin for result in successful_results) / success_count),
        success_count=int(success_count),
    )


def build_report(
    *,
    task_id: str,
    validator_hotkey: str,
    score_histories: list[list[float]],
    avg_window: int,
    results_by_uid: Sequence[tuple[int, Any]],
    image_url_by_uid: dict[int, str],
    min_avg_window: int = C.MIN_WEIGHT_HISTORY_SIZE,
) -> LeaderboardReport:
    window = effective_avg_window(
        score_histories,
        [uid for uid, _ in results_by_uid],
        avg_window,
        min_avg_window,
    )
    miners: list[LeaderboardMinerResult] = []
    for uid, result in results_by_uid:
        miners.append(
            LeaderboardMinerResult(
                uid=int(uid),
                avg_score=avg_score(score_histories, uid, window),
                last_score=float(result.score),
                graph=score_graph(score_histories, uid),
                rmse=float(result.rmse),
                norm=float(result.norm),
                margin=float(result.margin),
                result=result_status(result),
                image_url=image_url_by_uid.get(uid) or C.LEADERBOARD_NO_IMAGE_URL,
            )
        )
    return LeaderboardReport(
        task_id=task_id,
        validator_hotkey=validator_hotkey,
        network=network_metrics(
            results_by_uid=results_by_uid,
        ),
        miners=miners,
    )
