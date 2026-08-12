from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BLOCK_TIME_SECONDS = 12


@dataclass(frozen=True)
class EpochCountdown:
    current_block: int
    tempo: int
    last_epoch_block: int
    next_epoch_block: int
    blocks_remaining: int
    seconds_remaining: int


def epoch_countdown(subtensor: Any, netuid: int) -> EpochCountdown:
    """Blocks/time remaining until the subnet's next epoch.

    The chain runs a subnet's epoch every `tempo` blocks: `last_step` records
    the block the epoch last ran, and the next run happens `tempo` blocks
    later. Verified against finney: consecutive epochs land exactly `tempo`
    blocks apart, and the chain's `blocks_since_last_step` counter resets to 0
    on the epoch block. (The legacy `(block + netuid + 1) % (tempo + 1)`
    template formula no longer matches the live chain.)
    """
    info = subtensor.subnet(netuid)
    tempo = int(info.tempo)
    last_epoch_block = int(info.last_step)
    current_block = int(subtensor.get_current_block())
    next_epoch_block = last_epoch_block + tempo
    blocks_remaining = max(0, next_epoch_block - current_block)
    return EpochCountdown(
        current_block=current_block,
        tempo=tempo,
        last_epoch_block=last_epoch_block,
        next_epoch_block=next_epoch_block,
        blocks_remaining=blocks_remaining,
        seconds_remaining=blocks_remaining * BLOCK_TIME_SECONDS,
    )
