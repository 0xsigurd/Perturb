#!/usr/bin/env python3
"""Show how many blocks (and how much time) remain until the next epoch.

See perturbnet/epoch_timing.py for the calculation; validators use the same
helper to schedule weight setting inside the final WEIGHT_WINDOW_BLOCKS
blocks of each epoch.

Usage:
    python scripts/epoch_countdown.py --netuid 26 --network finney
    python scripts/epoch_countdown.py --netuid 26 --watch
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bittensor as bt

from perturbnet.epoch_timing import BLOCK_TIME_SECONDS, epoch_countdown


def format_duration(seconds: int) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def report_once(subtensor: bt.Subtensor, netuid: int) -> None:
    c = epoch_countdown(subtensor, netuid)
    print(f"current block      : {c.current_block}")
    print(f"tempo              : {c.tempo}")
    print(f"last epoch block   : {c.last_epoch_block}")
    print(f"next epoch block   : {c.next_epoch_block}")
    print(f"blocks remaining   : {c.blocks_remaining}")
    print(f"time remaining     : ~{format_duration(c.seconds_remaining)} ({c.seconds_remaining}s)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Blocks/time remaining until next epoch")
    parser.add_argument("--netuid", type=int, default=26)
    parser.add_argument("--network", type=str, default="finney")
    parser.add_argument("--watch", action="store_true", help="Refresh every block interval")
    args = parser.parse_args()

    subtensor = bt.Subtensor(network=args.network)

    if not args.watch:
        report_once(subtensor, args.netuid)
        return 0

    try:
        while True:
            report_once(subtensor, args.netuid)
            print("-" * 40)
            time.sleep(BLOCK_TIME_SECONDS)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
