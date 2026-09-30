#!/usr/bin/env python3
"""Thin CLI entrypoint for the miner-bench harness.

Equivalent to:
    python -m miner_bench.runner [args...]
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from miner_bench.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
