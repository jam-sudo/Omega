from __future__ import annotations

import cProfile
import pstats
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def run_with_profile(func: Callable[[], T], out_file: Path, top_n: int = 30) -> T:
    profiler = cProfile.Profile()
    profiler.enable()
    result = func()
    profiler.disable()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profiler, stream=handle).sort_stats("cumtime")
        stats.print_stats(top_n)
    return result
