"""Numpy compatibility helpers for cross-version support.

numpy < 2.0 : use np.trapz   (np.trapezoid does not exist)
numpy >= 2.0: use np.trapezoid (np.trapz deprecated in 2.0, removed in 2.4)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# Use getattr to avoid mypy attr-defined errors on cross-version numpy stubs.
# getattr returns Any so no type: ignore comments are needed.
np_trapz: Callable[..., Any] = getattr(np, "trapezoid", None) or getattr(  # noqa: B009
    np, "trapz", None
)

__all__ = ["np_trapz"]
