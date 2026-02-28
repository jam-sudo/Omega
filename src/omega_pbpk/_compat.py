"""Numpy compatibility helpers for cross-version support.

numpy < 2.0 : use np.trapz   (np.trapezoid does not exist)
numpy >= 2.0: use np.trapezoid (np.trapz deprecated in 2.0, removed in 2.4)
"""

from __future__ import annotations

import numpy as np

# Resolve the trapezoid function dynamically to support both numpy <2.0
# (np.trapz) and numpy >=2.0 (np.trapezoid, np.trapz removed in 2.4).
# Dynamic lookup avoids mypy attr-defined errors on cross-version stubs.
_names = ("trapezoid", "trapz")
np_trapz = next(getattr(np, n) for n in _names if hasattr(np, n))

__all__ = ["np_trapz"]
