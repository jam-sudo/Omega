"""Numpy compatibility helpers for cross-version support.

numpy < 2.0 : use np.trapz   (np.trapezoid does not exist)
numpy >= 2.0: use np.trapezoid (np.trapz deprecated in 2.0, removed in 2.4)
"""

import numpy as np

try:
    np_trapz = np.trapezoid  # numpy >= 2.0
except AttributeError:
    np_trapz = np.trapz  # type: ignore[attr-defined]  # numpy < 2.0

__all__ = ["np_trapz"]
