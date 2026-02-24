from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def emax_effect(
    concentration_mg_per_L: FloatArray,
    e0: float,
    emax: float,
    ec50_mg_per_L: float,
    hill: float,
) -> FloatArray:
    c_hill = np.power(np.maximum(concentration_mg_per_L, 0.0), hill)
    ec50_hill = ec50_mg_per_L**hill
    return cast(FloatArray, e0 + (emax * c_hill) / (ec50_hill + c_hill))
