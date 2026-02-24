from __future__ import annotations

import numpy as np


def emax_effect(
    concentration_mg_per_L: np.ndarray,
    e0: float,
    emax: float,
    ec50_mg_per_L: float,
    hill: float,
) -> np.ndarray:
    c_hill = np.power(np.maximum(concentration_mg_per_L, 0.0), hill)
    ec50_hill = ec50_mg_per_L**hill
    return e0 + (emax * c_hill) / (ec50_hill + c_hill)
