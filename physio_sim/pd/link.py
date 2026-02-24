from __future__ import annotations

import numpy as np


def effect_site_concentration(
    time_h: np.ndarray,
    plasma_conc_mg_per_L: np.ndarray,
    ke0_per_h: float | None,
) -> np.ndarray:
    if ke0_per_h is None:
        return plasma_conc_mg_per_L
    ce = np.zeros_like(plasma_conc_mg_per_L)
    for i in range(1, len(time_h)):
        dt = time_h[i] - time_h[i - 1]
        ce[i] = ce[i - 1] + dt * ke0_per_h * (plasma_conc_mg_per_L[i - 1] - ce[i - 1])
    return ce
