from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def cmax_tmax(time_h: FloatArray, conc: FloatArray) -> tuple[float, float]:
    idx = int(np.argmax(conc))
    return float(conc[idx]), float(time_h[idx])


def auc_trapezoid(time_h: FloatArray, conc: FloatArray) -> float:
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(conc, time_h))


def effect_summary(time_h: FloatArray, effect: FloatArray) -> dict[str, float]:
    emax, t_emax = cmax_tmax(time_h, effect)
    return {"Emax": emax, "t_Emax_h": t_emax}
