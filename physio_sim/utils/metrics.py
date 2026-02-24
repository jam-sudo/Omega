from __future__ import annotations

import numpy as np


def cmax_tmax(time_h: np.ndarray, conc: np.ndarray) -> tuple[float, float]:
    idx = int(np.argmax(conc))
    return float(conc[idx]), float(time_h[idx])


def auc_trapezoid(time_h: np.ndarray, conc: np.ndarray) -> float:
    return float(np.trapz(conc, time_h))


def effect_summary(time_h: np.ndarray, effect: np.ndarray) -> dict[str, float]:
    emax, t_emax = cmax_tmax(time_h, effect)
    return {"Emax": emax, "t_Emax_h": t_emax}
