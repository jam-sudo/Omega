"""Non-Compartmental Analysis (NCA) for PK parameter estimation.

Standard NCA methods per FDA bioanalytical guidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class NCAResult:
    """Standard NCA PK parameters."""

    cmax_mg_L: float
    tmax_h: float
    auc0t_mg_h_L: float  # AUC from 0 to last measurable point (linear-log trapezoidal)
    auc0inf_mg_h_L: float  # AUC extrapolated to infinity
    t_half_h: float  # terminal half-life
    kel_per_h: float  # terminal elimination rate constant
    vz_L: float  # apparent volume of distribution (F·Vz for oral)
    cl_L_per_h: float  # apparent clearance (F·CL for oral)
    mrt_h: float  # mean residence time
    r_squared: float  # R² of terminal log-linear regression


def run_nca(
    time_h: NDArray[np.float64],
    conc_mg_L: NDArray[np.float64],
    dose_mg: float,
    terminal_fraction: float = 0.3,
) -> NCAResult:
    """Run standard NCA on PK concentration-time data.

    Args:
        time_h: Time array (h), starting at 0
        conc_mg_L: Plasma concentration array (mg/L)
        dose_mg: Administered dose (mg)
        terminal_fraction: Fraction of data points to use for terminal slope

    Returns:
        NCAResult with all standard PK parameters
    """
    t = np.asarray(time_h, dtype=float)
    c = np.asarray(conc_mg_L, dtype=float)

    # Cmax and Tmax
    imax = int(np.argmax(c))
    cmax = float(c[imax])
    tmax = float(t[imax])

    # AUC0-t using linear-log trapezoidal rule
    auc0t = _auc_linlog(t, c)

    # Terminal elimination rate (kel) from log-linear regression
    n = len(t)
    t0 = max(imax + 1, int((1 - terminal_fraction) * n))
    t_term = t[t0:]
    c_term = c[t0:]
    valid = c_term > 0
    if valid.sum() >= 3:
        log_c = np.log(c_term[valid])
        t_fit = t_term[valid]
        slope, intercept = np.polyfit(t_fit, log_c, 1)
        kel = float(-slope) if slope < 0 else 0.01
        # R²
        log_c_pred = slope * t_fit + intercept
        ss_res = np.sum((log_c - log_c_pred) ** 2)
        ss_tot = np.sum((log_c - log_c.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-12))
    else:
        kel = 0.01
        r2 = 0.0

    t_half = float(math.log(2) / max(kel, 1e-9))

    # AUC extrapolation: AUC0-inf = AUC0-t + Clast/kel
    clast = float(c[c > 0][-1]) if np.any(c > 0) else 0.0
    auc_extrap = clast / max(kel, 1e-9)
    auc0inf = auc0t + auc_extrap

    # PK parameters
    cl = float(dose_mg / max(auc0inf, 1e-9))
    vz = float(cl / max(kel, 1e-9))

    # MRT = AUMC/AUC
    aumc = _aumc_linlog(t, c)
    mrt = float(aumc / max(auc0t, 1e-9))

    return NCAResult(
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc0t_mg_h_L=auc0t,
        auc0inf_mg_h_L=auc0inf,
        t_half_h=t_half,
        kel_per_h=kel,
        vz_L=vz,
        cl_L_per_h=cl,
        mrt_h=mrt,
        r_squared=r2,
    )


def _auc_linlog(t: NDArray, c: NDArray) -> float:
    """Linear-log trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        c1, c2 = c[i - 1], c[i]
        if c1 > 0 and c2 > 0 and c2 < c1:
            # Log trapezoidal for decreasing phase
            auc += dt * (c1 - c2) / math.log(c1 / c2)
        else:
            auc += dt * (c1 + c2) / 2  # linear trapezoidal
    return float(auc)


def _aumc_linlog(t: NDArray, c: NDArray) -> float:
    """AUMC using linear trapezoidal rule."""
    tc = t * c
    return float(np_trapz(tc, t))
