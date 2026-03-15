"""Bayesian individual PK parameter estimation.

Given population PK predictions (base_cl, base_vd) and 1-5 observed C(t) points,
estimate individual CL and Vd scaling factors using scipy L-BFGS-B optimization
in log-concentration space.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _one_cpt_oral(
    t: float,
    dose_mg: float,
    cl: float,
    vd: float,
    ka: float = 1.0,
    F: float = 0.8,
) -> float:
    """Analytical 1-compartment oral PK model.

    Parameters
    ----------
    t : float
        Time (hours).
    dose_mg : float
        Oral dose in mg.
    cl : float
        Clearance (L/h).
    vd : float
        Volume of distribution (L).
    ka : float
        Absorption rate constant (1/h).
    F : float
        Bioavailability fraction (0-1).

    Returns
    -------
    float
        Predicted plasma concentration (mg/L).
    """
    ke = cl / vd

    # Handle ka == ke singularity
    if abs(ka - ke) < 1e-6:
        ka = ke * 1.01

    c = (F * dose_mg / vd) * (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
    return max(float(c), 0.0)


def fit_individual(
    observations: list[tuple[float, float]],
    dose_mg: float,
    base_cl: float,
    base_vd: float,
    ka: float = 1.0,
    F: float = 0.8,
) -> dict:
    """Fit individual CL/Vd scaling factors to observed concentration-time data.

    Parameters
    ----------
    observations : list of (time_h, conc_mg_L)
        Observed concentration-time pairs (1-5 points).
    dose_mg : float
        Oral dose in mg.
    base_cl : float
        Population clearance estimate (L/h).
    base_vd : float
        Population volume of distribution estimate (L).
    ka : float
        Absorption rate constant (1/h).
    F : float
        Bioavailability fraction (0-1).

    Returns
    -------
    dict
        cl_scale, vd_scale, cl_individual, vd_individual, residual
    """
    times = np.array([obs[0] for obs in observations])
    concs = np.array([obs[1] for obs in observations])

    eps = 1e-12

    def objective(params: np.ndarray) -> float:
        cl_scale, vd_scale = params
        cl = base_cl * cl_scale
        vd = base_vd * vd_scale

        mse = 0.0
        for i in range(len(times)):
            c_pred = _one_cpt_oral(times[i], dose_mg, cl, vd, ka, F)
            log_pred = np.log(c_pred + eps)
            log_obs = np.log(concs[i] + eps)
            mse += (log_pred - log_obs) ** 2
        return mse / len(times)

    result = minimize(
        objective,
        x0=np.array([1.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(0.05, 20.0), (0.05, 20.0)],
    )

    cl_scale, vd_scale = result.x
    return {
        "cl_scale": float(cl_scale),
        "vd_scale": float(vd_scale),
        "cl_individual": float(base_cl * cl_scale),
        "vd_individual": float(base_vd * vd_scale),
        "residual": float(result.fun),
    }
