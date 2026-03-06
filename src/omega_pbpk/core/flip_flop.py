"""Flip-flop pharmacokinetics detection and simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "FlipFlopResult",
    "simulate_flip_flop",
    "detect_flip_flop_from_data",
    "compare_formulations",
]


@dataclass
class FlipFlopResult:
    """Results from a flip-flop pharmacokinetics simulation."""

    drug_name: str
    ka_per_h: float
    ke_per_h: float
    is_flip_flop: bool
    ka_apparent_per_h: float
    ke_apparent_per_h: float
    thalf_apparent_h: float
    thalf_true_absorption_h: float
    auc_mg_h_L: float
    cmax_mg_L: float
    tmax_h: float
    times_h: list = field(default_factory=list)
    cp: list = field(default_factory=list)


def simulate_flip_flop(
    drug_name: str,
    dose_mg: float,
    vd_L: float,
    ka_per_h: float,
    ke_per_h: float,
    f: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> FlipFlopResult:
    """Simulate a one-compartment oral model and detect flip-flop kinetics.

    Flip-flop occurs when ka < ke, making absorption the rate-limiting step.
    The observed terminal slope then reflects ka rather than ke.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in milligrams.
    vd_L : float
        Volume of distribution in litres.
    ka_per_h : float
        Absorption rate constant (h^-1).
    ke_per_h : float
        Elimination rate constant (h^-1).
    f : float
        Oral bioavailability fraction (default 1.0).
    t_end_h : float
        Simulation end time in hours (default 24).
    dt_h : float
        Time step in hours (default 0.05).

    Returns
    -------
    FlipFlopResult

    Raises
    ------
    ValueError
        If any of vd_L, dose_mg, ka_per_h, or ke_per_h are <= 0.
    """
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    if ke_per_h <= 0:
        raise ValueError(f"ke_per_h must be > 0, got {ke_per_h}")

    times = np.arange(0.0, t_end_h + dt_h, dt_h)

    # Analytical solution for 1-compartment oral model
    # Cp(t) = (F * dose * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))
    # Handle the special case ka == ke with L'Hopital
    if abs(ka_per_h - ke_per_h) < 1e-10:
        cp = (f * dose_mg * ka_per_h / vd_L) * times * np.exp(-ke_per_h * times)
    else:
        cp = (
            (f * dose_mg * ka_per_h)
            / (vd_L * (ka_per_h - ke_per_h))
            * (np.exp(-ke_per_h * times) - np.exp(-ka_per_h * times))
        )

    cp = np.maximum(cp, 0.0)

    # AUC via trapezoidal rule
    auc = float(np.trapz(cp, times))

    cmax_idx = int(np.argmax(cp))
    cmax = float(cp[cmax_idx])
    tmax = float(times[cmax_idx])

    # Flip-flop detection: ka < ke means absorption is rate-limiting
    is_flip_flop = ka_per_h < ke_per_h

    # Apparent rate constants as observed from terminal slope
    # In flip-flop: terminal slope = -ka/2.303 (the slower process governs)
    ka_apparent = min(ka_per_h, ke_per_h)
    ke_apparent = max(ka_per_h, ke_per_h)
    thalf_apparent = math.log(2) / ka_apparent
    thalf_true_absorption = math.log(2) / ka_per_h

    return FlipFlopResult(
        drug_name=drug_name,
        ka_per_h=ka_per_h,
        ke_per_h=ke_per_h,
        is_flip_flop=is_flip_flop,
        ka_apparent_per_h=ka_apparent,
        ke_apparent_per_h=ke_apparent,
        thalf_apparent_h=thalf_apparent,
        thalf_true_absorption_h=thalf_true_absorption,
        auc_mg_h_L=auc,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        times_h=times.tolist(),
        cp=cp.tolist(),
    )


def detect_flip_flop_from_data(
    times: list | np.ndarray,
    conc: list | np.ndarray,
    dose_mg: float,
    vd_L: float,
) -> dict:
    """Estimate ka and ke from observed concentration–time data and detect flip-flop.

    Uses log-linear regression on the terminal phase (last 40% of non-zero data)
    to estimate the observed terminal rate constant, then applies the method of
    residuals to the absorption phase to estimate ka.

    Parameters
    ----------
    times : array-like
        Observed time points (h).
    conc : array-like
        Observed plasma concentrations (mg/L).
    dose_mg : float
        Administered dose (mg).
    vd_L : float
        Volume of distribution (L).

    Returns
    -------
    dict with keys:
        ka_estimated (float), ke_estimated (float), is_flip_flop (bool),
        evidence_strength (str: 'strong' or 'weak')
    """
    times_arr = np.asarray(times, dtype=float)
    conc_arr = np.asarray(conc, dtype=float)

    # Keep only positive concentrations
    mask = conc_arr > 0
    t_pos = times_arr[mask]
    c_pos = conc_arr[mask]

    n = len(t_pos)
    if n < 4:
        return {
            "ka_estimated": float("nan"),
            "ke_estimated": float("nan"),
            "is_flip_flop": False,
            "evidence_strength": "weak",
        }

    # --- Terminal phase: last 40% of data points ---
    terminal_start = max(1, int(math.ceil(0.60 * n)))
    t_term = t_pos[terminal_start:]
    c_term = c_pos[terminal_start:]

    if len(t_term) < 2:
        t_term = t_pos[-2:]
        c_term = c_pos[-2:]

    # Log-linear fit on terminal phase
    log_c_term = np.log(c_term)
    slope_term, intercept_term = np.polyfit(t_term, log_c_term, 1)
    ke_terminal = float(-slope_term)  # positive value

    # --- Method of residuals for absorption phase ---
    # Use the first 60% of data
    t_early = t_pos[:terminal_start]
    c_early = c_pos[:terminal_start]

    # Back-extrapolate terminal line to early times
    c_extrap = np.exp(intercept_term + slope_term * t_early)
    residuals = c_extrap - c_early
    residuals = residuals[residuals > 0]
    t_residual = t_early[: len(residuals)]

    if len(t_residual) < 2:
        # Cannot estimate ka reliably
        ka_estimated = ke_terminal * 2.0  # placeholder assumption
        evidence_strength = "weak"
    else:
        log_res = np.log(residuals)
        slope_res, _ = np.polyfit(t_residual, log_res, 1)
        ka_estimated = float(-slope_res)

        # Assess quality by R² of the residual fit
        log_res_pred = np.polyval([slope_res, _], t_residual)
        ss_res = float(np.sum((log_res - log_res_pred) ** 2))
        ss_tot = float(np.sum((log_res - np.mean(log_res)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        evidence_strength = "strong" if r2 >= 0.80 else "weak"

    is_flip_flop = ke_terminal < ka_estimated  # flip-flop: terminal slope is ka (slower than ke)

    return {
        "ka_estimated": float(ka_estimated),
        "ke_estimated": float(ke_terminal),
        "is_flip_flop": bool(is_flip_flop),
        "evidence_strength": evidence_strength,
    }


def compare_formulations(
    drug_name: str,
    dose_mg: float,
    vd_L: float,
    ke_per_h: float,
    ka_values: list[float],
) -> list[FlipFlopResult]:
    """Simulate a drug with multiple formulations (different ka values).

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in milligrams.
    vd_L : float
        Volume of distribution (L).
    ke_per_h : float
        Elimination rate constant (h^-1).
    ka_values : list[float]
        List of absorption rate constants to compare.

    Returns
    -------
    list[FlipFlopResult]
        Results for each formulation, sorted by AUC descending.
    """
    results = [simulate_flip_flop(drug_name, dose_mg, vd_L, ka, ke_per_h) for ka in ka_values]
    results.sort(key=lambda r: r.auc_mg_h_L, reverse=True)
    return results
