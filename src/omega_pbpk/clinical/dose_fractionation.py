"""Dose fractionation optimization for PBPK/PK targets.

Optimizes how a total daily dose is split across multiple administrations
to achieve specific PK targets: minimize peak, maximize trough, or minimize
fluctuation.

Method: 1-compartment analytical superposition over 2 days to approximate
near-steady-state concentrations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FractionationResult:
    """PK results for a single fractionation scheme."""

    drug_name: str
    total_daily_dose_mg: float
    n_fractions: int
    dose_per_fraction_mg: float
    interval_h: float
    times_h: list
    c_plasma_mg_L: list
    css_max: float
    css_min: float
    css_avg: float
    fluctuation_pct: float  # (Css_max - Css_min) / Css_avg * 100
    accumulation_ratio: float  # Css_max relative to first dose Cmax


@dataclass
class FractionationOptResult:
    """Optimization result across multiple fractionation schemes."""

    drug_name: str
    optimal_n_fractions: int
    target: str
    all_results: list = field(default_factory=list)
    optimal_result: FractionationResult | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Helper: 1-cpt analytical plasma concentration
# ---------------------------------------------------------------------------


def _one_cpt_oral_profile(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float,
    t_arr: np.ndarray,
) -> np.ndarray:
    """1-compartment oral absorption PK profile.

    C(t) = (F * D * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))
    """
    ke = cl_L_per_h / vd_L
    if abs(ka_per_h - ke) < 1e-6:
        # Flip-flop or degenerate — perturb ka slightly
        ka_per_h = ke * 1.001

    coeff = (f_oral * dose_mg * ka_per_h) / (vd_L * (ka_per_h - ke))
    c = coeff * (np.exp(-ke * t_arr) - np.exp(-ka_per_h * t_arr))
    c = np.maximum(c, 0.0)
    return c


def _one_cpt_iv_profile(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    t_arr: np.ndarray,
) -> np.ndarray:
    """1-compartment IV bolus PK profile."""
    ke = cl_L_per_h / vd_L
    c0 = dose_mg / vd_L
    return c0 * np.exp(-ke * t_arr)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_fractionated_dosing(
    drug_name: str,
    total_daily_dose_mg: float,
    n_fractions: int,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> FractionationResult:
    """Simulate fractionated dosing and compute near-steady-state PK.

    Parameters
    ----------
    drug_name : str
        Name identifier for the drug.
    total_daily_dose_mg : float
        Total daily dose in mg (> 0).
    n_fractions : int
        Number of equal doses per day (1–12).
    cl_L_per_h : float
        Clearance in L/h (> 0).
    vd_L : float
        Volume of distribution in L (> 0).
    route : str
        "oral" or "iv".
    ka_per_h : float
        Absorption rate constant (oral only).
    f_oral : float
        Oral bioavailability fraction.
    t_end_h : float
        Simulation duration in hours (default 48 h = 2 days).
    dt_h : float
        Time step for output array.

    Returns
    -------
    FractionationResult
    """
    _validate_inputs(total_daily_dose_mg, n_fractions, cl_L_per_h, vd_L)

    dose_per_fraction_mg = total_daily_dose_mg / n_fractions
    interval_h = 24.0 / n_fractions

    n_points = int(t_end_h / dt_h) + 1
    t_arr = np.linspace(0.0, t_end_h, n_points)

    # Determine how many doses to superimpose over t_end_h
    n_doses = int(math.floor(t_end_h / interval_h)) + 1

    # Superposition: sum single-dose profiles shifted by each dose time
    c_total = np.zeros(n_points)
    for k in range(n_doses):
        t_dose = k * interval_h
        t_since_dose = t_arr - t_dose
        t_since_dose = np.where(t_since_dose < 0.0, 0.0, t_since_dose)
        mask = t_arr >= t_dose

        if route == "iv":
            contrib = _one_cpt_iv_profile(dose_per_fraction_mg, cl_L_per_h, vd_L, t_since_dose)
        else:
            contrib = _one_cpt_oral_profile(
                dose_per_fraction_mg, cl_L_per_h, vd_L, ka_per_h, f_oral, t_since_dose
            )

        c_total = np.where(mask, c_total + contrib, c_total)

    # Evaluate steady-state over the last complete dosing interval
    last_dose_start = (n_doses - 2) * interval_h
    last_dose_end = last_dose_start + interval_h
    ss_mask = (t_arr >= last_dose_start) & (t_arr <= last_dose_end)

    if ss_mask.sum() < 2:
        ss_mask = np.ones(n_points, dtype=bool)

    c_ss = c_total[ss_mask]
    css_max = float(np.max(c_ss))
    css_min = float(np.min(c_ss))
    t_ss = t_arr[ss_mask]
    auc = float(np.trapz(c_ss, t_ss))
    css_avg = auc / max(float(t_ss[-1] - t_ss[0]), 1e-12)
    fluctuation_pct = (css_max - css_min) / max(css_avg, 1e-12) * 100.0

    # Accumulation ratio: Css_max / first dose Cmax
    ke = cl_L_per_h / vd_L
    if route == "iv":
        first_cmax = dose_per_fraction_mg / vd_L
    else:
        _ka = ka_per_h
        if abs(_ka - ke) < 1e-6:
            _ka = ke * 1.001
        t_cmax = math.log(_ka / ke) / (_ka - ke) if _ka > ke else 1.0 / _ka
        t_cmax = max(t_cmax, 0.0)
        coeff = (f_oral * dose_per_fraction_mg * _ka) / (vd_L * (_ka - ke))
        first_cmax = coeff * (math.exp(-ke * t_cmax) - math.exp(-_ka * t_cmax))
        first_cmax = max(first_cmax, 1e-12)

    accumulation_ratio = css_max / max(first_cmax, 1e-12)

    return FractionationResult(
        drug_name=drug_name,
        total_daily_dose_mg=total_daily_dose_mg,
        n_fractions=n_fractions,
        dose_per_fraction_mg=dose_per_fraction_mg,
        interval_h=interval_h,
        times_h=t_arr.tolist(),
        c_plasma_mg_L=c_total.tolist(),
        css_max=css_max,
        css_min=css_min,
        css_avg=css_avg,
        fluctuation_pct=fluctuation_pct,
        accumulation_ratio=accumulation_ratio,
    )


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def optimize_fractionation(
    drug_name: str,
    total_daily_dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    target: str = "minimize_fluctuation",
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
) -> FractionationOptResult:
    """Find the best number of daily fractions to achieve a PK target.

    Parameters
    ----------
    drug_name : str
        Name identifier for the drug.
    total_daily_dose_mg : float
        Total daily dose in mg.
    cl_L_per_h : float
        Clearance in L/h.
    vd_L : float
        Volume of distribution in L.
    target : str
        Optimization goal: "minimize_fluctuation", "maximize_trough",
        or "minimize_peak".
    route : str
        "oral" or "iv".
    ka_per_h : float
        Absorption rate constant (oral only).
    f_oral : float
        Oral bioavailability.

    Returns
    -------
    FractionationOptResult
    """
    _validate_inputs(total_daily_dose_mg, 1, cl_L_per_h, vd_L)

    valid_targets = {"minimize_fluctuation", "maximize_trough", "minimize_peak"}
    if target not in valid_targets:
        raise ValueError(f"target must be one of {valid_targets}, got '{target}'")

    all_results = []
    for n in range(1, 5):  # 1 to 4 fractions
        r = simulate_fractionated_dosing(
            drug_name=drug_name,
            total_daily_dose_mg=total_daily_dose_mg,
            n_fractions=n,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            route=route,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
        )
        all_results.append(r)

    # Select optimal result based on target
    if target == "minimize_fluctuation":
        optimal = min(all_results, key=lambda r: r.fluctuation_pct)
        notes = "Lowest fluctuation % selected."
    elif target == "maximize_trough":
        optimal = max(all_results, key=lambda r: r.css_min)
        notes = "Highest Css_min (trough) selected."
    else:  # minimize_peak
        optimal = min(all_results, key=lambda r: r.css_max)
        notes = "Lowest Css_max (peak) selected."

    return FractionationOptResult(
        drug_name=drug_name,
        optimal_n_fractions=optimal.n_fractions,
        target=target,
        all_results=all_results,
        optimal_result=optimal,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_inputs(
    total_daily_dose_mg: float,
    n_fractions: int,
    cl_L_per_h: float,
    vd_L: float,
) -> None:
    if total_daily_dose_mg <= 0:
        raise ValueError(f"total_daily_dose_mg must be > 0, got {total_daily_dose_mg}")
    if not (1 <= n_fractions <= 12):
        raise ValueError(f"n_fractions must be in [1, 12], got {n_fractions}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")


__all__ = [
    "FractionationResult",
    "FractionationOptResult",
    "simulate_fractionated_dosing",
    "optimize_fractionation",
]
