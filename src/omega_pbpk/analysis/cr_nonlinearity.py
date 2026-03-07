"""Concentration-response nonlinearity analysis.

Provides tools to:
- Fit Hill equations to concentration-response data
- Quantify degree of nonlinearity (superlinear / linear / sublinear)
- Identify steepest response range
- Simulate 1-compartment PK + Hill PD and report nonlinear behaviour

All arithmetic uses pure Python stdlib (no numpy/scipy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CRNonlinearityResult:
    """Result of a nonlinear PK/PD simulation."""

    drug_name: str
    times_h: list
    c_plasma_mg_L: list
    effect: list
    peak_effect: float
    time_peak_effect_h: float
    auc_effect: float
    ec50_mg_L: float
    hill_n: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sum_sq_residuals(
    concentrations: list,
    responses: list,
    emax: float,
    ec50: float,
    hill_n: float,
) -> float:
    """Sum of squared residuals for Hill equation."""
    total = 0.0
    for c, r in zip(concentrations, responses):
        cn = c**hill_n
        ec50n = ec50**hill_n
        pred = emax * cn / (ec50n + cn) if (ec50n + cn) > 0 else 0.0
        total += (r - pred) ** 2
    return total


def _linspace(start: float, stop: float, n: int) -> list:
    """Return n evenly spaced values from start to stop inclusive."""
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _logspace(start: float, stop: float, n: int) -> list:
    """Return n logarithmically spaced values from start to stop inclusive."""
    if n == 1:
        return [start]
    if start <= 0 or stop <= 0:
        return _linspace(start, stop, n)
    log_start = math.log10(start)
    log_stop = math.log10(stop)
    step = (log_stop - log_start) / (n - 1)
    return [10 ** (log_start + i * step) for i in range(n)]


def _r_squared(
    responses: list,
    concentrations: list,
    emax: float,
    ec50: float,
    hill_n: float,
) -> float:
    """Coefficient of determination R² for Hill fit."""
    n = len(responses)
    if n == 0:
        return 0.0
    mean_r = sum(responses) / n
    ss_tot = sum((r - mean_r) ** 2 for r in responses)
    if ss_tot == 0.0:
        return 1.0
    ss_res = _sum_sq_residuals(concentrations, responses, emax, ec50, hill_n)
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def fit_hill_equation(
    concentrations: list,
    responses: list,
    hill_n_init: float = 1.0,
) -> dict:
    """Fit Hill equation E = Emax * C^n / (EC50^n + C^n) by grid search.

    Args:
        concentrations: List of concentration values (positive floats).
        responses: List of response values corresponding to each concentration.
        hill_n_init: Initial guess for Hill coefficient (unused — grid search
            explores the full range automatically; kept for API compatibility).

    Returns:
        dict with keys: emax, ec50, hill_n, r_squared, notes.
    """
    if len(concentrations) != len(responses):
        raise ValueError(
            f"concentrations and responses must have the same length, "
            f"got {len(concentrations)} vs {len(responses)}"
        )
    if len(concentrations) < 2:
        raise ValueError("Need at least 2 data points to fit Hill equation")

    max_resp = max(responses)
    min_conc = min(c for c in concentrations if c > 0)
    max_conc = max(concentrations)

    # Guard: if all concentrations are the same, EC50 search is degenerate
    if min_conc >= max_conc:
        max_conc = min_conc * 10.0

    emax_candidates = _linspace(max_resp * 0.5, max_resp * 2.0, 5)
    # Log-spaced EC50 candidates give better coverage over wide concentration ranges
    ec50_candidates = _logspace(min_conc, max_conc, 15)
    # Finer hill_n grid to distinguish n=1 vs n=3 regimes
    hill_n_candidates = _linspace(0.5, 4.0, 8)

    best_ssr = float("inf")
    best_emax = max_resp
    best_ec50 = (min_conc + max_conc) / 2.0
    best_n = 1.0

    for emax in emax_candidates:
        for ec50 in ec50_candidates:
            for n in hill_n_candidates:
                ssr = _sum_sq_residuals(concentrations, responses, emax, ec50, n)
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_emax = emax
                    best_ec50 = ec50
                    best_n = n

    r2 = _r_squared(responses, concentrations, best_emax, best_ec50, best_n)

    notes = "Hill equation fit via grid search"
    if r2 < 0.8:
        notes += "; poor fit (R²<0.80) — data may not follow Hill kinetics"

    return {
        "emax": best_emax,
        "ec50": best_ec50,
        "hill_n": best_n,
        "r_squared": r2,
        "notes": notes,
    }


def dose_response_index(concentrations: list, responses: list) -> dict:
    """Calculate degree of nonlinearity from concentration-response data.

    Args:
        concentrations: Concentration values.
        responses: Response values.

    Returns:
        dict with keys: hill_n, linearity_class, therapeutic_window_index,
        ec10, ec50, ec90.
    """
    if len(concentrations) != len(responses):
        raise ValueError("concentrations and responses must have the same length")

    fit = fit_hill_equation(concentrations, responses)
    n = fit["hill_n"]
    ec50 = fit["ec50"]

    # ECx = EC50 * (x / (100 - x))^(1/n)
    ec10 = ec50 * (10.0 / 90.0) ** (1.0 / n)
    ec90 = ec50 * (90.0 / 10.0) ** (1.0 / n)

    twi = (ec90 - ec10) / ec50 if ec50 > 0 else float("inf")

    if n < 0.8:
        linearity_class = "sublinear"
    elif n <= 1.2:
        linearity_class = "linear"
    else:
        linearity_class = "superlinear"

    return {
        "hill_n": n,
        "linearity_class": linearity_class,
        "therapeutic_window_index": twi,
        "ec10": ec10,
        "ec50": ec50,
        "ec90": ec90,
    }


def identify_nonlinear_range(concentrations: list, responses: list) -> dict:
    """Find the concentration range where the response is steepest.

    Uses finite differences on the midpoints between adjacent data points
    to estimate local slope dE/dC.

    Args:
        concentrations: Concentration values (should be sorted ascending for
            meaningful results; the function sorts them internally).
        responses: Response values paired with concentrations.

    Returns:
        dict with keys: steepest_conc_range, max_slope, linear_range_low,
        linear_range_high.
    """
    if len(concentrations) != len(responses):
        raise ValueError("concentrations and responses must have the same length")
    if len(concentrations) < 2:
        raise ValueError("Need at least 2 data points to identify nonlinear range")

    # Sort by concentration
    pairs = sorted(zip(concentrations, responses), key=lambda x: x[0])
    sorted_c = [p[0] for p in pairs]
    sorted_r = [p[1] for p in pairs]

    # Compute slopes between adjacent points
    slopes = []
    for i in range(len(sorted_c) - 1):
        dc = sorted_c[i + 1] - sorted_c[i]
        dr = sorted_r[i + 1] - sorted_r[i]
        slope = dr / dc if dc != 0 else 0.0
        slopes.append(abs(slope))

    max_slope = max(slopes)
    steep_idx = slopes.index(max_slope)

    # Steepest range: concentrations around the steepest transition
    c_low = sorted_c[steep_idx]
    c_high = sorted_c[steep_idx + 1]
    steepest_conc_range = (c_low, c_high)

    # Linear range approximation: where slope is within 20% of mean slope
    mean_slope = sum(slopes) / len(slopes)
    linear_threshold = 0.2 * mean_slope if mean_slope > 0 else 0.0

    linear_indices = [i for i, s in enumerate(slopes) if abs(s - mean_slope) <= linear_threshold]
    if linear_indices:
        linear_range_low = sorted_c[linear_indices[0]]
        linear_range_high = sorted_c[linear_indices[-1] + 1]
    else:
        linear_range_low = sorted_c[0]
        linear_range_high = sorted_c[-1]

    return {
        "steepest_conc_range": steepest_conc_range,
        "max_slope": max_slope,
        "linear_range_low": linear_range_low,
        "linear_range_high": linear_range_high,
    }


def simulate_nonlinear_pk_pd(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    emax: float,
    ec50_mg_L: float,
    hill_n: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
    route: str = "oral",
    ka_per_h: float = 1.0,
) -> CRNonlinearityResult:
    """Simulate 1-compartment PK with Hill PD.

    Uses forward Euler for ODE integration.  Supports oral (first-order
    absorption) and IV bolus routes.

    Args:
        drug_name: Name of the drug.
        dose_mg: Dose in mg.
        cl_L_per_h: Clearance in L/h.
        vd_L: Volume of distribution in L.
        emax: Maximum effect (dimensionless or appropriate units).
        ec50_mg_L: EC50 in mg/L.
        hill_n: Hill coefficient.
        t_end_h: Simulation end time in hours.
        dt_h: Time step in hours.
        route: 'oral' or 'iv'.
        ka_per_h: First-order absorption rate (oral only), per hour.

    Returns:
        CRNonlinearityResult dataclass.
    """
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if ec50_mg_L <= 0:
        raise ValueError(f"ec50_mg_L must be > 0, got {ec50_mg_L}")
    if dose_mg < 0:
        raise ValueError(f"dose_mg must be >= 0, got {dose_mg}")

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    n_steps = max(1, round(t_end_h / dt_h))

    times_h = []
    c_plasma = []
    effect_list = []

    if route == "oral":
        a_gut = dose_mg  # drug in gut (mg)
        c_central = 0.0  # plasma conc (mg/L)
    else:
        # IV bolus
        a_gut = 0.0
        c_central = dose_mg / vd_L

    def _hill_effect(c: float) -> float:
        if c <= 0.0:
            return 0.0
        cn = c**hill_n
        ec50n = ec50_mg_L**hill_n
        return emax * cn / (ec50n + cn)

    t = 0.0
    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma.append(c_central)
        effect_list.append(_hill_effect(c_central))

        if t >= t_end_h:
            break

        # Forward Euler step
        if route == "oral":
            d_a_gut = -ka_per_h * a_gut * dt_h
            d_c = (ka_per_h * a_gut / vd_L - ke * c_central) * dt_h
            a_gut = max(0.0, a_gut + d_a_gut)
            c_central = max(0.0, c_central + d_c)
        else:
            d_c = -ke * c_central * dt_h
            c_central = max(0.0, c_central + d_c)

        t = min(t + dt_h, t_end_h)

    # Peak effect
    peak_effect = max(effect_list)
    idx_peak = effect_list.index(peak_effect)
    time_peak = times_h[idx_peak]

    # Trapezoidal AUC of effect
    auc_effect = 0.0
    for i in range(1, len(times_h)):
        dt = times_h[i] - times_h[i - 1]
        auc_effect += 0.5 * (effect_list[i - 1] + effect_list[i]) * dt

    notes = f"1-cpt {route} PK; Hill n={hill_n:.2f}, EC50={ec50_mg_L:.3g} mg/L, Emax={emax:.3g}"

    return CRNonlinearityResult(
        drug_name=drug_name,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        effect=effect_list,
        peak_effect=peak_effect,
        time_peak_effect_h=time_peak,
        auc_effect=auc_effect,
        ec50_mg_L=ec50_mg_L,
        hill_n=hill_n,
        notes=notes,
    )
