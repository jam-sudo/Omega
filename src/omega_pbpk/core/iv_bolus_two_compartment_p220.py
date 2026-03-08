"""
Phase 220 — 2-Compartment IV Bolus Analytical Solution

Implements the analytical biexponential model C(t) = A*exp(-alpha*t) + B*exp(-beta*t)
and parameter estimation by method of residuals (feathering).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Result dataclass (plain — contains list fields)
# ---------------------------------------------------------------------------


@dataclass
class IVTwoCompResult:
    drug_name: str
    dose_mg: float
    alpha_per_h: float
    beta_per_h: float
    A_coeff_mg_L: float
    B_coeff_mg_L: float
    t_half_alpha_h: float
    t_half_beta_h: float
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    cmax_mg_L: float = 0.0
    auc_mg_h_per_L: float = 0.0
    cl_L_per_h: float = 0.0
    vd_central_L: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(xs: list[float], ys: list[float]) -> float:
    """Manual trapezoidal integration."""
    if len(xs) < 2:
        return 0.0
    total = 0.0
    for i in range(len(xs) - 1):
        total += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    return total


def _linreg(xs: list[float], ys: list[float]):
    """Simple linear regression; returns (slope, intercept)."""
    n = len(xs)
    if n < 2:
        raise ValueError("Need at least 2 points for linear regression.")
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        raise ValueError("Singular design matrix in linear regression.")
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _r_squared(observed: list[float], predicted: list[float]) -> float:
    """Coefficient of determination R²."""
    n = len(observed)
    if n < 2:
        return float("nan")
    mean_obs = sum(observed) / n
    ss_tot = sum((o - mean_obs) ** 2 for o in observed)
    ss_res = sum((o - p) ** 2 for o, p in zip(observed, predicted))
    if ss_tot < 1e-15:
        return 1.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_iv_two_compartment(
    drug_name: str,
    dose_mg: float,
    alpha: float,
    beta: float,
    A_coeff: float,
    B_coeff: float,
    t_end_h: float = 24.0,
    dt: float = 0.1,
) -> IVTwoCompResult:
    """Simulate a 2-compartment IV bolus using the analytical biexponential model.

    C(t) = A * exp(-alpha * t) + B * exp(-beta * t)

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    alpha:
        Rapid distribution rate constant (h⁻¹); must be > beta.
    beta:
        Slow elimination rate constant (h⁻¹); must be > 0.
    A_coeff:
        Coefficient for the alpha (distribution) exponential (mg/L).
    B_coeff:
        Coefficient for the beta (elimination) exponential (mg/L).
    t_end_h:
        Simulation end time (h).
    dt:
        Time step (h).

    Returns
    -------
    IVTwoCompResult
    """
    if alpha <= beta:
        raise ValueError(
            f"alpha ({alpha:.4f}) must be greater than beta ({beta:.4f}) for a "
            "two-compartment model."
        )
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive.")

    # Build time vector
    n_steps = max(2, int(round(t_end_h / dt)) + 1)
    times: list[float] = [i * dt for i in range(n_steps)]

    # Analytical concentration at each time point
    c_plasma: list[float] = [
        A_coeff * math.exp(-alpha * t) + B_coeff * math.exp(-beta * t) for t in times
    ]

    cmax = A_coeff + B_coeff  # C(0)
    auc_analytical = A_coeff / alpha + B_coeff / beta
    t_half_alpha = math.log(2.0) / alpha
    t_half_beta = math.log(2.0) / beta
    cl = dose_mg / auc_analytical
    vd_central = dose_mg / cmax  # = dose / (A + B)

    notes = (
        f"Analytical 2-cpt IV bolus. "
        f"t½α={t_half_alpha:.2f} h, t½β={t_half_beta:.2f} h. "
        f"CL={cl:.3f} L/h, Vd_central={vd_central:.3f} L."
    )

    return IVTwoCompResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        alpha_per_h=alpha,
        beta_per_h=beta,
        A_coeff_mg_L=A_coeff,
        B_coeff_mg_L=B_coeff,
        t_half_alpha_h=t_half_alpha,
        t_half_beta_h=t_half_beta,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=cmax,
        auc_mg_h_per_L=auc_analytical,
        cl_L_per_h=cl,
        vd_central_L=vd_central,
        notes=notes,
    )


def estimate_two_comp_params(
    drug_name: str,
    times: list[float],
    concentrations: list[float],
    dose_mg: float = 100.0,
    t_end_h: float = 24.0,
    dt: float = 0.1,
    terminal_fraction: float = 0.4,
) -> IVTwoCompResult:
    """Estimate 2-compartment parameters from observed IV bolus data by method of residuals.

    The method of residuals (feathering) proceeds as:
    1. Identify the terminal (late) phase — the last ``terminal_fraction`` of
       data points — and fit log(C) vs t to get beta and B.
    2. Subtract the terminal phase from the full profile, fit residuals to get
       alpha and A.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    times:
        Observed sampling times (h). Must have at least 4 points.
    concentrations:
        Observed plasma concentrations (mg/L) corresponding to *times*.
    dose_mg:
        Dose administered (mg); used for CL/Vd calculation.
    t_end_h:
        End time for simulated profile.
    dt:
        Time step for simulated profile.
    terminal_fraction:
        Fraction of data points used for terminal-phase regression (default 0.4).

    Returns
    -------
    IVTwoCompResult
    """
    if len(times) != len(concentrations):
        raise ValueError(
            f"times and concentrations must have the same length "
            f"(got {len(times)} vs {len(concentrations)})."
        )
    if len(times) < 4:
        raise ValueError(
            f"At least 4 data points are required for biexponential fitting (got {len(times)})."
        )
    # Filter out non-positive concentrations
    pairs = [(t, c) for t, c in zip(times, concentrations) if c > 0]
    if len(pairs) < 4:
        raise ValueError("At least 4 positive concentration values are required.")

    pairs.sort(key=lambda p: p[0])
    ts = [p[0] for p in pairs]
    cs = [p[1] for p in pairs]

    n = len(pairs)

    # ------------------------------------------------------------------
    # Step 1 — Terminal phase: fit last terminal_fraction of points
    # ------------------------------------------------------------------
    n_terminal = max(2, int(round(n * terminal_fraction)))
    t_term = ts[n - n_terminal :]
    log_c_term = [math.log(c) for c in cs[n - n_terminal :]]

    slope_beta, intercept_beta = _linreg(t_term, log_c_term)
    beta = -slope_beta  # beta > 0
    B_coeff = math.exp(intercept_beta)

    if beta <= 0:
        beta = 1e-6  # safety clamp

    # ------------------------------------------------------------------
    # Step 2 — Residuals: subtract terminal phase from early points
    # ------------------------------------------------------------------
    n_early = n - n_terminal
    if n_early < 2:
        # Fallback: use all points for both phases with a small separation
        n_early = max(2, n // 2)

    t_early = ts[:n_early]
    residuals = [cs[i] - B_coeff * math.exp(-beta * ts[i]) for i in range(n_early)]

    # Keep only positive residuals
    pos_pairs = [(t, r) for t, r in zip(t_early, residuals) if r > 0]
    if len(pos_pairs) < 2:
        # Fallback: estimate alpha from first two points
        alpha = beta * 10.0
        A_coeff = cs[0] - B_coeff
        if A_coeff <= 0:
            A_coeff = cs[0]
    else:
        t_pos = [p[0] for p in pos_pairs]
        log_r = [math.log(p[1]) for p in pos_pairs]
        slope_alpha, intercept_alpha = _linreg(t_pos, log_r)
        alpha = -slope_alpha
        A_coeff = math.exp(intercept_alpha)

        if alpha <= beta:
            alpha = beta * 5.0  # ensure alpha > beta

    # ------------------------------------------------------------------
    # Build result via simulate_iv_two_compartment
    # ------------------------------------------------------------------
    result = simulate_iv_two_compartment(
        drug_name=drug_name,
        dose_mg=dose_mg,
        alpha=alpha,
        beta=beta,
        A_coeff=A_coeff,
        B_coeff=B_coeff,
        t_end_h=t_end_h,
        dt=dt,
    )

    # Annotate notes with fit quality
    predicted = [A_coeff * math.exp(-alpha * t) + B_coeff * math.exp(-beta * t) for t in ts]
    r2 = _r_squared(cs, predicted)
    result.notes = f"Method-of-residuals fit. R²={r2:.3f}. " + result.notes
    return result
