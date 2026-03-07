"""Phase 374 — Pharmacokinetic Parameter Calculator (NCA).

Non-Compartmental Analysis (NCA) of PK concentration-time data
for IV and oral routes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NCAResult:
    """Results from Non-Compartmental Analysis."""

    drug_name: str
    dose_mg: float
    route: str

    # Input
    times_h: list[float]
    concentrations_mg_L: list[float]

    # Primary NCA parameters
    cmax_mg_L: float
    tmax_h: float
    auc_last_mg_h_per_L: float
    auc_inf_mg_h_per_L: float
    aumc_mg_h2_per_L: float

    # Secondary parameters
    t_half_h: float
    lambda_z_per_h: float
    mrt_h: float
    vd_ss_L: float
    cl_obs_L_per_h: float

    # Extrapolation quality
    pct_auc_extrapolated: float
    n_points_for_lambda: int
    r2_terminal: float

    # Bioavailability (oral only)
    f_oral: float | None

    # Summary flag
    adequate_sampling: bool

    notes: str


def _trapezoid(times: list[float], concs: list[float]) -> float:
    """Linear trapezoid integration."""
    total = 0.0
    for i in range(1, len(times)):
        total += 0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1])
    return total


def _trapezoid_first_moment(times: list[float], concs: list[float]) -> float:
    """Trapezoid integration of t*C(t)."""
    tc = [times[i] * concs[i] for i in range(len(times))]
    return _trapezoid(times, tc)


def _linear_fit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Fit y = a + b*x via OLS.

    Returns (a, b, r2).
    """
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    sxy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    if sxx == 0.0:
        return mean_y, 0.0, 1.0
    b = sxy / sxx
    a = mean_y - b * mean_x
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    if ss_tot == 0.0:
        return a, b, 1.0
    ss_res = sum((y[i] - (a + b * x[i])) ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot
    return a, b, r2


def calculate_nca(
    drug_name: str,
    dose_mg: float,
    times_h: list[float],
    concentrations_mg_L: list[float],
    route: str = "oral",
    f_for_oral: float | None = None,
    n_terminal_points: int = 3,
) -> NCAResult:
    """Perform Non-Compartmental Analysis of PK data.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    times_h:
        Sampling times in hours, must be sorted ascending.
    concentrations_mg_L:
        Plasma concentrations in mg/L corresponding to times_h.
    route:
        Dosing route: "iv" or "oral".
    f_for_oral:
        Oral bioavailability fraction (0–1). Required for CL calculation
        in oral route. If None for oral, CL is reported as apparent CL/F.
    n_terminal_points:
        Number of terminal data points to use for lambda_z estimation.
        If 0, use all available positive-concentration points.

    Returns
    -------
    NCAResult
    """
    # --- Input validation ---
    if len(times_h) < 3 or len(concentrations_mg_L) < 3:
        raise ValueError("At least 3 data points are required.")
    if len(times_h) != len(concentrations_mg_L):
        raise ValueError("times_h and concentrations_mg_L must have the same length.")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive; got {dose_mg}.")
    if route not in {"iv", "oral"}:
        raise ValueError(f"route must be 'iv' or 'oral'; got '{route}'.")
    for i in range(1, len(times_h)):
        if times_h[i] <= times_h[i - 1]:
            raise ValueError("times_h must be strictly ascending.")
    for c in concentrations_mg_L:
        if c < 0:
            raise ValueError(f"Concentrations must be non-negative; got {c}.")

    times = list(times_h)
    concs = list(concentrations_mg_L)
    n = len(times)

    # --- Drop trailing BLQ (zero/negative) concentrations ---
    last_pos = n - 1
    while last_pos > 0 and concs[last_pos] <= 0.0:
        last_pos -= 1
    # Trim to last positive index
    times_valid = times[: last_pos + 1]
    concs_valid = concs[: last_pos + 1]
    n_valid = len(times_valid)

    if n_valid < 2:
        raise ValueError("Insufficient non-zero concentration data for NCA.")

    # --- Cmax, Tmax ---
    cmax = max(concs_valid)
    tmax_idx = concs_valid.index(cmax)
    tmax = times_valid[tmax_idx]

    # --- Terminal elimination: lambda_z ---
    # Use last n_terminal_points positive-concentration points
    if n_terminal_points == 0 or n_terminal_points >= n_valid:
        n_term = n_valid
    else:
        n_term = n_terminal_points

    # Must have at least 2 terminal points for regression
    n_term = max(2, min(n_term, n_valid))

    term_times = times_valid[-n_term:]
    term_concs = concs_valid[-n_term:]

    # Check all terminal concs > 0
    if any(c <= 0 for c in term_concs):
        raise ValueError(
            "Terminal concentration points must all be positive for lambda_z estimation."
        )

    log_term_concs = [math.log(c) for c in term_concs]
    # Fit log(C) = intercept - lambda_z * t
    a_fit, b_fit, r2_term = _linear_fit(term_times, log_term_concs)
    # b_fit = -lambda_z  (should be negative for elimination)
    lambda_z = -b_fit
    if lambda_z <= 0:
        # Force positive by using the magnitude
        lambda_z = abs(lambda_z) if abs(lambda_z) > 1e-12 else 1e-6

    t_half = math.log(2.0) / lambda_z

    # --- AUC_last (0 → t_last) ---
    auc_last = _trapezoid(times_valid, concs_valid)

    # --- AUC_inf: extrapolate from Clast ---
    c_last = concs_valid[-1]
    t_last = times_valid[-1]
    auc_extrapolated = c_last / lambda_z
    auc_inf = auc_last + auc_extrapolated

    pct_extrapolated = (auc_extrapolated / auc_inf * 100.0) if auc_inf > 0 else 0.0

    # --- AUMC (0 → t_last), then extrapolate to inf ---
    # AUMC_last = integral of t*C from 0 to t_last (linear trapezoid)
    aumc_last = _trapezoid_first_moment(times_valid, concs_valid)
    # Extrapolation: AUMC_extra = t_last*C_last/lambda_z + C_last/lambda_z^2
    aumc_extra = t_last * c_last / lambda_z + c_last / (lambda_z**2)
    aumc_total = aumc_last + aumc_extra

    # --- MRT ---
    mrt = aumc_total / auc_inf if auc_inf > 0 else 0.0

    # --- CL ---
    if route == "iv":
        cl = dose_mg / auc_inf
        f_oral_result: float | None = None
    else:
        # Oral: CL/F (apparent) if F not given
        if f_for_oral is not None:
            cl = f_for_oral * dose_mg / auc_inf
        else:
            cl = dose_mg / auc_inf  # apparent CL/F
        f_oral_result = f_for_oral

    # --- Vd_ss = CL * MRT ---
    vd_ss = cl * mrt

    # --- Adequate sampling check ---
    adequate = r2_term > 0.95 and pct_extrapolated < 20.0 and n_term >= 3

    notes_parts: list[str] = []
    if route == "oral" and f_for_oral is None:
        notes_parts.append("CL reported as apparent CL/F (bioavailability unknown).")
    if pct_extrapolated > 20.0:
        notes_parts.append(f"High extrapolation: {pct_extrapolated:.1f}% of AUC_inf extrapolated.")
    if r2_term < 0.95:
        notes_parts.append(
            f"Poor terminal fit: R2={r2_term:.3f}. Lambda_z estimate may be unreliable."
        )

    return NCAResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        concentrations_mg_L=concs,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_last_mg_h_per_L=auc_last,
        auc_inf_mg_h_per_L=auc_inf,
        aumc_mg_h2_per_L=aumc_total,
        t_half_h=t_half,
        lambda_z_per_h=lambda_z,
        mrt_h=mrt,
        vd_ss_L=vd_ss,
        cl_obs_L_per_h=cl,
        pct_auc_extrapolated=pct_extrapolated,
        n_points_for_lambda=n_term,
        r2_terminal=r2_term,
        f_oral=f_oral_result,
        adequate_sampling=adequate,
        notes=" ".join(notes_parts),
    )


def compare_nca_parameters(nca_results: list[NCAResult]) -> dict:
    """Compare NCA parameters across multiple subjects or studies.

    Parameters
    ----------
    nca_results:
        List of NCAResult objects.

    Returns
    -------
    dict
        {param: {mean, cv_pct, range}} for CL, Vd, t_half, AUC.
    """
    if not nca_results:
        return {}

    params = {
        "cl_obs_L_per_h": [r.cl_obs_L_per_h for r in nca_results],
        "vd_ss_L": [r.vd_ss_L for r in nca_results],
        "t_half_h": [r.t_half_h for r in nca_results],
        "auc_inf_mg_h_per_L": [r.auc_inf_mg_h_per_L for r in nca_results],
    }

    summary: dict = {}
    for param, values in params.items():
        n = len(values)
        mean_v = sum(values) / n
        if n > 1:
            variance = sum((v - mean_v) ** 2 for v in values) / (n - 1)
            std_v = math.sqrt(variance)
        else:
            std_v = 0.0
        cv_pct = (std_v / mean_v * 100.0) if mean_v != 0.0 else 0.0
        summary[param] = {
            "mean": mean_v,
            "cv_pct": cv_pct,
            "range": (min(values), max(values)),
        }

    return summary
