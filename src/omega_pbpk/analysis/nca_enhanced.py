"""Enhanced Non-Compartmental Analysis (NCA) — Phase 813.

Provides NCAEnhancedResult and nca_enhanced() with:
- Linear-log trapezoidal AUC
- AUMC and MRT
- Auto-selected terminal slope (R² > 0.9, min 3 points)
- AUC extrapolation to infinity
- Vd_area and CL_obs
- Bioavailability ratio
- batch_nca() for multiple profiles
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NCAEnhancedResult:
    compound_name: str
    route: str
    dose_mg: float
    cmax: float
    tmax_h: float
    auc_0_t: float
    auc_0_inf: float
    auc_extrapolation_pct: float
    aumc: float
    mrt_h: float
    t_half_h: float
    kel: float
    vd_area_L: float
    cl_obs_L_per_h: float
    bioavailability_ratio: float
    kel_r2: float
    n_timepoints_terminal: int
    notes: str


def _linlog_trap(t1: float, t2: float, c1: float, c2: float) -> float:
    """Linear-log trapezoidal rule for one interval.

    Uses log method when both concentrations are positive and c2 < c1
    (declining phase), otherwise uses linear trapezoidal.
    """
    dt = t2 - t1
    if dt <= 0.0:
        return 0.0
    if c1 > 0.0 and c2 > 0.0 and c2 < c1:
        # Log trapezoidal
        ln_ratio = math.log(c1 / c2)
        if ln_ratio == 0.0:
            return dt * (c1 + c2) / 2.0
        return dt * (c1 - c2) / ln_ratio
    # Linear trapezoidal
    return dt * (c1 + c2) / 2.0


def _linlog_trap_moment(t1: float, t2: float, c1: float, c2: float) -> float:
    """Trapezoidal area for moment curve (t*C)."""
    dt = t2 - t1
    if dt <= 0.0:
        return 0.0
    tc1 = t1 * c1
    tc2 = t2 * c2
    return dt * (tc1 + tc2) / 2.0


def _linear_regression(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) for lists x, y."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return 0.0, 0.0, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    y_mean = sy / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    if ss_tot == 0.0:
        return slope, intercept, 1.0
    y_pred = [slope * xi + intercept for xi in x]
    ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
    r2 = 1.0 - ss_res / ss_tot
    return slope, intercept, r2


def _auto_terminal(
    times: list[float],
    log_concs: list[float],
) -> tuple[int, float, float, float]:
    """Find the best terminal subset (max R², min 3 points, R² > 0.9).

    Returns (n_points, kel, intercept_ln, r2) for the last n_points.
    """
    n = len(times)
    best_r2 = -1.0
    best_n = 3
    best_kel = 0.0
    best_intercept = 0.0

    max_n = n - 1  # at least 1 point before terminal phase
    if max_n < 3:
        max_n = n

    for k in range(3, max_n + 1):
        t_sub = times[n - k :]
        lc_sub = log_concs[n - k :]
        slope, intercept, r2 = _linear_regression(t_sub, lc_sub)
        kel = -slope
        if kel <= 0.0:
            continue
        if r2 > best_r2:
            best_r2 = r2
            best_n = k
            best_kel = kel
            best_intercept = intercept

    # If no subset with r2 > 0, use last 3 points
    if best_r2 < 0.0:
        k = min(3, n)
        t_sub = times[n - k :]
        lc_sub = log_concs[n - k :]
        slope, intercept, r2 = _linear_regression(t_sub, lc_sub)
        best_kel = max(-slope, 1e-9)
        best_intercept = intercept
        best_r2 = max(r2, 0.0)
        best_n = k

    return best_n, best_kel, best_intercept, best_r2


def nca_enhanced(
    compound_name: str,
    times_h: list[float],
    concentrations_mg_L: list[float],
    dose_mg: float,
    route: str = "iv",
    n_terminal: int | None = None,
    iv_auc_for_bioavailability: float | None = None,
    iv_dose_for_bioavailability: float | None = None,
) -> NCAEnhancedResult:
    """Perform enhanced NCA on a single PK profile.

    Parameters
    ----------
    compound_name : str
    times_h : list[float]
        Time points in hours (must be sorted ascending).
    concentrations_mg_L : list[float]
        Plasma concentrations in mg/L.
    dose_mg : float
        Administered dose in mg.
    route : str
        "iv" or "oral".
    n_terminal : int or None
        Number of points for terminal slope. Auto-selected if None.
    iv_auc_for_bioavailability : float or None
        AUC_0_inf from IV reference for bioavailability calculation.
    iv_dose_for_bioavailability : float or None
        IV dose for bioavailability calculation.

    Returns
    -------
    NCAEnhancedResult
    """
    if len(times_h) != len(concentrations_mg_L):
        raise ValueError(
            f"times_h (len={len(times_h)}) and concentrations_mg_L "
            f"(len={len(concentrations_mg_L)}) must have the same length."
        )
    if len(times_h) < 2:
        raise ValueError("At least 2 time points are required for NCA.")
    for i in range(1, len(times_h)):
        if times_h[i] <= times_h[i - 1]:
            raise ValueError(
                f"times_h must be strictly increasing. "
                f"Found times_h[{i - 1}]={times_h[i - 1]} >= times_h[{i}]={times_h[i]}."
            )

    n = len(times_h)
    # Cmax and Tmax
    cmax = max(concentrations_mg_L)
    tmax_h = times_h[concentrations_mg_L.index(cmax)]

    # AUC_0_t (linear-log trapezoidal)
    auc_0_t = 0.0
    for i in range(n - 1):
        auc_0_t += _linlog_trap(
            times_h[i],
            times_h[i + 1],
            concentrations_mg_L[i],
            concentrations_mg_L[i + 1],
        )

    # AUMC (moment curve)
    aumc_0_t = 0.0
    for i in range(n - 1):
        aumc_0_t += _linlog_trap_moment(
            times_h[i],
            times_h[i + 1],
            concentrations_mg_L[i],
            concentrations_mg_L[i + 1],
        )

    # Terminal slope — use log(C) vs t regression
    # Filter out non-positive concentrations for log transform
    valid_indices = [i for i, c in enumerate(concentrations_mg_L) if c > 0.0]
    if len(valid_indices) < 2:
        raise ValueError("At least 2 positive concentrations are required for terminal slope.")

    t_valid = [times_h[i] for i in valid_indices]
    lc_valid = [math.log(concentrations_mg_L[i]) for i in valid_indices]

    notes_parts: list[str] = []

    if n_terminal is not None:
        # Use user-specified number of terminal points
        if n_terminal < 2:
            raise ValueError("n_terminal must be at least 2.")
        k = min(n_terminal, len(t_valid))
        t_sub = t_valid[len(t_valid) - k :]
        lc_sub = lc_valid[len(lc_valid) - k :]
        slope, intercept, r2 = _linear_regression(t_sub, lc_sub)
        kel = -slope
        if kel <= 0.0:
            kel = 1e-9
            notes_parts.append("kel forced positive from terminal regression")
        n_pts = k
    else:
        n_pts, kel, intercept, r2 = _auto_terminal(t_valid, lc_valid)
        if r2 < 0.9:
            notes_parts.append(f"Terminal R²={r2:.3f} < 0.9; best available fit used.")

    kel_r2 = r2

    # Clast from last valid concentration
    clast = concentrations_mg_L[valid_indices[-1]]
    tlast = times_h[valid_indices[-1]]

    # AUC extrapolation: Clast / kel
    auc_extra = clast / kel
    auc_0_inf = auc_0_t + auc_extra
    auc_extrapolation_pct = (auc_extra / auc_0_inf * 100.0) if auc_0_inf > 0.0 else 0.0

    # AUMC extrapolation: tlast * Clast / kel + Clast / kel²
    aumc_extra = tlast * clast / kel + clast / (kel * kel)
    aumc_0_inf = aumc_0_t + aumc_extra

    # MRT
    mrt_h = aumc_0_inf / auc_0_inf if auc_0_inf > 0.0 else 0.0
    if route == "iv":
        # For IV bolus, subtract DOSING_DURATION/2 — for bolus assume 0
        pass  # MRT for IV bolus = AUMC/AUC
    # For oral: MRT_oral - MRT_iv gives MAT, but we don't have IV here

    # t½
    t_half_h = math.log(2.0) / kel if kel > 0.0 else float("inf")

    # Vd_area and CL_obs
    if route == "iv":
        cl_obs = dose_mg / auc_0_inf if auc_0_inf > 0.0 else float("inf")
        vd_area = dose_mg / (kel * auc_0_inf) if (kel > 0.0 and auc_0_inf > 0.0) else float("inf")
    else:
        # oral: CL/F = dose / AUC
        cl_obs = dose_mg / auc_0_inf if auc_0_inf > 0.0 else float("inf")
        vd_area = dose_mg / (kel * auc_0_inf) if (kel > 0.0 and auc_0_inf > 0.0) else float("inf")

    # Bioavailability
    if iv_auc_for_bioavailability is not None and iv_dose_for_bioavailability is not None:
        if (
            route == "oral"
            and iv_auc_for_bioavailability > 0.0
            and iv_dose_for_bioavailability > 0.0
        ):
            bioavailability_ratio = (auc_0_inf * iv_dose_for_bioavailability) / (
                iv_auc_for_bioavailability * dose_mg
            )
        else:
            bioavailability_ratio = 1.0
            notes_parts.append("Bioavailability set to 1.0 (IV route or missing reference).")
    else:
        bioavailability_ratio = 1.0

    notes = "; ".join(notes_parts) if notes_parts else "OK"

    return NCAEnhancedResult(
        compound_name=compound_name,
        route=route,
        dose_mg=dose_mg,
        cmax=cmax,
        tmax_h=tmax_h,
        auc_0_t=auc_0_t,
        auc_0_inf=auc_0_inf,
        auc_extrapolation_pct=auc_extrapolation_pct,
        aumc=aumc_0_inf,
        mrt_h=mrt_h,
        t_half_h=t_half_h,
        kel=kel,
        vd_area_L=vd_area,
        cl_obs_L_per_h=cl_obs,
        bioavailability_ratio=bioavailability_ratio,
        kel_r2=kel_r2,
        n_timepoints_terminal=n_pts,
        notes=notes,
    )


def batch_nca(profiles: list[dict]) -> list[NCAEnhancedResult]:
    """Run nca_enhanced on a list of profile dictionaries.

    Each dict must have keys matching the nca_enhanced() parameters:
    - compound_name, times_h, concentrations_mg_L, dose_mg
    Optional: route, n_terminal, iv_auc_for_bioavailability,
              iv_dose_for_bioavailability

    Returns
    -------
    list[NCAEnhancedResult]
    """
    results = []
    for profile in profiles:
        result = nca_enhanced(
            compound_name=profile["compound_name"],
            times_h=profile["times_h"],
            concentrations_mg_L=profile["concentrations_mg_L"],
            dose_mg=profile["dose_mg"],
            route=profile.get("route", "iv"),
            n_terminal=profile.get("n_terminal", None),
            iv_auc_for_bioavailability=profile.get("iv_auc_for_bioavailability", None),
            iv_dose_for_bioavailability=profile.get("iv_dose_for_bioavailability", None),
        )
        results.append(result)
    return results
