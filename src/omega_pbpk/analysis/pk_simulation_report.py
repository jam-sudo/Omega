"""Structured PK simulation report with key metrics — Phase 406."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PKReport",
    "generate_pk_report",
    "compare_simulations",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PKReport:
    """Structured PK simulation report with computed metrics."""

    drug_name: str
    route: str
    dose_mg: float
    interval_h: float
    cmax_mg_L: float
    tmax_h: float
    auc_inf_mg_h_per_L: float
    t_half_h: float
    ke_per_h: float
    cl_L_per_h: float
    vd_L: float
    mrt_h: float
    accumulation_ratio: float
    steady_state_trough_mg_L: float
    notes: tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC integration."""
    auc = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        auc += 0.5 * (concs[i] + concs[i + 1]) * dt
    return auc


def _terminal_ke(times: list[float], concs: list[float]) -> float:
    """Estimate terminal elimination rate constant from log-linear regression.

    Uses the last 3–5 non-zero data points in the terminal phase.
    Falls back to two-point log-linear if insufficient points.
    """
    # Find Cmax index
    cmax_idx = 0
    for i, c in enumerate(concs):
        if c >= concs[cmax_idx]:
            cmax_idx = i

    # Terminal phase: after Cmax, use up to 5 points
    terminal_t = []
    terminal_lnc = []
    for i in range(cmax_idx, len(concs)):
        if concs[i] > 1e-12:
            terminal_t.append(times[i])
            terminal_lnc.append(math.log(concs[i]))

    if len(terminal_t) < 2:
        return 0.1  # fallback

    # Use last min(5, n) points
    n_pts = min(5, len(terminal_t))
    t_pts = terminal_t[-n_pts:]
    lnc_pts = terminal_lnc[-n_pts:]

    # Linear regression: lnC = lnC0 - ke*t
    n = len(t_pts)
    t_mean = sum(t_pts) / n
    lnc_mean = sum(lnc_pts) / n
    num = sum((t_pts[i] - t_mean) * (lnc_pts[i] - lnc_mean) for i in range(n))
    denom = sum((t_pts[i] - t_mean) ** 2 for i in range(n))

    if denom < 1e-12:
        # Two-point fallback
        ke_est = (lnc_pts[0] - lnc_pts[-1]) / (t_pts[-1] - t_pts[0])
        return max(ke_est, 1e-4)

    slope = num / denom  # negative slope = -ke
    ke_est = max(-slope, 1e-4)
    return ke_est


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------


def generate_pk_report(
    drug_name: str,
    times_h: list[float],
    c_plasma: list[float],
    dose_mg: float,
    route: str,
    cl_L_per_h: float,
    vd_L: float,
    interval_h: float = 0.0,
) -> PKReport:
    """Generate a structured PK simulation report with key metrics.

    Computes NCA-style parameters from a simulated concentration–time profile.
    If interval_h > 0 (multi-dose), also computes accumulation ratio and
    steady-state trough concentration.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    times_h : list[float]
        Time points (h). Must be sorted, length >= 2.
    c_plasma : list[float]
        Plasma concentrations (mg/L) corresponding to times_h.
    dose_mg : float
        Dose (mg). Must be > 0.
    route : str
        Administration route (e.g. "oral", "iv", "sc").
    cl_L_per_h : float
        Systemic clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    interval_h : float
        Dosing interval (h). If > 0, multi-dose metrics are computed.
        Default 0.0 (single dose).

    Returns
    -------
    PKReport
    """
    # --- Validation ---
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if len(times_h) < 2:
        raise ValueError("times_h must have at least 2 elements")
    if len(times_h) != len(c_plasma):
        raise ValueError("times_h and c_plasma must have the same length")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not route:
        raise ValueError("route must be a non-empty string")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if interval_h < 0:
        raise ValueError("interval_h must be >= 0")

    # --- Basic NCA ---
    # Cmax and Tmax
    cmax_idx = max(range(len(c_plasma)), key=lambda i: c_plasma[i])
    cmax_mg_L = c_plasma[cmax_idx]
    tmax_h = times_h[cmax_idx]

    # AUC0-last (trapezoidal)
    auc_0_last = _trapezoidal_auc(list(times_h), list(c_plasma))

    # Terminal ke
    ke_per_h = _terminal_ke(list(times_h), list(c_plasma))
    t_half_h = math.log(2.0) / ke_per_h if ke_per_h > 0 else float("inf")

    # AUC extrapolation: AUC_inf = AUC_0_last + Clast/ke
    c_last = c_plasma[-1] if c_plasma[-1] > 0 else 0.0
    auc_extrap = c_last / ke_per_h if ke_per_h > 0 else 0.0
    auc_inf = auc_0_last + auc_extrap

    # MRT = AUMC / AUC; AUMC = integral(t*C dt)
    # Computed manually (trapezoidal)
    aumc = 0.0
    for i in range(len(times_h) - 1):
        dt = times_h[i + 1] - times_h[i]
        aumc += 0.5 * (times_h[i] * c_plasma[i] + times_h[i + 1] * c_plasma[i + 1]) * dt
    # Extrapolation of AUMC: (Clast * tlast / ke) + (Clast / ke^2)
    t_last = times_h[-1]
    aumc_extrap = (c_last * t_last / ke_per_h + c_last / (ke_per_h**2)) if ke_per_h > 0 else 0.0
    aumc_inf = aumc + aumc_extrap
    mrt_h = aumc_inf / auc_inf if auc_inf > 0 else 0.0

    # --- Multi-dose ---
    accumulation_ratio = 0.0
    ss_trough_mg_L = 0.0
    if interval_h > 0:
        # Accumulation ratio R = 1 / (1 - exp(-ke * tau))
        exp_term = math.exp(-ke_per_h * interval_h)
        if abs(1.0 - exp_term) > 1e-9:
            accumulation_ratio = 1.0 / (1.0 - exp_term)
        else:
            accumulation_ratio = float("inf")
        # Steady-state trough: Ctrough,ss = (dose/Vd) * exp(-ke*tau) / (1 - exp(-ke*tau))
        # This is the 1-compartment IV approximation; for oral route it serves as estimate
        c0_approx = dose_mg / vd_L
        denom = 1.0 - exp_term
        ss_trough_mg_L = c0_approx * exp_term / denom if abs(denom) > 1e-9 else 0.0

    # --- Notes ---
    notes_list: list[str] = []
    if t_half_h > interval_h > 0:
        notes_list.append("t_half > dosing interval — significant accumulation expected")
    if accumulation_ratio > 5:
        notes_list.append(f"High accumulation ratio ({accumulation_ratio:.1f}×)")
    if auc_extrap / auc_inf > 0.2 and auc_inf > 0:
        notes_list.append("AUC extrapolation >20% — consider longer sampling")
    if cmax_mg_L < 1e-9:
        notes_list.append("Very low Cmax — check input concentrations")

    return PKReport(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        interval_h=interval_h,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_inf_mg_h_per_L=auc_inf,
        t_half_h=t_half_h,
        ke_per_h=ke_per_h,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        mrt_h=mrt_h,
        accumulation_ratio=accumulation_ratio,
        steady_state_trough_mg_L=ss_trough_mg_L,
        notes=tuple(notes_list),
    )


# ---------------------------------------------------------------------------
# Comparison function
# ---------------------------------------------------------------------------


def compare_simulations(reports: list[PKReport]) -> dict:
    """Compare PK parameters across multiple simulation reports.

    Parameters
    ----------
    reports : list[PKReport]
        List of PKReport objects to compare. Must be non-empty.

    Returns
    -------
    dict with keys:
        "table" — list of dicts, one per report, with key metrics
        "best_auc" — drug_name with highest AUC
        "best_cmax" — drug_name with highest Cmax
        "longest_t_half" — drug_name with longest half-life
        "summary" — str overview
    """
    if not reports:
        raise ValueError("reports must be a non-empty list")

    table = []
    for r in reports:
        row = {
            "drug_name": r.drug_name,
            "route": r.route,
            "dose_mg": r.dose_mg,
            "cmax_mg_L": r.cmax_mg_L,
            "tmax_h": r.tmax_h,
            "auc_inf_mg_h_per_L": r.auc_inf_mg_h_per_L,
            "t_half_h": r.t_half_h,
            "ke_per_h": r.ke_per_h,
            "cl_L_per_h": r.cl_L_per_h,
            "vd_L": r.vd_L,
            "mrt_h": r.mrt_h,
            "accumulation_ratio": r.accumulation_ratio,
            "steady_state_trough_mg_L": r.steady_state_trough_mg_L,
        }
        table.append(row)

    best_auc = max(reports, key=lambda r: r.auc_inf_mg_h_per_L).drug_name
    best_cmax = max(reports, key=lambda r: r.cmax_mg_L).drug_name
    longest_half = max(
        reports, key=lambda r: r.t_half_h if math.isfinite(r.t_half_h) else 0.0
    ).drug_name

    summary = (
        f"Compared {len(reports)} simulation(s). "
        f"Highest AUC: {best_auc}. "
        f"Highest Cmax: {best_cmax}. "
        f"Longest t½: {longest_half}."
    )

    return {
        "table": table,
        "best_auc": best_auc,
        "best_cmax": best_cmax,
        "longest_t_half": longest_half,
        "summary": summary,
    }
