"""Flip-flop pharmacokinetics model.

Flip-flop occurs when ka < ke: the terminal slope reflects absorption, not elimination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "FlipFlopResult",
    "simulate_flip_flop_pk",
    "screen_flip_flop_risk",
    "identify_flip_flop_from_data",
]


@dataclass
class FlipFlopResult:
    """Result from flip-flop PK simulation."""

    drug_name: str
    dose_mg: float
    ka_per_h: float
    ke_per_h: float
    vd_L: float
    times_h: list
    c_plasma_mg_L: list
    is_flip_flop: bool
    ka_ke_ratio: float
    apparent_t_half_h: float
    true_t_half_h: float
    absorption_t_half_h: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    misidentification_risk: str
    notes: str


def _log_linear_terminal_slope(times_h: list, concs: list, fraction: float = 0.30) -> float:
    """Estimate terminal slope (lambda_z) from log-linear regression on last fraction of data."""
    n = len(times_h)
    start_idx = int(n * (1.0 - fraction))
    if start_idx >= n - 2:
        start_idx = max(0, n - 3)

    # Filter positive concentrations
    t_pts = []
    ln_c_pts = []
    for i in range(start_idx, n):
        if concs[i] > 1e-15:
            t_pts.append(times_h[i])
            ln_c_pts.append(math.log(concs[i]))

    if len(t_pts) < 2:
        return 0.0

    # Simple linear regression: ln(C) = intercept - slope * t  → slope = lambda_z
    n_pts = len(t_pts)
    sum_t = sum(t_pts)
    sum_y = sum(ln_c_pts)
    sum_tt = sum(t * t for t in t_pts)
    sum_ty = sum(t_pts[i] * ln_c_pts[i] for i in range(n_pts))

    denom = n_pts * sum_tt - sum_t * sum_t
    if abs(denom) < 1e-15:
        return 0.0

    slope = (n_pts * sum_ty - sum_t * sum_y) / denom
    # slope is negative for declining curve; lambda_z = -slope
    return -slope


def simulate_flip_flop_pk(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float,
    ke_per_h: float,
    vd_L: float,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> FlipFlopResult:
    """Simulate 1-compartment oral PK and detect flip-flop.

    Forward Euler: dA_gut/dt = -ka * A_gut; dC/dt = (ka * A_gut) / Vd - ke * C
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if ke_per_h <= 0:
        raise ValueError("ke_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")

    # Initial conditions
    a_gut = dose_mg * f_oral  # mg in gut
    c = 0.0  # mg/L in plasma

    times_h = []
    c_plasma = []

    t = 0.0
    steps = int(t_end_h / dt_h) + 1

    for _ in range(steps):
        times_h.append(t)
        c_plasma.append(c)

        # Forward Euler
        da_gut = -ka_per_h * a_gut
        dc = (ka_per_h * a_gut) / vd_L - ke_per_h * c

        a_gut += da_gut * dt_h
        if a_gut < 0.0:
            a_gut = 0.0
        c += dc * dt_h
        if c < 0.0:
            c = 0.0
        t += dt_h

    # Cmax and Tmax
    cmax = max(c_plasma)
    tmax = times_h[c_plasma.index(cmax)]

    # AUC via trapezoidal
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma[i - 1] + c_plasma[i]) * (times_h[i] - times_h[i - 1])

    # Flip-flop detection
    is_flip_flop = ka_per_h < ke_per_h
    ka_ke_ratio = ka_per_h / ke_per_h

    # Terminal half-life from log-linear regression
    lambda_z = _log_linear_terminal_slope(times_h, c_plasma, fraction=0.30)
    apparent_t_half = math.log(2.0) / lambda_z if lambda_z > 1e-10 else float("inf")

    true_t_half = math.log(2.0) / ke_per_h
    absorption_t_half = math.log(2.0) / ka_per_h

    # Misidentification risk
    if is_flip_flop and ka_ke_ratio < 0.5:
        misidentification_risk = "high"
    elif is_flip_flop:
        misidentification_risk = "moderate"
    else:
        misidentification_risk = "low"

    notes_parts = []
    if is_flip_flop:
        notes_parts.append(
            f"Flip-flop PK detected: ka ({ka_per_h:.3f}/h) < ke ({ke_per_h:.3f}/h). "
            f"Terminal slope reflects absorption (t½_abs={absorption_t_half:.2f}h), "
            f"not elimination (t½_elim={true_t_half:.2f}h)."
        )
    else:
        notes_parts.append(
            f"No flip-flop: ka ({ka_per_h:.3f}/h) >= ke ({ke_per_h:.3f}/h). "
            f"Terminal slope reflects elimination (t½={true_t_half:.2f}h)."
        )

    return FlipFlopResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_per_h=ka_per_h,
        ke_per_h=ke_per_h,
        vd_L=vd_L,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        is_flip_flop=is_flip_flop,
        ka_ke_ratio=ka_ke_ratio,
        apparent_t_half_h=apparent_t_half,
        true_t_half_h=true_t_half,
        absorption_t_half_h=absorption_t_half,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        misidentification_risk=misidentification_risk,
        notes=" ".join(notes_parts),
    )


def screen_flip_flop_risk(compounds: list) -> list:
    """Screen a list of compounds for flip-flop risk.

    Each dict: {"compound_name": str, "ka_per_h": float, "ke_per_h": float,
                "dose_mg": float, "vd_L": float}

    Returns sorted by ka_ke_ratio ascending (most flip-flop prone first).
    """
    results = []
    for c in compounds:
        result = simulate_flip_flop_pk(
            drug_name=c.get("compound_name", "unknown"),
            dose_mg=c["dose_mg"],
            ka_per_h=c["ka_per_h"],
            ke_per_h=c["ke_per_h"],
            vd_L=c["vd_L"],
        )
        results.append(result)

    results.sort(key=lambda r: r.ka_ke_ratio)
    return results


def identify_flip_flop_from_data(
    times_h: list,
    concs_mg_L: list,
    ke_true: float,
) -> dict:
    """Detect flip-flop from observed PK data when true ke is known.

    Fits terminal slope (lambda_z) from last 30% of data.
    Flip-flop is identified when apparent_ke < ke_true * 0.8.

    Returns:
        dict with apparent_ke, true_ke, is_flip_flop, confidence
    """
    lambda_z = _log_linear_terminal_slope(times_h, concs_mg_L, fraction=0.30)
    apparent_ke = lambda_z

    is_flip_flop = apparent_ke < ke_true * 0.8

    ratio = apparent_ke / ke_true if ke_true > 1e-15 else 1.0
    if ratio < 0.5:
        confidence = "high"
    elif ratio < 0.8:
        confidence = "moderate"
    else:
        confidence = "low"

    return {
        "apparent_ke": apparent_ke,
        "true_ke": ke_true,
        "is_flip_flop": is_flip_flop,
        "confidence": confidence,
    }
