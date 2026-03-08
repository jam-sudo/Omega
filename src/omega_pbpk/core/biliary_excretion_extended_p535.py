"""
Phase 535 — Biliary Excretion and Enterohepatic Recirculation (Extended)

Extended model for biliary excretion with MW threshold and deconjugation,
using a 3-compartment Forward Euler ODE system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BiliaryExcretionExtendedResult:
    """Result of extended biliary excretion simulation."""

    drug_name: str
    dose_mg: float
    mw_Da: float
    f_bile: float
    f_deconj: float
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    a_bile_mg: list[float] = field(default_factory=list)
    cmax_plasma: float = 0.0
    tmax_plasma_h: float = 0.0
    auc_plasma: float = 0.0
    biliary_fraction_pct: float = 0.0
    recirculation_cycles: float = 0.0
    effective_t_half_h: float = 0.0
    ehr_amplification: bool = False
    notes: str = ""


def _compute_f_bile(mw_Da: float) -> float:
    """Fraction of hepatic clearance going to bile based on MW threshold."""
    if mw_Da <= 400.0:
        return 0.0
    return max(0.0, min(0.8, (mw_Da - 400.0) / 500.0))


def simulate_biliary_excretion_extended(
    drug_name: str,
    dose_mg: float,
    mw_Da: float,
    cl_total_L_per_h: float,
    vd_L: float,
    f_deconj: float = 0.7,
    t_end_h: float = 72.0,
    dt_h: float = 0.2,
) -> BiliaryExcretionExtendedResult:
    """
    Simulate biliary excretion with enterohepatic recirculation.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    mw_Da : float
    cl_total_L_per_h : float
    vd_L : float
    f_deconj : float, optional
        Fraction of biliary drug reabsorbed after deconjugation (default 0.7)
    t_end_h : float, optional
    dt_h : float, optional

    Returns
    -------
    BiliaryExcretionExtendedResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if cl_total_L_per_h <= 0:
        raise ValueError("cl_total_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0.0 <= f_deconj <= 1.0):
        raise ValueError("f_deconj must be between 0 and 1")

    f_bile = _compute_f_bile(mw_Da)

    # Rate constants
    cl_bile = cl_total_L_per_h * f_bile
    cl_nonbile = cl_total_L_per_h * (1.0 - f_bile)
    k_el = cl_nonbile / vd_L  # non-biliary elimination (/h)
    k_bile_clearance = cl_bile / vd_L  # biliary clearance (/h)
    k_transit = 0.1  # gut transit (/h)
    k_portal = 1.0  # portal to plasma (/h)

    # Initial conditions
    c_plasma = dose_mg / vd_L
    a_bile = 0.0
    a_portal = 0.0

    times_h: list[float] = []
    c_plasma_list: list[float] = []
    a_bile_list: list[float] = []

    t = 0.0
    steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_bile_list.append(a_bile)

        # Derivatives
        dc_plasma_dt = -k_el * c_plasma - k_bile_clearance * c_plasma + k_portal * a_portal / vd_L
        da_bile_dt = k_bile_clearance * c_plasma * vd_L - k_transit * a_bile
        da_portal_dt = k_transit * a_bile * f_deconj - k_portal * a_portal

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + dc_plasma_dt * dt_h)
        a_bile = max(0.0, a_bile + da_bile_dt * dt_h)
        a_portal = max(0.0, a_portal + da_portal_dt * dt_h)

        t += dt_h

    # Post-processing
    cmax_plasma = max(c_plasma_list)
    tmax_plasma_h = times_h[c_plasma_list.index(cmax_plasma)]

    # Trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    biliary_fraction_pct = f_bile * 100.0
    recirculation_cycles = f_bile * f_deconj

    # Effective half-life: approximate from terminal slope
    # Use last 25% of simulation to estimate lambda_z
    n = len(times_h)
    start_idx = int(n * 0.75)
    effective_t_half_h = float("nan")
    if start_idx < n - 1:
        c_end = c_plasma_list[-1]
        c_start_seg = c_plasma_list[start_idx]
        dt_seg = times_h[-1] - times_h[start_idx]
        if c_end > 0 and c_start_seg > 0 and c_end < c_start_seg and dt_seg > 0:
            lambda_z = math.log(c_start_seg / c_end) / dt_seg
            if lambda_z > 0:
                effective_t_half_h = math.log(2.0) / lambda_z

    if math.isnan(effective_t_half_h):
        # Fallback to theoretical t½ based on non-biliary elimination
        effective_t_half_h = math.log(2.0) / k_el if k_el > 0 else float("inf")

    ehr_amplification = f_bile > 0.3 and f_deconj > 0.5

    notes_parts = []
    if f_bile == 0.0:
        notes_parts.append("MW<=400 Da: no significant biliary excretion")
    if ehr_amplification:
        notes_parts.append("Significant enterohepatic recirculation detected")
    notes = "; ".join(notes_parts) if notes_parts else "Normal biliary excretion profile"

    return BiliaryExcretionExtendedResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_Da=mw_Da,
        f_bile=f_bile,
        f_deconj=f_deconj,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_bile_mg=a_bile_list,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        biliary_fraction_pct=biliary_fraction_pct,
        recirculation_cycles=recirculation_cycles,
        effective_t_half_h=effective_t_half_h,
        ehr_amplification=ehr_amplification,
        notes=notes,
    )


def compare_mw_biliary(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    mw_values: list,
) -> list:
    """
    Simulate biliary excretion for multiple MW values and return results sorted
    by biliary_fraction_pct descending (highest biliary excretion first).

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    cl_L_per_h : float
    vd_L : float
    mw_values : list of float

    Returns
    -------
    list of BiliaryExcretionExtendedResult sorted by biliary_fraction_pct descending
    """
    results = []
    for mw in mw_values:
        result = simulate_biliary_excretion_extended(
            drug_name=drug_name,
            dose_mg=dose_mg,
            mw_Da=mw,
            cl_total_L_per_h=cl_L_per_h,
            vd_L=vd_L,
        )
        results.append(result)
    results.sort(key=lambda r: r.biliary_fraction_pct, reverse=True)
    return results
