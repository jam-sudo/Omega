"""Phase 278 — Bile Acid Recycling PK Model.

Simulate enterohepatic circulation (EHC) of bile acid-conjugated drugs using
a two-compartment model: systemic + GI lumen, with hepatic secretion into bile,
intestinal reabsorption (ileum), and first-pass effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BileAcidRecyclingResult:
    """Result of bile acid recycling PK simulation."""

    drug_name: str
    dose_mg: float
    f_ehc: float
    k_bile_per_h: float
    times_h: list[float]
    c_systemic_mg_L: list[float]
    bile_pool_mg: list[float]
    auc_systemic_mg_h_per_L: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    secondary_peak_present: bool
    ehc_contribution_pct: float
    notes: str


def _validate_inputs(
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vd_sys_L: float,
    ka_per_h: float,
    f_ehc: float,
    k_bile_per_h: float,
    t_end_h: float,
) -> None:
    """Validate simulation input parameters."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError(f"cl_hepatic_L_per_h must be > 0, got {cl_hepatic_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    if not (0.0 <= f_ehc <= 1.0):
        raise ValueError(f"f_ehc must be in [0, 1], got {f_ehc}")
    if k_bile_per_h <= 0:
        raise ValueError(f"k_bile_per_h must be > 0, got {k_bile_per_h}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")


def _detect_secondary_peak(times: list[float], concentrations: list[float], tmax_h: float) -> bool:
    """Detect secondary peak after tmax using local maxima detection."""
    # Find index of tmax
    tmax_idx = 0
    for i, t in enumerate(times):
        if t >= tmax_h:
            tmax_idx = i
            break

    # Look for local maximum after tmax
    start_idx = tmax_idx + 2
    if start_idx >= len(concentrations) - 1:
        return False

    for i in range(start_idx, len(concentrations) - 1):
        if concentrations[i] > concentrations[i - 1] and concentrations[i] > concentrations[i + 1]:
            # Must be meaningfully above surrounding values
            if concentrations[i] > 1e-6:
                return True
    return False


def _trapezoidal_auc(times: list[float], values: list[float]) -> float:
    """Compute AUC via manual trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_bile_acid_recycling(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vd_sys_L: float,
    ka_per_h: float,
    f_ehc: float,
    k_bile_per_h: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> BileAcidRecyclingResult:
    """Simulate enterohepatic circulation PK.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    dose_mg : float
        Oral dose in mg.
    cl_hepatic_L_per_h : float
        Hepatic clearance in L/h.
    vd_sys_L : float
        Volume of distribution (systemic) in L.
    ka_per_h : float
        Absorption rate constant from gut to systemic in 1/h.
    f_ehc : float
        Fraction of bile pool reabsorbed in ileum (0-1).
    k_bile_per_h : float
        Rate of hepatic secretion into bile in 1/h.
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Integration time step in hours.

    Returns
    -------
    BileAcidRecyclingResult
        PK time course and summary statistics.
    """
    _validate_inputs(dose_mg, cl_hepatic_L_per_h, vd_sys_L, ka_per_h, f_ehc, k_bile_per_h, t_end_h)

    # First-pass fraction: default 0.6 (40% bioavailability before EHC)
    first_pass_fraction = 0.6
    a_gut_0 = dose_mg * (1.0 - first_pass_fraction)

    # Elimination rate constant
    k_elim = cl_hepatic_L_per_h / vd_sys_L  # 1/h

    # Bile reabsorption rate constant
    k_absorb = 0.5  # 1/h (fixed)

    # Initial conditions
    a_gut = a_gut_0  # mg in gut
    a_sys = 0.0  # mg in systemic
    bile_pool = 0.0  # mg in bile pool

    times: list[float] = []
    c_sys_list: list[float] = []
    bile_list: list[float] = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        c_sys = a_sys / vd_sys_L
        times.append(t)
        c_sys_list.append(c_sys)
        bile_list.append(bile_pool)

        if t >= t_end_h:
            break

        # Rates
        gut_to_sys = ka_per_h * a_gut
        elim_from_sys = k_elim * a_sys
        bile_secretion = k_bile_per_h * a_sys
        bile_reabsorption = k_absorb * bile_pool
        reabsorbed_to_sys = f_ehc * bile_reabsorption

        # Forward Euler update
        d_a_gut = -gut_to_sys
        d_a_sys = gut_to_sys - elim_from_sys - bile_secretion + reabsorbed_to_sys
        d_bile = bile_secretion - bile_reabsorption

        a_gut = max(0.0, a_gut + d_a_gut * dt_h)
        a_sys = max(0.0, a_sys + d_a_sys * dt_h)
        bile_pool = max(0.0, bile_pool + d_bile * dt_h)

        t = min(t + dt_h, t_end_h)

    # Summary statistics
    auc = _trapezoidal_auc(times, c_sys_list)
    cmax = max(c_sys_list)
    cmax_idx = c_sys_list.index(cmax)
    tmax = times[cmax_idx]

    secondary_peak = _detect_secondary_peak(times, c_sys_list, tmax)

    # EHC contribution estimate: compare with f_ehc=0 scenario
    # Analytical approximation: without EHC, first-pass AUC dominates
    # We estimate EHC contribution as a fraction of total AUC
    if auc > 0 and f_ehc > 0:
        # Simple estimate: EHC contribution scales with f_ehc and k_bile relative to k_elim
        ehc_factor = f_ehc * k_bile_per_h / (k_elim + k_bile_per_h)
        ehc_contribution_pct = min(100.0, ehc_factor * 100.0)
    else:
        ehc_contribution_pct = 0.0

    # Build notes
    notes_parts = []
    if f_ehc == 0.0:
        notes_parts.append("No enterohepatic recirculation (f_ehc=0)")
    elif f_ehc >= 0.8:
        notes_parts.append("High enterohepatic recirculation fraction")
    if secondary_peak:
        notes_parts.append("Secondary absorption peak detected from EHC")
    if k_bile_per_h > k_elim:
        notes_parts.append("Bile secretion faster than elimination")
    if not notes_parts:
        notes_parts.append("Standard EHC simulation")
    notes = "; ".join(notes_parts)

    return BileAcidRecyclingResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_ehc=f_ehc,
        k_bile_per_h=k_bile_per_h,
        times_h=times,
        c_systemic_mg_L=c_sys_list,
        bile_pool_mg=bile_list,
        auc_systemic_mg_h_per_L=auc,
        cmax_systemic_mg_L=cmax,
        tmax_systemic_h=tmax,
        secondary_peak_present=secondary_peak,
        ehc_contribution_pct=ehc_contribution_pct,
        notes=notes,
    )


def compare_ehc_fractions(
    drug_name: str,
    dose_mg: float,
    f_ehc_list: list[float],
    cl_hepatic_L_per_h: float,
    vd_sys_L: float,
    ka_per_h: float = 1.0,
    k_bile_per_h: float = 0.3,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list[BileAcidRecyclingResult]:
    """Compare simulations across different EHC fractions.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    dose_mg : float
        Oral dose in mg.
    f_ehc_list : list of float
        List of EHC fractions to compare.
    cl_hepatic_L_per_h : float
        Hepatic clearance in L/h.
    vd_sys_L : float
        Volume of distribution in L.
    ka_per_h : float
        Absorption rate constant in 1/h.
    k_bile_per_h : float
        Bile secretion rate in 1/h.
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Time step in hours.

    Returns
    -------
    list of BileAcidRecyclingResult
        Results sorted by auc_systemic_mg_h_per_L descending.
    """
    results = []
    for f_ehc in f_ehc_list:
        result = simulate_bile_acid_recycling(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_hepatic_L_per_h=cl_hepatic_L_per_h,
            vd_sys_L=vd_sys_L,
            ka_per_h=ka_per_h,
            f_ehc=f_ehc,
            k_bile_per_h=k_bile_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_systemic_mg_h_per_L, reverse=True)
    return results
