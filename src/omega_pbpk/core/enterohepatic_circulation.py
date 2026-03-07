"""Enterohepatic circulation (EHC) PBPK model — Phase 380.

4-compartment forward-Euler simulation of bile secretion, intestinal
reabsorption, and the resulting secondary plasma concentration peaks.
Pure Python stdlib only (math, dataclasses).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EnterohepaticResult:
    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_bile_mg_L: list
    c_gut_lumen_mg_L: list
    secondary_peaks: int
    auc_total_mg_h_per_L: float
    auc_without_ehc_mg_h_per_L: float
    ehc_auc_increase_pct: float
    t_half_apparent_h: float
    cmax_mg_L: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def _count_secondary_peaks(times: list, conc: list, skip_before_h: float = 2.0) -> int:
    """Count local maxima in conc after skip_before_h."""
    count = 0
    for i in range(1, len(conc) - 1):
        if times[i] <= skip_before_h:
            continue
        if conc[i] > conc[i - 1] and conc[i] > conc[i + 1]:
            count += 1
    return count


def _apparent_half_life(times: list, conc: list, cmax: float, t_cmax: float) -> float:
    """Estimate apparent t1/2 from the post-Cmax region.

    Finds the first time after Cmax where C <= cmax/2 using linear interpolation.
    Falls back to a terminal-slope estimate if not found.
    """
    if cmax <= 0.0:
        return float("nan")

    half_cmax = cmax / 2.0

    # Collect indices after Cmax
    post_idx = [i for i, t in enumerate(times) if t >= t_cmax]

    if len(post_idx) < 2:
        return float("nan")

    # Find first crossing below half_cmax
    for k in range(1, len(post_idx)):
        i_prev = post_idx[k - 1]
        i_curr = post_idx[k]
        c_prev = conc[i_prev]
        c_curr = conc[i_curr]
        if c_prev >= half_cmax >= c_curr:
            if abs(c_prev - c_curr) < 1e-20:
                t_half_reach = times[i_prev]
            else:
                frac = (c_prev - half_cmax) / (c_prev - c_curr)
                t_half_reach = times[i_prev] + frac * (times[i_curr] - times[i_prev])
            return t_half_reach - t_cmax

    # Fallback: terminal slope from last two points
    i_last = post_idx[-1]
    i_prev2 = post_idx[-2]
    c_last = conc[i_last]
    c_prev2 = conc[i_prev2]
    t_last = times[i_last]
    t_prev2 = times[i_prev2]
    dt = t_last - t_prev2
    if c_last > 0.0 and c_prev2 > c_last and dt > 0.0:
        lam_z = (math.log(c_prev2) - math.log(c_last)) / dt
        if lam_z > 0.0:
            return math.log(2.0) / lam_z
    return float("nan")


def _run_simulation(
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    ka_absorption_per_h: float,
    f_bile_secretion: float,
    k_bile_to_gut_per_h: float,
    f_gut_reabsorption: float,
    t_end_h: float,
    dt_h: float,
) -> tuple:
    """Run 4-compartment Forward Euler and return (times, c_plasma, c_bile, c_lumen)."""
    n_steps = int(math.ceil(t_end_h / dt_h))

    times: list = []
    c_plasma_list: list = []
    c_bile_list: list = []
    c_lumen_list: list = []

    bile_vol_L = vd_L * 0.05  # proxy for bile volume (~5% of Vd)
    lumen_vol_L = vd_L * 0.1  # proxy for gut lumen volume (~10% of Vd)

    # State (amounts in mg; plasma as concentration mg/L)
    m_gut = dose_mg
    c_plasma = 0.0  # mg/L
    m_bile = 0.0  # mg
    m_lumen = 0.0  # mg

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_bile_list.append(m_bile / bile_vol_L if bile_vol_L > 0.0 else 0.0)
        c_lumen_list.append(m_lumen / lumen_vol_L if lumen_vol_L > 0.0 else 0.0)

        if i == n_steps:
            break

        # ODEs (Forward Euler)
        # dM_gut/dt = -ka * M_gut
        d_m_gut = -ka_absorption_per_h * m_gut

        # dC_plasma/dt = ka*M_gut/Vd + f_reabs*k_b2g*M_lumen/Vd - CL_hep*C/Vd
        d_c_plasma = (
            ka_absorption_per_h * m_gut / vd_L
            + f_gut_reabsorption * k_bile_to_gut_per_h * m_lumen / vd_L
            - cl_hepatic_L_per_h * c_plasma / vd_L
        )

        # dM_bile/dt = f_bile * CL_hep * C - k_b2g * M_bile
        d_m_bile = f_bile_secretion * cl_hepatic_L_per_h * c_plasma - k_bile_to_gut_per_h * m_bile

        # dM_lumen/dt = k_b2g * M_bile - k_b2g * M_lumen
        d_m_lumen = k_bile_to_gut_per_h * m_bile - k_bile_to_gut_per_h * m_lumen

        # Forward Euler update (clamp to 0)
        m_gut = max(0.0, m_gut + d_m_gut * dt_h)
        c_plasma = max(0.0, c_plasma + d_c_plasma * dt_h)
        m_bile = max(0.0, m_bile + d_m_bile * dt_h)
        m_lumen = max(0.0, m_lumen + d_m_lumen * dt_h)

    return times, c_plasma_list, c_bile_list, c_lumen_list


def simulate_enterohepatic_circulation(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    ka_absorption_per_h: float = 1.0,
    f_bile_secretion: float = 0.3,
    k_bile_to_gut_per_h: float = 0.5,
    f_gut_reabsorption: float = 0.6,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> EnterohepaticResult:
    """Simulate enterohepatic circulation with a 4-compartment Forward Euler model.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Oral dose in mg.
    cl_hepatic_L_per_h:
        Total hepatic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    ka_absorption_per_h:
        First-order GI absorption rate constant (1/h).
    f_bile_secretion:
        Fraction of hepatic clearance directed into bile (0-1).
    k_bile_to_gut_per_h:
        Rate constant for bile release into gut lumen (1/h).
    f_gut_reabsorption:
        Fraction of lumen drug that is reabsorbed (0-1).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    EnterohepaticResult
    """
    if dose_mg <= 0.0:
        raise ValueError("dose_mg must be > 0")
    if cl_hepatic_L_per_h <= 0.0:
        raise ValueError("cl_hepatic_L_per_h must be > 0")
    if vd_L <= 0.0:
        raise ValueError("vd_L must be > 0")
    if ka_absorption_per_h <= 0.0:
        raise ValueError("ka_absorption_per_h must be > 0")
    if not (0.0 <= f_bile_secretion <= 1.0):
        raise ValueError("f_bile_secretion must be in [0, 1]")
    if k_bile_to_gut_per_h <= 0.0:
        raise ValueError("k_bile_to_gut_per_h must be > 0")
    if not (0.0 <= f_gut_reabsorption <= 1.0):
        raise ValueError("f_gut_reabsorption must be in [0, 1]")

    # Run with EHC
    times, c_plasma, c_bile, c_lumen = _run_simulation(
        dose_mg=dose_mg,
        cl_hepatic_L_per_h=cl_hepatic_L_per_h,
        vd_L=vd_L,
        ka_absorption_per_h=ka_absorption_per_h,
        f_bile_secretion=f_bile_secretion,
        k_bile_to_gut_per_h=k_bile_to_gut_per_h,
        f_gut_reabsorption=f_gut_reabsorption,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Run without EHC for reference AUC
    _, c_plasma_no_ehc, _, _ = _run_simulation(
        dose_mg=dose_mg,
        cl_hepatic_L_per_h=cl_hepatic_L_per_h,
        vd_L=vd_L,
        ka_absorption_per_h=ka_absorption_per_h,
        f_bile_secretion=0.0,
        k_bile_to_gut_per_h=k_bile_to_gut_per_h,
        f_gut_reabsorption=f_gut_reabsorption,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    auc_total = _trapezoidal_auc(times, c_plasma)
    auc_no_ehc = _trapezoidal_auc(times, c_plasma_no_ehc)

    if auc_no_ehc > 0.0:
        ehc_increase_pct = (auc_total - auc_no_ehc) / auc_no_ehc * 100.0
    else:
        ehc_increase_pct = 0.0

    # Cmax and tmax
    cmax = max(c_plasma) if c_plasma else 0.0
    cmax_idx = c_plasma.index(cmax) if cmax > 0.0 else 0
    t_cmax = times[cmax_idx]

    # Apparent half-life
    t_half = _apparent_half_life(times, c_plasma, cmax, t_cmax)

    # Count secondary plasma peaks (skip initial absorption)
    n_secondary = _count_secondary_peaks(times, c_plasma, skip_before_h=2.0)

    notes = (
        f"EHC simulation: f_bile={f_bile_secretion:.2f}, "
        f"f_reabs={f_gut_reabsorption:.2f}, "
        f"k_bile_to_gut={k_bile_to_gut_per_h:.2f}/h. "
        f"AUC increase due to EHC: {ehc_increase_pct:.1f}%."
    )

    return EnterohepaticResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_bile_mg_L=c_bile,
        c_gut_lumen_mg_L=c_lumen,
        secondary_peaks=n_secondary,
        auc_total_mg_h_per_L=auc_total,
        auc_without_ehc_mg_h_per_L=auc_no_ehc,
        ehc_auc_increase_pct=ehc_increase_pct,
        t_half_apparent_h=t_half,
        cmax_mg_L=cmax,
        notes=notes,
    )


def compare_ehc_fractions(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    f_bile_values: list | None = None,
) -> list:
    """Compare EHC simulations across a range of bile secretion fractions.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Oral dose in mg.
    cl_hepatic_L_per_h:
        Total hepatic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    f_bile_values:
        List of f_bile_secretion values. Defaults to [0.0, 0.1, 0.3, 0.5, 0.7].

    Returns
    -------
    list of EnterohepaticResult sorted by auc_total_mg_h_per_L descending.
    """
    if f_bile_values is None:
        f_bile_values = [0.0, 0.1, 0.3, 0.5, 0.7]

    results = []
    for fb in f_bile_values:
        result = simulate_enterohepatic_circulation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_hepatic_L_per_h=cl_hepatic_L_per_h,
            vd_L=vd_L,
            f_bile_secretion=fb,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_total_mg_h_per_L, reverse=True)
    return results


__all__ = [
    "EnterohepaticResult",
    "simulate_enterohepatic_circulation",
    "compare_ehc_fractions",
]
