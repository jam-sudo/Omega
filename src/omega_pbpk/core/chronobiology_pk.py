"""
Phase 746 — Chronobiology Effect on PK

Models circadian rhythm effects on drug pharmacokinetics. Many metabolic enzymes
and transporters oscillate with the biological clock:
  - CYP3A4: peak at 15:00 (amplitude 30%)
  - CYP2D6: peak at 12:00 (amplitude 15%)
  - GFR:    peak at 14:00 (amplitude 25%)
  - OATP1B1: peak at 09:00 (amplitude 20%)

This module is distinct from chronopk.py (Phase 57). It focuses on the composite
CL rhythm driven by fractional contributions of CYP3A4 and renal elimination,
and provides optimal dosing window recommendations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ChronobiologyPKResult:
    """Result of chronobiology-modulated PK simulation."""

    drug_name: str
    dose_mg: float
    dosing_clock_h: float
    times_h: list  # wall-clock hours elapsed from dose time
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_avg_h: float  # approximation: Vd / CL_base × ln2
    cl_at_tmax: float  # CL at the time of peak concentration
    optimal_dosing_window: str
    notes: str


def _circadian_factor(clock_h: float, amplitude: float, peak_h: float) -> float:
    """
    Compute circadian modulation factor at a given clock hour.

    factor = 1 + amplitude × cos(2π × (clock_h - peak_h) / 24)
    """
    return 1.0 + amplitude * math.cos(2.0 * math.pi * (clock_h - peak_h) / 24.0)


def _effective_cl(
    clock_h: float,
    cl_base: float,
    fm_cyp3a4: float,
    fm_renal: float,
    amplitude_cyp3a4: float,
    amplitude_gfr: float,
) -> float:
    """
    Compute effective clearance at a given clock hour.

    CL(t) = CL_base × (fm_cyp3a4 × f_cyp3a4(t) + fm_renal × f_gfr(t) + f_other)
    where f_other fraction = 1 - fm_cyp3a4 - fm_renal (no circadian variation)
    """
    f_cyp3a4 = _circadian_factor(clock_h, amplitude_cyp3a4, peak_h=15.0)
    f_gfr = _circadian_factor(clock_h, amplitude_gfr, peak_h=14.0)
    fm_other = 1.0 - fm_cyp3a4 - fm_renal
    composite = fm_cyp3a4 * f_cyp3a4 + fm_renal * f_gfr + fm_other
    return cl_base * composite


def _validate_params(
    dose_mg: float,
    dosing_clock_h: float,
    cl_base_L_per_h: float,
    vd_L: float,
    fm_cyp3a4: float,
    fm_renal: float,
) -> None:
    """Validate simulation parameters."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0.0 <= dosing_clock_h < 24.0):
        raise ValueError(f"dosing_clock_h must be in [0, 24), got {dosing_clock_h}")
    if cl_base_L_per_h <= 0:
        raise ValueError(f"cl_base_L_per_h must be > 0, got {cl_base_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0.0 <= fm_cyp3a4 <= 1.0):
        raise ValueError(f"fm_cyp3a4 must be in [0, 1], got {fm_cyp3a4}")
    if not (0.0 <= fm_renal <= 1.0):
        raise ValueError(f"fm_renal must be in [0, 1], got {fm_renal}")
    if fm_cyp3a4 + fm_renal > 1.0 + 1e-9:
        raise ValueError(f"fm_cyp3a4 + fm_renal must be <= 1, got {fm_cyp3a4 + fm_renal:.4f}")


def simulate_chronobiology_pk(
    drug_name: str,
    dose_mg: float,
    dosing_clock_h: float = 8.0,
    cl_base_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ka_per_h: float = 1.0,
    fm_cyp3a4: float = 0.5,
    fm_renal: float = 0.3,
    amplitude_cyp3a4: float = 0.3,
    amplitude_gfr: float = 0.25,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> ChronobiologyPKResult:
    """
    Simulate chronobiology-modulated 1-compartment oral PK.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    dosing_clock_h : float
        Hour of day at which dose is administered (0–23.99).
    cl_base_L_per_h : float
        Baseline clearance in L/h.
    vd_L : float
        Volume of distribution in L.
    ka_per_h : float
        First-order absorption rate constant (/h).
    fm_cyp3a4 : float
        Fraction of clearance mediated by CYP3A4.
    fm_renal : float
        Fraction of clearance via renal (GFR) elimination.
    amplitude_cyp3a4 : float
        Circadian amplitude for CYP3A4 (default 0.3 = 30%).
    amplitude_gfr : float
        Circadian amplitude for GFR (default 0.25 = 25%).
    t_end_h : float
        Simulation duration in hours (from dose time).
    dt_h : float
        Forward Euler step size in hours.

    Returns
    -------
    ChronobiologyPKResult
    """
    _validate_params(dose_mg, dosing_clock_h, cl_base_L_per_h, vd_L, fm_cyp3a4, fm_renal)

    # Initial conditions
    a_gut = dose_mg
    c_pl = 0.0

    n_steps = int(math.ceil(t_end_h / dt_h))
    times_h: list = []
    c_plasma_list: list = []

    t_elapsed = 0.0
    for _ in range(n_steps + 1):
        times_h.append(t_elapsed)
        c_plasma_list.append(c_pl)

        # Current clock time (wraps around 24h)
        clock_h = (dosing_clock_h + t_elapsed) % 24.0

        cl_t = _effective_cl(
            clock_h, cl_base_L_per_h, fm_cyp3a4, fm_renal, amplitude_cyp3a4, amplitude_gfr
        )
        ke_t = cl_t / vd_L

        d_a_gut = -ka_per_h * a_gut
        d_c_pl = ka_per_h * a_gut / vd_L - ke_t * c_pl

        a_gut = max(0.0, a_gut + d_a_gut * dt_h)
        c_pl = max(0.0, c_pl + d_c_pl * dt_h)

        t_elapsed += dt_h

    # Summary metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_idx = c_plasma_list.index(cmax_mg_L)
    tmax_h_val = times_h[tmax_idx]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * (times_h[i] - times_h[i - 1])

    # Average half-life approximation (use baseline CL)
    t_half_avg_h = math.log(2.0) * vd_L / cl_base_L_per_h

    # CL at time of peak concentration
    clock_at_tmax = (dosing_clock_h + tmax_h_val) % 24.0
    cl_at_tmax = _effective_cl(
        clock_at_tmax, cl_base_L_per_h, fm_cyp3a4, fm_renal, amplitude_cyp3a4, amplitude_gfr
    )

    # Optimal dosing window recommendation
    if fm_cyp3a4 > 0.5:
        optimal_dosing_window = "evening (20:00-22:00) for higher exposure; morning for lower"
    else:
        optimal_dosing_window = "timing has minimal PK impact (<15% variation)"

    notes = (
        f"fm_CYP3A4={fm_cyp3a4:.2f}, fm_renal={fm_renal:.2f}; "
        f"dosing at {dosing_clock_h:.1f}:00; "
        f"CYP3A4 peaks at 15:00 (amplitude {amplitude_cyp3a4 * 100:.0f}%), "
        f"GFR peaks at 14:00 (amplitude {amplitude_gfr * 100:.0f}%)."
    )

    return ChronobiologyPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_clock_h=dosing_clock_h,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h_val,
        auc_mg_h_per_L=auc,
        t_half_avg_h=t_half_avg_h,
        cl_at_tmax=cl_at_tmax,
        optimal_dosing_window=optimal_dosing_window,
        notes=notes,
    )


def compare_dosing_times(
    drug_name: str,
    dose_mg: float,
    clock_hours: list,
    cl_base_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ka_per_h: float = 1.0,
    fm_cyp3a4: float = 0.5,
    fm_renal: float = 0.3,
    amplitude_cyp3a4: float = 0.3,
    amplitude_gfr: float = 0.25,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list:
    """
    Compare PK outcomes across multiple dosing times.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    clock_hours : list of float
        List of clock hours (0–23.99) at which to simulate dosing.
    ... (remaining parameters same as simulate_chronobiology_pk)

    Returns
    -------
    list of ChronobiologyPKResult, one per clock hour.
    """
    results = []
    for ch in clock_hours:
        result = simulate_chronobiology_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            dosing_clock_h=ch,
            cl_base_L_per_h=cl_base_L_per_h,
            vd_L=vd_L,
            ka_per_h=ka_per_h,
            fm_cyp3a4=fm_cyp3a4,
            fm_renal=fm_renal,
            amplitude_cyp3a4=amplitude_cyp3a4,
            amplitude_gfr=amplitude_gfr,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)
    return results
