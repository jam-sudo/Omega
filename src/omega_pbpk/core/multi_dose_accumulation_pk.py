"""
Phase 947 — Multi-dose accumulation and washout kinetics.

Predicts steady state, accumulation ratio, and time to steady state
using 1-compartment analytical superposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["MultiDoseAccumulationResult", "simulate_accumulation", "find_steady_state_dose"]


@dataclass
class MultiDoseAccumulationResult:
    drug_name: str
    dose_mg: float
    dosing_interval_h: float
    n_doses: int
    times_h: list
    c_plasma_mg_L: list
    css_max_theoretical: float
    css_min_theoretical: float
    accumulation_ratio: float
    time_to_90ss_h: float
    cmax_last_dose: float
    cmin_last_dose: float
    fluctuation_pct: float
    t_half_h: float
    notes: str


def _validate_inputs(
    dose_mg: float,
    dosing_interval_h: float,
    n_doses: int,
    cl_L_per_h: float,
    vd_L: float,
    bioavailability: float,
    route: str,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 < bioavailability <= 1):
        raise ValueError("bioavailability must be in (0, 1]")
    if route not in {"iv", "oral"}:
        raise ValueError("route must be 'iv' or 'oral'")


def _single_dose_iv(t: float, c0: float, ke: float) -> float:
    """IV bolus single-dose concentration."""
    if t < 0:
        return 0.0
    return c0 * math.exp(-ke * t)


def _single_dose_oral(
    t: float, dose_mg: float, vd_L: float, ke: float, ka: float, f: float
) -> float:
    """Oral single-dose concentration via 1-cpt model."""
    if t <= 0:
        return 0.0
    if abs(ka - ke) < 1e-9:
        # Avoid division by zero: use limit form
        return ka * f * dose_mg / vd_L * t * math.exp(-ke * t)
    return (ka * f * dose_mg) / (vd_L * (ka - ke)) * (math.exp(-ke * t) - math.exp(-ka * t))


def simulate_accumulation(
    drug_name: str,
    dose_mg: float,
    dosing_interval_h: float = 12.0,
    n_doses: int = 10,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    bioavailability: float = 1.0,
    simulate_washout: bool = True,
    washout_duration_h: float = 24.0,
    dt_h: float = 0.1,
) -> MultiDoseAccumulationResult:
    """Simulate multi-dose accumulation kinetics via superposition."""
    _validate_inputs(dose_mg, dosing_interval_h, n_doses, cl_L_per_h, vd_L, bioavailability, route)

    ke = cl_L_per_h / vd_L
    t_half = math.log(2.0) / ke

    # Analytical steady-state metrics (infinite doses, IV-equivalent)
    _c0 = dose_mg / vd_L  # IV peak; oral uses F but for accumulation ratio we use IV basis
    # For oral: effective c0 for SS calculation uses bioavailable dose
    if route == "oral":
        c0_eff = bioavailability * dose_mg / vd_L
    else:
        c0_eff = dose_mg / vd_L

    exp_ke_tau = math.exp(-ke * dosing_interval_h)
    # Accumulation ratio (steady-state Cmax / single-dose Cmax)
    accumulation_ratio = 1.0 / (1.0 - exp_ke_tau)
    css_max_theoretical = c0_eff * accumulation_ratio
    css_min_theoretical = css_max_theoretical * exp_ke_tau

    # Time to 90% steady state: t_90ss = -ln(1 - 0.9) / ke = ln(10) / ke
    time_to_90ss_h = math.log(10.0) / ke

    # Build time array: n_doses intervals + optional washout
    total_duration = n_doses * dosing_interval_h
    if simulate_washout:
        total_duration += washout_duration_h

    n_steps = max(1, int(round(total_duration / dt_h)))
    times = [i * dt_h for i in range(n_steps + 1)]

    # Dose times
    dose_times = [i * dosing_interval_h for i in range(n_doses)]

    # Build concentration profile via superposition
    c_profile = []
    for t in times:
        c = 0.0
        for dose_time in dose_times:
            dt_since = t - dose_time
            if dt_since < 0:
                continue
            if route == "iv":
                c += _single_dose_iv(dt_since, c0_eff, ke)
            else:
                c += _single_dose_oral(dt_since, dose_mg, vd_L, ke, ka_per_h, bioavailability)
        c_profile.append(max(0.0, c))

    # Cmax at last dose interval
    last_dose_time = dose_times[-1]
    last_dose_end = last_dose_time + dosing_interval_h
    last_interval_times = [t for t in times if last_dose_time <= t <= last_dose_end]
    if last_interval_times:
        idxs = [i for i, t in enumerate(times) if last_dose_time <= t <= last_dose_end]
        last_interval_c = [c_profile[i] for i in idxs]
        cmax_last_dose = max(last_interval_c)
        cmin_last_dose = (
            c_profile[idxs[0]]
            if route == "iv"
            else min(last_interval_c[: max(1, len(last_interval_c) // 4)])
        )
        # For IV: trough is just before the last dose (concentration at last_dose_time)
        if route == "iv":
            cmin_last_dose = c_profile[idxs[0]]
    else:
        cmax_last_dose = c_profile[-1]
        cmin_last_dose = c_profile[-1]

    # Fluctuation: 100 * (Cmax - Cmin) / Css_avg
    css_avg = (css_max_theoretical + css_min_theoretical) / 2.0
    if css_avg > 0:
        fluctuation_pct = 100.0 * (css_max_theoretical - css_min_theoretical) / css_avg
    else:
        fluctuation_pct = 0.0

    _cmax_plasma_mg_L = max(c_profile) if c_profile else 0.0

    notes = (
        f"Route={route}, ke={ke:.4f}/h, t½={t_half:.2f}h, "
        f"R_acc={accumulation_ratio:.2f}, t90ss={time_to_90ss_h:.1f}h"
    )

    return MultiDoseAccumulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dosing_interval_h=dosing_interval_h,
        n_doses=n_doses,
        times_h=times,
        c_plasma_mg_L=c_profile,
        css_max_theoretical=css_max_theoretical,
        css_min_theoretical=css_min_theoretical,
        accumulation_ratio=accumulation_ratio,
        time_to_90ss_h=time_to_90ss_h,
        cmax_last_dose=cmax_last_dose,
        cmin_last_dose=cmin_last_dose,
        fluctuation_pct=fluctuation_pct,
        t_half_h=t_half,
        notes=notes,
    )


def find_steady_state_dose(
    drug_name: str,
    target_css_min_mg_L: float,
    target_css_max_mg_L: float,
    cl_L_per_h: float,
    vd_L: float,
    dosing_interval_h: float = 12.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    bioavailability: float = 1.0,
) -> dict:
    """Find the optimal dose to achieve target Css_min and Css_max."""
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")

    ke = cl_L_per_h / vd_L
    exp_ke_tau = math.exp(-ke * dosing_interval_h)
    accumulation_ratio = 1.0 / (1.0 - exp_ke_tau)

    # Css_max = (F * dose / Vd) * R_acc  → dose = Css_max * Vd / (F * R_acc)
    f = bioavailability if route == "oral" else 1.0
    # Target: average of desired min/max as the "Css_max" target
    css_max_target = target_css_max_mg_L
    optimal_dose_mg = css_max_target * vd_L / (f * accumulation_ratio)
    optimal_dose_mg = max(optimal_dose_mg, 0.001)

    # Predict actual Css at this dose
    c0_eff = f * optimal_dose_mg / vd_L
    predicted_css_max = c0_eff * accumulation_ratio
    predicted_css_min = predicted_css_max * exp_ke_tau

    within_window = (
        predicted_css_min >= target_css_min_mg_L and predicted_css_max <= target_css_max_mg_L
    )

    return {
        "optimal_dose_mg": optimal_dose_mg,
        "predicted_css_min": predicted_css_min,
        "predicted_css_max": predicted_css_max,
        "within_therapeutic_window": within_window,
    }
