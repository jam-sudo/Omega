"""
Phase 937 — Critical Care PK
PK model for critically ill ICU patients with altered physiology from
sepsis, organ failure, and mechanical ventilation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["CriticalCarePKResult", "simulate_critical_care_pk", "compare_icu_states"]

# ICU state multipliers: (CL_multiplier, Vd_multiplier)
_ICU_MULTIPLIERS = {
    "normal_icu": (1.0, 1.0),
    "sepsis_early": (1.8, 2.0),
    "sepsis_late_aki": (0.3, 2.5),
    "ards": (0.7, 2.2),
}

_VALID_ROUTES = {"iv_bolus", "iv_infusion"}

_ICU_NOTES = {
    "normal_icu": "Reference ICU patient; no significant physiological alterations.",
    "sepsis_early": "Early sepsis: augmented renal clearance (ARC), increased Vd due to fluid resuscitation.",
    "sepsis_late_aki": "Late sepsis with AKI: severely reduced renal clearance, markedly expanded Vd.",
    "ards": "ARDS: moderate reduction in clearance, expanded Vd from capillary leak and mechanical ventilation.",
}


@dataclass
class CriticalCarePKResult:
    drug_name: str
    icu_state: str
    dose_mg: float
    route: str
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    cl_effective_L_per_h: float
    vd_effective_L: float
    cl_fold_change: float
    vd_fold_change: float
    t_half_h: float
    notes: str


def _validate_inputs(
    dose_mg: float,
    cl_ref_L_per_h: float,
    vd_ref_L: float,
    icu_state: str,
    route: str,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_ref_L_per_h <= 0:
        raise ValueError("cl_ref_L_per_h must be > 0")
    if vd_ref_L <= 0:
        raise ValueError("vd_ref_L must be > 0")
    if icu_state not in _ICU_MULTIPLIERS:
        raise ValueError(
            f"icu_state must be one of {set(_ICU_MULTIPLIERS.keys())}, got '{icu_state}'"
        )
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got '{route}'")


def simulate_critical_care_pk(
    drug_name: str,
    dose_mg: float,
    icu_state: str = "normal_icu",
    cl_ref_L_per_h: float = 10.0,
    vd_ref_L: float = 50.0,
    route: str = "iv_bolus",
    infusion_rate_mg_h: float = 0.0,
    infusion_duration_h: float = 24.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> CriticalCarePKResult:
    """Simulate PK in a critically ill ICU patient using 1-cpt forward Euler."""
    _validate_inputs(dose_mg, cl_ref_L_per_h, vd_ref_L, icu_state, route)

    cl_mult, vd_mult = _ICU_MULTIPLIERS[icu_state]
    cl_eff = cl_ref_L_per_h * cl_mult
    vd_eff = vd_ref_L * vd_mult
    ke = cl_eff / vd_eff
    t_half = math.log(2.0) / ke

    # Build time array
    n_steps = int(round(t_end_h / dt_h)) + 1
    times = [i * dt_h for i in range(n_steps)]

    # Determine initial concentration and infusion rate
    if route == "iv_bolus":
        c = dose_mg / vd_eff
        actual_rate = 0.0
    else:
        # iv_infusion: start at 0
        c = 0.0
        if infusion_rate_mg_h > 0.0:
            actual_rate = infusion_rate_mg_h
        else:
            actual_rate = dose_mg / infusion_duration_h if infusion_duration_h > 0 else 0.0

    concentrations = []
    for _i, t in enumerate(times):
        concentrations.append(c)
        # Forward Euler step
        if route == "iv_bolus":
            dc_dt = -ke * c
        else:
            if t < infusion_duration_h:
                dc_dt = actual_rate / vd_eff - ke * c
            else:
                dc_dt = -ke * c
        c = max(0.0, c + dc_dt * dt_h)

    # PK metrics
    cmax = max(concentrations)
    tmax = times[concentrations.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(len(times) - 1):
        auc += 0.5 * (concentrations[i] + concentrations[i + 1]) * dt_h

    notes = _ICU_NOTES[icu_state]

    return CriticalCarePKResult(
        drug_name=drug_name,
        icu_state=icu_state,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=concentrations,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        cl_effective_L_per_h=cl_eff,
        vd_effective_L=vd_eff,
        cl_fold_change=cl_mult,
        vd_fold_change=vd_mult,
        t_half_h=t_half,
        notes=notes,
    )


def compare_icu_states(
    drug_name: str,
    dose_mg: float,
    cl_ref_L_per_h: float = 10.0,
    vd_ref_L: float = 50.0,
    route: str = "iv_bolus",
    infusion_rate_mg_h: float = 0.0,
    infusion_duration_h: float = 24.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> list[CriticalCarePKResult]:
    """Compare PK across all 4 ICU states, sorted by AUC descending."""
    results = []
    for state in _ICU_MULTIPLIERS:
        res = simulate_critical_care_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            icu_state=state,
            cl_ref_L_per_h=cl_ref_L_per_h,
            vd_ref_L=vd_ref_L,
            route=route,
            infusion_rate_mg_h=infusion_rate_mg_h,
            infusion_duration_h=infusion_duration_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(res)
    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
