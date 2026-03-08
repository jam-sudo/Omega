"""Phase 609 — Receptor Desensitization PD Model.

Simulates receptor desensitization (tolerance) during repeated drug exposure
using a turnover model with resensitization kinetics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ReceptorDesensitizationResult:
    """Result of receptor desensitization simulation."""

    drug_name: str
    dose_mg: float
    n_doses: int
    times_h: list
    c_drug_mg_L: list
    r_active: list
    effect: list
    tolerance_index: float
    peak_effect_dose1: float
    peak_effect_last_dose: float
    receptor_recovery_time_h: float
    notes: str


def simulate_receptor_desensitization(
    drug_name: str,
    dose_mg: float,
    n_doses: int,
    tau_h: float,
    vd_L: float,
    cl_L_per_h: float,
    emax: float,
    ec50_mg_L: float,
    k_desens_per_h: float,
    k_resens_per_h: float,
    t_end_h: float | None = None,
    dt_h: float = 0.1,
) -> ReceptorDesensitizationResult:
    """Simulate receptor desensitization during repeated dosing.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams per administration.
    n_doses:
        Number of doses administered.
    tau_h:
        Dosing interval in hours.
    vd_L:
        Volume of distribution (L).
    cl_L_per_h:
        Systemic clearance (L/h).
    emax:
        Maximum pharmacodynamic effect.
    ec50_mg_L:
        Concentration producing 50% of Emax (mg/L).
    k_desens_per_h:
        Desensitization rate constant (/h).
    k_resens_per_h:
        Resensitization rate constant (/h).
    t_end_h:
        Simulation end time (h). Defaults to n_doses * tau_h + 24.
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    ReceptorDesensitizationResult
    """
    # Validation
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")
    if tau_h <= 0:
        raise ValueError("tau_h must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if k_desens_per_h <= 0:
        raise ValueError("k_desens_per_h must be > 0")
    if k_resens_per_h <= 0:
        raise ValueError("k_resens_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")

    if t_end_h is None:
        t_end_h = n_doses * tau_h + 24.0

    # PK parameters
    f_oral = 0.9
    ka = 1.5  # /h

    ke = cl_L_per_h / vd_L  # elimination rate constant

    # Build list of dosing times
    dosing_times = [i * tau_h for i in range(n_doses)]
    last_dose_time = dosing_times[-1]

    # Initial conditions
    a_gut = 0.0
    c_drug = 0.0
    r_active = 1.0

    times_h = []
    c_drug_list = []
    r_active_list = []
    effect_list = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    # Track doses already administered
    dose_idx = 0

    for step in range(n_steps + 1):
        t = step * dt_h

        # Administer dose at dosing times
        for d_time in dosing_times:
            # Use a small tolerance to detect dosing time crossing
            if abs(t - d_time) < dt_h * 0.5 and t >= d_time:
                # Only add once — track which doses applied
                pass

        # Apply dose: add dose at start of each interval
        if dose_idx < n_doses and abs(t - dosing_times[dose_idx]) < dt_h * 0.5:
            a_gut += dose_mg * f_oral
            dose_idx += 1

        # Compute effect
        e = emax * r_active * c_drug / (ec50_mg_L + c_drug) if (ec50_mg_L + c_drug) > 0 else 0.0

        times_h.append(t)
        c_drug_list.append(max(0.0, c_drug))
        r_active_list.append(max(0.0, min(1.0, r_active)))
        effect_list.append(max(0.0, e))

        if step == n_steps:
            break

        # Forward Euler update
        da_gut = -ka * a_gut
        dc_drug = ka * a_gut / vd_L - ke * c_drug
        dr_active = (
            k_resens_per_h * (1.0 - r_active)
            - k_desens_per_h * c_drug / (ec50_mg_L + c_drug) * r_active
        )

        a_gut = max(0.0, a_gut + da_gut * dt_h)
        c_drug = max(0.0, c_drug + dc_drug * dt_h)
        r_active = max(0.0, min(1.0, r_active + dr_active * dt_h))

    # Find peak effect in first dosing interval
    dose1_end = tau_h if n_doses > 1 else t_end_h
    dose1_effects = [effect_list[i] for i, t in enumerate(times_h) if t <= dose1_end]
    peak_effect_dose1 = max(dose1_effects) if dose1_effects else 0.0

    # Find peak effect in last dosing interval
    last_dose_start = last_dose_time
    last_dose_end = last_dose_time + tau_h
    last_effects = [
        effect_list[i] for i, t in enumerate(times_h) if last_dose_start <= t <= last_dose_end
    ]
    peak_effect_last_dose = max(last_effects) if last_effects else 0.0

    # Tolerance index
    if peak_effect_dose1 > 0:
        tolerance_index = peak_effect_last_dose / peak_effect_dose1
    else:
        tolerance_index = 1.0

    # Receptor recovery time: after last dose time, first time r_active >= 0.9
    receptor_recovery_time_h = float("inf")
    for i, t in enumerate(times_h):
        if t > last_dose_time and r_active_list[i] >= 0.9:
            receptor_recovery_time_h = t - last_dose_time
            break

    notes = (
        f"Simulated {n_doses} doses of {dose_mg} mg {drug_name} every {tau_h} h. "
        f"Tolerance index: {tolerance_index:.3f}. "
        f"Recovery time: {receptor_recovery_time_h:.1f} h after last dose."
    )

    return ReceptorDesensitizationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        n_doses=n_doses,
        times_h=times_h,
        c_drug_mg_L=c_drug_list,
        r_active=r_active_list,
        effect=effect_list,
        tolerance_index=tolerance_index,
        peak_effect_dose1=peak_effect_dose1,
        peak_effect_last_dose=peak_effect_last_dose,
        receptor_recovery_time_h=receptor_recovery_time_h,
        notes=notes,
    )
