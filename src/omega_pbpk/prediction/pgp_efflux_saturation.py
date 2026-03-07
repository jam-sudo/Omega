"""Phase 870 — P-glycoprotein Efflux Saturation Model.

Models P-gp efflux saturation kinetics affecting intestinal absorption
using a simplified 2-state ODE system with forward Euler integration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PGPEffluxSaturationResult:
    """Result of P-gp efflux saturation simulation."""

    drug_name: str
    dose_mg: float
    vmax_pgp_mg_per_h: float
    km_pgp_mg_per_L: float
    times_h: list[float]
    c_blood_mg_L: list[float]
    a_dissolved_mg: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_absorbed: float
    pgp_saturation_at_cmax: float
    dose_proportional: bool
    notes: str


def _compute_auc_trapz(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1])
    return auc


def simulate_pgp_efflux_saturation(
    drug_name: str,
    dose_mg: float,
    ka_passive_per_h: float = 0.5,
    vmax_pgp_mg_per_h: float = 2.0,
    km_pgp_mg_per_L: float = 10.0,
    v_gi_L: float = 0.25,
    vd_L: float = 20.0,
    cl_L_per_h: float = 2.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
    dose_as_solution: bool = True,
) -> PGPEffluxSaturationResult:
    """Simulate P-gp efflux saturation kinetics during intestinal absorption.

    Uses a 2-state forward Euler ODE model:
    - A_dissolved: dissolved drug in GI (mg)
    - C_blood: systemic plasma concentration (mg/L)

    P-gp efflux follows Michaelis-Menten kinetics. Net absorption is passive
    permeation minus P-gp efflux.

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose in mg. Must be > 0.
        ka_passive_per_h: Passive absorption rate constant (per h). Must be > 0.
        vmax_pgp_mg_per_h: P-gp efflux Vmax (mg/h). Must be >= 0.
        km_pgp_mg_per_L: P-gp Km in lumen concentration (mg/L). Must be > 0.
        v_gi_L: GI fluid volume (L).
        vd_L: Volume of distribution (L).
        cl_L_per_h: Systemic clearance (L/h).
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).
        dose_as_solution: If True, all dose immediately dissolved; if False,
            use first-order dissolution with k_diss=0.5/h.

    Returns:
        PGPEffluxSaturationResult with concentration-time profiles and metrics.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_passive_per_h <= 0:
        raise ValueError("ka_passive_per_h must be > 0")
    if vmax_pgp_mg_per_h < 0:
        raise ValueError("vmax_pgp_mg_per_h must be >= 0")
    if km_pgp_mg_per_L <= 0:
        raise ValueError("km_pgp_mg_per_L must be > 0")

    ke = cl_L_per_h / vd_L

    # Initial conditions
    # A_dissolved: dissolved drug in GI (mg)
    # A_undissolved: undissolved drug in GI (mg) — only used when dose_as_solution=False
    if dose_as_solution:
        a_dissolved = dose_mg
        a_undissolved = 0.0
    else:
        a_dissolved = 0.0
        a_undissolved = dose_mg

    c_blood = 0.0
    k_diss = 0.5  # first-order dissolution rate (per h)

    times: list[float] = []
    c_blood_list: list[float] = []
    a_dissolved_list: list[float] = []

    total_absorbed = 0.0  # track cumulative absorbed drug

    t = 0.0
    while t <= t_end_h + 1e-9:
        times.append(round(t, 8))
        c_blood_list.append(c_blood)
        a_dissolved_list.append(a_dissolved)

        # Compute lumen concentration
        c_apical = a_dissolved / v_gi_L  # mg/L

        # P-gp efflux (Michaelis-Menten) in mg/h
        j_efflux_mg_per_h = vmax_pgp_mg_per_h * c_apical / (km_pgp_mg_per_L + c_apical)

        # Passive absorption flux in mg/h
        j_passive_mg_per_h = ka_passive_per_h * a_dissolved

        # Net absorption (cannot be negative — no back-flux from blood to gut)
        j_abs_mg_per_h = max(0.0, j_passive_mg_per_h - j_efflux_mg_per_h)

        # Dissolution flux (only if dose not immediately dissolved)
        if not dose_as_solution:
            j_diss_mg_per_h = k_diss * a_undissolved
        else:
            j_diss_mg_per_h = 0.0

        # ODEs
        da_dissolved = (-j_abs_mg_per_h + j_diss_mg_per_h) * dt_h
        dc_blood = (j_abs_mg_per_h / vd_L - ke * c_blood) * dt_h

        # Track absorbed drug for f_absorbed
        total_absorbed += j_abs_mg_per_h * dt_h

        # Update states
        a_dissolved = max(0.0, a_dissolved + da_dissolved)
        c_blood = max(0.0, c_blood + dc_blood)

        if not dose_as_solution:
            a_undissolved = max(0.0, a_undissolved - j_diss_mg_per_h * dt_h)

        t += dt_h

    # Find Cmax and Tmax
    cmax = 0.0
    tmax = 0.0
    for i, c in enumerate(c_blood_list):
        if c > cmax:
            cmax = c
            tmax = times[i]

    auc = _compute_auc_trapz(times, c_blood_list)
    f_absorbed = min(1.0, total_absorbed / dose_mg)

    # P-gp saturation at Cmax: compute efflux at Cmax time
    cmax_idx = 0
    for i, c in enumerate(c_blood_list):
        if c == cmax:
            cmax_idx = i
            break

    a_diss_at_cmax = a_dissolved_list[cmax_idx]
    c_apical_at_cmax = a_diss_at_cmax / v_gi_L
    if vmax_pgp_mg_per_h > 0:
        j_eff_at_cmax = vmax_pgp_mg_per_h * c_apical_at_cmax / (km_pgp_mg_per_L + c_apical_at_cmax)
        pgp_saturation = min(1.0, j_eff_at_cmax / vmax_pgp_mg_per_h)
    else:
        pgp_saturation = 0.0

    # Dose proportionality check
    # Non-proportional when dose saturates P-gp: dose concentration > Km
    dose_conc_in_gi = dose_mg / v_gi_L  # mg/L — initial lumen concentration
    dose_proportional = dose_conc_in_gi <= km_pgp_mg_per_L

    notes = (
        f"P-gp efflux saturation model. "
        f"Vmax={vmax_pgp_mg_per_h} mg/h, Km={km_pgp_mg_per_L} mg/L. "
        f"f_absorbed={f_absorbed:.3f}. "
        f"Dose proportional: {dose_proportional}."
    )

    return PGPEffluxSaturationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        vmax_pgp_mg_per_h=vmax_pgp_mg_per_h,
        km_pgp_mg_per_L=km_pgp_mg_per_L,
        times_h=times,
        c_blood_mg_L=c_blood_list,
        a_dissolved_mg=a_dissolved_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_absorbed=f_absorbed,
        pgp_saturation_at_cmax=pgp_saturation,
        dose_proportional=dose_proportional,
        notes=notes,
    )


def compare_dose_levels(
    drug_name: str,
    doses_mg: list[float],
    vmax_pgp_mg_per_h: float = 2.0,
    km_pgp_mg_per_L: float = 10.0,
) -> list[PGPEffluxSaturationResult]:
    """Compare P-gp efflux saturation across multiple dose levels.

    Args:
        drug_name: Name of the drug.
        doses_mg: List of dose levels (mg) to compare.
        vmax_pgp_mg_per_h: P-gp efflux Vmax (mg/h).
        km_pgp_mg_per_L: P-gp Km (mg/L).

    Returns:
        List of PGPEffluxSaturationResult sorted by dose_mg ascending.
    """
    results = []
    for dose in doses_mg:
        result = simulate_pgp_efflux_saturation(
            drug_name=drug_name,
            dose_mg=dose,
            vmax_pgp_mg_per_h=vmax_pgp_mg_per_h,
            km_pgp_mg_per_L=km_pgp_mg_per_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.dose_mg)
    return results
