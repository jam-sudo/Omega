"""
Phase 893 — Muscle Drug Distribution (core/muscle_pk.py)

Models drug distribution into skeletal muscle using a 3-compartment system:
  - Plasma (central)
  - Muscle interstitium
  - Muscle intracellular

Relevant for statins, aminoglycosides, local anesthetics.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MusclePKResult:
    """Result of a muscle PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_muscle_interstitial_mg_L: list = field(default_factory=list)
    c_muscle_intracell_mg_L: list = field(default_factory=list)
    cmax_plasma: float = 0.0
    auc_plasma: float = 0.0
    cmax_muscle_intracell: float = 0.0
    auc_muscle_intracell: float = 0.0
    muscle_plasma_kp: float = 0.0
    myopathy_risk_score: float = 0.0
    myopathy_risk_level: str = "low"
    notes: str = ""


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        total += 0.5 * (values[i] + values[i - 1]) * dt
    return total


def simulate_muscle_pk(
    drug_name: str,
    dose_mg: float,
    kp_muscle: float = 2.0,
    k_uptake_per_h: float = 0.3,
    k_release_per_h: float = 0.1,
    cl_sys_L_per_h: float = 10.0,
    vd_sys_L: float = 50.0,
    v_muscle_L: float = 28.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> MusclePKResult:
    """Simulate drug distribution into skeletal muscle.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg (IV bolus to plasma).
    kp_muscle:
        Muscle-to-plasma partition coefficient (dimensionless).
    k_uptake_per_h:
        First-order rate constant for plasma to interstitium (h^-1).
    k_release_per_h:
        First-order rate constant for interstitium efflux (h^-1).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution of central compartment (L).
    v_muscle_L:
        Total skeletal muscle volume (L); default 28 L for 70 kg human.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    MusclePKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if kp_muscle <= 0:
        raise ValueError("kp_muscle must be > 0")
    if v_muscle_L <= 0:
        raise ValueError("v_muscle_L must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")

    # Derived volumes
    v_interstitial_L = v_muscle_L * 0.1  # 10% interstitial
    v_intracell_L = v_muscle_L * 0.7  # 70% intracellular

    # Rate constants
    ke_sys = cl_sys_L_per_h / vd_sys_L
    k_ic_per_h = k_uptake_per_h * kp_muscle * 0.5  # intracellular uptake

    # Initial amounts (mg)
    a_plasma = dose_mg
    a_inter = 0.0
    a_intracell = 0.0

    times: list = []
    c_plasma_list: list = []
    c_inter_list: list = []
    c_ic_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _step in range(n_steps + 1):
        c_plasma = a_plasma / vd_sys_L
        c_inter = a_inter / v_interstitial_L if v_interstitial_L > 0 else 0.0
        c_ic = a_intracell / v_intracell_L if v_intracell_L > 0 else 0.0

        times.append(t)
        c_plasma_list.append(c_plasma)
        c_inter_list.append(c_inter)
        c_ic_list.append(c_ic)

        # ODEs (Forward Euler)
        da_plasma = (-ke_sys * a_plasma - k_uptake_per_h * a_plasma) * dt_h
        da_inter = (
            k_uptake_per_h * a_plasma - k_release_per_h * a_inter - k_ic_per_h * a_inter
        ) * dt_h
        da_intracell = (k_ic_per_h * a_inter - k_release_per_h * 0.3 * a_intracell) * dt_h

        a_plasma += da_plasma
        a_inter += da_inter
        a_intracell += da_intracell

        # Clamp to zero (numerical safety)
        if a_plasma < 0.0:
            a_plasma = 0.0
        if a_inter < 0.0:
            a_inter = 0.0
        if a_intracell < 0.0:
            a_intracell = 0.0

        t += dt_h

    # PK metrics
    cmax_plasma = max(c_plasma_list)
    auc_plasma = _trapz(times, c_plasma_list)
    cmax_muscle_intracell = max(c_ic_list)
    auc_muscle_intracell = _trapz(times, c_ic_list)

    muscle_plasma_kp = auc_muscle_intracell / auc_plasma if auc_plasma > 0.0 else 0.0

    # Risk scoring
    raw_score = muscle_plasma_kp * 20.0 + kp_muscle * 5.0
    myopathy_risk_score = max(0.0, min(100.0, raw_score))

    if myopathy_risk_score > 60.0:
        myopathy_risk_level = "high"
        notes = "High myopathy risk — monitor CK levels. Consider dose reduction."
    elif myopathy_risk_score > 30.0:
        myopathy_risk_level = "moderate"
        notes = "Moderate muscle accumulation — routine CK monitoring recommended."
    else:
        myopathy_risk_level = "low"
        notes = "Low myopathy risk."

    return MusclePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_muscle_interstitial_mg_L=c_inter_list,
        c_muscle_intracell_mg_L=c_ic_list,
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        cmax_muscle_intracell=cmax_muscle_intracell,
        auc_muscle_intracell=auc_muscle_intracell,
        muscle_plasma_kp=muscle_plasma_kp,
        myopathy_risk_score=myopathy_risk_score,
        myopathy_risk_level=myopathy_risk_level,
        notes=notes,
    )


def compare_statin_kp(
    drug_name: str,
    dose_mg: float,
    kp_values: list,
) -> list:
    """Simulate across a range of Kp values and sort by myopathy risk.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Dose in mg.
    kp_values:
        List of kp_muscle values to compare.

    Returns
    -------
    List of MusclePKResult sorted by myopathy_risk_score descending.
    """
    results = [
        simulate_muscle_pk(drug_name=drug_name, dose_mg=dose_mg, kp_muscle=kp) for kp in kp_values
    ]
    results.sort(key=lambda r: r.myopathy_risk_score, reverse=True)
    return results
