"""
Phase 894 — Cardiac Drug Distribution (core/cardiac_pk.py)

Models drug distribution in heart tissue and cardiac PK/PD for antiarrhythmics.

Compartments:
  - Plasma (central)
  - Myocardium (heart muscle)
  - Cardiac intracellular (ion channel targets)

Relevant for antiarrhythmics, hERG blockers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CardiacPKResult:
    """Result of a cardiac PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_myocardium_mg_L: list = field(default_factory=list)
    cmax_plasma: float = 0.0
    auc_plasma: float = 0.0
    cmax_myocardium: float = 0.0
    auc_myocardium: float = 0.0
    heart_plasma_kp: float = 0.0
    cardiac_exposure_score: float = 0.0
    qtc_risk_level: str = "low"
    notes: str = ""


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        total += 0.5 * (values[i] + values[i - 1]) * dt
    return total


def simulate_cardiac_pk(
    drug_name: str,
    dose_mg: float,
    kp_heart: float = 3.0,
    k_uptake_per_h: float = 5.0,
    k_efflux_per_h: float = 1.0,
    cl_sys_L_per_h: float = 15.0,
    vd_sys_L: float = 70.0,
    v_heart_L: float = 0.3,
    herg_ic50_uM: float = 10.0,
    mw_Da: float = 400.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> CardiacPKResult:
    """Simulate drug distribution into cardiac tissue.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg (IV bolus to plasma).
    kp_heart:
        Heart-to-plasma partition coefficient (dimensionless).
    k_uptake_per_h:
        First-order rate constant for plasma to myocardium (h^-1).
    k_efflux_per_h:
        First-order rate constant for myocardium efflux back to plasma (h^-1).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution of central compartment (L).
    v_heart_L:
        Myocardial volume (L); default 0.3 L (approx 300 mL).
    herg_ic50_uM:
        hERG channel IC50 in micromolar for QTc risk assessment.
    mw_Da:
        Molecular weight in Daltons for concentration unit conversion.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    CardiacPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if kp_heart <= 0:
        raise ValueError("kp_heart must be > 0")
    if v_heart_L <= 0:
        raise ValueError("v_heart_L must be > 0")
    if herg_ic50_uM <= 0:
        raise ValueError("herg_ic50_uM must be > 0")

    ke_sys = cl_sys_L_per_h / vd_sys_L

    # Initial amounts (mg)
    a_plasma = dose_mg
    a_heart = 0.0

    times: list = []
    c_plasma_list: list = []
    c_heart_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _step in range(n_steps + 1):
        c_plasma = a_plasma / vd_sys_L
        c_heart = a_heart / v_heart_L

        times.append(t)
        c_plasma_list.append(c_plasma)
        c_heart_list.append(c_heart)

        # ODEs (Forward Euler)
        da_plasma = (
            -ke_sys * a_plasma - k_uptake_per_h * a_plasma + k_efflux_per_h * a_heart
        ) * dt_h
        da_heart = (k_uptake_per_h * a_plasma * kp_heart - k_efflux_per_h * a_heart) * dt_h

        a_plasma += da_plasma
        a_heart += da_heart

        # Clamp to zero (numerical safety)
        if a_plasma < 0.0:
            a_plasma = 0.0
        if a_heart < 0.0:
            a_heart = 0.0

        t += dt_h

    # PK metrics
    cmax_plasma = max(c_plasma_list)
    auc_plasma = _trapz(times, c_plasma_list)
    cmax_myocardium = max(c_heart_list)
    auc_myocardium = _trapz(times, c_heart_list)

    heart_plasma_kp = auc_myocardium / auc_plasma if auc_plasma > 0.0 else kp_heart

    # Cardiac exposure score
    raw_score = heart_plasma_kp * 10.0 + kp_heart * 5.0
    cardiac_exposure_score = max(0.0, min(100.0, raw_score))

    # QTc risk via hERG occupancy
    # Convert cmax_myocardium from mg/L to uM: (mg/L) / (g/mol) * 1e3 = mmol/L * 1e3 = uM
    # (mg/L) / (Da) * 1e6 is equivalent: 1 mg/L / (g/mol) = 1 mmol/m^3 = 1e-3 mmol/L = 1e-3 uM... let us be precise
    # mg/L * (1 g / 1000 mg) * (1 mol / mw_Da g) * (1e6 umol / mol) * (1 L / 1 L) = mg/L / mw_Da * 1000 = uM
    cmax_uM = cmax_myocardium / mw_Da * 1000.0 if mw_Da > 0 else 0.0
    occupancy = cmax_uM / (cmax_uM + herg_ic50_uM) if (cmax_uM + herg_ic50_uM) > 0 else 0.0

    if occupancy > 0.5:
        qtc_risk_level = "high"
        notes = "High cardiac exposure — significant hERG occupancy. QTc monitoring required."
    elif occupancy > 0.2:
        qtc_risk_level = "moderate"
        notes = "Moderate cardiac accumulation — baseline ECG recommended."
    else:
        qtc_risk_level = "low"
        notes = "Low cardiac exposure risk."

    return CardiacPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_myocardium_mg_L=c_heart_list,
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        cmax_myocardium=cmax_myocardium,
        auc_myocardium=auc_myocardium,
        heart_plasma_kp=heart_plasma_kp,
        cardiac_exposure_score=cardiac_exposure_score,
        qtc_risk_level=qtc_risk_level,
        notes=notes,
    )


def compare_cardiac_scenarios(
    drug_name: str,
    dose_mg: float,
    scenarios: list,
) -> list:
    """Simulate cardiac PK under multiple parameter scenarios.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Default dose in mg (overridden per scenario if provided).
    scenarios:
        List of dicts with optional keys: "kp_heart", "dose_mg",
        "herg_ic50_uM", "name".

    Returns
    -------
    List of CardiacPKResult sorted by cardiac_exposure_score descending.
    """
    results = []
    for scenario in scenarios:
        scenario_dose = scenario.get("dose_mg", dose_mg)
        kp_heart = scenario.get("kp_heart", 3.0)
        herg_ic50_uM = scenario.get("herg_ic50_uM", 10.0)
        name = scenario.get("name", drug_name)

        result = simulate_cardiac_pk(
            drug_name=name,
            dose_mg=scenario_dose,
            kp_heart=kp_heart,
            herg_ic50_uM=herg_ic50_uM,
        )
        results.append(result)

    results.sort(key=lambda r: r.cardiac_exposure_score, reverse=True)
    return results
