"""
Phase 383 — Blood-Brain Barrier Efflux Transporter Model

Models drug transport across the blood-brain barrier with active efflux
transporters (P-gp, BCRP, MRP4) and passive diffusion.

Pure Python, stdlib only (math, dataclasses).
Forward Euler ODE integration.
"""

import math
from dataclasses import dataclass


@dataclass
class BBBEflluxResult:
    """Result of BBB efflux transporter simulation. Note: double-l in 'Efllux'."""

    drug_name: str
    times_h: list
    c_plasma_mg_L: list
    c_brain_mg_L: list
    kp_brain: float
    efflux_ratio: float
    passive_permeability_cm_per_s: float
    total_brain_influx_rate: float
    total_efflux_rate: float
    p_gp_substrate: bool
    bcrp_substrate: bool
    cns_exposure_class: str
    auc_brain_mg_h_per_L: float
    auc_plasma_mg_h_per_L: float
    notes: str


def _classify_cns_exposure(kp_brain: float) -> str:
    """Classify CNS exposure based on brain/plasma Kp ratio."""
    if kp_brain > 0.5:
        return "high"
    elif kp_brain > 0.1:
        return "moderate"
    elif kp_brain > 0.01:
        return "low"
    else:
        return "minimal"


def _trapezoidal_auc(times: list, concentrations: list) -> float:
    """Compute AUC using manual trapezoidal rule."""
    return sum(
        0.5 * (concentrations[i] + concentrations[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )


def simulate_bbb_efflux(
    drug_name: str,
    dose_mg: float,
    logp: float,
    mw_da: float,
    psa_A2: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    p_gp_ic50_uM: float | None = None,
    bcrp_ic50_uM: float | None = None,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> BBBEflluxResult:
    """
    Simulate drug concentration in plasma and brain over time accounting for
    BBB passive permeability and active efflux transporters (P-gp, BCRP).

    Parameters
    ----------
    drug_name : str
        Identifier for the drug.
    dose_mg : float
        IV bolus dose in mg.
    logp : float
        Octanol-water partition coefficient.
    mw_da : float
        Molecular weight in Daltons.
    psa_A2 : float
        Polar surface area in Angstrom^2.
    cl_sys_L_per_h : float
        Systemic clearance in L/h.
    vd_sys_L : float
        Volume of distribution in L.
    p_gp_ic50_uM : float, optional
        P-gp inhibitor IC50 in uM (if drug inhibits P-gp at this concentration).
    bcrp_ic50_uM : float, optional
        BCRP inhibitor IC50 in uM.
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Forward Euler time step in hours.

    Returns
    -------
    BBBEflluxResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if psa_A2 < 0:
        raise ValueError("psa_A2 must be >= 0")
    if mw_da <= 0:
        raise ValueError("mw_da must be > 0")

    # --- Brain volume ---
    V_brain = 1.35  # L

    # --- Passive permeability from logP/MW/PSA (Caco-2 surrogate) ---
    logPe = 0.3 * logp - 0.01 * psa_A2 - 0.005 * mw_da + 0.5
    passive_perm_cm_s = max(1e-8, 10**logPe * 1e-6)

    # BBB surface area = 20000 cm2
    # influx_passive_per_h [h^-1] = perm_cm_s * 3600 s/h * surface_cm2 / (V_brain_L * 1000 mL/L)
    surface_area_cm2 = 20000.0
    influx_passive_per_h = (passive_perm_cm_s * 3600.0 * surface_area_cm2) / (V_brain * 1000.0)

    # --- Substrate classification ---
    p_gp_substrate = (logp > 1.5) and (mw_da > 350) and (psa_A2 < 120)
    bcrp_substrate = (logp > 0) and (mw_da > 300) and (psa_A2 < 100)

    # --- Active efflux rates ---
    k_pgp = 0.0
    if p_gp_substrate:
        k_pgp = 0.5  # baseline P-gp efflux per h
        if p_gp_ic50_uM is not None:
            k_pgp *= 0.1  # inhibited

    k_bcrp = 0.0
    if bcrp_substrate:
        k_bcrp = 0.2  # baseline BCRP efflux per h
        if bcrp_ic50_uM is not None:
            k_bcrp *= 0.1  # inhibited

    total_efflux_per_h = k_pgp + k_bcrp

    # --- Rate constants for ODE ---
    # plasma→brain: passive influx
    k_plasma_to_brain = influx_passive_per_h
    # brain→plasma: passive back-flux + active efflux
    k_brain_to_plasma = influx_passive_per_h + total_efflux_per_h

    # --- Initial conditions ---
    C_plasma = dose_mg / vd_sys_L  # mg/L
    C_brain = 0.0  # mg/L

    # Elimination rate constant
    k_elim = cl_sys_L_per_h / vd_sys_L  # per h

    # --- Forward Euler simulation ---
    times = [0.0]
    plasma_concs = [C_plasma]
    brain_concs = [C_brain]

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    for _ in range(n_steps):
        # dC_plasma/dt = -k_elim * C_plasma  (brain contribution negligible, V_brain << vd)
        dC_plasma = -k_elim * C_plasma

        # dC_brain/dt = k_plasma_to_brain * C_plasma - k_brain_to_plasma * C_brain
        dC_brain = k_plasma_to_brain * C_plasma - k_brain_to_plasma * C_brain

        C_plasma = max(0.0, C_plasma + dC_plasma * dt_h)
        C_brain = max(0.0, C_brain + dC_brain * dt_h)

        t = min(t + dt_h, t_end_h)
        times.append(t)
        plasma_concs.append(C_plasma)
        brain_concs.append(C_brain)

    # --- Derived metrics ---
    final_plasma = plasma_concs[-1]
    final_brain = brain_concs[-1]
    kp_brain = (final_brain / final_plasma) if final_plasma > 0.0 else 0.0

    efflux_ratio = (k_brain_to_plasma / k_plasma_to_brain) if k_plasma_to_brain > 0.0 else 1.0

    cns_exposure_class = _classify_cns_exposure(kp_brain)

    auc_brain = _trapezoidal_auc(times, brain_concs)
    auc_plasma = _trapezoidal_auc(times, plasma_concs)

    # --- Build notes ---
    note_parts = []
    if p_gp_substrate:
        status = "inhibited" if p_gp_ic50_uM is not None else "active"
        note_parts.append(f"P-gp substrate (efflux {status})")
    if bcrp_substrate:
        status = "inhibited" if bcrp_ic50_uM is not None else "active"
        note_parts.append(f"BCRP substrate (efflux {status})")
    if not p_gp_substrate and not bcrp_substrate:
        note_parts.append("No active efflux transporter substrate")
    notes = "; ".join(note_parts) if note_parts else "Passive diffusion only"

    return BBBEflluxResult(
        drug_name=drug_name,
        times_h=times,
        c_plasma_mg_L=plasma_concs,
        c_brain_mg_L=brain_concs,
        kp_brain=kp_brain,
        efflux_ratio=efflux_ratio,
        passive_permeability_cm_per_s=passive_perm_cm_s,
        total_brain_influx_rate=k_plasma_to_brain,
        total_efflux_rate=total_efflux_per_h,
        p_gp_substrate=p_gp_substrate,
        bcrp_substrate=bcrp_substrate,
        cns_exposure_class=cns_exposure_class,
        auc_brain_mg_h_per_L=auc_brain,
        auc_plasma_mg_h_per_L=auc_plasma,
        notes=notes,
    )


def screen_cns_penetration(drugs_list: list) -> list:
    """
    Screen a list of drugs for CNS penetration.

    Parameters
    ----------
    drugs_list : list of dict
        Each dict must have keys: name, dose_mg, logp, mw_da, psa_A2,
        cl_sys_L_per_h, vd_sys_L.

    Returns
    -------
    list of BBBEflluxResult sorted by kp_brain descending.
    """
    results = []
    for drug in drugs_list:
        result = simulate_bbb_efflux(
            drug_name=drug["name"],
            dose_mg=drug["dose_mg"],
            logp=drug["logp"],
            mw_da=drug["mw_da"],
            psa_A2=drug["psa_A2"],
            cl_sys_L_per_h=drug["cl_sys_L_per_h"],
            vd_sys_L=drug["vd_sys_L"],
        )
        results.append(result)

    results.sort(key=lambda r: r.kp_brain, reverse=True)
    return results
