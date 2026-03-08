"""
Phase 895 — Adipose Tissue Drug Distribution

Models drug accumulation in adipose tissue, critical for lipophilic drugs
and for understanding PK changes in obese patients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AdiposeDistributionResult:
    """Result of adipose tissue drug distribution simulation."""

    drug_name: str
    dose_mg: float
    logp: float
    bmi: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_adipose_mg_L: list[float]
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    cmax_adipose: float
    auc_adipose: float
    adipose_plasma_kp: float
    vd_apparent_L: float
    t_half_terminal_h: float
    obesity_dose_adjustment: float
    notes: str


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def simulate_adipose_distribution(
    drug_name: str,
    dose_mg: float,
    logp: float = 3.0,
    bmi: float = 25.0,
    k_abs_per_h: float = 1.0,
    cl_L_per_h: float = 10.0,
    vd_lean_L: float = 30.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> AdiposeDistributionResult:
    """
    Simulate drug distribution into adipose tissue.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    logp : float
        Lipophilicity (log octanol-water partition coefficient).
    bmi : float
        Patient BMI (kg/m^2).
    k_abs_per_h : float
        First-order absorption rate constant (h^-1).
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_lean_L : float
        Lean-body volume of distribution (L).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        ODE integration time step (h).

    Returns
    -------
    AdiposeDistributionResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if bmi <= 10:
        raise ValueError("bmi must be > 10 (physiological range)")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")

    # --- Adipose tissue volume estimation ---
    if bmi < 22:
        adipose_kg = 10.0
    else:
        adipose_kg = (bmi - 22) * 0.5 + 10.0
    v_adipose_L = adipose_kg * 0.9  # density ~1.1 kg/L

    # --- Partition coefficient ---
    kp_adipose = max(0.1, math.pow(10.0, logp * 0.3))

    # --- Effective Vd and rate constants ---
    effective_vd = vd_lean_L + v_adipose_L * kp_adipose
    ke_sys = cl_L_per_h / effective_vd
    k_adipose_uptake = 0.1 * kp_adipose
    k_adipose_release = 0.1

    # --- Forward Euler ODE integration ---
    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = []
    c_plasma: list[float] = []
    c_adipose: list[float] = []

    a_gut = dose_mg  # amount in gut (mg)
    A_plasma = 0.0  # amount in plasma (mg)
    A_adipose = 0.0  # amount in adipose (mg)

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)
        cp = A_plasma / vd_lean_L if vd_lean_L > 0 else 0.0
        ca = A_adipose / v_adipose_L if v_adipose_L > 0 else 0.0
        c_plasma.append(cp)
        c_adipose.append(ca)

        # Derivatives
        d_gut = -k_abs_per_h * a_gut
        d_plasma = (
            k_abs_per_h * a_gut
            - ke_sys * A_plasma
            - k_adipose_uptake * A_plasma
            + k_adipose_release * A_adipose
        )
        d_adipose = k_adipose_uptake * A_plasma - k_adipose_release * A_adipose

        # Update
        a_gut += d_gut * dt_h
        A_plasma += d_plasma * dt_h
        A_adipose += d_adipose * dt_h

        # Clamp negatives
        a_gut = max(0.0, a_gut)
        A_plasma = max(0.0, A_plasma)
        A_adipose = max(0.0, A_adipose)

    # --- Derived PK metrics ---
    cmax_plasma = max(c_plasma)
    tmax_plasma_h = times[c_plasma.index(cmax_plasma)]
    cmax_adipose = max(c_adipose)
    auc_plasma = _trapezoidal_auc(times, c_plasma)
    auc_adipose = _trapezoidal_auc(times, c_adipose)

    adipose_plasma_kp = auc_adipose / auc_plasma if auc_plasma > 0 else kp_adipose
    vd_apparent_L = effective_vd
    t_half_terminal_h = 0.693 * effective_vd / cl_L_per_h

    # Obesity dose adjustment relative to normal BMI (BMI=25 reference)
    ref_bmi = 25.0
    if ref_bmi < 22:
        ref_adipose_kg = 10.0
    else:
        ref_adipose_kg = (ref_bmi - 22) * 0.5 + 10.0
    ref_v_adipose_L = ref_adipose_kg * 0.9
    ref_vd = vd_lean_L + ref_v_adipose_L * kp_adipose
    obesity_dose_adjustment = effective_vd / ref_vd

    # --- Notes ---
    if logp > 4:
        notes = (
            "Highly lipophilic — significant adipose accumulation expected. "
            "Consider dose adjustment in obese patients."
        )
    elif logp > 2:
        notes = "Moderate lipophilicity — some adipose distribution"
    else:
        notes = "Low adipose partitioning"

    return AdiposeDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        bmi=bmi,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_adipose_mg_L=c_adipose,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        cmax_adipose=cmax_adipose,
        auc_adipose=auc_adipose,
        adipose_plasma_kp=adipose_plasma_kp,
        vd_apparent_L=vd_apparent_L,
        t_half_terminal_h=t_half_terminal_h,
        obesity_dose_adjustment=obesity_dose_adjustment,
        notes=notes,
    )


def compare_bmi_groups(
    drug_name: str,
    dose_mg: float,
    logp: float = 3.0,
    bmis: list[float] | None = None,
) -> list[AdiposeDistributionResult]:
    """
    Compare adipose distribution across BMI groups.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    logp : float
        Lipophilicity.
    bmis : list of float, optional
        BMI values to simulate (default: [20, 25, 30, 35, 40]).

    Returns
    -------
    list of AdiposeDistributionResult sorted by bmi ascending
    """
    if bmis is None:
        bmis = [20.0, 25.0, 30.0, 35.0, 40.0]

    results = [
        simulate_adipose_distribution(drug_name=drug_name, dose_mg=dose_mg, logp=logp, bmi=b)
        for b in bmis
    ]
    return sorted(results, key=lambda r: r.bmi)
