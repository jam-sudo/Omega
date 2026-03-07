"""
Phase 948 — Drug-target binding kinetics prediction.

Predicts kon, koff, residence time, and receptor occupancy time-course.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["DrugTargetBindingResult", "simulate_target_binding", "compare_binding_kinetics"]


@dataclass
class DrugTargetBindingResult:
    drug_name: str
    mw_Da: float
    ic50_nM: float
    kd_nM: float
    kon_per_nM_per_h: float
    koff_per_h: float
    residence_time_h: float
    times_h: list
    receptor_occupancy: list
    cmax_plasma_mg_L: float
    peak_occupancy: float
    time_to_peak_occupancy_h: float
    time_above_50pct_occupancy_h: float
    auc_occupancy: float
    kinetic_selectivity_index: float
    notes: str


def _validate_inputs(
    dose_mg: float,
    mw_Da: float,
    ic50_nM: float,
    vd_L: float,
    cl_L_per_h: float,
    r_total_nM: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if ic50_nM <= 0:
        raise ValueError("ic50_nM must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if r_total_nM <= 0:
        raise ValueError("r_total_nM must be > 0")


def _estimate_kon(mw_Da: float) -> float:
    """Estimate kon in M^-1 s^-1 from MW, then convert to L/(nmol·h)."""
    log_kon = 6.0 - 0.002 * mw_Da
    kon_M_per_s = 10**log_kon
    # Clamp to [1e3, 1e8]
    kon_M_per_s = max(1e3, min(1e8, kon_M_per_s))
    # Convert M^-1 s^-1 → nM^-1 h^-1
    # 1 M^-1 s^-1 = 1e-9 nM^-1 s^-1 = 1e-9 * 3600 nM^-1 h^-1
    kon_nM_per_h = kon_M_per_s * 1e-9 * 3600.0
    return kon_nM_per_h


def simulate_target_binding(
    drug_name: str,
    dose_mg: float,
    mw_Da: float,
    ic50_nM: float,
    vd_L: float = 50.0,
    cl_L_per_h: float = 10.0,
    r_total_nM: float = 10.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> DrugTargetBindingResult:
    """Simulate drug-target binding kinetics via Forward Euler ODE."""
    _validate_inputs(dose_mg, mw_Da, ic50_nM, vd_L, cl_L_per_h, r_total_nM)

    # Kinetic parameters
    kd_nM = ic50_nM  # Cheng-Prusoff simplification
    kon_nM_h = _estimate_kon(mw_Da)  # L/(nmol·h) i.e. nM^-1 h^-1
    koff_h = kon_nM_h * kd_nM  # h^-1

    residence_time_h = 1.0 / koff_h if koff_h > 0 else float("inf")

    # PK: 1-cpt IV bolus
    ke = cl_L_per_h / vd_L
    # Cmax in mg/L
    cmax_mg_L = dose_mg / vd_L

    # Convert dose to nM for ODE:
    # C_nM(t) = [dose_mg / vd_L * exp(-ke*t)] / (MW_g/mol * 1e-6)
    #          = C_mg_L / (mw_Da * 1e-6)   [since 1 g/mol = 1 mg/mmol, etc.]
    # More carefully:
    # C_mg_L / mw_Da_g_mol = mmol/L = µmol/mL → µM
    # 1 µM = 1000 nM
    # C_nM = (C_mg_L / mw_Da) * 1e6
    # Because: mg/L ÷ (g/mol) = mg/L * mol/g = (mg·mol)/(L·g) = mmol/L = mM
    # mM * 1e6 → nM? No: 1 mM = 1e6 nM.
    # Actually: 1 mM = 1000 µM = 1e6 nM — yes.
    # So C_nM = C_mg_L / mw_Da * 1e6
    def c_plasma_nM(t: float) -> float:
        c_mg_L = cmax_mg_L * math.exp(-ke * t)
        return c_mg_L / mw_Da * 1e6

    # Forward Euler for receptor occupancy ODE
    # d[DR]/dt = kon * [D] * (R_total - [DR]) - koff * [DR]
    # We track DR in normalized units (fraction of R_total)
    # [D] in nM, kon in nM^-1 h^-1, koff in h^-1
    n_steps = max(1, int(round(t_end_h / dt_h)))
    times = [i * dt_h for i in range(n_steps + 1)]
    occupancy = [0.0] * len(times)  # DR/R_total

    dr_frac = 0.0  # initial occupancy fraction
    for i in range(1, len(times)):
        t = times[i - 1]
        d_nM = c_plasma_nM(t)
        r_free_nM = r_total_nM * (1.0 - dr_frac)
        dr_nM = r_total_nM * dr_frac
        d_dr_dt = kon_nM_h * d_nM * r_free_nM - koff_h * dr_nM
        dr_nM_new = dr_nM + d_dr_dt * dt_h
        # Clamp
        dr_nM_new = max(0.0, min(r_total_nM, dr_nM_new))
        dr_frac = dr_nM_new / r_total_nM
        occupancy[i] = dr_frac

    # Metrics
    peak_occupancy = max(occupancy)
    peak_idx = occupancy.index(peak_occupancy)
    time_to_peak_h = times[peak_idx]

    # Time above 50% occupancy
    above_50 = [i for i, o in enumerate(occupancy) if o >= 0.5]
    if above_50:
        time_above_50pct_h = (above_50[-1] - above_50[0]) * dt_h
    else:
        time_above_50pct_h = 0.0

    # AUC occupancy (trapezoidal)
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (occupancy[i - 1] + occupancy[i]) * dt_h

    kinetic_selectivity_index = residence_time_h * peak_occupancy

    notes = (
        f"kd={kd_nM:.2f}nM, kon={kon_nM_h:.3e}nM⁻¹h⁻¹, "
        f"koff={koff_h:.4f}/h, RT={residence_time_h:.2f}h, "
        f"peak_occ={peak_occupancy:.3f}"
    )

    return DrugTargetBindingResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        ic50_nM=ic50_nM,
        kd_nM=kd_nM,
        kon_per_nM_per_h=kon_nM_h,
        koff_per_h=koff_h,
        residence_time_h=residence_time_h,
        times_h=times,
        receptor_occupancy=occupancy,
        cmax_plasma_mg_L=cmax_mg_L,
        peak_occupancy=peak_occupancy,
        time_to_peak_occupancy_h=time_to_peak_h,
        time_above_50pct_occupancy_h=time_above_50pct_h,
        auc_occupancy=auc,
        kinetic_selectivity_index=kinetic_selectivity_index,
        notes=notes,
    )


def compare_binding_kinetics(compounds: list) -> list:
    """Compare binding kinetics for multiple compounds, sorted by KSI descending."""
    results = []
    for comp in compounds:
        result = simulate_target_binding(
            drug_name=comp["drug_name"],
            dose_mg=comp["dose_mg"],
            mw_Da=comp["mw_Da"],
            ic50_nM=comp["ic50_nM"],
            vd_L=comp.get("vd_L", 50.0),
            cl_L_per_h=comp.get("cl_L_per_h", 10.0),
        )
        results.append(result)
    results.sort(key=lambda r: r.kinetic_selectivity_index, reverse=True)
    return results
