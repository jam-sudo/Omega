"""
Phase 740 — Tumor Microenvironment PK (core/tumor_microenv_pk.py)

Models drug penetration into solid tumors accounting for elevated IFP,
MW, and logP effects on vascular permeability.

3-compartment: plasma + tumor rim + tumor core.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TumorMicroenvPKResult:
    """Result of tumor microenvironment PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_rim_mg_L: list[float]
    c_core_mg_L: list[float]
    cmax_plasma: float
    cmax_rim: float
    cmax_core: float
    auc_plasma: float
    auc_rim: float
    auc_core: float
    rim_to_plasma_ratio: float
    core_to_rim_ratio: float
    core_to_plasma_ratio: float
    tumor_penetration_index: float
    ps_rim_actual: float
    notes: str


def _validate_inputs(dose_mg: float) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")


def _trapezoidal_auc(times: list[float], values: list[float]) -> float:
    """Compute AUC using trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


def _compute_ps_rim_actual(
    ps_rim_L_per_h: float,
    ifp_mmHg: float,
    mw_da: float,
    logp: float,
) -> float:
    """Apply IFP, MW, and logP modulation to rim PS."""
    ifp_factor = min(1.0, ifp_mmHg / 20.0)
    mw_factor = max(0.1, 1.0 - (mw_da - 300.0) / 3000.0)
    logp_factor = max(0.5, min(1.5, 1.0 + (logp - 2.0) * 0.1))
    return ps_rim_L_per_h * (1.0 - 0.5 * ifp_factor) * mw_factor * logp_factor


def simulate_tumor_microenv_pk(
    drug_name: str = "Drug",
    dose_mg: float = 100.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ps_rim_L_per_h: float = 0.5,
    ps_core_L_per_h: float = 0.1,
    v_rim_L: float = 0.05,
    v_core_L: float = 0.2,
    ke_tumor_per_h: float = 0.1,
    ifp_mmHg: float = 15.0,
    mw_da: float = 300.0,
    logp: float = 2.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> TumorMicroenvPKResult:
    """Simulate tumor microenvironment PK (IV bolus).

    States:
        C_plasma: systemic plasma (mg/L)
        C_rim   : tumor rim (mg/L)
        C_core  : tumor core (mg/L)
    """
    _validate_inputs(dose_mg)

    ke = cl_L_per_h / vd_L
    ps_rim_actual = _compute_ps_rim_actual(ps_rim_L_per_h, ifp_mmHg, mw_da, logp)

    # Initial conditions — IV bolus
    c_plasma = dose_mg / vd_L
    c_rim = 0.0
    c_core = 0.0

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_rim_list: list[float] = []
    c_core_list: list[float] = []

    n_steps = int(round(t_end_h / dt_h))
    t = 0.0

    for _ in range(n_steps + 1):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_rim_list.append(c_rim)
        c_core_list.append(c_core)

        if _ == n_steps:
            break

        # ODEs
        dc_plasma_dt = -ke * c_plasma
        dc_rim_dt = ps_rim_actual * (c_plasma - c_rim) / v_rim_L - ke_tumor_per_h * c_rim
        dc_core_dt = ps_core_L_per_h * (c_rim - c_core) / v_core_L - ke_tumor_per_h * c_core

        # Forward Euler
        c_plasma = max(0.0, c_plasma + dc_plasma_dt * dt_h)
        c_rim = max(0.0, c_rim + dc_rim_dt * dt_h)
        c_core = max(0.0, c_core + dc_core_dt * dt_h)

        t += dt_h

    # Metrics
    cmax_plasma = max(c_plasma_list)
    cmax_rim = max(c_rim_list)
    cmax_core = max(c_core_list)

    auc_plasma = _trapezoidal_auc(times, c_plasma_list)
    auc_rim = _trapezoidal_auc(times, c_rim_list)
    auc_core = _trapezoidal_auc(times, c_core_list)

    rim_to_plasma_ratio = auc_rim / auc_plasma if auc_plasma > 0.0 else 0.0
    core_to_rim_ratio = auc_core / auc_rim if auc_rim > 0.0 else 0.0
    core_to_plasma_ratio = auc_core / auc_plasma if auc_plasma > 0.0 else 0.0
    tumor_penetration_index = core_to_plasma_ratio * 100.0

    notes = (
        f"Tumor microenv PK: dose={dose_mg}mg IV bolus, IFP={ifp_mmHg}mmHg, "
        f"MW={mw_da}Da, logP={logp}. "
        f"PS_rim_actual={ps_rim_actual:.3f}L/h, "
        f"TPI={tumor_penetration_index:.1f}%, "
        f"core/plasma AUC ratio={core_to_plasma_ratio:.3f}."
    )

    return TumorMicroenvPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_rim_mg_L=c_rim_list,
        c_core_mg_L=c_core_list,
        cmax_plasma=cmax_plasma,
        cmax_rim=cmax_rim,
        cmax_core=cmax_core,
        auc_plasma=auc_plasma,
        auc_rim=auc_rim,
        auc_core=auc_core,
        rim_to_plasma_ratio=rim_to_plasma_ratio,
        core_to_rim_ratio=core_to_rim_ratio,
        core_to_plasma_ratio=core_to_plasma_ratio,
        tumor_penetration_index=tumor_penetration_index,
        ps_rim_actual=ps_rim_actual,
        notes=notes,
    )


def assess_tumor_penetration(result: TumorMicroenvPKResult) -> dict:
    """Assess tumor penetration quality and provide recommendations.

    Penetration grades based on tumor_penetration_index (%):
        poor     : < 5%
        moderate : 5-20%
        good     : > 20%
    """
    tpi = result.tumor_penetration_index

    if tpi < 5.0:
        penetration_grade = "poor"
        recommendation = (
            "Poor tumor core penetration. Consider nanoparticle formulation, "
            "EPR-targeted delivery, or agents that reduce tumor IFP (e.g., VEGF inhibitors). "
            "Evaluate lower MW analogues or more lipophilic derivatives."
        )
    elif tpi <= 20.0:
        penetration_grade = "moderate"
        recommendation = (
            "Moderate tumor core penetration. Dose escalation or fractionated dosing may "
            "improve efficacy. Combination with IFP-reducing agents could enhance delivery."
        )
    else:
        penetration_grade = "good"
        recommendation = (
            "Good tumor core penetration. Drug physicochemical properties are favorable. "
            "Monitor for systemic toxicity at doses required for tumor concentrations."
        )

    return {
        "penetration_grade": penetration_grade,
        "tumor_penetration_index_pct": tpi,
        "core_to_plasma_ratio": result.core_to_plasma_ratio,
        "core_to_rim_ratio": result.core_to_rim_ratio,
        "rim_to_plasma_ratio": result.rim_to_plasma_ratio,
        "ps_rim_actual_L_per_h": result.ps_rim_actual,
        "recommendation": recommendation,
    }
