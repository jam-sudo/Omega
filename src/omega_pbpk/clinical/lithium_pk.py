"""
Phase 967 — Lithium PK model.

Narrow therapeutic index drug with renal clearance.
Therapeutic range: 0.6-1.2 mEq/L = 4.16-8.33 mg/L (serum lithium).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["LithiumPKResult", "simulate_lithium_pk", "find_therapeutic_dose"]

# Therapeutic range in mg/L (1 mEq/L = 6.941 mg/L)
_THERAPEUTIC_MIN_MG_L = 0.6 * 6.941  # ~4.16
_THERAPEUTIC_MAX_MG_L = 1.2 * 6.941  # ~8.33

# Lithium carbonate molecular weights:
# Li2CO3 = 73.89 g/mol, so each formula unit contributes 2 Li ions
# 1 formula unit Li2CO3 = 2 Li+ = 2 * 6.941 = 13.882 mg Li
# dose_mg_Li = dose_mg_Li2CO3 * 2 * 6.941 / 73.89
_LI2CO3_MW = 73.89
_LI_MW = 6.941


def _calc_crcl_ml_min(age_years: float, weight_kg: float, sex: str, scr_mg_dL: float) -> float:
    """Cockcroft-Gault CrCl in mL/min."""
    crcl = (140.0 - age_years) * weight_kg / (72.0 * scr_mg_dL)
    if sex == "female":
        crcl *= 0.85
    return crcl


def _dose_mg_carbonate_to_mg_li(dose_mg: float) -> float:
    """Convert mg lithium carbonate to mg elemental lithium."""
    # Li2CO3 MW = 73.89, each formula unit gives 2 Li atoms
    return dose_mg * 2.0 * _LI_MW / _LI2CO3_MW


@dataclass
class LithiumPKResult:
    """Result of lithium PK simulation."""

    drug_name: str
    dose_mg: float
    weight_kg: float
    crcl_ml_min: float
    cl_Li_L_per_h: float
    vd_L: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    cmin_mg_L: float
    t_half_h: float
    css_min_mg_L: float
    css_max_mg_L: float
    in_therapeutic_range: bool
    notes: str


def simulate_lithium_pk(
    dose_mg: float = 300.0,
    weight_kg: float = 70.0,
    age_years: float = 40.0,
    sex: str = "male",
    scr_mg_dL: float = 1.0,
    vd_per_kg_L: float = 0.8,
    ka_per_h: float = 0.5,
    dosing_interval_h: float = 12.0,
    n_doses: int = 5,
    t_end_h: float = 120.0,
    dt_h: float = 0.5,
    drug_name: str = "lithium_carbonate",
) -> LithiumPKResult:
    """Simulate lithium PK using 1-compartment forward Euler model."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive")
    if age_years <= 0:
        raise ValueError("age_years must be positive")
    if scr_mg_dL <= 0:
        raise ValueError("scr_mg_dL must be positive")
    if sex not in {"male", "female"}:
        raise ValueError("sex must be 'male' or 'female'")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")

    # Pharmacokinetic parameters
    crcl_ml_min = _calc_crcl_ml_min(age_years, weight_kg, sex, scr_mg_dL)
    crcl_L_per_h = crcl_ml_min * 60.0 / 1000.0
    cl_Li_L_per_h = 0.25 * crcl_L_per_h
    vd_L = vd_per_kg_L * weight_kg
    ke = cl_Li_L_per_h / vd_L
    t_half_h = math.log(2.0) / ke

    # Convert dose: mg Li2CO3 → mg elemental Li (used as amount in compartment)
    dose_mg_Li = _dose_mg_carbonate_to_mg_li(dose_mg)

    # Forward Euler simulation with 1-cpt oral absorption
    # States: A_gut (mg Li), C_plasma (mg/L Li)
    # dA_gut/dt = -ka * A_gut  (at dose times)
    # dC/dt = (ka * A_gut) / Vd - ke * C
    n_steps = int(round(t_end_h / dt_h)) + 1
    times_h = [i * dt_h for i in range(n_steps)]

    A_gut = 0.0
    C_plasma = 0.0
    c_values = []

    dose_times = {i * dosing_interval_h for i in range(n_doses)}

    for _i, t in enumerate(times_h):
        # Check if a dose is given at this time
        for dt_dose in dose_times:
            if abs(t - dt_dose) < dt_h * 0.5:
                A_gut += dose_mg_Li

        c_values.append(C_plasma)

        # Forward Euler step
        dA_gut = -ka_per_h * A_gut
        dC = (ka_per_h * A_gut) / vd_L - ke * C_plasma
        A_gut += dA_gut * dt_h
        A_gut = max(A_gut, 0.0)
        C_plasma += dC * dt_h
        C_plasma = max(C_plasma, 0.0)

    # Compute Cmax and Cmin
    cmax_mg_L = max(c_values) if c_values else 0.0

    # Trough = value just before last dose
    last_dose_time = (n_doses - 1) * dosing_interval_h
    trough_idx = int(round(last_dose_time / dt_h))
    if trough_idx > 0 and trough_idx < len(c_values):
        cmin_mg_L = c_values[trough_idx]
    else:
        cmin_mg_L = min(c_values) if c_values else 0.0

    # Analytical steady-state (1-cpt oral, multiple dose)
    # Css_max = F * dose / (Vd * (1 - exp(-ke*tau))) * ka/(ka-ke) * [exp(-ke*t_max) - exp(-ka*t_max)]
    # Css_min = Css_max * exp(-ke*tau)  approximately
    # More precisely: at steady state for 1-cpt oral:
    # A(t) in gut is periodic, but approximate with superposition
    # Use simple analytical approximation:
    # AUC_ss = F * dose / CL  → average Css = F * dose / (CL * tau)
    # Css_min ≈ F * dose / (Vd * (exp(ke*tau) - 1))
    # Css_max ≈ Css_min * exp(ke * tau) * [ka/(ka-ke)] correction...
    # For simplicity: use numerical trough from simulation if enough doses given
    # Otherwise use analytical for SS
    tau = dosing_interval_h
    F = 1.0  # bioavailability

    # Analytical SS trough (1-cpt, first-order absorption, SS):
    # Css_min = F * dose * ka / (Vd * (ka - ke)) * [exp(-ke*tau)/(1-exp(-ke*tau)) - exp(-ka*tau)/(1-exp(-ka*tau))]
    if abs(ka_per_h - ke) < 1e-9:
        # Degenerate case
        css_min_mg_L = F * dose_mg_Li / (vd_L * (1.0 - math.exp(-ke * tau)))
        css_max_mg_L = css_min_mg_L * math.exp(ke * tau)
    else:
        term1 = math.exp(-ke * tau) / (1.0 - math.exp(-ke * tau))
        term2 = math.exp(-ka_per_h * tau) / (1.0 - math.exp(-ka_per_h * tau))
        css_min_mg_L = (F * dose_mg_Li * ka_per_h / (vd_L * (ka_per_h - ke))) * (term1 - term2)
        css_min_mg_L = max(css_min_mg_L, 0.0)

        # Css_max at tmax:
        # tmax_ss = ln(ka/ke) / (ka - ke)
        tmax_ss = math.log(ka_per_h / ke) / (ka_per_h - ke)
        css_max_mg_L = (F * dose_mg_Li * ka_per_h / (vd_L * (ka_per_h - ke))) * (
            math.exp(-ke * tmax_ss) / (1.0 - math.exp(-ke * tau))
            - math.exp(-ka_per_h * tmax_ss) / (1.0 - math.exp(-ka_per_h * tau))
        )
        css_max_mg_L = max(css_max_mg_L, css_min_mg_L)

    in_therapeutic_range = _THERAPEUTIC_MIN_MG_L <= css_min_mg_L <= _THERAPEUTIC_MAX_MG_L

    notes = (
        f"CrCl={crcl_ml_min:.1f} mL/min; CL_Li={cl_Li_L_per_h:.3f} L/h; "
        f"Vd={vd_L:.1f} L; t½={t_half_h:.1f} h; "
        f"Therapeutic range: {_THERAPEUTIC_MIN_MG_L:.2f}-{_THERAPEUTIC_MAX_MG_L:.2f} mg/L; "
        f"CSS_min={css_min_mg_L:.2f} mg/L ({'IN' if in_therapeutic_range else 'OUT OF'} range)"
    )

    return LithiumPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        weight_kg=weight_kg,
        crcl_ml_min=crcl_ml_min,
        cl_Li_L_per_h=cl_Li_L_per_h,
        vd_L=vd_L,
        times_h=times_h,
        c_plasma_mg_L=c_values,
        cmax_mg_L=cmax_mg_L,
        cmin_mg_L=cmin_mg_L,
        t_half_h=t_half_h,
        css_min_mg_L=css_min_mg_L,
        css_max_mg_L=css_max_mg_L,
        in_therapeutic_range=in_therapeutic_range,
        notes=notes,
    )


def find_therapeutic_dose(
    weight_kg: float = 70.0,
    age_years: float = 40.0,
    sex: str = "male",
    scr_mg_dL: float = 1.0,
    dosing_interval_h: float = 12.0,
    vd_per_kg_L: float = 0.8,
    ka_per_h: float = 0.5,
    target_css_min_mg_L: float = 6.0,
) -> dict:
    """Find recommended lithium carbonate dose to achieve therapeutic CSS_min.

    Iterates over candidate doses (150 to 1500 mg in 150 mg steps)
    and selects the one whose CSS_min is closest to target and within range.
    """
    # Validation
    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive")
    if age_years <= 0:
        raise ValueError("age_years must be positive")
    if scr_mg_dL <= 0:
        raise ValueError("scr_mg_dL must be positive")
    if sex not in {"male", "female"}:
        raise ValueError("sex must be 'male' or 'female'")

    candidate_doses = [150 * i for i in range(1, 11)]  # 150 to 1500 mg
    best_dose = candidate_doses[0]
    best_result = None
    best_score = float("inf")

    for dose_mg in candidate_doses:
        result = simulate_lithium_pk(
            dose_mg=dose_mg,
            weight_kg=weight_kg,
            age_years=age_years,
            sex=sex,
            scr_mg_dL=scr_mg_dL,
            vd_per_kg_L=vd_per_kg_L,
            ka_per_h=ka_per_h,
            dosing_interval_h=dosing_interval_h,
            n_doses=10,
            t_end_h=200.0,
        )
        # Prefer doses in therapeutic range, then closest to target
        score = abs(result.css_min_mg_L - target_css_min_mg_L)
        if result.in_therapeutic_range:
            score -= 1000  # strong preference for in-range
        if score < best_score:
            best_score = score
            best_dose = dose_mg
            best_result = result

    if best_result is None:
        best_result = simulate_lithium_pk(
            dose_mg=best_dose,
            weight_kg=weight_kg,
            age_years=age_years,
            sex=sex,
            scr_mg_dL=scr_mg_dL,
        )

    return {
        "recommended_dose_mg": float(best_dose),
        "predicted_css_min_mg_L": best_result.css_min_mg_L,
        "predicted_css_max_mg_L": best_result.css_max_mg_L,
        "in_therapeutic_range": best_result.in_therapeutic_range,
    }
