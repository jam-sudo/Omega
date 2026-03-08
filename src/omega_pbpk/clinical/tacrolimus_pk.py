"""
Phase 1005 — Tacrolimus PK Model
2-compartment oral absorption model for transplant immunosuppression.
CYP3A5 genotype-dependent clearance; whole blood monitoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TacrolimusPKResult:
    """Result of tacrolimus PK simulation."""

    drug_name: str = "Tacrolimus"
    dose_mg: float = 0.0
    times_h: list = field(default_factory=list)
    c_whole_blood_ng_mL: list = field(default_factory=list)
    c0_trough_ng_mL: float = 0.0
    cmax_ng_mL: float = 0.0
    tmax_h: float = 0.0
    auc_0_12h_ng_h_per_mL: float = 0.0
    within_therapeutic_range: bool = False
    dose_recommendation: str = "maintain"
    cyp3a5_phenotype: str = "poor_metabolizer"
    notes: str = ""


def simulate_tacrolimus_pk(
    dose_mg: float = 5.0,
    weight_kg: float = 70.0,
    cyp3a5_star1: bool = False,
    hematocrit: float = 0.45,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> TacrolimusPKResult:
    """Simulate tacrolimus PK using a 2-compartment oral model.

    Parameters
    ----------
    dose_mg:
        Oral dose in mg.
    weight_kg:
        Patient body weight in kg.
    cyp3a5_star1:
        True if patient carries CYP3A5*1 (extensive metabolizer).
    hematocrit:
        Fractional hematocrit (0 < hct < 1).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler step size in hours.

    Returns
    -------
    TacrolimusPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if not (0 < hematocrit < 1):
        raise ValueError(f"hematocrit must be in (0, 1), got {hematocrit}")

    # CYP3A5-dependent clearance
    if cyp3a5_star1:
        cl_l_per_h = 0.04 * weight_kg  # extensive metabolizer
        cyp3a5_phenotype = "extensive_metabolizer"
    else:
        cl_l_per_h = 0.018 * weight_kg  # poor metabolizer
        cyp3a5_phenotype = "poor_metabolizer"

    # PK parameters
    v1 = 0.5 * weight_kg  # L, central volume
    v2 = 8.0 * weight_kg  # L, peripheral volume
    q = 0.3  # L/h, inter-compartmental clearance
    f = 0.25  # oral bioavailability
    ka = 0.5  # /h, absorption rate constant
    blood_plasma_ratio = 15.0 + 20.0 * hematocrit

    ke = cl_l_per_h / v1
    k12 = q / v1
    k21 = q / v2

    # Initial conditions
    # A_gut in mg, A1 in mg (central compartment amount), A2 in mg
    a_gut = f * dose_mg
    a1 = 0.0
    a2 = 0.0

    n_steps = int(math.ceil(t_end_h / dt_h))
    times_h = []
    c_whole_blood = []

    for step in range(n_steps + 1):
        t = step * dt_h
        # Plasma concentration in mg/L
        c_plasma_mg_l = a1 / v1 if v1 > 0 else 0.0
        # Convert to whole blood ng/mL
        c_blood_ng_ml = c_plasma_mg_l * blood_plasma_ratio * 1000.0

        times_h.append(t)
        c_whole_blood.append(c_blood_ng_ml)

        if step < n_steps:
            # ODEs
            d_gut = -ka * a_gut
            d_a1 = ka * a_gut - (ke + k12) * a1 + k21 * a2
            d_a2 = k12 * a1 - k21 * a2

            # Forward Euler
            a_gut = a_gut + dt_h * d_gut
            a1 = a1 + dt_h * d_a1
            a2 = a2 + dt_h * d_a2

            # Clamp to non-negative
            if a_gut < 0.0:
                a_gut = 0.0
            if a1 < 0.0:
                a1 = 0.0
            if a2 < 0.0:
                a2 = 0.0

    # Trough at end of interval
    c0_trough = c_whole_blood[-1]

    # Cmax and tmax
    cmax = max(c_whole_blood)
    tmax_idx = c_whole_blood.index(cmax)
    tmax = times_h[tmax_idx]

    # AUC 0-12h via trapezoidal rule
    auc = 0.0
    for i in range(1, len(times_h)):
        dt = times_h[i] - times_h[i - 1]
        auc += 0.5 * (c_whole_blood[i - 1] + c_whole_blood[i]) * dt

    # Therapeutic range: C0 5–15 ng/mL
    within_range = 5.0 <= c0_trough <= 15.0

    # Dose recommendation
    if c0_trough < 5.0:
        dose_recommendation = "increase"
    elif c0_trough > 15.0:
        dose_recommendation = "decrease"
    else:
        dose_recommendation = "maintain"

    notes_parts = [
        f"CYP3A5 {'*1/*3 or *1/*1' if cyp3a5_star1 else '*3/*3'}: CL={cl_l_per_h:.2f} L/h",
        f"Blood:plasma ratio={blood_plasma_ratio:.1f}",
        f"C0 trough={c0_trough:.1f} ng/mL (target 5-15 ng/mL)",
    ]
    notes = "; ".join(notes_parts)

    return TacrolimusPKResult(
        drug_name="Tacrolimus",
        dose_mg=dose_mg,
        times_h=times_h,
        c_whole_blood_ng_mL=c_whole_blood,
        c0_trough_ng_mL=c0_trough,
        cmax_ng_mL=cmax,
        tmax_h=tmax,
        auc_0_12h_ng_h_per_mL=auc,
        within_therapeutic_range=within_range,
        dose_recommendation=dose_recommendation,
        cyp3a5_phenotype=cyp3a5_phenotype,
        notes=notes,
    )
