"""Phase 1102 — Drug Microsomal Stability Prediction.

Predict intrinsic clearance from in vitro microsomal stability (t½) assay
and scale to in vivo using the well-stirred liver model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Constants
_MICROSOMAL_PROTEIN_PER_G_LIVER = 52.5  # mg protein per gram liver
_LIVER_WEIGHT_G = 1500.0  # grams (human)
_TOTAL_MICROSOMAL_PROTEIN_MG = _MICROSOMAL_PROTEIN_PER_G_LIVER * _LIVER_WEIGHT_G  # 78750 mg


@dataclass(frozen=True)
class MicrosomalStabilityResult:
    """Result of microsomal stability prediction."""

    drug_name: str
    t_half_in_vitro_min: float
    protein_conc_mg_per_mL: float
    volume_incubation_uL: float
    fu_mic: float
    clint_in_vitro_uL_per_min_per_mg: float
    clint_liver_L_per_h: float
    cl_hepatic_L_per_h: float
    hepatic_extraction_ratio: float
    f_oral_predicted: float
    metabolic_stability: str  # "high" t½>60min, "moderate" 30-60min, "low" <30min
    drug_class: str  # "low_clearance" E<0.3, "intermediate" 0.3-0.7, "high_clearance" >0.7
    notes: str


def predict_microsomal_stability(
    drug_name: str,
    t_half_in_vitro_min: float,
    protein_conc_mg_per_mL: float = 0.5,
    volume_incubation_uL: float = 100.0,
    fu_mic: float = 1.0,
    q_hepatic_L_per_h: float = 90.0,
) -> MicrosomalStabilityResult:
    """Predict hepatic clearance and oral bioavailability from microsomal stability.

    Parameters
    ----------
    drug_name:              compound identifier
    t_half_in_vitro_min:    microsomal half-life from stability assay [min]
    protein_conc_mg_per_mL: microsomal protein concentration in incubation [mg/mL]
    volume_incubation_uL:   incubation volume [µL]
    fu_mic:                 unbound fraction in microsomes [0, 1]
    q_hepatic_L_per_h:      hepatic blood flow [L/h]

    Returns
    -------
    MicrosomalStabilityResult
    """
    # --- validation ---
    if t_half_in_vitro_min <= 0:
        raise ValueError("t_half_in_vitro_min must be > 0")
    if protein_conc_mg_per_mL <= 0:
        raise ValueError("protein_conc_mg_per_mL must be > 0")
    if volume_incubation_uL <= 0:
        raise ValueError("volume_incubation_uL must be > 0")
    if not (0 < fu_mic <= 1.0):
        raise ValueError("fu_mic must be in (0, 1]")
    if q_hepatic_L_per_h <= 0:
        raise ValueError("q_hepatic_L_per_h must be > 0")

    # Step 1: Degradation rate constant from in vitro t½
    k_deg_per_min = math.log(2) / t_half_in_vitro_min  # [1/min]

    # Step 2: In vitro CLint
    # CLint_in_vitro = k_deg * V_incubation / protein_conc  [µL/min/mg protein]
    # V_incubation in µL, protein_conc in mg/mL → convert to mg/µL for consistent units
    protein_conc_mg_per_uL = protein_conc_mg_per_mL / 1000.0  # [mg/µL]
    clint_in_vitro = (
        k_deg_per_min * volume_incubation_uL / (protein_conc_mg_per_uL * volume_incubation_uL)
    )
    # Simplifies to: clint_in_vitro = k_deg / protein_conc_mg_per_uL [µL/min/mg]
    # = k_deg * 1000 / protein_conc_mg_per_mL
    clint_in_vitro = k_deg_per_min * 1000.0 / protein_conc_mg_per_mL  # [µL/min/mg protein]

    # Step 3: Scale to whole liver
    # CLint_liver [µL/min] = CLint_in_vitro [µL/min/mg] * total_protein [mg]
    clint_liver_uL_per_min = clint_in_vitro * _TOTAL_MICROSOMAL_PROTEIN_MG
    # Convert to L/h: ×60 min/h ÷ 1e6 µL/L
    clint_liver_L_per_h = clint_liver_uL_per_min * 60.0 / 1.0e6

    # Step 4: Well-stirred liver model
    # CLhep = Q_h * fu_mic * CLint / (Q_h + fu_mic * CLint)
    fu_clint = fu_mic * clint_liver_L_per_h
    cl_hepatic = q_hepatic_L_per_h * fu_clint / (q_hepatic_L_per_h + fu_clint)

    # Step 5: Hepatic extraction ratio and oral bioavailability
    hepatic_extraction_ratio = cl_hepatic / q_hepatic_L_per_h
    # Clamp to [0, 1] due to floating point
    hepatic_extraction_ratio = max(0.0, min(1.0, hepatic_extraction_ratio))
    f_hepatic = 1.0 - hepatic_extraction_ratio
    # Simplified: assume complete gut absorption
    f_oral_predicted = max(0.0, min(1.0, f_hepatic))

    # Step 6: Classification
    if t_half_in_vitro_min > 60.0:
        metabolic_stability = "high"
    elif t_half_in_vitro_min >= 30.0:
        metabolic_stability = "moderate"
    else:
        metabolic_stability = "low"

    if hepatic_extraction_ratio < 0.3:
        drug_class = "low_clearance"
    elif hepatic_extraction_ratio <= 0.7:
        drug_class = "intermediate"
    else:
        drug_class = "high_clearance"

    notes = (
        f"In vitro t½={t_half_in_vitro_min:.1f} min; "
        f"CLint_iv={clint_in_vitro:.2f} µL/min/mg; "
        f"CLhep={cl_hepatic:.2f} L/h; "
        f"E_h={hepatic_extraction_ratio:.3f}; "
        f"F_oral={f_oral_predicted:.3f}; "
        f"stability={metabolic_stability}; class={drug_class}."
    )

    return MicrosomalStabilityResult(
        drug_name=drug_name,
        t_half_in_vitro_min=t_half_in_vitro_min,
        protein_conc_mg_per_mL=protein_conc_mg_per_mL,
        volume_incubation_uL=volume_incubation_uL,
        fu_mic=fu_mic,
        clint_in_vitro_uL_per_min_per_mg=clint_in_vitro,
        clint_liver_L_per_h=clint_liver_L_per_h,
        cl_hepatic_L_per_h=cl_hepatic,
        hepatic_extraction_ratio=hepatic_extraction_ratio,
        f_oral_predicted=f_oral_predicted,
        metabolic_stability=metabolic_stability,
        drug_class=drug_class,
        notes=notes,
    )


def screen_compounds(compounds: list) -> list:
    """Screen a list of compounds by microsomal stability.

    Parameters
    ----------
    compounds:  list of dicts with keys:
                  "name" (str), "t_half_min" (float), "fu_mic" (float, optional)

    Returns
    -------
    list of MicrosomalStabilityResult sorted by cl_hepatic_L_per_h ascending
    (most stable = lowest hepatic clearance = first)
    """
    results = []
    for compound in compounds:
        name = compound["name"]
        t_half = compound["t_half_min"]
        fu_mic = compound.get("fu_mic", 1.0)
        res = predict_microsomal_stability(
            drug_name=name,
            t_half_in_vitro_min=t_half,
            fu_mic=fu_mic,
        )
        results.append(res)

    # Sort ascending by hepatic clearance (most stable first)
    results.sort(key=lambda r: r.cl_hepatic_L_per_h)
    return results
