"""Phase 333 — Drug Crystallization Risk Predictor.

Predicts crystallization risk of a drug in biological fluids based on
physicochemical properties. Uses supersaturation ratio and amorphous
contribution to estimate risk class and formulation recommendation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CrystallizationRiskResult:
    """Result of crystallization risk assessment."""

    drug_name: str
    dose_mg: float
    solubility_mg_mL: float
    solubility_at_ph_mg_mL: float
    supersaturation_ratio: float
    amorphous_factor: float
    crystallization_rate_per_h: float
    time_to_crystallization_h: float
    risk_score: float
    risk_class: str
    formulation_recommendation: str
    notes: str


_VALID_DRUG_TYPES = {"acid", "base", "neutral"}

_FORMULATION_MAP = {
    "low": "No special formulation required",
    "moderate": "Consider solubilizing excipients or amorphous form",
    "high": "Use ASD (amorphous solid dispersion) or nano-formulation",
}


def predict_crystallization_risk(
    drug_name: str,
    solubility_mg_mL: float,
    logP: float,
    pKa: float,
    drug_type: str,
    dose_mg: float,
    volume_of_distribution_mL: float,
) -> CrystallizationRiskResult:
    """Predict crystallization risk in biological fluids.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    solubility_mg_mL:
        Thermodynamic (intrinsic) aqueous solubility in mg/mL.
    logP:
        Logarithm of octanol-water partition coefficient.
    pKa:
        Acid dissociation constant.
    drug_type:
        Ionisation type: "acid", "base", or "neutral".
    dose_mg:
        Administered dose in milligrams.
    volume_of_distribution_mL:
        Local volume used to estimate concentration in mL.

    Returns
    -------
    CrystallizationRiskResult
        Frozen dataclass containing all risk metrics.
    """
    # --- Input validation ---
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if volume_of_distribution_mL <= 0:
        raise ValueError(f"volume_of_distribution_mL must be > 0, got {volume_of_distribution_mL}")
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {_VALID_DRUG_TYPES}, got {drug_type!r}")

    # --- pH-adjusted solubility at intestinal pH 6.8 ---
    intestinal_ph = 6.8
    if drug_type == "acid":
        sol_at_ph = solubility_mg_mL * (1.0 + 10.0 ** (intestinal_ph - pKa))
    elif drug_type == "base":
        sol_at_ph = solubility_mg_mL * (1.0 + 10.0 ** (pKa - intestinal_ph))
    else:  # neutral
        sol_at_ph = solubility_mg_mL

    # --- Estimated local concentration ---
    local_conc_mg_mL = dose_mg / volume_of_distribution_mL

    # --- Supersaturation ratio (SR) against pH-adjusted solubility ---
    sr = local_conc_mg_mL / sol_at_ph

    # --- Amorphous contribution factor ---
    amorphous_factor = max(1.0, 10.0 ** (logP / 4.0))

    # --- Crystal growth rate ---
    k_crystal = 0.01 * sr * amorphous_factor  # 1/h

    # --- Time to crystallization ---
    if sr > 1.0:
        t_cryst_h = 1.0 / k_crystal
    else:
        t_cryst_h = math.inf

    # --- Risk score (0-100) ---
    risk_score = min(100.0, sr * 20.0)

    # --- Risk class ---
    if sr < 1.0:
        risk_class = "low"
    elif sr < 3.0:
        risk_class = "moderate"
    else:
        risk_class = "high"

    formulation_recommendation = _FORMULATION_MAP[risk_class]

    notes_parts = [
        f"SR={sr:.3f}",
        f"local_conc={local_conc_mg_mL:.4f} mg/mL",
        f"sol_at_pH6.8={sol_at_ph:.4f} mg/mL",
        f"amorphous_factor={amorphous_factor:.3f}",
    ]
    if sr <= 1.0:
        notes_parts.append("No crystallization expected under these conditions.")
    else:
        notes_parts.append(f"Estimated crystallization onset ~{t_cryst_h:.2f} h post-dose.")
    notes = "; ".join(notes_parts)

    return CrystallizationRiskResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        solubility_mg_mL=solubility_mg_mL,
        solubility_at_ph_mg_mL=sol_at_ph,
        supersaturation_ratio=sr,
        amorphous_factor=amorphous_factor,
        crystallization_rate_per_h=k_crystal,
        time_to_crystallization_h=t_cryst_h,
        risk_score=risk_score,
        risk_class=risk_class,
        formulation_recommendation=formulation_recommendation,
        notes=notes,
    )
