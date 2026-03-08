"""Phase 551 - Drug Crystallization Risk in Formulation.

Predicts risk of drug crystallization during formulation or in vivo
for amorphous solid dispersions and supersaturated solutions.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CrystallizationRiskResult:
    drug_name: str
    logP: float
    mw_Da: float
    melting_point_C: float
    dose_mg: float
    solubility_amorphous_mg_mL: float
    supersaturation_ratio: float
    crystallization_rate_constant: float
    induction_time_h: float
    risk_score: float
    risk_class: str
    formulation_recommendation: str
    notes: str


def predict_crystallization_risk(
    drug_name: str,
    logP: float,
    mw_Da: float,
    melting_point_C: float,
    dose_mg: float = 100.0,
    crystalline_solubility_mg_mL: float = 0.01,
) -> CrystallizationRiskResult:
    """Predict crystallization risk for an amorphous drug formulation."""
    # Supersaturation ratio: SSR = exp(0.05 * (Tm - 25)), clamped [1, 100]
    ssr_raw = math.exp(0.05 * (melting_point_C - 25.0))
    ssr = max(1.0, min(100.0, ssr_raw))

    solubility_amorphous = crystalline_solubility_mg_mL * ssr

    # Crystallization rate constant
    k_cryst_raw = 0.05 * max(1.0, logP)
    k_cryst = max(0.01, min(2.0, k_cryst_raw))

    # Induction time
    t_induction_raw = 1.0 / (k_cryst * max(1.0, ssr - 1.0))
    t_induction = max(0.1, min(48.0, t_induction_raw))

    # Risk score
    risk_raw = 20.0 * logP + 0.5 * (melting_point_C - 100.0) / 10.0 + 30.0 * (ssr - 1.0) / 99.0
    risk_score = max(0.0, min(100.0, risk_raw))

    # Risk classification
    if risk_score > 60.0:
        risk_class = "high"
        recommendation = "Use polymer matrix ASD (HPMC-AS) with surfactant"
    elif risk_score > 30.0:
        risk_class = "moderate"
        recommendation = "Consider amorphous dispersion or co-solvent"
    else:
        risk_class = "low"
        recommendation = "Standard crystalline formulation acceptable"

    notes = (
        f"SSR={ssr:.2f}, k_cryst={k_cryst:.4f}/h, induction={t_induction:.2f}h, dose={dose_mg}mg"
    )

    return CrystallizationRiskResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        melting_point_C=melting_point_C,
        dose_mg=dose_mg,
        solubility_amorphous_mg_mL=solubility_amorphous,
        supersaturation_ratio=ssr,
        crystallization_rate_constant=k_cryst,
        induction_time_h=t_induction,
        risk_score=risk_score,
        risk_class=risk_class,
        formulation_recommendation=recommendation,
        notes=notes,
    )


def screen_crystallization(compounds: list) -> list:
    """Screen multiple compounds for crystallization risk, sorted by risk_score descending."""
    results = [predict_crystallization_risk(**comp) for comp in compounds]
    results.sort(key=lambda r: r.risk_score, reverse=True)
    return results
