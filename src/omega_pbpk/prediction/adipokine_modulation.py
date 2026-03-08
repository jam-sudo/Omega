"""Phase 918 — Drug Adipokine Modulation Score.

Predicts drug effects on adipokine secretion (leptin, adiponectin, TNF-alpha)
from adipose tissue — relevant for metabolic drugs, antidiabetics, anti-obesity drugs.
"""

from dataclasses import dataclass

_VALID_DRUG_CLASSES = {"antidiabetic", "statin", "NSAID", "antipsychotic", "other"}

# Drug class effect profiles: (leptin, adiponectin, tnf_alpha, metabolic_modifier)
_CLASS_EFFECTS = {
    "antidiabetic": ("decrease", "increase", "decrease", 0.7),
    "statin": ("neutral", "increase", "decrease", 0.85),
    "NSAID": ("neutral", "neutral", "decrease", 0.9),
    "antipsychotic": ("increase", "decrease", "increase", 1.4),
    "other": ("neutral", "neutral", "neutral", 1.0),
}


@dataclass(frozen=True)
class AdipokineModulationResult:
    """Result of adipokine modulation prediction for a drug."""

    drug_name: str
    logp: float
    mw_Da: float
    drug_class: str
    kp_adipose_estimate: float
    leptin_effect_direction: str
    adiponectin_effect_direction: str
    tnf_alpha_effect_direction: str
    metabolic_syndrome_risk_modifier: float
    adipose_accumulation_index: float
    notes: str


def predict_adipokine_modulation(
    drug_name: str,
    logp: float = 2.0,
    mw_Da: float = 400.0,
    drug_class: str = "other",
) -> AdipokineModulationResult:
    """Predict drug effects on adipokine modulation.

    Args:
        drug_name: Name of the drug.
        logp: Octanol-water partition coefficient.
        mw_Da: Molecular weight in Daltons.
        drug_class: One of "antidiabetic", "statin", "NSAID", "antipsychotic", "other".

    Returns:
        AdipokineModulationResult with adipokine effect predictions.

    Raises:
        ValueError: If drug_class is not valid.
    """
    if drug_class not in _VALID_DRUG_CLASSES:
        raise ValueError(
            f"drug_class must be one of {sorted(_VALID_DRUG_CLASSES)}, got '{drug_class}'"
        )

    # Adipose:plasma Kp estimate — lipophilic drugs accumulate in fat
    kp_adipose = 10.0 ** (logp * 0.3)
    kp_adipose = max(0.1, min(50.0, kp_adipose))

    # Drug class effects
    leptin, adiponectin, tnf_alpha, metabolic_modifier = _CLASS_EFFECTS[drug_class]

    # Adipose accumulation index (0-100)
    accumulation_index = kp_adipose * 5.0 + logp * 10.0
    accumulation_index = max(0.0, min(100.0, accumulation_index))

    # Notes based on priority order
    if drug_class == "antipsychotic":
        notes = (
            "Antipsychotic class: risk of metabolic syndrome — monitor BMI, fasting glucose, lipids"
        )
    elif drug_class == "antidiabetic":
        notes = "Antidiabetic: favorable adipokine profile — may improve metabolic syndrome"
    elif metabolic_modifier > 1.2:
        notes = "Adverse adipokine modulation — metabolic monitoring required"
    else:
        notes = "Neutral or favorable metabolic profile"

    return AdipokineModulationResult(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        drug_class=drug_class,
        kp_adipose_estimate=kp_adipose,
        leptin_effect_direction=leptin,
        adiponectin_effect_direction=adiponectin,
        tnf_alpha_effect_direction=tnf_alpha,
        metabolic_syndrome_risk_modifier=metabolic_modifier,
        adipose_accumulation_index=accumulation_index,
        notes=notes,
    )


def screen_metabolic_risk(candidates: list) -> list:
    """Screen drug candidates for metabolic syndrome risk via adipokine modulation.

    Args:
        candidates: List of dicts with keys:
            - "name": drug name (required)
            - "logp": logP value (required)
            - "mw_Da": molecular weight, default 400
            - "drug_class": drug class, default "other"

    Returns:
        List of AdipokineModulationResult sorted by metabolic_syndrome_risk_modifier
        descending (highest risk first).
    """
    results = []
    for cand in candidates:
        result = predict_adipokine_modulation(
            drug_name=cand["name"],
            logp=cand["logp"],
            mw_Da=cand.get("mw_Da", 400.0),
            drug_class=cand.get("drug_class", "other"),
        )
        results.append(result)
    results.sort(key=lambda r: r.metabolic_syndrome_risk_modifier, reverse=True)
    return results
