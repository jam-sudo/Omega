"""Phase 737 — Drug Crystallization Risk Predictor.

Predicts crystallization risk for IV and oral formulations based on
physicochemical properties. High crystallization risk indicates the drug
may precipitate upon dilution, mixing, or in GI fluids.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrystallizationRiskResult:
    """Result of crystallization risk prediction."""

    compound_name: str
    logp: float
    mw_da: float
    tm_celsius: float
    logs_mg_L: float
    risk_score: float
    risk_level: str
    supersaturation_ratio: float
    crystallization_likely: bool
    recommended_excipients: str
    notes: str


def predict_crystallization_risk(
    compound_name: str,
    logp: float = 2.0,
    mw_da: float = 300.0,
    tm_celsius: float = 150.0,
    logs_mg_L: float = 10.0,
    target_conc_mg_L: float = 1.0,
) -> CrystallizationRiskResult:
    """Predict crystallization risk based on physicochemical properties.

    Args:
        compound_name: Name of the compound.
        logp: Octanol-water partition coefficient (log scale).
        mw_da: Molecular weight in Daltons.
        tm_celsius: Melting point in Celsius. Higher Tm indicates stronger
            crystal lattice energy, harder to dissolve.
        logs_mg_L: Aqueous solubility in mg/L (not log scale). Low values
            indicate poor solubility.
        target_conc_mg_L: Target concentration in the formulation or GI fluid
            (mg/L). Used to compute supersaturation ratio.

    Returns:
        CrystallizationRiskResult with risk score, level, and recommendations.
    """
    base_score = 0

    # logP contribution: lipophilic drugs are poorly water-soluble
    if logp > 4:
        base_score += 30
    elif logp > 2:
        base_score += 15

    # Melting point contribution: high Tm → high crystal lattice energy
    if tm_celsius > 200:
        base_score += 25
    elif tm_celsius > 150:
        base_score += 15

    # Solubility contribution: low solubility → high crystallization risk
    if logs_mg_L < 0.1:
        base_score += 40
    elif logs_mg_L < 1.0:
        base_score += 20

    # Molecular weight contribution: large molecules have poor dissolution kinetics
    if mw_da > 500:
        base_score += 10

    risk_score = float(min(100, base_score))

    # Determine risk level
    if risk_score < 30:
        risk_level = "low"
    elif risk_score <= 60:
        risk_level = "moderate"
    else:
        risk_level = "high"

    # Supersaturation ratio: target concentration / solubility
    # >1 means drug exceeds solubility at target concentration
    if logs_mg_L > 0.0:
        supersaturation_ratio = target_conc_mg_L / logs_mg_L
    else:
        supersaturation_ratio = float("inf") if target_conc_mg_L > 0 else 0.0

    # Crystallization is likely when supersaturated AND risk is not low
    crystallization_likely = supersaturation_ratio > 1.0 and risk_level != "low"

    # Recommend excipients based on risk level
    if risk_level == "high":
        recommended_excipients = "HPC+SDS"
    elif risk_level == "moderate":
        recommended_excipients = "PVP K30"
    else:
        recommended_excipients = "none"

    # Build notes
    notes_parts: list[str] = []
    if logp > 4:
        notes_parts.append("High logP indicates significant hydrophobicity")
    if tm_celsius > 200:
        notes_parts.append("High melting point suggests strong crystal lattice energy")
    if logs_mg_L < 0.1:
        notes_parts.append("Very low solubility (<0.1 mg/L); crystallization highly probable")
    elif logs_mg_L < 1.0:
        notes_parts.append("Low solubility (<1 mg/L); consider solubilizing excipients")
    if supersaturation_ratio > 1.0:
        notes_parts.append(
            f"Supersaturation ratio {supersaturation_ratio:.2f} exceeds unity; "
            "precipitation risk present"
        )
    if mw_da > 500:
        notes_parts.append("High MW (>500 Da) may slow dissolution kinetics")
    if not notes_parts:
        notes_parts.append("Favorable physicochemical profile; crystallization risk is low")

    notes = "; ".join(notes_parts)

    return CrystallizationRiskResult(
        compound_name=compound_name,
        logp=logp,
        mw_da=mw_da,
        tm_celsius=tm_celsius,
        logs_mg_L=logs_mg_L,
        risk_score=risk_score,
        risk_level=risk_level,
        supersaturation_ratio=supersaturation_ratio,
        crystallization_likely=crystallization_likely,
        recommended_excipients=recommended_excipients,
        notes=notes,
    )


def screen_crystallization_risk(
    compounds: list[dict],
) -> list[CrystallizationRiskResult]:
    """Screen a list of compounds for crystallization risk.

    Args:
        compounds: List of dicts with keys matching predict_crystallization_risk
            parameters (compound_name required; others optional).

    Returns:
        List of CrystallizationRiskResult sorted by risk_score descending
        (highest risk first).
    """
    results: list[CrystallizationRiskResult] = []
    for comp in compounds:
        result = predict_crystallization_risk(
            compound_name=comp["compound_name"],
            logp=comp.get("logp", 2.0),
            mw_da=comp.get("mw_da", 300.0),
            tm_celsius=comp.get("tm_celsius", 150.0),
            logs_mg_L=comp.get("logs_mg_L", 10.0),
            target_conc_mg_L=comp.get("target_conc_mg_L", 1.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.risk_score, reverse=True)
    return results
