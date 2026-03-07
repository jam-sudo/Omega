"""Drug stability in plasma: esterase lability, hydrolysis prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PlasmaStabilityResult",
    "predict_plasma_stability",
    "screen_plasma_stability",
]


@dataclass(frozen=True)
class PlasmaStabilityResult:
    """Result of plasma stability prediction."""

    drug_name: str
    logp: float
    structural_alerts: list[str]
    hydrolysis_susceptibility: str
    plasma_t_half_h: float
    stability_at_37C_pct_1h: float
    stability_at_37C_pct_4h: float
    esterase_lability_score: float
    recommended_storage_temp_C: float
    in_vitro_handling_notes: str
    notes: str


def predict_plasma_stability(
    drug_name: str,
    logp: float = 2.0,
    has_ester: bool = False,
    has_amide: bool = False,
    has_lactone: bool = False,
    has_carbonate: bool = False,
    has_thioester: bool = False,
    mw_Da: float = 400.0,
    pka: float | None = None,
) -> PlasmaStabilityResult:
    """Predict plasma stability based on structural features.

    Returns a PlasmaStabilityResult with predicted t½, stability percentages,
    esterase lability score, and handling recommendations.
    """
    # Build structural alerts list
    structural_alerts: list[str] = []
    if has_ester:
        structural_alerts.append("ester")
    if has_amide:
        structural_alerts.append("amide")
    if has_lactone:
        structural_alerts.append("lactone")
    if has_carbonate:
        structural_alerts.append("carbonate")
    if has_thioester:
        structural_alerts.append("thioester")

    # Esterase lability score
    base = 10.0
    if has_ester:
        base += 35.0
    if has_thioester:
        base += 25.0
    if has_lactone:
        base += 20.0
    if has_carbonate:
        base += 15.0
    if has_amide:
        base += 5.0
    # logP modifier: hydrophobic esters more labile
    base += (logp - 2.0) * 3.0
    esterase_lability_score = max(0.0, min(100.0, base))

    # Plasma half-life prediction
    base_t_half = 100.0
    if has_ester:
        base_t_half = max(0.5, 20.0 / (1.0 + logp * 0.2))
    if has_thioester:
        base_t_half = min(base_t_half, 5.0)
    if has_lactone:
        base_t_half = min(base_t_half, 10.0)
    if has_carbonate:
        base_t_half = min(base_t_half, 15.0)
    if has_amide:
        base_t_half = min(base_t_half, 50.0)
    plasma_t_half_h = max(0.1, min(200.0, base_t_half))

    # Stability percentages at 37°C using first-order decay
    ln2 = math.log(2.0)
    k = ln2 / plasma_t_half_h
    stability_1h = 100.0 * math.exp(-k * 1.0)
    stability_4h = 100.0 * math.exp(-k * 4.0)

    # Hydrolysis susceptibility classification
    if esterase_lability_score > 60.0:
        hydrolysis_susceptibility = "high"
    elif esterase_lability_score > 30.0:
        hydrolysis_susceptibility = "moderate"
    else:
        hydrolysis_susceptibility = "low"

    # Recommended storage temperature
    if esterase_lability_score > 60.0:
        recommended_storage_temp_C = -80.0
    elif esterase_lability_score > 30.0:
        recommended_storage_temp_C = -20.0
    else:
        recommended_storage_temp_C = 4.0

    # In vitro handling notes
    if esterase_lability_score > 60.0:
        in_vitro_handling_notes = "Add esterase inhibitor (PMSF/NaF) to plasma. Use ice throughout."
    elif esterase_lability_score > 30.0:
        in_vitro_handling_notes = "Keep samples on ice. Process within 2h."
    else:
        in_vitro_handling_notes = "Standard handling acceptable."

    # Notes
    if hydrolysis_susceptibility == "high":
        notes = (
            "Extensive plasma hydrolysis expected — PK interpretation requires stabilized samples"
        )
    elif hydrolysis_susceptibility == "moderate":
        notes = "Moderate stability — monitor for hydrolysis during sample preparation"
    else:
        notes = "Good plasma stability"

    return PlasmaStabilityResult(
        drug_name=drug_name,
        logp=logp,
        structural_alerts=structural_alerts,
        hydrolysis_susceptibility=hydrolysis_susceptibility,
        plasma_t_half_h=plasma_t_half_h,
        stability_at_37C_pct_1h=stability_1h,
        stability_at_37C_pct_4h=stability_4h,
        esterase_lability_score=esterase_lability_score,
        recommended_storage_temp_C=recommended_storage_temp_C,
        in_vitro_handling_notes=in_vitro_handling_notes,
        notes=notes,
    )


def screen_plasma_stability(candidates: list[dict]) -> list[PlasmaStabilityResult]:
    """Screen a list of candidate compounds for plasma stability.

    Each dict must have "name" and "logp"; optionally any of the has_* flags.
    Returns results sorted by plasma_t_half_h descending (most stable first).
    """
    results = []
    for candidate in candidates:
        name = candidate["name"]
        logp = candidate["logp"]
        result = predict_plasma_stability(
            drug_name=name,
            logp=logp,
            has_ester=candidate.get("has_ester", False),
            has_amide=candidate.get("has_amide", False),
            has_lactone=candidate.get("has_lactone", False),
            has_carbonate=candidate.get("has_carbonate", False),
            has_thioester=candidate.get("has_thioester", False),
        )
        results.append(result)

    results.sort(key=lambda r: r.plasma_t_half_h, reverse=True)
    return results
