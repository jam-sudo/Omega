"""Phase 245: Pharmaceutical cocrystal design for improving drug properties."""

from dataclasses import dataclass

# Coformer lookup table: name -> (delta_pka_threshold, solubility_modifier, stability_modifier, coformer_pka)
_COFORMER_TABLE = {
    "nicotinamide": (3.0, 2.5, 1.2, 3.35),
    "saccharin": (2.0, 1.8, 1.0, 1.60),
    "caffeine": (2.5, 2.0, 1.1, 0.5),
    "glutaric_acid": (2.0, 3.0, 0.9, 4.34),
    "urea": (1.5, 1.5, 1.3, 0.1),
}


@dataclass(frozen=True)
class CocrystalResult:
    drug_name: str
    coformer: str
    drug_pka: float
    coformer_pka: float
    delta_pka: float
    cocrystal_feasibility: str
    predicted_solubility_mg_mL: float
    stability_score: float
    melting_point_reduction_C: float
    bioavailability_improvement_pct: float
    recommended: bool
    notes: str


def design_cocrystal(
    drug_name: str,
    drug_pka: float,
    baseline_solubility_mg_mL: float,
    coformer: str,
) -> CocrystalResult:
    """Design a cocrystal for the given drug and coformer."""
    if not (0.0 <= drug_pka <= 14.0):
        raise ValueError(f"drug_pka must be in [0, 14], got {drug_pka}")
    if baseline_solubility_mg_mL <= 0:
        raise ValueError(f"baseline_solubility_mg_mL must be > 0, got {baseline_solubility_mg_mL}")
    if coformer not in _COFORMER_TABLE:
        raise ValueError(
            f"Unknown coformer '{coformer}'. Valid options: {list(_COFORMER_TABLE.keys())}"
        )

    threshold, sol_mod, stab_mod, cf_pka = _COFORMER_TABLE[coformer]

    delta_pka = abs(drug_pka - cf_pka)
    feasibility = "high" if delta_pka > threshold else "low"

    predicted_solubility = baseline_solubility_mg_mL * sol_mod * (1.0 + 0.1 * delta_pka)

    raw_stability = stab_mod * 70.0 * (min(delta_pka, 5.0) / 5.0)
    stability_score = max(0.0, min(100.0, raw_stability))

    if feasibility == "high":
        melting_point_reduction_C = 5.0 * delta_pka * stab_mod
    else:
        melting_point_reduction_C = 0.0

    bioavailability_improvement_pct = (
        predicted_solubility / baseline_solubility_mg_mL - 1.0
    ) * 100.0

    recommended = (feasibility == "high") and (stability_score >= 50.0)

    notes_parts = [f"Coformer: {coformer}, feasibility={feasibility}"]
    if recommended:
        notes_parts.append("Recommended for further development.")
    else:
        if feasibility != "high":
            notes_parts.append("Low feasibility: delta_pKa below threshold.")
        if stability_score < 50.0:
            notes_parts.append("Stability score below threshold (50).")
    notes = " ".join(notes_parts)

    return CocrystalResult(
        drug_name=drug_name,
        coformer=coformer,
        drug_pka=drug_pka,
        coformer_pka=cf_pka,
        delta_pka=delta_pka,
        cocrystal_feasibility=feasibility,
        predicted_solubility_mg_mL=predicted_solubility,
        stability_score=stability_score,
        melting_point_reduction_C=melting_point_reduction_C,
        bioavailability_improvement_pct=bioavailability_improvement_pct,
        recommended=recommended,
        notes=notes,
    )


def screen_coformers(
    drug_name: str,
    drug_pka: float,
    baseline_solubility_mg_mL: float,
) -> list[CocrystalResult]:
    """Screen all 5 coformers and return results sorted by stability_score descending."""
    results = []
    for coformer in _COFORMER_TABLE:
        result = design_cocrystal(drug_name, drug_pka, baseline_solubility_mg_mL, coformer)
        results.append(result)
    results.sort(key=lambda r: r.stability_score, reverse=True)
    return results
