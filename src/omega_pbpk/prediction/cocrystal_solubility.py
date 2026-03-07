"""Pharmaceutical cocrystal solubility prediction.

Predicts solubility enhancement when an API forms a cocrystal with a coformer
via non-covalent interactions. Uses empirical enhancement factors and Ksp-based
eutectic point calculations.
"""

from __future__ import annotations

from dataclasses import dataclass

# Coformer database: name -> (enhancement_factor, ksp_factor)
_COFORMER_DB: dict[str, tuple[float, float]] = {
    "saccharin": (3.0, 0.3),
    "nicotinamide": (5.0, 0.5),
    "citric_acid": (8.0, 0.7),
    "caffeine": (4.0, 0.4),
    "glutaric_acid": (6.0, 0.6),
    "oxalic_acid": (2.0, 0.2),
    "succinic_acid": (4.0, 0.4),
    "malonic_acid": (3.0, 0.3),
}

_VALID_STOICHIOMETRIES = {"1:1", "1:2", "2:1"}


@dataclass(frozen=True)
class CocrystalSolubilityResult:
    """Result of cocrystal solubility prediction."""

    drug_name: str
    coformer_name: str
    stoichiometry: str
    drug_solubility_mg_mL: float
    cocrystal_solubility_mg_mL: float
    solubility_enhancement_ratio: float
    eutectic_point_fraction: float
    stability_constant: float
    phase_solubility_type: str
    supersaturation_risk: bool
    recommendation: str
    notes: str


def predict_cocrystal_solubility(
    drug_name: str,
    coformer_name: str,
    drug_logp: float = 2.0,
    drug_pka: float = 7.0,
    stoichiometry: str = "1:1",
) -> CocrystalSolubilityResult:
    """Predict solubility of a pharmaceutical cocrystal.

    Args:
        drug_name: Name of the API.
        coformer_name: Name of the coformer (must be in the supported list).
        drug_logp: Partition coefficient (log P) of the drug.
        drug_pka: pKa of the drug (used for future extensions).
        stoichiometry: Drug:coformer molar ratio — "1:1", "1:2", or "2:1".

    Returns:
        CocrystalSolubilityResult with solubility and stability metrics.

    Raises:
        ValueError: If coformer_name or stoichiometry is invalid.
    """
    if coformer_name not in _COFORMER_DB:
        supported = ", ".join(sorted(_COFORMER_DB))
        raise ValueError(f"Unsupported coformer '{coformer_name}'. Supported: {supported}")
    if stoichiometry not in _VALID_STOICHIOMETRIES:
        raise ValueError(
            f"Invalid stoichiometry '{stoichiometry}'. Must be one of: "
            f"{sorted(_VALID_STOICHIOMETRIES)}"
        )

    # Step 2: Drug intrinsic solubility from logP
    drug_solubility = 10 ** (-0.7 * drug_logp + 0.44)  # mg/mL

    # Step 3: Get base enhancement factor and Ksp factor
    base_enhancement, ksp_factor = _COFORMER_DB[coformer_name]

    # Step 4: Stoichiometry modifier
    if stoichiometry == "1:2":
        enhancement_factor = base_enhancement * 0.7
    elif stoichiometry == "2:1":
        enhancement_factor = base_enhancement * 1.3
    else:
        enhancement_factor = base_enhancement

    # Step 5-6: Cocrystal solubility and ratio
    cocrystal_solubility = drug_solubility * enhancement_factor
    solubility_enhancement_ratio = cocrystal_solubility / drug_solubility  # = enhancement_factor

    # Step 7: Eutectic point fraction
    eutectic_point_fraction = 1.0 / (1.0 + enhancement_factor)

    # Step 8: Stability constant
    stability_constant = ksp_factor * (enhancement_factor**2)

    # Step 9: Phase solubility type
    if enhancement_factor > 5.0 and ksp_factor > 0.4:
        phase_solubility_type = "type_A"
    else:
        phase_solubility_type = "type_B"

    # Step 10: Supersaturation risk
    supersaturation_risk = solubility_enhancement_ratio > 10.0

    # Step 11: Recommendation
    if enhancement_factor >= 6.0:
        recommendation = "recommended"
    elif enhancement_factor >= 4.0:
        recommendation = "consider"
    else:
        recommendation = "limited benefit"

    notes = (
        f"Drug logP={drug_logp:.2f}, pKa={drug_pka:.2f}. "
        f"Coformer: {coformer_name} (stoichiometry {stoichiometry}). "
        f"Enhancement factor: {enhancement_factor:.2f}x. "
        f"Phase type: {phase_solubility_type}."
    )

    return CocrystalSolubilityResult(
        drug_name=drug_name,
        coformer_name=coformer_name,
        stoichiometry=stoichiometry,
        drug_solubility_mg_mL=drug_solubility,
        cocrystal_solubility_mg_mL=cocrystal_solubility,
        solubility_enhancement_ratio=solubility_enhancement_ratio,
        eutectic_point_fraction=eutectic_point_fraction,
        stability_constant=stability_constant,
        phase_solubility_type=phase_solubility_type,
        supersaturation_risk=supersaturation_risk,
        recommendation=recommendation,
        notes=notes,
    )


def screen_coformers(
    drug_name: str,
    drug_logp: float,
    drug_pka: float,
    coformers: list | None = None,
) -> list[CocrystalSolubilityResult]:
    """Screen multiple coformers for a given drug, sorted by enhancement ratio (descending).

    Args:
        drug_name: Name of the API.
        drug_logp: Partition coefficient (log P) of the drug.
        drug_pka: pKa of the drug.
        coformers: List of coformer names to screen. If None, screens all supported coformers.

    Returns:
        List of CocrystalSolubilityResult sorted by solubility_enhancement_ratio descending.
    """
    if coformers is None:
        coformers = list(_COFORMER_DB.keys())

    results = []
    for coformer in coformers:
        result = predict_cocrystal_solubility(
            drug_name=drug_name,
            coformer_name=coformer,
            drug_logp=drug_logp,
            drug_pka=drug_pka,
            stoichiometry="1:1",
        )
        results.append(result)

    results.sort(key=lambda r: r.solubility_enhancement_ratio, reverse=True)
    return results
