"""
Phase 416 — Co-crystal solubility enhancement prediction.

Co-crystals combine API with coformer to improve solubility.
Enhancement depends on ionization states and eutectic point.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CocrystalSolubilityResult:
    """Result of co-crystal solubility prediction."""

    drug_name: str
    coformer_name: str
    drug_pka: float
    coformer_pka: float
    drug_solubility_mg_mL: float
    cocrystal_solubility_mg_mL: float
    enhancement_ratio: float
    eutectic_ph: float
    solubility_class: str
    cocrystal_class: str
    dissolution_improvement: str
    notes: str


def predict_cocrystal_solubility(
    drug_name,
    coformer_name,
    drug_pka,
    coformer_pka,
    drug_solubility_mg_mL,
    coformer_solubility_mg_mL,
    ionization_type_drug,
    ionization_type_coformer,
    temperature_C=25.0,
):
    """
    Predict co-crystal solubility enhancement.

    Parameters
    ----------
    drug_name : str
    coformer_name : str
    drug_pka : float
    coformer_pka : float
    drug_solubility_mg_mL : float  -- intrinsic drug solubility
    coformer_solubility_mg_mL : float  -- coformer solubility
    ionization_type_drug : str  -- "acid", "base", or "neutral"
    ionization_type_coformer : str  -- "acid", "base", or "neutral"
    temperature_C : float  -- temperature in Celsius (4 to 80)

    Returns
    -------
    CocrystalSolubilityResult
    """
    if drug_solubility_mg_mL <= 0:
        raise ValueError(f"drug_solubility_mg_mL must be > 0, got {drug_solubility_mg_mL}")
    if coformer_solubility_mg_mL <= 0:
        raise ValueError(f"coformer_solubility_mg_mL must be > 0, got {coformer_solubility_mg_mL}")
    if temperature_C < 4 or temperature_C > 80:
        raise ValueError(f"temperature_C must be between 4 and 80, got {temperature_C}")

    sol_ratio = coformer_solubility_mg_mL / drug_solubility_mg_mL

    # Determine ionization type pairing
    drug_type = ionization_type_drug.lower()
    coformer_type = ionization_type_coformer.lower()

    is_salt_forming = (drug_type == "acid" and coformer_type == "base") or (
        drug_type == "base" and coformer_type == "acid"
    )
    is_same_type = (drug_type == "acid" and coformer_type == "acid") or (
        drug_type == "base" and coformer_type == "base"
    )

    if is_salt_forming:
        enhancement = (sol_ratio**0.5) * 2.0
    elif is_same_type:
        enhancement = sol_ratio**0.25
    else:
        # neutral coformer or mixed neutral cases
        enhancement = 1.2 + 0.1 * (sol_ratio**0.1)

    # Temperature correction
    if temperature_C > 25.0:
        enhancement = enhancement * (1.0 + 0.01 * (temperature_C - 25.0))

    # Cap enhancement
    if enhancement > 100.0:
        enhancement = 100.0
    if enhancement < 1.0:
        enhancement = 1.0

    cocrystal_solubility_mg_mL = drug_solubility_mg_mL * enhancement

    # Eutectic pH
    if drug_type == "acid":
        eutectic_ph = (drug_pka + coformer_pka) / 2.0
    else:
        eutectic_ph = coformer_pka - 0.5

    # Classify co-crystal by coformer pKa
    if coformer_pka < 4.0:
        cocrystal_class = "acidic_coformer"
    elif coformer_pka > 8.0:
        cocrystal_class = "basic_coformer"
    else:
        cocrystal_class = "neutral_coformer"

    # Dissolution improvement by enhancement ratio
    if enhancement < 1.5:
        dissolution_improvement = "none"
    elif enhancement < 5.0:
        dissolution_improvement = "moderate"
    elif enhancement < 20.0:
        dissolution_improvement = "significant"
    else:
        dissolution_improvement = "major"

    # Solubility class by cocrystal solubility
    if cocrystal_solubility_mg_mL < 0.1:
        solubility_class = "poor"
    elif cocrystal_solubility_mg_mL <= 1.0:
        solubility_class = "moderate"
    else:
        solubility_class = "good"

    notes = (
        f"Co-crystal enhancement ratio = {enhancement:.2f}x ({dissolution_improvement} class). "
        f"Coformer class: {cocrystal_class}."
    )

    return CocrystalSolubilityResult(
        drug_name=drug_name,
        coformer_name=coformer_name,
        drug_pka=drug_pka,
        coformer_pka=coformer_pka,
        drug_solubility_mg_mL=drug_solubility_mg_mL,
        cocrystal_solubility_mg_mL=cocrystal_solubility_mg_mL,
        enhancement_ratio=enhancement,
        eutectic_ph=eutectic_ph,
        solubility_class=solubility_class,
        cocrystal_class=cocrystal_class,
        dissolution_improvement=dissolution_improvement,
        notes=notes,
    )


def screen_coformers(
    drug_name,
    drug_pka,
    drug_solubility_mg_mL,
    ionization_type_drug,
    coformer_list,
):
    """
    Screen a list of coformers for co-crystal solubility enhancement.

    Parameters
    ----------
    drug_name : str
    drug_pka : float
    drug_solubility_mg_mL : float
    ionization_type_drug : str  -- "acid", "base", or "neutral"
    coformer_list : list of dict with keys:
        name, pka, solubility_mg_mL, ionization_type

    Returns
    -------
    list of CocrystalSolubilityResult sorted by enhancement_ratio descending
    """
    results = []
    for cf in coformer_list:
        result = predict_cocrystal_solubility(
            drug_name=drug_name,
            coformer_name=cf["name"],
            drug_pka=drug_pka,
            coformer_pka=cf["pka"],
            drug_solubility_mg_mL=drug_solubility_mg_mL,
            coformer_solubility_mg_mL=cf["solubility_mg_mL"],
            ionization_type_drug=ionization_type_drug,
            ionization_type_coformer=cf["ionization_type"],
        )
        results.append(result)

    results.sort(key=lambda r: r.enhancement_ratio, reverse=True)
    return results
