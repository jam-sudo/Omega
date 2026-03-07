"""Phase 872: Extended BCS Classification with BDDCS and Biowaiver Eligibility."""

from dataclasses import dataclass

# Physical constants for BCS
INTESTINAL_RADIUS_CM = 0.175
TRANSIT_TIME_H = 3.5
DOSE_VOLUME_ML = 250.0


@dataclass(frozen=True)
class BCSExtendedResult:
    drug_name: str
    dose_mg: float
    solubility_mg_mL: float
    peff_cm_s: float
    f_metabolized: float
    dose_number: float
    absorption_number: float
    bcs_class: str
    bddcs_class: str
    high_solubility: bool
    high_permeability: bool
    biowaiver_eligible: bool
    predicted_fa: float
    notes: str


def classify_bcs_extended(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    peff_cm_s: float,
    f_metabolized: float = 0.5,
    rapidly_dissolving: bool = True,
) -> BCSExtendedResult:
    """Classify drug using extended BCS with BDDCS and biowaiver eligibility."""
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if peff_cm_s < 0:
        raise ValueError("peff_cm_s must be >= 0")

    # Dose number: dimensionless ratio of dose to solubility in 250 mL
    dose_number = dose_mg / (solubility_mg_mL * DOSE_VOLUME_ML)

    # Absorption number: dimensionless ratio based on intestinal transit
    # An = Peff * t_transit_s / R_cm
    t_transit_s = TRANSIT_TIME_H * 3600.0
    absorption_number = peff_cm_s * t_transit_s / INTESTINAL_RADIUS_CM

    # BCS classification
    high_solubility = dose_number < 1.0
    high_permeability = absorption_number > 1.0

    if high_solubility and high_permeability:
        bcs_class = "I"
    elif (not high_solubility) and high_permeability:
        bcs_class = "II"
    elif high_solubility and (not high_permeability):
        bcs_class = "III"
    else:
        bcs_class = "IV"

    # BDDCS classification based on metabolism
    extensively_metabolized = f_metabolized >= 0.7
    if high_solubility and extensively_metabolized:
        bddcs_class = "1"
    elif (not high_solubility) and extensively_metabolized:
        bddcs_class = "2"
    elif high_solubility and (not extensively_metabolized):
        bddcs_class = "3"
    else:
        bddcs_class = "4"

    # Biowaiver eligibility (FDA 2017)
    if bcs_class == "I" and rapidly_dissolving:
        biowaiver_eligible = True
    elif bcs_class == "III" and rapidly_dissolving:
        biowaiver_eligible = True
    else:
        biowaiver_eligible = False

    # Predicted fraction absorbed (sigmoid approximation)
    if absorption_number + 1.0 == 0.0:
        predicted_fa = 0.0
    else:
        predicted_fa = absorption_number / (absorption_number + 1.0)

    # Clamp to [0, 1]
    if predicted_fa < 0.0:
        predicted_fa = 0.0
    if predicted_fa > 1.0:
        predicted_fa = 1.0

    # Notes
    notes_parts = [
        f"BCS Class {bcs_class}: {'high' if high_solubility else 'low'} solubility, "
        f"{'high' if high_permeability else 'low'} permeability.",
        f"BDDCS Class {bddcs_class} ({'extensive' if extensively_metabolized else 'poor'} metabolism).",
        f"Dose number: {dose_number:.3f}, Absorption number: {absorption_number:.3f}.",
    ]
    if biowaiver_eligible:
        notes_parts.append("Biowaiver eligible.")
    else:
        notes_parts.append("Not eligible for biowaiver.")

    notes = " ".join(notes_parts)

    return BCSExtendedResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        solubility_mg_mL=solubility_mg_mL,
        peff_cm_s=peff_cm_s,
        f_metabolized=f_metabolized,
        dose_number=dose_number,
        absorption_number=absorption_number,
        bcs_class=bcs_class,
        bddcs_class=bddcs_class,
        high_solubility=high_solubility,
        high_permeability=high_permeability,
        biowaiver_eligible=biowaiver_eligible,
        predicted_fa=predicted_fa,
        notes=notes,
    )


def screen_bcs_portfolio(compounds: list[dict]) -> dict:
    """Screen a portfolio of compounds, returning results grouped by BCS class."""
    class_I: list[BCSExtendedResult] = []
    class_II: list[BCSExtendedResult] = []
    class_III: list[BCSExtendedResult] = []
    class_IV: list[BCSExtendedResult] = []
    biowaiver_candidates: list[BCSExtendedResult] = []

    for c in compounds:
        result = classify_bcs_extended(
            drug_name=c["drug_name"],
            dose_mg=c["dose_mg"],
            solubility_mg_mL=c["solubility_mg_mL"],
            peff_cm_s=c["peff_cm_s"],
            f_metabolized=c.get("f_metabolized", 0.5),
            rapidly_dissolving=c.get("rapidly_dissolving", True),
        )
        if result.bcs_class == "I":
            class_I.append(result)
        elif result.bcs_class == "II":
            class_II.append(result)
        elif result.bcs_class == "III":
            class_III.append(result)
        else:
            class_IV.append(result)

        if result.biowaiver_eligible:
            biowaiver_candidates.append(result)

    return {
        "class_I": class_I,
        "class_II": class_II,
        "class_III": class_III,
        "class_IV": class_IV,
        "biowaiver_candidates": biowaiver_candidates,
    }
