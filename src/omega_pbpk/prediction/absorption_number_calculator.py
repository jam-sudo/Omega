"""Phase 576 - Oral Drug Absorption Number Calculator.

Calculates the absorption number (An) and dose number (Do) from BCS theory,
predicting oral absorption.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AbsorptionNumberResult:
    """Result of absorption number calculation."""

    drug_name: str
    dose_mg: float
    solubility_mg_mL: float
    peff_cm_per_s: float
    dose_number: float
    absorption_number: float
    fraction_absorbed: float
    fa_corrected: float
    bcs_class: str
    absorption_limited_by: str
    notes: str


def calculate_absorption_number(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    peff_cm_per_s: float,
    t_transit_min: float = 199.0,
    r_intestine_cm: float = 1.75,
) -> AbsorptionNumberResult:
    """Calculate absorption number, dose number, and predicted fraction absorbed.

    Uses BCS (Biopharmaceutics Classification System) theory.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be positive")
    if peff_cm_per_s <= 0:
        raise ValueError("peff_cm_per_s must be positive")

    t_transit_s = t_transit_min * 60.0
    an = peff_cm_per_s * 2.0 * t_transit_s / r_intestine_cm
    fa_perm_limited = max(0.0, min(1.0, 1.0 - math.exp(-2.0 * an)))

    do = dose_mg / (250.0 * solubility_mg_mL)

    if do <= 1.0:
        fa_corrected = fa_perm_limited
    else:
        fa_corrected = min(fa_perm_limited, 1.0 / do)

    # BCS classification
    high_perm = an > 1.0
    high_sol = do <= 1.0

    if high_perm and high_sol:
        bcs_class = "I"
        absorption_limited_by = "neither"
    elif high_perm and not high_sol:
        bcs_class = "II"
        absorption_limited_by = "solubility"
    elif not high_perm and high_sol:
        bcs_class = "III"
        absorption_limited_by = "permeability"
    else:
        bcs_class = "IV"
        absorption_limited_by = "both"

    notes = f"BCS Class {bcs_class}: An={an:.3f}, Do={do:.3f}"

    return AbsorptionNumberResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        solubility_mg_mL=solubility_mg_mL,
        peff_cm_per_s=peff_cm_per_s,
        dose_number=do,
        absorption_number=an,
        fraction_absorbed=fa_perm_limited,
        fa_corrected=fa_corrected,
        bcs_class=bcs_class,
        absorption_limited_by=absorption_limited_by,
        notes=notes,
    )


def screen_compounds(compounds: list) -> list:
    """Screen multiple compounds and sort by fa_corrected descending.

    Args:
        compounds: list of dicts with keys drug_name, dose_mg,
                   solubility_mg_mL, peff_cm_per_s.

    Returns:
        List of AbsorptionNumberResult sorted by fa_corrected descending.
    """
    results = []
    for comp in compounds:
        result = calculate_absorption_number(
            drug_name=comp["drug_name"],
            dose_mg=comp["dose_mg"],
            solubility_mg_mL=comp["solubility_mg_mL"],
            peff_cm_per_s=comp["peff_cm_per_s"],
        )
        results.append(result)
    results.sort(key=lambda r: r.fa_corrected, reverse=True)
    return results
