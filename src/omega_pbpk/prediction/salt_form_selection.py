"""Phase 428: Pharmaceutical salt form selection and solubility enhancement prediction.

The pHmax model predicts optimal pH for maximum solubility of ionizable drugs.
Salt forms can dramatically improve solubility of ionizable drugs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

COUNTERIONS = {
    "hydrochloride": {
        "pka": -7,
        "type": "acid",
        "hygroscopicity": "moderate",
        "stability": "low",
        "solubility_factor": 3.0,
    },
    "sodium": {
        "pka": 15,
        "type": "base",
        "hygroscopicity": "moderate",
        "stability": "low",
        "solubility_factor": 4.0,
    },
    "potassium": {
        "pka": 15,
        "type": "base",
        "hygroscopicity": "low",
        "stability": "low",
        "solubility_factor": 3.5,
    },
    "acetate": {
        "pka": 4.75,
        "type": "acid",
        "hygroscopicity": "moderate",
        "stability": "moderate",
        "solubility_factor": 2.0,
    },
    "mesylate": {
        "pka": -1.2,
        "type": "acid",
        "hygroscopicity": "low",
        "stability": "low",
        "solubility_factor": 3.5,
    },
    "tosylate": {
        "pka": -1.3,
        "type": "acid",
        "hygroscopicity": "low",
        "stability": "low",
        "solubility_factor": 3.0,
    },
    "maleate": {
        "pka": 1.9,
        "type": "acid",
        "hygroscopicity": "moderate",
        "stability": "moderate",
        "solubility_factor": 2.5,
    },
    "fumarate": {
        "pka": 3.0,
        "type": "acid",
        "hygroscopicity": "low",
        "stability": "low",
        "solubility_factor": 2.5,
    },
    "tartrate": {
        "pka": 2.98,
        "type": "acid",
        "hygroscopicity": "high",
        "stability": "moderate",
        "solubility_factor": 2.0,
    },
    "citrate": {
        "pka": 3.13,
        "type": "acid",
        "hygroscopicity": "high",
        "stability": "moderate",
        "solubility_factor": 2.5,
    },
}


@dataclass(frozen=True)
class SaltFormResult:
    """Result of salt form prediction."""

    drug_name: str
    salt_counterion: str
    drug_pka: float
    ionization_type: str
    intrinsic_solubility_mg_mL: float
    salt_solubility_mg_mL: float
    solubility_ratio: float
    ph_max: float
    ph_optimal_dissolution: float
    salt_class: str
    hygroscopicity_risk: str
    stability_risk: str
    disproportionation_risk: str
    recommended: bool
    notes: str


def predict_salt_form(
    drug_name: str,
    salt_counterion: str,
    drug_pka: float,
    ionization_type: str,
    intrinsic_solubility_mg_mL: float,
) -> SaltFormResult:
    """Predict properties of a specific drug-counterion salt form.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    salt_counterion : str
        Name of the counterion (must be in COUNTERIONS database).
    drug_pka : float
        Drug pKa value.
    ionization_type : str
        "acid" or "base" ionization type.
    intrinsic_solubility_mg_mL : float
        Intrinsic (unionized) solubility in mg/mL.

    Returns
    -------
    SaltFormResult
    """
    # Validation
    if salt_counterion not in COUNTERIONS:
        raise ValueError(
            f"Unknown counterion '{salt_counterion}'. Valid options: {sorted(COUNTERIONS.keys())}"
        )
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    valid_ionization = {"acid", "base"}
    if ionization_type not in valid_ionization:
        raise ValueError(
            f"ionization_type must be one of {valid_ionization}, got '{ionization_type}'"
        )

    counterion = COUNTERIONS[salt_counterion]
    counterion_pka = counterion["pka"]
    counterion_type = counterion["type"]
    solubility_factor = counterion["solubility_factor"]
    hygroscopicity = counterion["hygroscopicity"]
    stability = counterion["stability"]

    # Check compatibility: acid drug needs basic counterion; base drug needs acid counterion
    # Acid drug (donates H+) -> needs base counterion (sodium, potassium)
    # Base drug (accepts H+)  -> needs acid counterion (HCl, mesylate, etc.)
    compatible = (ionization_type == "acid" and counterion_type == "base") or (
        ionization_type == "base" and counterion_type == "acid"
    )

    if not compatible:
        # Incompatible pair: no salt formation, solubility factor = 1.0
        effective_factor = 1.0
    else:
        effective_factor = solubility_factor

    # Salt solubility calculation
    pka_diff = abs(drug_pka - counterion_pka)
    raw_salt_solubility = intrinsic_solubility_mg_mL * effective_factor * (1.0 + pka_diff * 0.05)

    # Cap at 50x intrinsic solubility
    max_solubility = intrinsic_solubility_mg_mL * 50.0
    salt_solubility = min(raw_salt_solubility, max_solubility)

    solubility_ratio = salt_solubility / intrinsic_solubility_mg_mL

    # pH of maximum solubility (pHmax)
    if solubility_ratio > 1.0:
        log_ratio = math.log10(solubility_ratio)
    else:
        log_ratio = 0.0

    if ionization_type == "base":
        ph_max = drug_pka - 0.5 * log_ratio if solubility_ratio > 1.0 else drug_pka
    else:  # acid
        ph_max = drug_pka + 0.5 * log_ratio

    # Optimal pH for GI dissolution
    if ionization_type == "base":
        if 1.0 <= ph_max <= 7.4:
            ph_optimal_dissolution = ph_max
        else:
            ph_optimal_dissolution = 5.0
    else:  # acid
        if 6.0 <= ph_max <= 7.4:
            ph_optimal_dissolution = ph_max
        else:
            ph_optimal_dissolution = 6.5

    # Disproportionation risk: based on |drug_pKa - counterion_pKa|
    if pka_diff < 2.0:
        disproportionation_risk = "high"
    elif pka_diff < 4.0:
        disproportionation_risk = "moderate"
    else:
        disproportionation_risk = "low"

    # Recommendation
    recommended = (
        solubility_ratio > 2.0
        and stability != "high"
        and hygroscopicity != "high"
        and disproportionation_risk != "high"
    )

    notes = (
        f"Salt form {drug_name} {salt_counterion}: solubility ratio {solubility_ratio:.2f}x. "
        f"Hygroscopicity: {hygroscopicity}. Stability: {stability}. "
        f"Disproportionation risk: {disproportionation_risk}. "
        f"{'Recommended.' if recommended else 'Not recommended.'}"
    )

    return SaltFormResult(
        drug_name=drug_name,
        salt_counterion=salt_counterion,
        drug_pka=drug_pka,
        ionization_type=ionization_type,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        salt_solubility_mg_mL=salt_solubility,
        solubility_ratio=solubility_ratio,
        ph_max=ph_max,
        ph_optimal_dissolution=ph_optimal_dissolution,
        salt_class=salt_counterion,
        hygroscopicity_risk=hygroscopicity,
        stability_risk=stability,
        disproportionation_risk=disproportionation_risk,
        recommended=recommended,
        notes=notes,
    )


def screen_salt_forms(
    drug_name: str,
    drug_pka: float,
    ionization_type: str,
    intrinsic_solubility_mg_mL: float,
) -> list[SaltFormResult]:
    """Screen all available counterions for a drug.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    drug_pka : float
        Drug pKa value.
    ionization_type : str
        "acid" or "base".
    intrinsic_solubility_mg_mL : float
        Intrinsic solubility in mg/mL.

    Returns
    -------
    list of SaltFormResult
        Results for all counterions sorted by solubility_ratio descending.
    """
    results = []
    for counterion_name in COUNTERIONS:
        result = predict_salt_form(
            drug_name=drug_name,
            salt_counterion=counterion_name,
            drug_pka=drug_pka,
            ionization_type=ionization_type,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        )
        results.append(result)

    results.sort(key=lambda r: r.solubility_ratio, reverse=True)
    return results
