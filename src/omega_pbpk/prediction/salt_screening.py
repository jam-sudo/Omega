"""Phase 175: Pharmaceutical salt screening for poorly soluble drugs."""

from dataclasses import dataclass

_COUNTERIONS: dict[str, dict] = {
    # For acid drugs (cation counterions):
    "sodium": {
        "type": "cation",
        "pka_counterion": 14.0,
        "solubility_factor": 3.0,
        "regulatory": "preferred",
    },
    "potassium": {
        "type": "cation",
        "pka_counterion": 14.0,
        "solubility_factor": 2.5,
        "regulatory": "preferred",
    },
    "calcium": {
        "type": "cation",
        "pka_counterion": 14.0,
        "solubility_factor": 1.5,
        "regulatory": "preferred",
    },
    "meglumine": {
        "type": "cation",
        "pka_counterion": 9.5,
        "solubility_factor": 4.0,
        "regulatory": "preferred",
    },
    "lysine": {
        "type": "cation",
        "pka_counterion": 10.5,
        "solubility_factor": 3.5,
        "regulatory": "preferred",
    },
    # For base drugs (anion counterions):
    "hydrochloride": {
        "type": "anion",
        "pka_counterion": -7.0,
        "solubility_factor": 3.0,
        "regulatory": "preferred",
    },
    "sulfate": {
        "type": "anion",
        "pka_counterion": -3.0,
        "solubility_factor": 2.0,
        "regulatory": "preferred",
    },
    "mesylate": {
        "type": "anion",
        "pka_counterion": -1.2,
        "solubility_factor": 3.5,
        "regulatory": "preferred",
    },
    "tosylate": {
        "type": "anion",
        "pka_counterion": -1.3,
        "solubility_factor": 2.5,
        "regulatory": "acceptable",
    },
    "maleate": {
        "type": "anion",
        "pka_counterion": 1.9,
        "solubility_factor": 3.0,
        "regulatory": "preferred",
    },
}

_HYGROSCOPICITY: dict[str, str] = {
    "calcium": "high",
    "meglumine": "moderate",
    "lysine": "moderate",
}


def _hygroscopicity(counterion_name: str) -> str:
    return _HYGROSCOPICITY.get(counterion_name, "low")


def _validate_inputs(
    pka: float,
    ionization_type: str,
    intrinsic_solubility_mg_mL: float,
    logP: float,
) -> None:
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got {ionization_type!r}"
        )
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")


def _compute_salt_form_result(
    drug_name: str,
    pka: float,
    ionization_type: str,
    intrinsic_solubility_mg_mL: float,
    counterion_name: str,
) -> "SaltFormResult":
    ci = _COUNTERIONS[counterion_name]
    pka_counterion = ci["pka_counterion"]
    solubility_factor = ci["solubility_factor"]
    regulatory = ci["regulatory"]
    counterion_type = ci["type"]

    delta_pka = abs(pka - pka_counterion)
    salt_viable = delta_pka >= 3.0

    if salt_viable:
        solubility_enhancement = solubility_factor * min(10.0, delta_pka / 3.0)
    else:
        solubility_enhancement = 1.0

    predicted_solubility_mg_mL = intrinsic_solubility_mg_mL * solubility_enhancement

    if salt_viable:
        notes = (
            f"Salt formation viable (delta_pKa={delta_pka:.1f}). "
            f"Predicted {solubility_enhancement:.1f}x solubility enhancement."
        )
    else:
        notes = f"Salt formation not viable: delta_pKa={delta_pka:.1f} < 3.0 required."

    return SaltFormResult(
        drug_name=drug_name,
        ionization_type=ionization_type,
        pka_drug=pka,
        counterion_name=counterion_name,
        counterion_type=counterion_type,
        delta_pka=delta_pka,
        salt_viable=salt_viable,
        solubility_enhancement=solubility_enhancement,
        predicted_solubility_mg_mL=predicted_solubility_mg_mL,
        regulatory_status=regulatory,
        hygroscopicity_risk=_hygroscopicity(counterion_name),
        notes=notes,
    )


@dataclass(frozen=True)
class SaltFormResult:
    drug_name: str
    ionization_type: str
    pka_drug: float
    counterion_name: str
    counterion_type: str
    delta_pka: float
    salt_viable: bool
    solubility_enhancement: float
    predicted_solubility_mg_mL: float
    regulatory_status: str
    hygroscopicity_risk: str
    notes: str


def screen_salt_forms(
    drug_name: str,
    pka: float,
    ionization_type: str,
    intrinsic_solubility_mg_mL: float,
    logP: float,
) -> list[SaltFormResult]:
    """Screen all applicable salt forms for a drug, sorted by solubility_enhancement descending."""
    _validate_inputs(pka, ionization_type, intrinsic_solubility_mg_mL, logP)

    if ionization_type == "neutral":
        return []

    # Acid drugs use cation counterions; base drugs use anion counterions
    if ionization_type == "acid":
        applicable_type = "cation"
    else:
        applicable_type = "anion"

    results: list[SaltFormResult] = []
    for counterion_name, ci in _COUNTERIONS.items():
        if ci["type"] != applicable_type:
            continue
        result = _compute_salt_form_result(
            drug_name, pka, ionization_type, intrinsic_solubility_mg_mL, counterion_name
        )
        results.append(result)

    results.sort(key=lambda r: r.solubility_enhancement, reverse=True)
    return results


def predict_salt_solubility(
    drug_name: str,
    pka: float,
    ionization_type: str,
    counterion_name: str,
    intrinsic_solubility_mg_mL: float,
) -> SaltFormResult:
    """Predict solubility for a specific drug-counterion pair."""
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got {ionization_type!r}"
        )
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if counterion_name not in _COUNTERIONS:
        raise ValueError(
            f"counterion_name {counterion_name!r} not in database. "
            f"Valid options: {sorted(_COUNTERIONS)}"
        )

    return _compute_salt_form_result(
        drug_name, pka, ionization_type, intrinsic_solubility_mg_mL, counterion_name
    )
