"""
Phase 938 — Crystal Form Solubility Prediction
Predict solubility differences between crystal polymorphs and amorphous forms.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CrystalFormSolubilityResult",
    "predict_crystal_form_solubility",
    "compare_crystal_forms",
]

# Crystal form relative solubility multipliers vs Form_I
_FORM_MULTIPLIERS = {
    "amorphous": 100.0,
    "Form_I": 1.0,
    "Form_II": 2.0,
    "Form_III": 3.5,
    "hydrate": 0.7,
    "solvate": 0.5,
    "co_crystal": 5.0,
}

# Stability risk by form
_STABILITY_RISK = {
    "amorphous": "high",
    "Form_I": "low",
    "Form_II": "moderate",
    "Form_III": "moderate",
    "hydrate": "low",
    "solvate": "moderate",
    "co_crystal": "moderate",
}

_ALL_FORMS = ["amorphous", "Form_I", "Form_II", "Form_III", "hydrate", "solvate", "co_crystal"]

_VALID_IONIZATION = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class CrystalFormSolubilityResult:
    drug_name: str
    crystal_form: str
    logp: float
    pka: float
    ionization_type: str
    ph: float
    intrinsic_solubility_mg_mL: float
    ph_adjusted_solubility_mg_mL: float
    relative_to_form_I: float
    dissolution_advantage: str
    stability_risk: str
    notes: str


def _validate_inputs(crystal_form: str, ph: float, ionization_type: str) -> None:
    if crystal_form not in _FORM_MULTIPLIERS:
        raise ValueError(
            f"crystal_form must be one of {set(_FORM_MULTIPLIERS.keys())}, got '{crystal_form}'"
        )
    if not (0 <= ph <= 14):
        raise ValueError("ph must be in [0, 14]")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got '{ionization_type}'"
        )


def _compute_intrinsic_solubility_mg_mL(logp: float) -> float:
    """Simplified Yalkowsky: log S0 = -logP + 0.5; S0 in mg/mL."""
    log_s0 = -logp + 0.5
    return 10.0**log_s0


def _apply_ph_adjustment(s0: float, ph: float, pka: float, ionization_type: str) -> float:
    """Apply Henderson-Hasselbalch pH adjustment to intrinsic solubility."""
    if ionization_type == "acid":
        # S(pH) = S0 * (1 + 10^(pH - pKa))
        exponent = ph - pka
        # Clamp exponent to avoid overflow
        exponent = max(-50.0, min(50.0, exponent))
        s_ph = s0 * (1.0 + 10.0**exponent)
    elif ionization_type == "base":
        # S(pH) = S0 * (1 + 10^(pKa - pH))
        exponent = pka - ph
        exponent = max(-50.0, min(50.0, exponent))
        s_ph = s0 * (1.0 + 10.0**exponent)
    else:
        # neutral: no pH effect
        s_ph = s0
    return s_ph


def predict_crystal_form_solubility(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    crystal_form: str,
    ph: float = 6.8,
) -> CrystalFormSolubilityResult:
    """Predict solubility for a specific crystal form at the given pH."""
    _validate_inputs(crystal_form, ph, ionization_type)

    # Intrinsic solubility (neutral form, Form_I reference)
    s0 = _compute_intrinsic_solubility_mg_mL(logp)

    # Apply crystal form multiplier to get form-specific intrinsic solubility
    form_mult = _FORM_MULTIPLIERS[crystal_form]
    s_form = s0 * form_mult

    # Apply pH adjustment to the form-specific S0
    s_ph = _apply_ph_adjustment(s_form, ph, pka, ionization_type)

    # Compute Form_I solubility at same pH for relative comparison
    s_form_I = _apply_ph_adjustment(s0 * _FORM_MULTIPLIERS["Form_I"], ph, pka, ionization_type)
    relative = s_ph / s_form_I if s_form_I > 0 else form_mult

    # Dissolution advantage based on relative solubility vs Form_I
    if relative >= 5.0:
        dissolution_advantage = "high"
    elif relative > 2.0:
        dissolution_advantage = "moderate"
    else:
        dissolution_advantage = "low"

    stability_risk = _STABILITY_RISK[crystal_form]

    # Build notes
    notes_parts = [
        f"Crystal form: {crystal_form}; logP={logp:.2f}; pKa={pka:.2f}; "
        f"ionization={ionization_type}; pH={ph:.1f}."
    ]
    notes_parts.append(
        f"Form multiplier vs Form_I: {form_mult:.1f}x. "
        f"Relative to Form_I at pH {ph:.1f}: {relative:.2f}x."
    )
    if crystal_form == "amorphous":
        notes_parts.append(
            "Amorphous form offers maximum solubility but carries crystallization risk."
        )
    elif crystal_form in {"Form_II", "Form_III"}:
        notes_parts.append("Metastable polymorph: monitor for form conversion during storage.")
    notes = " ".join(notes_parts)

    return CrystalFormSolubilityResult(
        drug_name=drug_name,
        crystal_form=crystal_form,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        ph=ph,
        intrinsic_solubility_mg_mL=s0,
        ph_adjusted_solubility_mg_mL=s_ph,
        relative_to_form_I=relative,
        dissolution_advantage=dissolution_advantage,
        stability_risk=stability_risk,
        notes=notes,
    )


def compare_crystal_forms(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    ph: float = 6.8,
) -> list[CrystalFormSolubilityResult]:
    """Compare all 7 crystal forms, sorted by ph_adjusted_solubility_mg_mL descending."""
    _validate_inputs("Form_I", ph, ionization_type)  # validate ph and ionization_type
    results = []
    for form in _ALL_FORMS:
        res = predict_crystal_form_solubility(
            drug_name=drug_name,
            logp=logp,
            pka=pka,
            ionization_type=ionization_type,
            crystal_form=form,
            ph=ph,
        )
        results.append(res)
    results.sort(key=lambda r: r.ph_adjusted_solubility_mg_mL, reverse=True)
    return results
