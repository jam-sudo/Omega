"""Phase 547 - Gastric pH Effect on Drug Absorption.

Models pH-dependent drug absorption from the stomach using
Henderson-Hasselbalch ionization and solubility calculations.
"""

from dataclasses import dataclass

_VALID_DRUG_TYPES = {"acid", "base"}


@dataclass(frozen=True)
class GastricPHAbsorptionResult:
    """Result of gastric pH-dependent absorption prediction."""

    drug_name: str
    pKa: float
    drug_type: str
    gastric_ph: float
    ionized_fraction: float
    dissolved_fraction: float
    fa_gastric: float
    absorption_class: str
    notes: str


def predict_gastric_absorption(
    drug_name: str,
    pKa: float,
    drug_type: str,
    gastric_ph: float = 2.0,
    intrinsic_solubility_mg_mL: float = 1.0,
    dose_mg: float = 100.0,
) -> GastricPHAbsorptionResult:
    """Predict gastric absorption based on pH-dependent ionization and solubility.

    Args:
        drug_name: Name of the drug.
        pKa: Acid dissociation constant.
        drug_type: Either 'acid' or 'base'.
        gastric_ph: Stomach pH (default 2.0).
        intrinsic_solubility_mg_mL: Intrinsic solubility in mg/mL.
        dose_mg: Oral dose in mg.

    Returns:
        GastricPHAbsorptionResult with absorption prediction.
    """
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {_VALID_DRUG_TYPES}, got '{drug_type}'")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")

    # Henderson-Hasselbalch ionization
    if drug_type == "acid":
        ionized_fraction = 1.0 / (1.0 + 10.0 ** (pKa - gastric_ph))
    else:  # base
        ionized_fraction = 1.0 / (1.0 + 10.0 ** (gastric_ph - pKa))

    fa_neutral = 1.0 - ionized_fraction

    # pH-dependent solubility
    if drug_type == "acid":
        sol = intrinsic_solubility_mg_mL * (1.0 + 10.0 ** (gastric_ph - pKa))
    else:  # base
        sol = intrinsic_solubility_mg_mL * (1.0 + 10.0 ** (pKa - gastric_ph))

    dissolved_fraction = min(1.0, 250.0 * sol / dose_mg)

    fa_gastric = fa_neutral * dissolved_fraction
    fa_gastric = max(0.0, min(1.0, fa_gastric))

    if fa_gastric >= 0.6:
        absorption_class = "good"
    elif fa_gastric >= 0.3:
        absorption_class = "moderate"
    else:
        absorption_class = "poor"

    notes_parts = []
    if ionized_fraction > 0.9:
        notes_parts.append("Highly ionized at this pH")
    if dissolved_fraction < 0.5:
        notes_parts.append("Dissolution-limited absorption")
    if not notes_parts:
        notes_parts.append("Favorable absorption conditions")
    notes = "; ".join(notes_parts)

    return GastricPHAbsorptionResult(
        drug_name=drug_name,
        pKa=pKa,
        drug_type=drug_type,
        gastric_ph=gastric_ph,
        ionized_fraction=ionized_fraction,
        dissolved_fraction=dissolved_fraction,
        fa_gastric=fa_gastric,
        absorption_class=absorption_class,
        notes=notes,
    )


def compare_gastric_ph(
    drug_name: str,
    pKa: float,
    drug_type: str,
    ph_values: list = None,
) -> list:
    """Compare absorption across multiple gastric pH values.

    Args:
        drug_name: Name of the drug.
        pKa: Acid dissociation constant.
        drug_type: Either 'acid' or 'base'.
        ph_values: List of pH values to compare (default [1.0, 2.0, 4.0, 6.0]).

    Returns:
        List of GastricPHAbsorptionResult sorted by fa_gastric descending.
    """
    if ph_values is None:
        ph_values = [1.0, 2.0, 4.0, 6.0]

    results = [
        predict_gastric_absorption(drug_name, pKa, drug_type, gastric_ph=ph) for ph in ph_values
    ]
    results.sort(key=lambda r: r.fa_gastric, reverse=True)
    return results
