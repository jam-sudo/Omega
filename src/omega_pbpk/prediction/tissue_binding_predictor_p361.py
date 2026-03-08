"""Phase 361 — Tissue Binding Predictor.

Predicts tissue-to-plasma partition coefficients (Kp) and drug binding
in various tissues using the Poulin-Theil simplified approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Tissue composition: (lipid_fraction, water_fraction, protein_fraction)
_TISSUE_COMPOSITION: dict[str, tuple[float, float, float]] = {
    "muscle": (0.04, 0.75, 0.21),
    "adipose": (0.79, 0.18, 0.03),
    "liver": (0.04, 0.71, 0.25),
    "kidney": (0.04, 0.77, 0.19),
    "brain": (0.09, 0.77, 0.14),
    "lung": (0.05, 0.83, 0.12),
    "heart": (0.04, 0.73, 0.23),
}

# Tissue volumes in litres (70 kg human)
_TISSUE_VOLUMES_L: dict[str, float] = {
    "muscle": 28.0,
    "adipose": 14.5,
    "liver": 1.5,
    "kidney": 0.3,
    "brain": 1.4,
    "lung": 1.0,
    "heart": 0.3,
}

_V_PLASMA_L: float = 3.0

# Reference plasma composition denominator constants (plasma approximation)
_REF_LIPID: float = 0.04
_REF_PROTEIN: float = 0.18
_REF_WATER: float = 0.73


@dataclass
class TissueBindingResult:
    """Result of tissue binding / Kp prediction."""

    drug_name: str
    logP: float
    fu_plasma: float
    kp_tissues: dict = field(default_factory=dict)
    highest_kp_tissue: str = ""
    lowest_kp_tissue: str = ""
    vss_predicted_L: float = 0.0
    adipose_accumulation: bool = False
    brain_penetration_index: float = 0.0
    notes: str = ""


def _validate_inputs(logP: float, fu_plasma: float, mw_Da: float) -> None:
    """Raise ValueError on invalid inputs."""
    if logP < -5.0 or logP > 10.0:
        raise ValueError(f"logP must be in [-5, 10]; got {logP}")
    if fu_plasma <= 0.0 or fu_plasma > 1.0:
        raise ValueError(f"fu_plasma must be in (0, 1]; got {fu_plasma}")
    if mw_Da <= 0.0:
        raise ValueError(f"mw_Da must be > 0; got {mw_Da}")


def _compute_kp(
    p_oct: float,
    lipid: float,
    water: float,
    protein: float,
    fu_plasma: float,
    drug_type: str,
    pKa: float,
) -> float:
    """Compute Kp for a single tissue using Poulin-Theil simplified model."""
    numerator = lipid * p_oct + 0.3 * protein * (1.0 / fu_plasma) + water
    denominator = _REF_LIPID * p_oct + 0.3 * _REF_PROTEIN * (1.0 / fu_plasma) + _REF_WATER
    kp = numerator / denominator

    # Lysosomal trapping correction for bases
    if drug_type == "base" and pKa > 7.4:
        kp *= 1.0 + 0.1 * pKa

    # Acid correction: more plasma-bound
    if drug_type == "acid":
        kp *= fu_plasma

    return kp


def predict_tissue_binding(
    drug_name: str,
    logP: float,
    pKa: float,
    drug_type: str,
    fu_plasma: float,
    mw_Da: float,
) -> TissueBindingResult:
    """Predict Kp values for 7 tissues and Vss.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    logP:
        Octanol-water partition coefficient (log units), range [-5, 10].
    pKa:
        Acid/base dissociation constant.
    drug_type:
        One of "acid", "base", or "neutral".
    fu_plasma:
        Fraction unbound in plasma, in (0, 1].
    mw_Da:
        Molecular weight in Da (must be > 0; used for validation).

    Returns
    -------
    TissueBindingResult
    """
    if drug_type not in ("acid", "base", "neutral"):
        raise ValueError(f"drug_type must be 'acid', 'base', or 'neutral'; got '{drug_type}'")
    _validate_inputs(logP, fu_plasma, mw_Da)

    p_oct = 10.0**logP

    kp_tissues: dict[str, float] = {}
    for tissue, (lipid, water, protein) in _TISSUE_COMPOSITION.items():
        kp_tissues[tissue] = _compute_kp(p_oct, lipid, water, protein, fu_plasma, drug_type, pKa)

    # Vss = sum(Kp_i * V_i) + V_plasma
    vss = _V_PLASMA_L
    for tissue, vol in _TISSUE_VOLUMES_L.items():
        vss += kp_tissues[tissue] * vol

    highest = max(kp_tissues, key=lambda t: kp_tissues[t])
    lowest = min(kp_tissues, key=lambda t: kp_tissues[t])

    adipose_accum = kp_tissues["adipose"] > 5.0
    brain_pi = kp_tissues["brain"]

    notes_parts: list[str] = []
    if drug_type == "base" and pKa > 7.4:
        notes_parts.append("lysosomal trapping applied")
    if drug_type == "acid":
        notes_parts.append("acid plasma-binding correction applied")
    if adipose_accum:
        notes_parts.append("adipose accumulation predicted")
    notes = "; ".join(notes_parts) if notes_parts else "no special corrections"

    return TissueBindingResult(
        drug_name=drug_name,
        logP=logP,
        fu_plasma=fu_plasma,
        kp_tissues=kp_tissues,
        highest_kp_tissue=highest,
        lowest_kp_tissue=lowest,
        vss_predicted_L=round(vss, 4),
        adipose_accumulation=adipose_accum,
        brain_penetration_index=round(brain_pi, 4),
        notes=notes,
    )
