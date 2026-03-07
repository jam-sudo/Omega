"""Lipophilicity-based volume of distribution and tissue partitioning prediction.

Predicts distribution volume (Vss) and tissue Kp values from logP/logD
using empirical and semi-mechanistic relationships.

References
----------
- Lombardo F et al. (2002) J Med Chem 45:2867-2876 (Vss from logD/ppb)
- Poulin P, Theil FP (2002) J Pharm Sci 91:1358-1370 (tissue Kp)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VssPredictionResult:
    """Predicted distribution parameters from lipophilicity."""

    logP: float
    fup: float
    mw: float
    pka: float
    compound_type: str
    logD_at_pH74: float
    vss_L_per_kg: float
    vss_70kg_L: float
    kp_liver: float
    kp_kidney: float
    kp_muscle: float
    kp_fat: float
    kp_brain: float
    kp_lung: float
    distribution_category: str  # "low" (<0.6), "moderate" (0.6-3), "high" (>3) L/kg
    notes: str


# ---------------------------------------------------------------------------
# logD at pH
# ---------------------------------------------------------------------------


def logD_at_pH(
    logP: float,
    pka: float,
    ph: float,
    compound_type: str = "acid",
) -> float:
    """Calculate logD at a given pH.

    Parameters
    ----------
    logP : float
        Partition coefficient (octanol/water) at neutral form.
    pka : float
        Acid dissociation constant.
    ph : float
        pH of the aqueous phase.
    compound_type : str
        "acid", "base", or "neutral".

    Returns
    -------
    float
        logD at the specified pH.

    Notes
    -----
    Acid:    logD = logP - log10(1 + 10^(pH - pKa))
    Base:    logD = logP - log10(1 + 10^(pKa - pH))
    Neutral: logD = logP
    """
    _validate_compound_type(compound_type)

    if compound_type == "neutral":
        return logP

    if compound_type == "acid":
        exponent = ph - pka
    else:  # base
        exponent = pka - ph

    # Clamp to avoid overflow
    exponent = min(exponent, 700.0)
    return logP - math.log10(1.0 + 10.0**exponent)


# ---------------------------------------------------------------------------
# Tissue Kp prediction
# ---------------------------------------------------------------------------

_VALID_TISSUES = {"liver", "kidney", "muscle", "fat", "brain", "lung"}

_TISSUE_FORMULAS = {
    # (base_scale, logP_exponent)  =>  Kp = base * fup * 10^(exp * logP)
    # fat:   10^(0.8*logP) * fup / 0.1  (high lipophilic accumulation)
    # brain: 0.29 * 10^(0.7*logP)       (simplified without PSA)
    # liver: 2.0 * 10^(0.3*logP)
    # kidney:1.5 * 10^(0.3*logP)
    # muscle:0.5 * 10^(0.2*logP)
    # lung:  3.0 * 10^(0.2*logP)
    "fat": ("fat", None),
    "brain": ("brain", None),
    "liver": (2.0, 0.3),
    "kidney": (1.5, 0.3),
    "muscle": (0.5, 0.2),
    "lung": (3.0, 0.2),
}


def predict_tissue_Kp(logP: float, fup: float, tissue: str) -> float:
    """Predict tissue-to-plasma Kp from logP and free fraction.

    Parameters
    ----------
    logP : float
        Octanol-water partition coefficient.
    fup : float
        Fraction unbound in plasma (0, 1].
    tissue : str
        Target tissue: "liver", "kidney", "muscle", "fat", "brain", "lung".

    Returns
    -------
    float
        Tissue partition coefficient (Kp = Ctissue / Cplasma).
    """
    if tissue not in _VALID_TISSUES:
        raise ValueError(f"tissue must be one of {sorted(_VALID_TISSUES)}, got '{tissue}'")

    if tissue == "fat":
        # fat: Kp = 10^(0.8*logP) * fup / 0.1
        exponent = min(0.8 * logP, 700.0)
        kp = (10.0**exponent) * fup / 0.1
    elif tissue == "brain":
        # brain: 0.29 * 10^(0.7*logP)
        exponent = min(0.7 * logP, 700.0)
        kp = 0.29 * (10.0**exponent)
    else:
        base, exp = _TISSUE_FORMULAS[tissue]
        exponent = min(exp * logP, 700.0)
        kp = base * (10.0**exponent)

    return max(kp, 1e-6)


# ---------------------------------------------------------------------------
# Vss prediction
# ---------------------------------------------------------------------------


def predict_vss(
    logP: float,
    fup: float,
    mw: float,
    pka: float = 7.0,
    compound_type: str = "neutral",
) -> VssPredictionResult:
    """Predict Vss and tissue Kp values from lipophilicity.

    Regression-based Vss estimate (L/kg, 70 kg human):
        Vss = 0.17 + logP * 0.49 - fup * 2.3
    with compound-type adjustments:
        - acid:  Vss *= 0.65 (less tissue distribution)
        - base:  Vss *= 1.35 (more tissue distribution)
        - neutral: no adjustment

    Parameters
    ----------
    logP : float
        Octanol-water partition coefficient.
    fup : float
        Fraction unbound in plasma (0, 1].
    mw : float
        Molecular weight (Da), must be > 0.
    pka : float
        Acid dissociation constant (default 7.0 for neutral).
    compound_type : str
        "acid", "base", or "neutral".

    Returns
    -------
    VssPredictionResult
    """
    _validate_compound_type(compound_type)
    if not (0.0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    # logD at physiological pH 7.4
    logd74 = logD_at_pH(logP, pka, ph=7.4, compound_type=compound_type)

    # Regression-based Vss (L/kg)
    vss_base = 0.17 + logP * 0.49 - fup * 2.3

    # Compound-type adjustment
    if compound_type == "acid":
        vss_base *= 0.65
        type_note = "Acids distribute less; Vss scaled down."
    elif compound_type == "base":
        vss_base *= 1.35
        type_note = "Bases distribute more; Vss scaled up."
    else:
        type_note = "Neutral compound; no type adjustment."

    # Floor at physiological minimum (~plasma volume ~0.04 L/kg)
    vss_L_per_kg = max(vss_base, 0.04)
    vss_70kg_L = vss_L_per_kg * 70.0

    # Tissue Kp predictions
    kp_liver = predict_tissue_Kp(logP, fup, "liver")
    kp_kidney = predict_tissue_Kp(logP, fup, "kidney")
    kp_muscle = predict_tissue_Kp(logP, fup, "muscle")
    kp_fat = predict_tissue_Kp(logP, fup, "fat")
    kp_brain = predict_tissue_Kp(logP, fup, "brain")
    kp_lung = predict_tissue_Kp(logP, fup, "lung")

    # Distribution category
    if vss_L_per_kg < 0.6:
        dist_cat = "low"
    elif vss_L_per_kg <= 3.0:
        dist_cat = "moderate"
    else:
        dist_cat = "high"

    notes = (
        f"logD7.4={logd74:.2f}; {type_note} Vss regression: 0.17 + {logP:.2f}*0.49 - {fup:.3f}*2.3."
    )

    return VssPredictionResult(
        logP=logP,
        fup=fup,
        mw=mw,
        pka=pka,
        compound_type=compound_type,
        logD_at_pH74=logd74,
        vss_L_per_kg=vss_L_per_kg,
        vss_70kg_L=vss_70kg_L,
        kp_liver=kp_liver,
        kp_kidney=kp_kidney,
        kp_muscle=kp_muscle,
        kp_fat=kp_fat,
        kp_brain=kp_brain,
        kp_lung=kp_lung,
        distribution_category=dist_cat,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_compound_type(compound_type: str) -> None:
    valid = {"acid", "base", "neutral"}
    if compound_type not in valid:
        raise ValueError(f"compound_type must be one of {valid}, got '{compound_type}'")


__all__ = [
    "VssPredictionResult",
    "logD_at_pH",
    "predict_tissue_Kp",
    "predict_vss",
]
