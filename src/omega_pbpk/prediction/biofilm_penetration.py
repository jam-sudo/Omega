"""
Phase 966 — Biofilm penetration predictor.

Predicts drug penetration into bacterial biofilms based on physicochemical properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

__all__ = ["BiofilmPenetrationResult", "predict_biofilm_penetration", "screen_biofilm_activity"]

_VALID_IONIZATION = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class BiofilmPenetrationResult:
    """Result of biofilm penetration prediction (frozen dataclass)."""

    drug_name: str
    logp: float
    mw: float
    pka: float
    ionization_type: str
    c_bulk_mg_L: float
    f_penetration: float
    c_biofilm_effective_mg_L: float
    mbec_mic_ratio: float
    penetration_category: str
    biofilm_activity_factor: float
    recommended_dose_multiplier: float
    notes: str


def _compute_f_penetration(logp: float, mw: float, ionization_type: str) -> float:
    """Compute penetration fraction based on drug physicochemical properties."""
    # Base penetration from logP
    if -1.0 <= logp <= 2.0:
        f_pen = 0.8
    elif logp > 2.0:
        f_pen = 0.5
    else:
        # logP < -1
        f_pen = 0.6

    # MW correction
    if mw < 500:
        f_pen *= 1.2
    elif mw > 800:
        f_pen *= 0.5

    # Charge correction for cationic compounds (bases)
    if ionization_type == "base":
        f_pen *= 0.7

    # Clamp to [0.01, 1.0]
    f_pen = max(0.01, min(1.0, f_pen))
    return f_pen


def predict_biofilm_penetration(
    drug_name: str,
    logp: float,
    mw: float,
    pka: float,
    ionization_type: str,
    c_bulk_mg_L: float = 1.0,
) -> BiofilmPenetrationResult:
    """Predict drug penetration into bacterial biofilm.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logp : float
        Octanol-water partition coefficient.
    mw : float
        Molecular weight (g/mol).
    pka : float
        pKa of the drug.
    ionization_type : str
        One of "acid", "base", "neutral".
    c_bulk_mg_L : float
        Bulk plasma/medium drug concentration (mg/L).

    Returns
    -------
    BiofilmPenetrationResult
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if c_bulk_mg_L <= 0:
        raise ValueError("c_bulk_mg_L must be > 0")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got '{ionization_type}'"
        )

    f_pen = _compute_f_penetration(logp, mw, ionization_type)
    c_biofilm = c_bulk_mg_L * f_pen

    # MBEC/MIC ratio: max(2, 10 / sqrt(f_pen))
    mbec_mic_ratio = max(2.0, 10.0 / sqrt(f_pen))

    # Penetration category
    if f_pen > 0.6:
        category = "good"
    elif f_pen > 0.3:
        category = "moderate"
    else:
        category = "poor"

    biofilm_activity_factor = 1.0 / mbec_mic_ratio
    recommended_dose_multiplier = mbec_mic_ratio

    notes = (
        f"logP={logp:.2f}, MW={mw:.1f}, pKa={pka:.2f}, type={ionization_type}. "
        f"f_penetration={f_pen:.3f} ({category}). "
        f"MBEC/MIC={mbec_mic_ratio:.1f}x. "
        f"Effective biofilm conc={c_biofilm:.3f} mg/L at bulk {c_bulk_mg_L:.3f} mg/L."
    )

    return BiofilmPenetrationResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        pka=pka,
        ionization_type=ionization_type,
        c_bulk_mg_L=c_bulk_mg_L,
        f_penetration=f_pen,
        c_biofilm_effective_mg_L=c_biofilm,
        mbec_mic_ratio=mbec_mic_ratio,
        penetration_category=category,
        biofilm_activity_factor=biofilm_activity_factor,
        recommended_dose_multiplier=recommended_dose_multiplier,
        notes=notes,
    )


def screen_biofilm_activity(compounds: list) -> list:
    """Screen a list of compounds for biofilm penetration.

    Parameters
    ----------
    compounds : list[dict]
        Each dict must contain: drug_name, logp, mw, pka, ionization_type.
        Optional: c_bulk_mg_L (defaults to 1.0).

    Returns
    -------
    list[BiofilmPenetrationResult]
        Sorted by f_penetration descending.
    """
    results = []
    for compound in compounds:
        result = predict_biofilm_penetration(
            drug_name=compound["drug_name"],
            logp=compound["logp"],
            mw=compound["mw"],
            pka=compound["pka"],
            ionization_type=compound["ionization_type"],
            c_bulk_mg_L=compound.get("c_bulk_mg_L", 1.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.f_penetration, reverse=True)
    return results
