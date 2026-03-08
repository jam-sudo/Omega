"""
Phase 931 — Intestinal permeability prediction from molecular descriptors.

Uses Sun (2002) Caco-2 to Peff correlation and Amidon fa model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "IntestinalPermeabilityResult",
    "predict_intestinal_permeability",
    "screen_permeability",
]


@dataclass(frozen=True)
class IntestinalPermeabilityResult:
    """Result from intestinal permeability prediction (frozen dataclass)."""

    drug_name: str
    logp: float
    mw: float
    pka: float
    ionization_type: str
    caco2_papp_cm_s: float
    peff_cm_s: float
    fa: float
    dose_number: float
    bcs_class: str
    high_permeability: bool
    high_solubility: bool
    absorption_category: str
    notes: str


def _ionization_correction(ionization_type: str, pka: float) -> float:
    """Compute ionization correction for log Papp estimation."""
    if ionization_type == "neutral":
        return 0.0
    elif ionization_type == "acid":
        return max(0.0, 0.5 * (pka - 7.4))
    else:  # base
        return max(0.0, 0.5 * (7.4 - pka))


def _predict_papp(logp: float, mw: float, ionization_correction: float) -> float:
    """Estimate Caco-2 Papp (cm/s) from logP, MW, and ionization correction.

    log10(Papp [cm/s]) = 0.5*logP - 0.01*MW + 0.8 - ionization_correction
    Returns Papp in cm/s.
    """
    log_papp = 0.5 * logp - 0.01 * mw + 0.8 - ionization_correction
    return 10.0**log_papp


def _predict_peff(papp_cm_s: float) -> float:
    """Convert Caco-2 Papp to human intestinal Peff using Sun (2002) correlation.

    log10(Peff) = 0.4926 * log10(Papp) - 0.1454
    Returns Peff in cm/s.
    """
    log_papp = math.log10(papp_cm_s)
    log_peff = 0.4926 * log_papp - 0.1454
    return 10.0**log_peff


def _predict_fa(peff_cm_s: float) -> float:
    """Predict fraction absorbed using simplified Amidon model.

    Pn = 2 * Peff / 0.0001 (normalised to reference)
    fa = Pn / (1 + Pn), clamped to [0, 1]
    """
    pn = 2.0 * peff_cm_s / 0.0001
    fa = pn / (1.0 + pn)
    return max(0.0, min(1.0, fa))


def _absorption_category(fa: float) -> str:
    if fa > 0.9:
        return "excellent"
    elif fa > 0.7:
        return "good"
    elif fa > 0.4:
        return "moderate"
    else:
        return "poor"


def predict_intestinal_permeability(
    drug_name: str,
    logp: float,
    mw: float,
    pka: float,
    ionization_type: str = "neutral",
    dose_mg: float = 100.0,
    cs_mg_mL: float = 1.0,
) -> IntestinalPermeabilityResult:
    """Predict intestinal permeability and BCS classification.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logp : float
        Octanol-water partition coefficient (log scale).
    mw : float
        Molecular weight (g/mol). Must be > 0.
    pka : float
        pKa of the ionizable group.
    ionization_type : str
        "neutral", "acid", or "base".
    dose_mg : float
        Dose in mg. Must be > 0.
    cs_mg_mL : float
        Aqueous solubility (mg/mL). Must be > 0.

    Returns
    -------
    IntestinalPermeabilityResult
    """
    # Validation
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cs_mg_mL <= 0:
        raise ValueError("cs_mg_mL must be > 0")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got '{ionization_type}'"
        )

    ion_corr = _ionization_correction(ionization_type, pka)
    papp = _predict_papp(logp, mw, ion_corr)
    peff = _predict_peff(papp)
    fa = _predict_fa(peff)

    # Dose number: Dn = dose_mg / (cs_mg_mL * 250 mL)
    dose_number = dose_mg / (cs_mg_mL * 250.0)

    # BCS classification
    high_permeability = fa > 0.9 or peff > 2e-4
    high_solubility = dose_number <= 1.0

    if high_permeability and high_solubility:
        bcs_class = "I"
    elif high_permeability and not high_solubility:
        bcs_class = "II"
    elif not high_permeability and high_solubility:
        bcs_class = "III"
    else:
        bcs_class = "IV"

    absorption_cat = _absorption_category(fa)

    notes_parts = [
        f"logP={logp:.2f}, MW={mw:.1f}, pKa={pka:.1f} ({ionization_type})",
        f"Caco-2 Papp={papp:.2e} cm/s",
        f"Human Peff={peff:.2e} cm/s",
        f"fa={fa:.3f} ({absorption_cat})",
        f"BCS class {bcs_class} (Dn={dose_number:.2f})",
    ]
    if ion_corr > 0:
        notes_parts.append(f"Ionization correction applied: {ion_corr:.2f} log units")

    notes = "; ".join(notes_parts)

    return IntestinalPermeabilityResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        pka=pka,
        ionization_type=ionization_type,
        caco2_papp_cm_s=papp,
        peff_cm_s=peff,
        fa=fa,
        dose_number=dose_number,
        bcs_class=bcs_class,
        high_permeability=high_permeability,
        high_solubility=high_solubility,
        absorption_category=absorption_cat,
        notes=notes,
    )


def screen_permeability(
    compounds: list[dict],
) -> list[IntestinalPermeabilityResult]:
    """Screen a list of compounds for intestinal permeability.

    Parameters
    ----------
    compounds : list[dict]
        Each dict must have: drug_name, logp, mw, pka, ionization_type.
        Optional: dose_mg, cs_mg_mL.

    Returns
    -------
    list[IntestinalPermeabilityResult]
        Results sorted by fa descending.
    """
    results = []
    for compound in compounds:
        result = predict_intestinal_permeability(
            drug_name=compound["drug_name"],
            logp=compound["logp"],
            mw=compound["mw"],
            pka=compound["pka"],
            ionization_type=compound.get("ionization_type", "neutral"),
            dose_mg=compound.get("dose_mg", 100.0),
            cs_mg_mL=compound.get("cs_mg_mL", 1.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.fa, reverse=True)
    return results
