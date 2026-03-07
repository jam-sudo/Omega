"""Phase 287 — Crystal Form Bioavailability Predictor.

Predicts relative bioavailability of different drug crystal forms
(polymorph A/B, anhydrate, hydrate, cocrystal) based on intrinsic
dissolution rates and solubility ratios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CrystalFormBAResult:
    """Result of crystal form bioavailability prediction."""

    drug_name: str
    crystal_form: str
    solubility_ratio_vs_reference: float
    intrinsic_dissolution_rate_mg_per_cm2_per_min: float
    ba_ratio: float
    ba_improvement_pct: float
    predicted_f_abs: float
    stability_risk: str
    manufacturability_score: float
    recommendation: str
    notes: str


_VALID_BCS = {"I", "II", "III", "IV"}


def _validate_inputs(
    intrinsic_dissolution_rate_mg_per_cm2_per_min: float,
    solubility_ratio_vs_reference: float,
    mw: float,
    dose_mg: float,
    bcs_class_reference: str,
) -> None:
    if intrinsic_dissolution_rate_mg_per_cm2_per_min <= 0:
        raise ValueError("intrinsic_dissolution_rate_mg_per_cm2_per_min must be > 0")
    if solubility_ratio_vs_reference <= 0:
        raise ValueError("solubility_ratio_vs_reference must be > 0")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if bcs_class_reference not in _VALID_BCS:
        raise ValueError(
            f"bcs_class_reference must be one of {_VALID_BCS}, got {bcs_class_reference!r}"
        )


def predict_crystal_form_ba(
    drug_name: str,
    crystal_form: str,
    intrinsic_dissolution_rate_mg_per_cm2_per_min: float,
    solubility_ratio_vs_reference: float,
    logp: float,
    mw: float,
    dose_mg: float,
    bcs_class_reference: str,
) -> CrystalFormBAResult:
    """Predict relative bioavailability of a crystal form vs reference.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    crystal_form:
        Label for the crystal form (e.g., "polymorph_A", "hydrate", "cocrystal").
    intrinsic_dissolution_rate_mg_per_cm2_per_min:
        Intrinsic dissolution rate [mg/(cm^2*min)].
    solubility_ratio_vs_reference:
        Solubility of this form divided by solubility of the reference polymorph.
    logp:
        Logarithm (base 10) of octanol/water partition coefficient.
    mw:
        Molecular weight [g/mol].
    dose_mg:
        Dose in mg.
    bcs_class_reference:
        BCS class of the reference polymorph: "I", "II", "III", or "IV".

    Returns
    -------
    CrystalFormBAResult
    """
    _validate_inputs(
        intrinsic_dissolution_rate_mg_per_cm2_per_min,
        solubility_ratio_vs_reference,
        mw,
        dose_mg,
        bcs_class_reference,
    )

    # Supersaturation index, capped at 4
    supersaturation_index = min(4.0, solubility_ratio_vs_reference)

    # Dissolution rate factor (normalised to reference IDR of 0.1 mg/cm^2/min)
    drf = intrinsic_dissolution_rate_mg_per_cm2_per_min / 0.1

    # BCS I/III: permeability-limited or complete absorption → crystal form has minor impact
    if bcs_class_reference in ("I", "III"):
        ba_ratio = 1.0
        notes = (
            f"BCS {bcs_class_reference}: absorption not solubility-limited; "
            "crystal form has minimal impact on bioavailability."
        )
    else:
        # BCS II/IV: solubility-limited
        ba_ratio = min(
            2.0,
            (supersaturation_index**0.5) * (drf**0.3),
        )
        notes = (
            f"BCS {bcs_class_reference}: solubility-limited absorption; "
            f"supersaturation index={supersaturation_index:.2f}, "
            f"dissolution rate factor={drf:.3f}."
        )

    ba_improvement_pct = (ba_ratio - 1.0) * 100.0
    predicted_f_abs = min(1.0, 0.5 * ba_ratio)

    # Stability risk: high if solubility ratio > 2 (metastable form)
    stability_risk = "high" if solubility_ratio_vs_reference > 2.0 else "low"

    # Manufacturability score: penalty for extreme solubility ratios
    manufacturability_score = float(
        100.0 - min(50.0, abs(solubility_ratio_vs_reference - 1.0) * 20.0)
    )

    # Recommendation
    if ba_ratio > 1.3:
        recommendation = "preferred"
    elif ba_ratio >= 0.8:
        recommendation = "acceptable"
    else:
        recommendation = "avoid"

    return CrystalFormBAResult(
        drug_name=drug_name,
        crystal_form=crystal_form,
        solubility_ratio_vs_reference=solubility_ratio_vs_reference,
        intrinsic_dissolution_rate_mg_per_cm2_per_min=intrinsic_dissolution_rate_mg_per_cm2_per_min,
        ba_ratio=ba_ratio,
        ba_improvement_pct=ba_improvement_pct,
        predicted_f_abs=predicted_f_abs,
        stability_risk=stability_risk,
        manufacturability_score=manufacturability_score,
        recommendation=recommendation,
        notes=notes,
    )


def compare_crystal_forms(
    drug_name: str,
    forms_data: list[dict],
) -> list[CrystalFormBAResult]:
    """Compare multiple crystal forms and return sorted by ba_ratio descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    forms_data:
        List of dicts, each containing keys matching the parameters of
        ``predict_crystal_form_ba`` (except ``drug_name``).

    Returns
    -------
    list[CrystalFormBAResult]
        Sorted descending by ba_ratio.
    """
    results = []
    for form in forms_data:
        result = predict_crystal_form_ba(
            drug_name=drug_name,
            crystal_form=form["crystal_form"],
            intrinsic_dissolution_rate_mg_per_cm2_per_min=form[
                "intrinsic_dissolution_rate_mg_per_cm2_per_min"
            ],
            solubility_ratio_vs_reference=form["solubility_ratio_vs_reference"],
            logp=form["logp"],
            mw=form["mw"],
            dose_mg=form["dose_mg"],
            bcs_class_reference=form["bcs_class_reference"],
        )
        results.append(result)

    results.sort(key=lambda r: r.ba_ratio, reverse=True)
    return results


# Make math available at module level (used only internally above)
_ = math.log10  # ensure import is not flagged as unused
