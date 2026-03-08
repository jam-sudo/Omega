"""Phase 644 — Intestinal Permeability Prediction.

Predicts effective intestinal permeability (Peff) from physicochemical
descriptors. Assigns BCS classification and estimates fraction absorbed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IntestinalPermResult:
    """Result from intestinal permeability prediction."""

    drug_name: str
    logP: float
    mw_Da: float
    psa_A2: float
    n_hbd: int
    peff_cm_s: float
    log_peff: float
    bcs_class: str
    fa_predicted_pct: float
    absorption_class: str
    permeability_category: str
    notes: str


def _validate_inputs(
    mw_Da: float,
    psa_A2: float,
    n_hbd: int,
) -> None:
    if mw_Da <= 0.0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if psa_A2 < 0.0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be >= 0, got {n_hbd}")


def predict_intestinal_permeability(
    drug_name: str,
    logP: float,
    mw_Da: float,
    psa_A2: float,
    n_hbd: int = 2,
    dose_mg: float = 100.0,
    solubility_mg_mL: float = 1.0,
) -> IntestinalPermResult:
    """Predict intestinal permeability and BCS classification."""
    _validate_inputs(mw_Da, psa_A2, n_hbd)

    # Permeability prediction
    log_peff_raw = -4.36 - 0.011 * psa_A2 - 0.016 * mw_Da + 0.46 * logP - 0.044 * n_hbd
    peff_raw = 10.0**log_peff_raw

    # Clamp to [1e-7, 1e-3]
    peff = max(1e-7, min(1e-3, peff_raw))
    log_peff = math.log10(peff)

    # Dose number
    do = dose_mg / (solubility_mg_mL * 250.0)

    # BCS classification
    high_perm = peff >= 1e-4
    high_sol = do <= 1.0

    if high_perm and high_sol:
        bcs_class = "I"
    elif high_perm and not high_sol:
        bcs_class = "II"
    elif not high_perm and high_sol:
        bcs_class = "III"
    else:
        bcs_class = "IV"

    # Fraction absorbed
    fa = min(100.0, max(0.0, (peff / 1e-4) * 80.0))

    # Permeability category
    if peff >= 1e-4:
        perm_cat = "high"
    elif peff >= 1e-5:
        perm_cat = "medium"
    else:
        perm_cat = "low"

    # Absorption class
    if fa > 80.0:
        abs_class = "complete"
    elif fa >= 60.0:
        abs_class = "good"
    elif fa >= 30.0:
        abs_class = "moderate"
    else:
        abs_class = "poor"

    notes = f"BCS {bcs_class}; Do={do:.2f}; perm={perm_cat}"

    return IntestinalPermResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        psa_A2=psa_A2,
        n_hbd=n_hbd,
        peff_cm_s=peff,
        log_peff=log_peff,
        bcs_class=bcs_class,
        fa_predicted_pct=fa,
        absorption_class=abs_class,
        permeability_category=perm_cat,
        notes=notes,
    )


def screen_permeability(
    compounds: list[dict],
) -> list[IntestinalPermResult]:
    """Screen multiple compounds, return sorted by Peff descending."""
    results = []
    for comp in compounds:
        r = predict_intestinal_permeability(
            drug_name=comp["drug_name"],
            logP=comp["logP"],
            mw_Da=comp["mw_Da"],
            psa_A2=comp["psa_A2"],
            n_hbd=comp.get("n_hbd", 2),
            dose_mg=comp.get("dose_mg", 100.0),
            solubility_mg_mL=comp.get("solubility_mg_mL", 1.0),
        )
        results.append(r)
    results.sort(key=lambda x: x.peff_cm_s, reverse=True)
    return results
