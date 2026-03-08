"""Phase 991 — Drug Permeability Prediction.

Predict apparent intestinal permeability (Papp) from physicochemical descriptors
using the Palm et al. 1997 / Egan 2000 multilinear approach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PermeabilityPredictionResult",
    "predict_permeability",
    "screen_permeability",
]

_VALID_IONIZATION = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class PermeabilityPredictionResult:
    """Result of Caco-2 / human intestinal permeability prediction."""

    drug_name: str
    logp: float
    mw: float
    psa: float
    hbd: int
    papp_cm_s: float
    log_papp: float
    peff_cm_s: float
    fa_predicted: float
    permeability_class: str
    bcs_permeability: str
    p_gp_efflux_correction: float
    notes: str


def predict_permeability(
    drug_name: str,
    logp: float,
    mw: float,
    psa: float = 80.0,
    hbd: int = 2,
    ionization_type: str = "neutral",
    pgp_substrate: bool = False,
) -> PermeabilityPredictionResult:
    """Predict intestinal permeability from physicochemical descriptors.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    logp:
        Octanol-water partition coefficient (log scale).
    mw:
        Molecular weight (Da).
    psa:
        Polar surface area (Angstrom^2).
    hbd:
        Hydrogen bond donors (count).
    ionization_type:
        One of "acid", "base", "neutral".
    pgp_substrate:
        If True, applies a 2.5-fold efflux correction (reduces Papp).

    Returns
    -------
    PermeabilityPredictionResult
    """
    # --- Validation ---
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if hbd < 0:
        raise ValueError(f"hbd must be >= 0, got {hbd}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got {ionization_type!r}"
        )

    # --- Multilinear Papp model (Palm et al. 1997 / Egan 2000) ---
    log_papp = -0.90 - 0.011 * psa - 0.24 * hbd + 0.27 * logp - 0.0015 * mw
    papp_cm_s = 10.0**log_papp

    # --- P-gp efflux correction ---
    p_gp_efflux_correction = 2.5 if pgp_substrate else 1.0
    if pgp_substrate:
        papp_effective = papp_cm_s / 2.5
    else:
        papp_effective = papp_cm_s

    # --- Human Peff scaling (Sun 2004) ---
    peff_cm_s = papp_effective * 1.8e-4

    # --- Fraction absorbed (simple saturation model) ---
    fa_predicted = peff_cm_s / (peff_cm_s + 5e-4)
    fa_predicted = max(0.01, min(0.99, fa_predicted))

    # --- Permeability classification ---
    if papp_effective > 20e-6:
        permeability_class = "high"
    elif papp_effective >= 2e-6:
        permeability_class = "medium"
    else:
        permeability_class = "low"

    # --- BCS permeability boundary (Peff > 2e-4 cm/s = BCS high) ---
    bcs_permeability = "high" if peff_cm_s > 2e-4 else "low"

    # --- Notes ---
    note_parts: list[str] = [f"Palm/Egan multilinear model; ionization={ionization_type}"]
    if pgp_substrate:
        note_parts.append("P-gp efflux applied (ER=2.5)")
    notes = "; ".join(note_parts)

    return PermeabilityPredictionResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        psa=psa,
        hbd=hbd,
        papp_cm_s=papp_effective,
        log_papp=math.log10(papp_effective),
        peff_cm_s=peff_cm_s,
        fa_predicted=fa_predicted,
        permeability_class=permeability_class,
        bcs_permeability=bcs_permeability,
        p_gp_efflux_correction=p_gp_efflux_correction,
        notes=notes,
    )


def screen_permeability(compounds: list[dict]) -> list[PermeabilityPredictionResult]:
    """Screen a list of compounds by permeability, sorted by Papp descending.

    Each compound dict must have at minimum: "drug_name", "logp", "mw".
    Optional keys: "psa", "hbd", "ionization_type", "pgp_substrate".
    """
    results: list[PermeabilityPredictionResult] = []
    for c in compounds:
        result = predict_permeability(
            drug_name=c["drug_name"],
            logp=c["logp"],
            mw=c["mw"],
            psa=c.get("psa", 80.0),
            hbd=c.get("hbd", 2),
            ionization_type=c.get("ionization_type", "neutral"),
            pgp_substrate=c.get("pgp_substrate", False),
        )
        results.append(result)
    results.sort(key=lambda r: r.papp_cm_s, reverse=True)
    return results
