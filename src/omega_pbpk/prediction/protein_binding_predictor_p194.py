"""
Phase 194 — Protein Binding Predictor
Predict plasma protein binding (fu, fraction unbound) from molecular descriptors.

Rule-based prediction:
  - Base fu from logP: fu_base = 1 / (1 + exp(0.7 * logP - 1.5))
  - Acidic drugs bind albumin more  (multiply by 0.7 if acid and pKa < 5)
  - Basic drugs bind AAG more      (multiply by 0.8 if base and pKa > 8)
  - Large MW (>600): extra binding  (multiply by 0.9)
  - High PSA (>100): slightly less bound (multiply by 1.2, capped at 1.0)
  - Clamp final fu to [0.001, 1.0]
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

_VALID_IONIZATION_TYPES = ("acid", "base", "neutral", "zwitterion")


@dataclass(frozen=True)
class ProteinBindingResult:
    drug_name: str
    mw_Da: float
    logP: float
    psa_A2: float
    pka: float
    ionization_type: str
    fu_plasma: float
    ppb_pct: float
    binding_class: str  # "highly_bound" | "moderately_bound" | "poorly_bound"
    primary_binding_protein: str  # "albumin" | "AAG" | "non-specific"
    vd_impact: str  # "high_vd_expected" | "low_vd_expected"
    notes: str


def predict_fu_plasma(
    drug_name: str,
    mw_Da: float,
    logP: float,
    psa_A2: float,
    pka: float,
    ionization_type: str,
) -> ProteinBindingResult:
    """Predict fraction unbound in plasma using rule-based physicochemical model.

    Parameters
    ----------
    drug_name       : str   Drug identifier.
    mw_Da           : float Molecular weight (Da). Must be > 0.
    logP            : float Octanol-water partition coefficient.
    psa_A2          : float Polar surface area (Angstrom^2). Must be >= 0.
    pka             : float Most relevant pKa.
    ionization_type : str   One of 'acid', 'base', 'neutral', 'zwitterion'.

    Returns
    -------
    ProteinBindingResult (frozen dataclass)
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")

    ion_lower = ionization_type.lower()
    if ion_lower not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"Invalid ionization_type '{ionization_type}'. "
            f"Must be one of: {list(_VALID_IONIZATION_TYPES)}"
        )

    # --- Step 1: base fu from logP (logistic) ---
    fu = 1.0 / (1.0 + math.exp(0.7 * logP - 1.5))

    # --- Step 2: ionization-based modifiers ---
    primary_protein = "non-specific"

    if ion_lower == "acid" and pka < 5:
        fu *= 0.7
        primary_protein = "albumin"
    elif ion_lower == "base" and pka > 8:
        fu *= 0.8
        primary_protein = "AAG"
    elif ion_lower == "acid":
        primary_protein = "albumin"
    elif ion_lower == "base":
        primary_protein = "AAG"

    # --- Step 3: large MW penalty ---
    if mw_Da > 600:
        fu *= 0.9

    # --- Step 4: high PSA → slightly less bound ---
    if psa_A2 > 100:
        fu *= 1.2
        fu = min(fu, 1.0)

    # --- Step 5: clamp ---
    fu = max(0.001, min(fu, 1.0))

    ppb_pct = (1.0 - fu) * 100.0

    # --- Classification ---
    if fu < 0.1:
        binding_class = "highly_bound"
    elif fu <= 0.5:
        binding_class = "moderately_bound"
    else:
        binding_class = "poorly_bound"

    # --- Vd impact (low fu → high Vd because tissue binding) ---
    vd_impact = "high_vd_expected" if fu < 0.3 else "low_vd_expected"

    notes = (
        f"logP={logP:.2f}, pKa={pka:.2f}, ionization={ionization_type}, "
        f"MW={mw_Da:.1f} Da, PSA={psa_A2:.1f} A2; "
        f"fu={fu:.4f}, PPB={ppb_pct:.1f}%, "
        f"primary protein={primary_protein}"
    )

    return ProteinBindingResult(
        drug_name=drug_name,
        mw_Da=mw_Da,
        logP=logP,
        psa_A2=psa_A2,
        pka=pka,
        ionization_type=ion_lower,
        fu_plasma=fu,
        ppb_pct=ppb_pct,
        binding_class=binding_class,
        primary_binding_protein=primary_protein,
        vd_impact=vd_impact,
        notes=notes,
    )


def screen_protein_binding(
    drugs: Sequence[dict],
) -> list[ProteinBindingResult]:
    """Screen a list of drug descriptor dicts and return results sorted by fu ascending.

    Each dict must contain keys:
        drug_name, mw_Da, logP, psa_A2, pka, ionization_type

    Returns list sorted by fu_plasma ascending (most bound first).
    """
    results = [
        predict_fu_plasma(
            drug_name=d["drug_name"],
            mw_Da=d["mw_Da"],
            logP=d["logP"],
            psa_A2=d["psa_A2"],
            pka=d["pka"],
            ionization_type=d["ionization_type"],
        )
        for d in drugs
    ]
    results.sort(key=lambda r: r.fu_plasma)
    return results
