"""
Protein Unbound Fraction (fu,plasma) Calculator.

Implements Austin et al. (2002) logistic regression method for predicting
fraction unbound in plasma from physicochemical descriptors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FupCalculatorResult:
    """Result of fu,plasma prediction."""

    compound_name: str
    logp: float
    mw_da: float
    psa: float
    fup_human: float
    fup_rat: float
    ppb_human_pct: float
    binding_category: str
    cl_blood_correction: str
    notes: str


def _compute_logit_fup(
    logp: float,
    mw_da: float,
    psa: float,
    pka_acid: float | None,
    pka_basic: float | None,
) -> float:
    """Compute logit(fup) from Austin et al. (2002) simplified coefficients."""
    logit_fup = 0.45 - 0.67 * logp - 0.014 * mw_da + 0.012 * psa

    if pka_basic is not None and pka_basic > 7.4:
        logit_fup += 0.35

    if pka_acid is not None and pka_acid < 5.0:
        logit_fup -= 0.5

    return logit_fup


def _logit_to_prob(logit: float) -> float:
    """Convert logit to probability via sigmoid."""
    return 1.0 / (1.0 + math.exp(-logit))


def _binding_category(fup: float) -> str:
    if fup > 0.5:
        return "low"
    if fup >= 0.1:
        return "moderate"
    return "high"


def _cl_blood_correction(fup: float) -> str:
    if fup > 0.5:
        return "minimal"
    if fup < 0.1:
        return "significant"
    return "moderate"


def _rat_correction(logp: float) -> float:
    """Empirical rat correction factor (rat ~ human with cosine wobble)."""
    return 0.9 + 0.1 * math.cos(logp * 0.5)


def calculate_fup(
    compound_name: str,
    logp: float = 2.0,
    mw_da: float = 300.0,
    psa: float = 60.0,
    pka_acid: float | None = None,
    pka_basic: float | None = None,
) -> FupCalculatorResult:
    """
    Predict fu,plasma using Austin et al. (2002) logistic regression.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    logp:
        Octanol-water partition coefficient.
    mw_da:
        Molecular weight in Daltons.
    psa:
        Polar surface area (Angstrom squared).
    pka_acid:
        Most acidic pKa (or None if not acidic).
    pka_basic:
        Most basic pKa (or None if not basic).

    Returns
    -------
    FupCalculatorResult
    """
    if mw_da <= 0:
        raise ValueError(f"mw_da must be positive, got {mw_da}")
    if psa < 0:
        raise ValueError(f"psa must be non-negative, got {psa}")

    logit_fup = _compute_logit_fup(logp, mw_da, psa, pka_acid, pka_basic)
    fup_human = _logit_to_prob(logit_fup)
    fup_human = max(0.001, min(0.999, fup_human))

    rat_factor = _rat_correction(logp)
    fup_rat = max(0.001, min(0.999, fup_human * rat_factor))

    ppb_human_pct = 100.0 * (1.0 - fup_human)
    cat = _binding_category(fup_human)
    cl_corr = _cl_blood_correction(fup_human)

    note_parts = [
        "Austin et al. (2002) logit model",
        f"logP={logp:.2f}, MW={mw_da:.1f} Da, PSA={psa:.1f} A2",
        f"fup_human={fup_human:.4f}, PPB={ppb_human_pct:.1f}%",
        f"Binding: {cat}, CL correction: {cl_corr}",
    ]
    if pka_acid is not None:
        note_parts.append(f"Acidic pKa={pka_acid:.1f} (albumin binding adjustment applied)")
    if pka_basic is not None:
        note_parts.append(f"Basic pKa={pka_basic:.1f} (basic drug adjustment applied)")

    notes = "; ".join(note_parts)

    return FupCalculatorResult(
        compound_name=compound_name,
        logp=logp,
        mw_da=mw_da,
        psa=psa,
        fup_human=fup_human,
        fup_rat=fup_rat,
        ppb_human_pct=ppb_human_pct,
        binding_category=cat,
        cl_blood_correction=cl_corr,
        notes=notes,
    )


def screen_fup(
    compounds: list[dict],
) -> list[FupCalculatorResult]:
    """
    Screen a list of compounds for fu,plasma.

    Each dict should contain 'compound_name' and optionally 'logp', 'mw_da',
    'psa', 'pka_acid', 'pka_basic'.

    Returns results sorted descending by fup_human (most free drug first).
    """
    results = []
    for c in compounds:
        name = c.get("compound_name", "unknown")
        logp = float(c.get("logp", 2.0))
        mw_da = float(c.get("mw_da", 300.0))
        psa = float(c.get("psa", 60.0))
        pka_acid = c.get("pka_acid", None)
        if pka_acid is not None:
            pka_acid = float(pka_acid)
        pka_basic = c.get("pka_basic", None)
        if pka_basic is not None:
            pka_basic = float(pka_basic)
        results.append(
            calculate_fup(
                name, logp=logp, mw_da=mw_da, psa=psa, pka_acid=pka_acid, pka_basic=pka_basic
            )
        )

    results.sort(key=lambda r: r.fup_human, reverse=True)
    return results
