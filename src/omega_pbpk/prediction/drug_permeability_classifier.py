"""
Phase 934 — Drug Membrane Permeability Classifier

Classify drug membrane permeability from molecular descriptors using
Lipinski Ro5, Veber rules, and physicochemical property-based rules.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DrugPermeabilityClassResult",
    "classify_permeability",
    "screen_permeability_class",
]

_PERM_ORDER = {"high": 0, "medium": 1, "low": 2}

VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class DrugPermeabilityClassResult:
    """Frozen result of permeability classification."""

    drug_name: str
    logp: float
    mw: float
    psa_estimated: float
    hbd: int
    hba: int
    rotatable_bonds: int
    permeability_class: str  # "high", "medium", or "low"
    lipinski_violations: int
    fails_ro5: bool
    oral_bioavailability_risk: str  # "low", "moderate", or "high"
    veber_compliant: bool
    bbb_likelihood: str  # "likely" or "unlikely"
    notes: str


def _estimate_psa(mw: float, logp: float, ionization_type: str) -> float:
    """
    Estimate PSA from MW, logP, and ionization type.

    PSA ≈ MW * 0.12 * (1 - 0.08 * logP) + ionization_contribution
    ionization_contribution: acid=20, base=15, neutral=5
    Clamped to [0, 200].
    """
    ionization_contrib = {"acid": 20.0, "base": 15.0, "neutral": 5.0}
    contrib = ionization_contrib.get(ionization_type, 5.0)
    psa = mw * 0.12 * (1.0 - 0.08 * logp) + contrib
    return max(0.0, min(200.0, psa))


def _classify_permeability_class(logp: float, mw: float, psa: float, hbd: int) -> str:
    """Classify permeability based on physicochemical descriptors."""
    if logp >= 1.0 and logp <= 3.0 and mw < 500.0 and psa < 60.0 and hbd <= 2:
        return "high"
    elif mw < 500.0 and psa < 90.0 and hbd <= 3:
        return "medium"
    else:
        return "low"


def _count_lipinski_violations(logp: float, mw: float, hbd: int, hba: int) -> int:
    """Count Lipinski Rule of 5 violations."""
    violations = 0
    if mw > 500.0:
        violations += 1
    if logp > 5.0:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
    return violations


def classify_permeability(
    drug_name: str,
    logp: float,
    mw: float,
    hbd: int,
    hba: int,
    rotatable_bonds: int,
    pka: float = 7.0,
    ionization_type: str = "neutral",
) -> DrugPermeabilityClassResult:
    """
    Classify drug membrane permeability class from molecular descriptors.

    Parameters
    ----------
    drug_name : str
    logp : float
    mw : float          Molecular weight (Da)
    hbd : int           H-bond donors
    hba : int           H-bond acceptors
    rotatable_bonds : int
    pka : float         (unused in current model, kept for API compatibility)
    ionization_type : str   "acid", "base", or "neutral"

    Returns
    -------
    DrugPermeabilityClassResult (frozen)
    """
    # Validation
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if hbd < 0:
        raise ValueError("hbd must be >= 0")
    if hba < 0:
        raise ValueError("hba must be >= 0")
    if rotatable_bonds < 0:
        raise ValueError("rotatable_bonds must be >= 0")
    if ionization_type not in VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {VALID_IONIZATION_TYPES}, got '{ionization_type}'"
        )

    psa = _estimate_psa(mw, logp, ionization_type)
    perm_class = _classify_permeability_class(logp, mw, psa, hbd)
    violations = _count_lipinski_violations(logp, mw, hbd, hba)
    fails_ro5 = violations >= 2

    # Oral bioavailability risk
    if fails_ro5:
        oral_risk = "high"
    elif violations == 1 or perm_class == "low":
        oral_risk = "moderate"
    else:
        oral_risk = "low"

    # Veber compliance: PSA <= 140 AND rotatable_bonds <= 10
    veber_compliant = psa <= 140.0 and rotatable_bonds <= 10

    # BBB likelihood: PSA < 90 AND MW < 450 AND logP in [0, 5]
    bbb_likely = psa < 90.0 and mw < 450.0 and 0.0 <= logp <= 5.0
    bbb_likelihood = "likely" if bbb_likely else "unlikely"

    # Notes
    notes_parts = []
    notes_parts.append(f"Permeability class: {perm_class}.")
    notes_parts.append(f"Lipinski violations: {violations}.")
    if fails_ro5:
        notes_parts.append("Fails Ro5 (>= 2 violations).")
    if not veber_compliant:
        notes_parts.append("Does not meet Veber criteria (PSA > 140 or RotB > 10).")
    notes_parts.append(f"BBB penetration: {bbb_likelihood}.")
    notes_parts.append(f"Oral bioavailability risk: {oral_risk}.")
    notes = " ".join(notes_parts)

    return DrugPermeabilityClassResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        psa_estimated=psa,
        hbd=hbd,
        hba=hba,
        rotatable_bonds=rotatable_bonds,
        permeability_class=perm_class,
        lipinski_violations=violations,
        fails_ro5=fails_ro5,
        oral_bioavailability_risk=oral_risk,
        veber_compliant=veber_compliant,
        bbb_likelihood=bbb_likelihood,
        notes=notes,
    )


def screen_permeability_class(
    compounds: list[dict],
) -> list[DrugPermeabilityClassResult]:
    """
    Screen a list of compounds for permeability class.

    Each dict must contain: drug_name, logp, mw, hbd, hba, rotatable_bonds.
    Optional: pka, ionization_type.

    Returns list sorted by permeability_class ("high" > "medium" > "low").
    """
    results = []
    for comp in compounds:
        result = classify_permeability(
            drug_name=comp["drug_name"],
            logp=comp["logp"],
            mw=comp["mw"],
            hbd=comp["hbd"],
            hba=comp["hba"],
            rotatable_bonds=comp["rotatable_bonds"],
            pka=comp.get("pka", 7.0),
            ionization_type=comp.get("ionization_type", "neutral"),
        )
        results.append(result)

    results.sort(key=lambda r: _PERM_ORDER.get(r.permeability_class, 99))
    return results
