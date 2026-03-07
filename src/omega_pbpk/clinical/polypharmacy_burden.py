"""Polypharmacy burden scoring — Phase 263.

Computes an aggregate polypharmacy risk score for patients on multiple
medications, combining DDI severity, drug count, and high-risk drug classes.

References
----------
- Masnoon N et al., BMC Geriatrics. 2017;17(1):230
- Hanlon JT et al., J Clin Epidemiol. 1992 (medication appropriateness index)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["DrugEntry", "PolypharmacyResult", "assess_polypharmacy_burden"]


# High-risk drug classes (Beers criteria / STOPP criteria)
_HIGH_RISK_CLASSES = {
    "anticoagulant": 2.0,
    "nsaid": 1.5,
    "benzodiazepine": 2.0,
    "opioid": 1.5,
    "antipsychotic": 1.5,
    "digoxin": 2.0,
    "insulin": 1.5,
    "anticholinergic": 1.0,
    "diuretic": 1.0,
    "antiepileptic": 1.5,
}

# DDI severity weights
_DDI_SEVERITY = {
    "major": 3.0,
    "moderate": 1.5,
    "minor": 0.5,
    "none": 0.0,
}


@dataclass
class DrugEntry:
    """A single drug in a polypharmacy regimen."""

    name: str
    drug_class: str = "other"  # From _HIGH_RISK_CLASSES keys or "other"
    ddi_pairs: list[str] = field(default_factory=list)  # Names of interacting drugs
    ddi_severity: list[str] = field(default_factory=list)  # Severity for each pair


@dataclass(frozen=True)
class PolypharmacyResult:
    """Result from polypharmacy burden assessment."""

    n_drugs: int
    drug_names: list[str]
    n_high_risk_drugs: int
    high_risk_drug_names: list[str]
    n_ddi_pairs: int
    ddi_burden_score: float
    high_risk_burden_score: float
    total_burden_score: float
    risk_category: str  # "low", "moderate", "high", "very_high"
    recommendation: str
    notes: str


def assess_polypharmacy_burden(
    drugs: list[DrugEntry],
    patient_age: int = 65,
) -> PolypharmacyResult:
    """Assess polypharmacy risk burden for a medication list.

    Parameters
    ----------
    drugs : List of DrugEntry objects.
    patient_age : Patient age (years). Elderly patients (>=65) have higher weight.

    Returns
    -------
    PolypharmacyResult
    """
    if not drugs:
        raise ValueError("drugs list must not be empty")
    if patient_age <= 0:
        raise ValueError("patient_age must be > 0")

    n_drugs = len(drugs)
    age_mult = 1.2 if patient_age >= 65 else 1.0

    # Drug count score: log-based
    count_score = math.log1p(n_drugs) * age_mult

    # High-risk drug score
    hr_score = 0.0
    hr_names = []
    for d in drugs:
        w = _HIGH_RISK_CLASSES.get(d.drug_class, 0.0)
        if w > 0:
            hr_score += w * age_mult
            hr_names.append(d.name)

    # DDI burden score
    ddi_score = 0.0
    n_ddi_pairs = 0
    for d in drugs:
        for sev in d.ddi_severity:
            ddi_score += _DDI_SEVERITY.get(sev, 0.0)
            if sev != "none":
                n_ddi_pairs += 1

    # Normalize to avoid double-counting (pairs counted twice)
    n_ddi_pairs = n_ddi_pairs // 2

    total_score = count_score + hr_score + ddi_score

    if total_score < 3:
        risk_cat = "low"
        rec = "Continue current regimen; review at next visit."
    elif total_score < 7:
        risk_cat = "moderate"
        rec = "Consider medication review; deprescribe lower-priority medications."
    elif total_score < 12:
        risk_cat = "high"
        rec = "Pharmacist-led medication review recommended. Prioritize deprescribing."
    else:
        risk_cat = "very_high"
        rec = "Urgent multidisciplinary medication review. High DDI and adverse event risk."

    notes = (
        f"{n_drugs} drugs ({len(hr_names)} high-risk). "
        f"DDI pairs: {n_ddi_pairs}. "
        f"Score: count={count_score:.1f}, HR={hr_score:.1f}, DDI={ddi_score:.1f}. "
        f"Total={total_score:.1f} \u2192 {risk_cat}."
    )

    return PolypharmacyResult(
        n_drugs=n_drugs,
        drug_names=[d.name for d in drugs],
        n_high_risk_drugs=len(hr_names),
        high_risk_drug_names=hr_names,
        n_ddi_pairs=n_ddi_pairs,
        ddi_burden_score=ddi_score,
        high_risk_burden_score=hr_score,
        total_burden_score=total_score,
        risk_category=risk_cat,
        recommendation=rec,
        notes=notes,
    )
