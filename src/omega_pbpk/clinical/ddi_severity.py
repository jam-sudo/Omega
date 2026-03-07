"""Drug Interaction Severity Classifier.

Classifies DDI severity using FDA 2020 AUCR thresholds with clinical
decision support recommendations.

FDA 2020 thresholds (substrate sensitivity):
- Negligible:  AUCR < 1.25
- Weak:        1.25 <= AUCR < 2.0
- Moderate:    2.0  <= AUCR < 5.0
- Strong:      AUCR >= 5.0

Narrow therapeutic index (narrow_ti) substrates apply the same fold-change
criteria but trigger clinical action at lower thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclasses (all frozen)
# ---------------------------------------------------------------------------

_VALID_SIGNIFICANCE = ("standard", "sensitive", "narrow_ti")

_SEVERITY_ORDER = {"strong": 0, "moderate": 1, "weak": 2, "negligible": 3}


@dataclass(frozen=True)
class DDISeverity:
    """Classification result for a single DDI pair."""

    aucr: float
    severity_category: str  # 'negligible' / 'weak' / 'moderate' / 'strong'
    clinical_significance: str  # 'standard' / 'sensitive' / 'narrow_ti'
    recommendation: str
    notes: str


@dataclass(frozen=True)
class DDIClassificationResult:
    """Per-pair DDI classification for batch results."""

    substrate: str
    perpetrator: str
    aucr: float
    severity_category: str
    recommendation: str
    notes: str


@dataclass(frozen=True)
class DDIRiskProfile:
    """Summary risk profile for a set of DDI interactions."""

    n_interactions: int
    n_strong: int
    n_moderate: int
    n_weak: int
    n_negligible: int
    highest_risk_pair: str  # "substrate + perpetrator"
    overall_risk_score: float  # 0.0 (no risk) to 1.0 (maximum risk)
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _severity_from_aucr(aucr: float) -> str:
    """Determine severity category from AUCR using FDA 2020 thresholds."""
    if aucr < 1.25:
        return "negligible"
    if aucr < 2.0:
        return "weak"
    if aucr < 5.0:
        return "moderate"
    return "strong"


def _build_notes(aucr: float, clinical_significance: str, severity: str) -> str:
    """Build informative notes string."""
    sig_label = {
        "standard": "standard substrate",
        "sensitive": "sensitive substrate",
        "narrow_ti": "narrow therapeutic index substrate",
    }[clinical_significance]
    return (
        f"AUCR={aucr:.2f} for {sig_label}; "
        f"FDA 2020 classification: {severity}."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_ddi_severity(
    aucr: float,
    clinical_significance: str = "standard",
) -> DDISeverity:
    """Classify DDI severity using FDA 2020 AUCR thresholds.

    Parameters
    ----------
    aucr:
        Predicted or observed area-under-the-curve ratio (victim with /
        without perpetrator). Must be > 0.
    clinical_significance:
        Substrate sensitivity category: 'standard', 'sensitive', or
        'narrow_ti'.

    Returns
    -------
    DDISeverity
    """
    if aucr <= 0:
        raise ValueError(f"aucr must be > 0, got {aucr}")
    if clinical_significance not in _VALID_SIGNIFICANCE:
        raise ValueError(
            f"clinical_significance must be one of {_VALID_SIGNIFICANCE}, "
            f"got '{clinical_significance}'"
        )

    severity = _severity_from_aucr(aucr)
    recommendation = recommend_clinical_action(severity, "substrate", "perpetrator")
    notes = _build_notes(aucr, clinical_significance, severity)

    # Narrow TI substrates: upgrade weak → moderate if AUCR >= 1.25
    if clinical_significance == "narrow_ti" and severity == "weak":
        severity = "moderate"
        recommendation = recommend_clinical_action(severity, "substrate", "perpetrator")
        notes += " Upgraded to moderate for narrow-TI substrate."

    return DDISeverity(
        aucr=aucr,
        severity_category=severity,
        clinical_significance=clinical_significance,
        recommendation=recommendation,
        notes=notes,
    )


def recommend_clinical_action(
    severity: str,
    drug_name: str,
    perpetrator: str,
) -> str:
    """Return clinical action recommendation for a given DDI severity level.

    Parameters
    ----------
    severity:
        One of 'negligible', 'weak', 'moderate', 'strong'.
    drug_name:
        Name of the victim (substrate) drug.
    perpetrator:
        Name of the perpetrator drug.

    Returns
    -------
    str
        Human-readable clinical recommendation.
    """
    actions: dict[str, str] = {
        "negligible": "No action required.",
        "weak": (
            f"Monitor {drug_name} for increased or decreased effect when "
            f"co-administered with {perpetrator}; consider dose titration if needed."
        ),
        "moderate": (
            f"Reduce {drug_name} dose by ~50% or use an alternative when "
            f"co-administered with {perpetrator}; monitor closely for efficacy "
            f"and toxicity."
        ),
        "strong": (
            f"Co-administration of {drug_name} with {perpetrator} is contraindicated "
            f"or requires a substantial dose reduction (75%+); consult prescribing "
            f"information and consider therapeutic alternatives."
        ),
    }
    return actions.get(
        severity,
        f"Unknown severity '{severity}'. Review prescribing information.",
    )


def batch_classify(ddi_list: list[dict]) -> list[DDIClassificationResult]:
    """Classify a list of DDI pairs and sort by severity (strong first).

    Parameters
    ----------
    ddi_list:
        List of dicts, each with keys:
        - substrate (str)
        - perpetrator (str)
        - aucr (float)
        - clinical_significance (str, optional, default 'standard')

    Returns
    -------
    list[DDIClassificationResult]
        Sorted: strong -> moderate -> weak -> negligible.
    """
    results: list[DDIClassificationResult] = []

    for entry in ddi_list:
        substrate = str(entry.get("substrate", "unknown"))
        perpetrator = str(entry.get("perpetrator", "unknown"))
        aucr = float(entry["aucr"])
        significance = str(entry.get("clinical_significance", "standard"))

        ddi_sev = classify_ddi_severity(aucr, significance)
        rec = recommend_clinical_action(ddi_sev.severity_category, substrate, perpetrator)

        results.append(
            DDIClassificationResult(
                substrate=substrate,
                perpetrator=perpetrator,
                aucr=aucr,
                severity_category=ddi_sev.severity_category,
                recommendation=rec,
                notes=ddi_sev.notes,
            )
        )

    results.sort(key=lambda r: _SEVERITY_ORDER[r.severity_category])
    return results


def summarize_risk_profile(ddi_list: list[dict]) -> DDIRiskProfile:
    """Compute overall risk summary for a set of DDI interactions.

    Parameters
    ----------
    ddi_list:
        Same format as for :func:`batch_classify`.

    Returns
    -------
    DDIRiskProfile
    """
    classified = batch_classify(ddi_list)
    n = len(classified)

    counts = {"strong": 0, "moderate": 0, "weak": 0, "negligible": 0}
    for r in classified:
        counts[r.severity_category] += 1

    # Highest risk pair = first item (already sorted strong-first)
    if classified:
        top = classified[0]
        highest_pair = f"{top.substrate} + {top.perpetrator}"
    else:
        highest_pair = "none"

    # Overall risk score: weighted sum normalised to [0, 1]
    # weights: strong=1.0, moderate=0.6, weak=0.2, negligible=0.0
    weights = {"strong": 1.0, "moderate": 0.6, "weak": 0.2, "negligible": 0.0}
    if n > 0:
        raw_score = sum(weights[r.severity_category] for r in classified) / n
    else:
        raw_score = 0.0

    # Cap at 1.0
    overall_risk_score = min(raw_score, 1.0)

    # Build notes
    severity_summary = (
        f"Strong: {counts['strong']}, Moderate: {counts['moderate']}, "
        f"Weak: {counts['weak']}, Negligible: {counts['negligible']}."
    )
    if counts["strong"] > 0:
        risk_label = "HIGH overall DDI risk."
    elif counts["moderate"] > 0:
        risk_label = "MODERATE overall DDI risk."
    elif counts["weak"] > 0:
        risk_label = "LOW overall DDI risk."
    else:
        risk_label = "NEGLIGIBLE overall DDI risk."

    notes = f"{severity_summary} {risk_label}"

    return DDIRiskProfile(
        n_interactions=n,
        n_strong=counts["strong"],
        n_moderate=counts["moderate"],
        n_weak=counts["weak"],
        n_negligible=counts["negligible"],
        highest_risk_pair=highest_pair,
        overall_risk_score=overall_risk_score,
        notes=notes,
    )


__all__ = [
    "DDISeverity",
    "DDIClassificationResult",
    "DDIRiskProfile",
    "classify_ddi_severity",
    "recommend_clinical_action",
    "batch_classify",
    "summarize_risk_profile",
]
