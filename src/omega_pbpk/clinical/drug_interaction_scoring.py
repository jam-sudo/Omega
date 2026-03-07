"""Drug-drug interaction scoring and classification.

Scores and classifies DDIs based on mechanism, severity, and PK impact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DDIScore:
    drug_a: str
    drug_b: str
    mechanism: str  # "cyp_inhibition","cyp_induction","pgp_inhibition","ppb_displacement","additive_pk","unknown"
    severity: str  # "contraindicated","major","moderate","minor","none"
    aucr: float  # predicted AUC ratio (victim AUC with/without perpetrator)
    cmax_r: float  # Cmax ratio
    clinical_significance: str  # "avoid","monitor_closely","monitor","no_action"
    management: str  # actionable recommendation
    confidence: str  # "high","medium","low"
    notes: str


@dataclass(frozen=True)
class PolypharmacyRiskResult:
    n_drugs: int
    drug_names: tuple
    pairwise_scores: tuple  # tuple of DDIScore for each pair
    n_contraindicated: int
    n_major: int
    n_moderate: int
    overall_risk: str  # "low","moderate","high","critical"
    priority_pair: str  # most severe interaction pair (e.g., "DrugA-DrugB")
    notes: str


def classify_severity_from_aucr(aucr: float) -> str:
    """AUCR >= 5 -> major; >= 2 -> moderate; >= 1.25 -> minor; else none."""
    if aucr >= 5.0:
        return "major"
    elif aucr >= 2.0:
        return "moderate"
    elif aucr >= 1.25:
        return "minor"
    else:
        return "none"


def _management_text(severity: str, mechanism: str) -> str:
    """Return actionable management recommendation based on severity and mechanism."""
    if severity == "contraindicated":
        return (
            "Avoid concomitant use. Consider alternative therapy. "
            "If unavoidable, ensure intensive monitoring and dose reduction of victim drug."
        )
    elif severity == "major":
        return (
            f"Significant interaction via {mechanism}. Reduce victim drug dose or use alternative. "
            "Monitor closely for toxicity or loss of efficacy."
        )
    elif severity == "moderate":
        return (
            f"Moderate interaction via {mechanism}. Monitor patient for altered response. "
            "Dose adjustment may be required."
        )
    elif severity == "minor":
        return (
            f"Minor interaction via {mechanism}. Standard monitoring is sufficient. "
            "Dose adjustment usually not required."
        )
    else:
        return "No clinically significant interaction expected. No special action required."


def score_ddi(
    drug_a: str,
    drug_b: str,
    aucr: float,
    mechanism: str = "cyp_inhibition",
    fm_victim: float = 0.8,
    ki_uM: float | None = None,
    c_max_inhibitor_uM: float | None = None,
) -> DDIScore:
    """Score a drug-drug interaction between drug_a and drug_b.

    Parameters
    ----------
    drug_a:
        Name of perpetrator drug.
    drug_b:
        Name of victim drug.
    aucr:
        Predicted AUC ratio (victim AUC with / without perpetrator).
    mechanism:
        Interaction mechanism.
    fm_victim:
        Fraction of victim drug metabolized by inhibited pathway.
    ki_uM:
        Inhibitor Ki in micromolar (optional, for confidence assessment).
    c_max_inhibitor_uM:
        Inhibitor Cmax in micromolar (optional, for confidence assessment).

    Returns
    -------
    DDIScore
    """
    # Validate inputs
    if aucr <= 0:
        raise ValueError(f"aucr must be > 0, got {aucr}")
    if not (0.0 <= fm_victim <= 1.0):
        raise ValueError(f"fm_victim must be in [0, 1], got {fm_victim}")

    valid_mechanisms = {
        "cyp_inhibition",
        "cyp_induction",
        "pgp_inhibition",
        "ppb_displacement",
        "additive_pk",
        "unknown",
    }
    if mechanism not in valid_mechanisms:
        mechanism = "unknown"

    # Determine severity (with contraindicated override)
    base_severity = classify_severity_from_aucr(aucr)

    # Contraindicated: aucr >= 10 OR (cyp_inhibition with aucr >= 5 and fm > 0.9)
    if aucr >= 10.0:
        severity = "contraindicated"
    elif mechanism == "cyp_inhibition" and aucr >= 5.0 and fm_victim > 0.9:
        severity = "contraindicated"
    else:
        severity = base_severity

    # Approximate Cmax ratio as sqrt(aucr)
    cmax_r = math.sqrt(aucr)

    # Clinical significance mapping
    if severity in ("contraindicated", "major"):
        clinical_significance = "avoid"
    elif severity == "moderate":
        clinical_significance = "monitor_closely"
    elif severity == "minor":
        clinical_significance = "monitor"
    else:
        clinical_significance = "no_action"

    # Management text
    management = _management_text(severity, mechanism)

    # Confidence assessment
    if ki_uM is not None and c_max_inhibitor_uM is not None:
        confidence = "high"
    elif aucr != 1.0:
        confidence = "medium"
    else:
        confidence = "low"

    notes = (
        f"AUCR={aucr:.2f}, fm_victim={fm_victim:.2f}, mechanism={mechanism}. "
        f"FDA 2020 DDI classification applied."
    )

    return DDIScore(
        drug_a=drug_a,
        drug_b=drug_b,
        mechanism=mechanism,
        severity=severity,
        aucr=aucr,
        cmax_r=cmax_r,
        clinical_significance=clinical_significance,
        management=management,
        confidence=confidence,
        notes=notes,
    )


def assess_polypharmacy(
    drug_names: list,
    aucr_matrix: list,
    mechanisms: list | None = None,
) -> PolypharmacyRiskResult:
    """Assess DDI risk for a polypharmacy regimen.

    Parameters
    ----------
    drug_names:
        List of drug names (n drugs).
    aucr_matrix:
        n×n matrix of pairwise AUCRs (list of lists). aucr_matrix[i][j] is the
        AUCR with drug i as perpetrator and drug j as victim.
    mechanisms:
        Optional n×n matrix of mechanism strings. If None, "unknown" is used.

    Returns
    -------
    PolypharmacyRiskResult
    """
    n = len(drug_names)
    if n < 2:
        raise ValueError(f"At least 2 drugs required, got {n}")
    if len(aucr_matrix) != n:
        raise ValueError(f"aucr_matrix rows ({len(aucr_matrix)}) must match n_drugs ({n})")
    for i, row in enumerate(aucr_matrix):
        if len(row) != n:
            raise ValueError(f"aucr_matrix row {i} length ({len(row)}) must match n_drugs ({n})")

    pairwise: list[DDIScore] = []
    for i in range(n):
        for j in range(i + 1, n):
            aucr_val = aucr_matrix[i][j]
            mech = "unknown"
            if mechanisms is not None:
                mech = mechanisms[i][j]
            score = score_ddi(
                drug_a=drug_names[i],
                drug_b=drug_names[j],
                aucr=aucr_val,
                mechanism=mech,
            )
            pairwise.append(score)

    n_contraindicated = sum(1 for s in pairwise if s.severity == "contraindicated")
    n_major = sum(1 for s in pairwise if s.severity == "major")
    n_moderate = sum(1 for s in pairwise if s.severity == "moderate")

    # Determine overall risk
    if n_contraindicated > 0:
        overall_risk = "critical"
    elif n_major > 0:
        overall_risk = "high"
    elif n_moderate > 0:
        overall_risk = "moderate"
    else:
        overall_risk = "low"

    # Priority pair: highest AUCR
    if pairwise:
        top = max(pairwise, key=lambda s: s.aucr)
        priority_pair = f"{top.drug_a}-{top.drug_b}"
    else:
        priority_pair = ""

    notes = (
        f"Polypharmacy analysis of {n} drugs ({len(pairwise)} pairs). "
        f"Contraindicated: {n_contraindicated}, Major: {n_major}, Moderate: {n_moderate}. "
        f"Overall risk: {overall_risk}."
    )

    return PolypharmacyRiskResult(
        n_drugs=n,
        drug_names=tuple(drug_names),
        pairwise_scores=tuple(pairwise),
        n_contraindicated=n_contraindicated,
        n_major=n_major,
        n_moderate=n_moderate,
        overall_risk=overall_risk,
        priority_pair=priority_pair,
        notes=notes,
    )
