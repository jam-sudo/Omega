"""Composite drug-drug interaction risk scoring system."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DDIRiskEntry", "DDIRiskReport", "score_ddi_risk", "screen_combinations"]


@dataclass(frozen=True)
class DDIRiskEntry:
    drug_a: str
    drug_b: str
    mechanism: str  # "cyp_inhibition", "cyp_induction", "transporter", "pk_additive", "pd_synergy"
    severity: str  # "none", "minor", "moderate", "major", "contraindicated"
    aucr_estimate: float
    risk_score: float  # 0-100
    clinical_relevance: str
    management: str


@dataclass(frozen=True)
class DDIRiskReport:
    primary_drug: str
    interacting_drugs: list
    interactions: list
    max_risk_score: float
    n_major_interactions: int
    overall_risk: str  # "low", "moderate", "high", "critical"
    priority_alert: str
    notes: str


def _score_interaction(
    primary_drug: str,
    interacting: dict,
    primary_cyp3a4_substrate: bool,
    primary_cyp2d6_substrate: bool,
    primary_pgp_substrate: bool,
) -> DDIRiskEntry:
    """Compute a DDIRiskEntry for a single interacting drug."""
    drug_b = interacting.get("name", "unknown")
    risk_score = 0.0
    mechanism = "none"
    aucr = 1.0

    # CYP3A4 inhibition of primary substrate
    ki3a4 = interacting.get("cyp3a4_inhibitor_ki") or interacting.get("cyp3a4_inhibition_ki_uM")
    if primary_cyp3a4_substrate and ki3a4:
        assumed_plasma = 1.0  # uM
        aucr_3a4 = 1.0 + assumed_plasma / ki3a4
        aucr = aucr_3a4
        risk_score += min(50.0, (aucr_3a4 - 1.0) * 10.0)
        mechanism = "cyp_inhibition"

    # CYP2D6 inhibition of primary substrate
    ki2d6 = interacting.get("cyp2d6_inhibitor_ki") or interacting.get("cyp2d6_inhibition_ki_uM")
    if primary_cyp2d6_substrate and ki2d6:
        aucr_2d6 = 1.0 + 1.0 / ki2d6
        risk_score += min(40.0, (aucr_2d6 - 1.0) * 8.0)
        if mechanism == "none":
            mechanism = "cyp_inhibition"

    # P-gp interaction
    if primary_pgp_substrate and interacting.get("pgp_inhibitor"):
        risk_score += 15.0
        aucr *= 1.5
        if mechanism == "none":
            mechanism = "transporter"

    # Cap at 100
    risk_score = min(100.0, risk_score)

    # Severity from score
    if risk_score >= 70.0:
        severity = "contraindicated"
    elif risk_score >= 50.0:
        severity = "major"
    elif risk_score >= 30.0:
        severity = "moderate"
    elif risk_score >= 10.0:
        severity = "minor"
    else:
        severity = "none"

    # Clinical relevance
    if severity in ("contraindicated", "major"):
        clinical_relevance = "Avoid combination"
    elif severity == "moderate":
        clinical_relevance = "Monitor closely"
    else:
        clinical_relevance = "Minimal clinical concern"

    # Management
    if severity == "contraindicated":
        management = "Contraindicated — use alternative"
    elif severity == "major":
        management = "Use with caution; consider dose reduction"
    else:
        management = "Routine monitoring"

    return DDIRiskEntry(
        drug_a=primary_drug,
        drug_b=drug_b,
        mechanism=mechanism,
        severity=severity,
        aucr_estimate=aucr,
        risk_score=risk_score,
        clinical_relevance=clinical_relevance,
        management=management,
    )


def score_ddi_risk(
    primary_drug: str,
    primary_cyp3a4_inhibition_ki_uM: float = None,
    primary_cyp2d6_inhibition_ki_uM: float = None,
    primary_cyp3a4_substrate: bool = False,
    primary_cyp2d6_substrate: bool = False,
    primary_pgp_inhibitor: bool = False,
    primary_pgp_substrate: bool = False,
    interacting_drugs: list = None,
) -> DDIRiskReport:
    """Score the DDI risk between a primary drug and a list of interacting drugs.

    Parameters
    ----------
    primary_drug:
        Name of the primary (victim) drug.
    primary_cyp3a4_inhibition_ki_uM:
        Ki for CYP3A4 inhibition by primary drug (uM), or None.
    primary_cyp2d6_inhibition_ki_uM:
        Ki for CYP2D6 inhibition by primary drug (uM), or None.
    primary_cyp3a4_substrate:
        Whether primary drug is a CYP3A4 substrate.
    primary_cyp2d6_substrate:
        Whether primary drug is a CYP2D6 substrate.
    primary_pgp_inhibitor:
        Whether primary drug is a P-gp inhibitor.
    primary_pgp_substrate:
        Whether primary drug is a P-gp substrate.
    interacting_drugs:
        List of dicts describing perpetrator drugs. Each dict may contain:
        - "name": drug name
        - "cyp3a4_inhibitor_ki": Ki in uM for CYP3A4
        - "cyp2d6_inhibitor_ki": Ki in uM for CYP2D6
        - "pgp_inhibitor": bool

    Returns
    -------
    DDIRiskReport
    """
    if interacting_drugs is None:
        interacting_drugs = []

    drug_names = [d.get("name", "unknown") for d in interacting_drugs]

    interactions: list[DDIRiskEntry] = []
    for drug in interacting_drugs:
        entry = _score_interaction(
            primary_drug=primary_drug,
            interacting=drug,
            primary_cyp3a4_substrate=primary_cyp3a4_substrate,
            primary_cyp2d6_substrate=primary_cyp2d6_substrate,
            primary_pgp_substrate=primary_pgp_substrate,
        )
        interactions.append(entry)

    # Overall risk assessment
    max_score = max((i.risk_score for i in interactions), default=0.0)
    n_major = sum(1 for i in interactions if i.severity in ("major", "contraindicated"))

    if max_score >= 70.0 or n_major >= 2:
        overall = "critical"
    elif max_score >= 50.0:
        overall = "high"
    elif max_score >= 30.0:
        overall = "moderate"
    else:
        overall = "low"

    # Priority alert: the most serious interaction description
    if interactions:
        worst = max(interactions, key=lambda x: x.risk_score)
        priority_alert = (
            f"{worst.drug_b}: {worst.severity} {worst.mechanism} interaction "
            f"(risk score={worst.risk_score:.1f}, AUCR={worst.aucr_estimate:.2f}x)"
        )
    else:
        priority_alert = "No interactions identified"

    notes = (
        f"{len(interactions)} interaction(s) evaluated; "
        f"primary drug CYP3A4_substrate={primary_cyp3a4_substrate}, "
        f"CYP2D6_substrate={primary_cyp2d6_substrate}, "
        f"P-gp_substrate={primary_pgp_substrate}"
    )

    return DDIRiskReport(
        primary_drug=primary_drug,
        interacting_drugs=drug_names,
        interactions=interactions,
        max_risk_score=max_score,
        n_major_interactions=n_major,
        overall_risk=overall,
        priority_alert=priority_alert,
        notes=notes,
    )


def screen_combinations(
    primary_drug: str,
    primary_drug_props: dict,
    candidate_drugs: list,
) -> list:
    """Score primary drug vs each candidate and return sorted by max_risk_score descending.

    Parameters
    ----------
    primary_drug:
        Name of the primary drug.
    primary_drug_props:
        Dict of primary drug properties (same keys as score_ddi_risk parameters).
    candidate_drugs:
        List of dicts, each describing one candidate interacting drug.

    Returns
    -------
    list of DDIRiskReport, sorted by max_risk_score descending.
    """
    reports: list[DDIRiskReport] = []
    for candidate in candidate_drugs:
        report = score_ddi_risk(
            primary_drug=primary_drug,
            primary_cyp3a4_inhibition_ki_uM=primary_drug_props.get("cyp3a4_inhibition_ki_uM"),
            primary_cyp2d6_inhibition_ki_uM=primary_drug_props.get("cyp2d6_inhibition_ki_uM"),
            primary_cyp3a4_substrate=primary_drug_props.get("cyp3a4_substrate", False),
            primary_cyp2d6_substrate=primary_drug_props.get("cyp2d6_substrate", False),
            primary_pgp_inhibitor=primary_drug_props.get("pgp_inhibitor", False),
            primary_pgp_substrate=primary_drug_props.get("pgp_substrate", False),
            interacting_drugs=[candidate],
        )
        reports.append(report)

    reports.sort(key=lambda r: r.max_risk_score, reverse=True)
    return reports
