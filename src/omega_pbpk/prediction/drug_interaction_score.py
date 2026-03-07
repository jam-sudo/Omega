"""Composite drug-drug interaction (DDI) risk scoring (Phase 712).

Predicts DDI risk from CYP/transporter/protein binding profiles.
Generates victim score (susceptibility to inhibition) and perpetrator score
(ability to inhibit others). Overall risk is capped at 100.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DDIScoreResult",
    "compute_ddi_score",
    "screen_ddi_scores",
    "estimate_clinical_aucr",
]

_VALID_TI = {"wide", "narrow", "moderate"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DDIScoreResult:
    """Result from DDI risk scoring."""

    compound_name: str
    victim_score: float
    perpetrator_score: float
    overall_ddi_risk: float
    risk_class: str
    requires_clinical_ddi_study: bool
    primary_interaction_pathway: str
    notes: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    fup: float,
    ki_3a4_uM: float | None,
    ki_2d6_uM: float | None,
    ki_2c9_uM: float | None,
    therapeutic_index: str,
) -> None:
    if not (0 < fup <= 1):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if ki_3a4_uM is not None and ki_3a4_uM <= 0:
        raise ValueError(f"ki_3a4_uM must be > 0, got {ki_3a4_uM}")
    if ki_2d6_uM is not None and ki_2d6_uM <= 0:
        raise ValueError(f"ki_2d6_uM must be > 0, got {ki_2d6_uM}")
    if ki_2c9_uM is not None and ki_2c9_uM <= 0:
        raise ValueError(f"ki_2c9_uM must be > 0, got {ki_2c9_uM}")
    if therapeutic_index not in _VALID_TI:
        raise ValueError(
            f"therapeutic_index must be one of {sorted(_VALID_TI)}, got '{therapeutic_index}'"
        )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _ki_to_perp_points(ki_uM: float | None) -> float:
    """Convert Ki (µM) to perpetrator sub-score."""
    if ki_uM is None:
        return 0.0
    if ki_uM < 1.0:
        return 30.0
    if ki_uM < 10.0:
        return 20.0
    if ki_uM < 100.0:
        return 10.0
    return 0.0


def _determine_primary_pathway(
    is_cyp3a4_substrate: bool,
    is_cyp3a4_inhibitor: bool,
    ki_3a4_uM: float | None,
    is_cyp2d6_substrate: bool,
    is_cyp2d6_inhibitor: bool,
    ki_2d6_uM: float | None,
    is_cyp2c9_substrate: bool,
    is_cyp2c9_inhibitor: bool,
    ki_2c9_uM: float | None,
    is_pgp_substrate: bool,
    is_pgp_inhibitor: bool,
    is_oatp_substrate: bool,
) -> str:
    """Identify primary interaction pathway based on highest risk contribution."""
    pathways: list[tuple[float, str]] = []

    # Substrate scores (victim)
    if is_cyp3a4_substrate:
        pathways.append((20.0, "CYP3A4 substrate"))
    if is_cyp2d6_substrate:
        pathways.append((15.0, "CYP2D6 substrate"))
    if is_cyp2c9_substrate:
        pathways.append((15.0, "CYP2C9 substrate"))
    if is_pgp_substrate:
        pathways.append((10.0, "P-gp substrate"))
    if is_oatp_substrate:
        pathways.append((10.0, "OATP substrate"))

    # Inhibitor scores (perpetrator)
    perp_3a4 = _ki_to_perp_points(ki_3a4_uM)
    if perp_3a4 > 0 or is_cyp3a4_inhibitor:
        pathways.append((max(perp_3a4, 10.0 if is_cyp3a4_inhibitor else 0.0), "CYP3A4 inhibitor"))
    perp_2d6 = _ki_to_perp_points(ki_2d6_uM)
    if perp_2d6 > 0 or is_cyp2d6_inhibitor:
        pathways.append((max(perp_2d6, 10.0 if is_cyp2d6_inhibitor else 0.0), "CYP2D6 inhibitor"))
    perp_2c9 = _ki_to_perp_points(ki_2c9_uM)
    if perp_2c9 > 0 or is_cyp2c9_inhibitor:
        pathways.append((max(perp_2c9, 10.0 if is_cyp2c9_inhibitor else 0.0), "CYP2C9 inhibitor"))
    if is_pgp_inhibitor:
        pathways.append((15.0, "P-gp inhibitor"))

    if not pathways:
        return "none"

    pathways.sort(key=lambda x: x[0], reverse=True)
    return pathways[0][1]


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------


def compute_ddi_score(
    compound_name: str,
    is_cyp3a4_substrate: bool = False,
    is_cyp3a4_inhibitor: bool = False,
    is_cyp2d6_substrate: bool = False,
    is_cyp2d6_inhibitor: bool = False,
    is_cyp2c9_substrate: bool = False,
    is_cyp2c9_inhibitor: bool = False,
    ki_3a4_uM: float | None = None,
    ki_2d6_uM: float | None = None,
    ki_2c9_uM: float | None = None,
    is_pgp_substrate: bool = False,
    is_pgp_inhibitor: bool = False,
    is_oatp_substrate: bool = False,
    fup: float = 0.5,
    therapeutic_index: str = "wide",
) -> DDIScoreResult:
    """Compute composite DDI risk score for a compound.

    Scoring components (0-100 scale):
    - Victim score: susceptibility to inhibition by perpetrators
    - Perpetrator score: ability to inhibit other drugs
    - Overall risk = max(victim, perpetrator) * TI_factor (narrow TI -> x1.5)

    Risk classification:
        < 20: "low"
        20-50: "moderate"
        > 50: "high"

    Regulatory flag: requires_clinical_ddi_study if overall_ddi_risk > 30.
    """
    _validate_inputs(fup, ki_3a4_uM, ki_2d6_uM, ki_2c9_uM, therapeutic_index)

    # --- Victim score ---
    victim_score = 0.0
    if is_cyp3a4_substrate:
        victim_score += 20.0
    if is_cyp2d6_substrate:
        victim_score += 15.0
    if is_cyp2c9_substrate:
        victim_score += 15.0
    if is_pgp_substrate:
        victim_score += 10.0
    if is_oatp_substrate:
        victim_score += 10.0
    victim_score = min(victim_score, 70.0)

    # --- Perpetrator score ---
    perpetrator_score = 0.0
    perpetrator_score += _ki_to_perp_points(ki_3a4_uM)
    perpetrator_score += _ki_to_perp_points(ki_2d6_uM)
    perpetrator_score += _ki_to_perp_points(ki_2c9_uM)
    if is_pgp_inhibitor:
        perpetrator_score += 15.0

    # --- TI amplification ---
    ti_factor = 1.5 if therapeutic_index == "narrow" else 1.0

    # Overall DDI risk
    base_risk = max(victim_score, perpetrator_score)
    overall_ddi_risk = min(base_risk * ti_factor, 100.0)

    # --- Classification ---
    if overall_ddi_risk < 20.0:
        risk_class = "low"
    elif overall_ddi_risk <= 50.0:
        risk_class = "moderate"
    else:
        risk_class = "high"

    requires_clinical_ddi_study = overall_ddi_risk > 30.0

    # --- Primary pathway ---
    primary_interaction_pathway = _determine_primary_pathway(
        is_cyp3a4_substrate=is_cyp3a4_substrate,
        is_cyp3a4_inhibitor=is_cyp3a4_inhibitor,
        ki_3a4_uM=ki_3a4_uM,
        is_cyp2d6_substrate=is_cyp2d6_substrate,
        is_cyp2d6_inhibitor=is_cyp2d6_inhibitor,
        ki_2d6_uM=ki_2d6_uM,
        is_cyp2c9_substrate=is_cyp2c9_substrate,
        is_cyp2c9_inhibitor=is_cyp2c9_inhibitor,
        ki_2c9_uM=ki_2c9_uM,
        is_pgp_substrate=is_pgp_substrate,
        is_pgp_inhibitor=is_pgp_inhibitor,
        is_oatp_substrate=is_oatp_substrate,
    )

    notes_parts = [
        f"Overall DDI risk: {overall_ddi_risk:.1f}/100 ({risk_class}).",
        f"Victim score: {victim_score:.1f}, Perpetrator score: {perpetrator_score:.1f}.",
        f"Primary pathway: {primary_interaction_pathway}.",
        f"Therapeutic index: {therapeutic_index}.",
    ]
    if requires_clinical_ddi_study:
        notes_parts.append("Clinical DDI study required (FDA guidance).")

    return DDIScoreResult(
        compound_name=compound_name,
        victim_score=victim_score,
        perpetrator_score=perpetrator_score,
        overall_ddi_risk=overall_ddi_risk,
        risk_class=risk_class,
        requires_clinical_ddi_study=requires_clinical_ddi_study,
        primary_interaction_pathway=primary_interaction_pathway,
        notes=" ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Batch screening
# ---------------------------------------------------------------------------


def screen_ddi_scores(compounds: list[dict]) -> list[DDIScoreResult]:
    """Screen multiple compounds and return results sorted by DDI risk (descending).

    Each dict in compounds should match the kwargs of compute_ddi_score,
    with 'compound_name' as a required key.
    """
    results: list[DDIScoreResult] = []
    for cpd in compounds:
        cpd_copy = dict(cpd)
        name = cpd_copy.pop("compound_name", "unknown")
        result = compute_ddi_score(compound_name=name, **cpd_copy)
        results.append(result)

    results.sort(key=lambda r: r.overall_ddi_risk, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Clinical AUCR estimation
# ---------------------------------------------------------------------------


def estimate_clinical_aucr(
    ki_uM: float,
    i_max_uM: float,
    fm_cyp: float = 0.8,
) -> float:
    """Estimate AUCR using the static DDI model (FDA 2020 guidance).

    AUCR = 1 / (1 - fm_cyp + fm_cyp / (1 + i_max_uM / ki_uM))

    Parameters
    ----------
    ki_uM:
        Inhibitor Ki in µM.
    i_max_uM:
        Maximum unbound inhibitor concentration at enzyme site (µM).
    fm_cyp:
        Fraction of victim drug metabolized by the inhibited CYP (0-1).

    Returns
    -------
    float
        Predicted AUCR (victim AUC with inhibitor / victim AUC without inhibitor).
    """
    if ki_uM <= 0:
        raise ValueError(f"ki_uM must be > 0, got {ki_uM}")
    if i_max_uM < 0:
        raise ValueError(f"i_max_uM must be >= 0, got {i_max_uM}")
    if not (0.0 <= fm_cyp <= 1.0):
        raise ValueError(f"fm_cyp must be in [0, 1], got {fm_cyp}")

    inhibition_factor = fm_cyp / (1.0 + i_max_uM / ki_uM)
    denominator = 1.0 - fm_cyp + inhibition_factor
    if denominator <= 0.0:
        return float("inf")
    return 1.0 / denominator
