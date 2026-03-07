"""Drug Abuse Potential Assessment (Phase 1103).

Based on FDA Draft Guidance on Abuse-Deterrent Opioid Formulations and
general principles of addiction pharmacology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_FORMULATIONS = {"IR", "ER", "abuse_deterrent"}

_FORMULATION_FACTORS = {
    "IR": 1.0,
    "ER": 0.5,
    "abuse_deterrent": 0.2,
}


@dataclass(frozen=True)
class AbuseResult:
    """Result of drug abuse potential assessment."""

    drug_name: str
    logP: float
    mw_Da: float
    tmax_h: float
    t_half_h: float
    formulation: str
    kp_brain: float
    abuse_score: float  # 0-100
    abuse_category: str  # "low", "moderate", "high", "very_high"
    cns_penetration_class: str  # "poor", "moderate", "good"
    physical_dependence_risk: str  # "low", "moderate", "high"
    scheduling_recommendation: str  # "unscheduled", "schedule_IV", "schedule_II", "schedule_I"
    notes: str


def _compute_kp_brain(logP: float, mw_Da: float) -> float:
    """Compute brain Kp from logP and MW."""
    raw = 0.5 * logP - 0.01 * mw_Da
    return max(0.01, min(2.0, math.exp(raw) if raw < 0.7 else min(2.0, raw + 0.5)))


def _compute_kp_brain_simple(logP: float, mw_Da: float) -> float:
    """Simple Kp_brain estimate clamped to [0.01, 2]."""
    # log(Kp_brain) = 0.5*logP - 0.01*MW
    log_kp = 0.5 * logP - 0.01 * mw_Da
    kp = math.exp(log_kp)
    return max(0.01, min(2.0, kp))


def _abuse_category(score: float) -> str:
    if score < 25:
        return "low"
    elif score < 50:
        return "moderate"
    elif score < 75:
        return "high"
    else:
        return "very_high"


def _cns_penetration_class(kp_brain: float) -> str:
    if kp_brain < 0.1:
        return "poor"
    elif kp_brain <= 0.5:
        return "moderate"
    else:
        return "good"


def _physical_dependence_risk(t_half_h: float) -> str:
    # Short t½ → higher withdrawal risk
    if t_half_h < 2.0:
        return "high"
    elif t_half_h < 6.0:
        return "moderate"
    else:
        return "low"


def _scheduling_recommendation(abuse_score: float) -> str:
    if abuse_score < 25:
        return "unscheduled"
    elif abuse_score < 50:
        return "schedule_IV"
    elif abuse_score < 75:
        return "schedule_II"
    else:
        return "schedule_I"


def _validate_inputs(
    logP: float,
    mw_Da: float,
    tmax_h: float,
    t_half_h: float,
    formulation: str,
) -> None:
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(f"formulation must be one of {_VALID_FORMULATIONS}, got '{formulation}'")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if tmax_h <= 0:
        raise ValueError(f"tmax_h must be > 0, got {tmax_h}")
    if t_half_h <= 0:
        raise ValueError(f"t_half_h must be > 0, got {t_half_h}")


def assess_abuse_potential(
    drug_name: str,
    logP: float,
    mw_Da: float,
    tmax_h: float = 1.0,
    t_half_h: float = 4.0,
    formulation: str = "IR",
) -> AbuseResult:
    """Assess drug abuse potential based on PK/PD properties.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logP:
        Octanol-water partition coefficient (lipophilicity).
    mw_Da:
        Molecular weight in Daltons.
    tmax_h:
        Time to peak concentration in hours.
    t_half_h:
        Elimination half-life in hours.
    formulation:
        Formulation type: "IR", "ER", or "abuse_deterrent".

    Returns
    -------
    AbuseResult with abuse score 0–100 and categorical risk assessment.
    """
    _validate_inputs(logP, mw_Da, tmax_h, t_half_h, formulation)

    kp_brain = _compute_kp_brain_simple(logP, mw_Da)

    # Component factors (each in [0, 1])
    cns_factor = min(1.0, kp_brain * 2.0)
    onset_factor = min(1.0, 1.0 / tmax_h)
    potency_factor = min(1.0, logP / 5.0) if logP > 0 else 0.0
    dependence_factor = min(1.0, max(0.0, (2.0 - t_half_h) / 2.0))
    formulation_factor = _FORMULATION_FACTORS[formulation]

    abuse_score = (
        20.0 * cns_factor
        + 20.0 * onset_factor
        + 20.0 * potency_factor
        + 20.0 * dependence_factor
        + 20.0 * formulation_factor
    )
    # Clamp to [0, 100]
    abuse_score = max(0.0, min(100.0, abuse_score))

    category = _abuse_category(abuse_score)
    cns_class = _cns_penetration_class(kp_brain)
    dep_risk = _physical_dependence_risk(t_half_h)
    sched = _scheduling_recommendation(abuse_score)

    notes_parts = []
    if kp_brain > 0.5:
        notes_parts.append("High CNS penetration increases euphorigenic potential.")
    if tmax_h < 0.5:
        notes_parts.append("Very rapid onset is associated with high abuse liability.")
    if t_half_h < 2.0:
        notes_parts.append("Short half-life heightens physical dependence and withdrawal risk.")
    if formulation == "abuse_deterrent":
        notes_parts.append("Abuse-deterrent formulation significantly reduces misuse risk.")
    if formulation == "ER":
        notes_parts.append("Extended-release formulation moderates peak exposure and onset.")
    if not notes_parts:
        notes_parts.append("No major abuse-risk flags identified.")

    return AbuseResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        tmax_h=tmax_h,
        t_half_h=t_half_h,
        formulation=formulation,
        kp_brain=kp_brain,
        abuse_score=abuse_score,
        abuse_category=category,
        cns_penetration_class=cns_class,
        physical_dependence_risk=dep_risk,
        scheduling_recommendation=sched,
        notes=" ".join(notes_parts),
    )


def compare_formulations(
    drug_name: str,
    logP: float,
    mw_Da: float,
    tmax_h: float = 1.0,
    t_half_h: float = 4.0,
) -> list[AbuseResult]:
    """Compare all three formulations for a drug, sorted by abuse_score ascending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons.
    tmax_h:
        Time to peak concentration in hours (for IR; ER and AD use same base PK).
    t_half_h:
        Elimination half-life in hours.

    Returns
    -------
    List of 3 AbuseResult objects sorted by abuse_score ascending (safest first).
    """
    results = [
        assess_abuse_potential(drug_name, logP, mw_Da, tmax_h, t_half_h, f)
        for f in ["IR", "ER", "abuse_deterrent"]
    ]
    results.sort(key=lambda r: r.abuse_score)
    return results
