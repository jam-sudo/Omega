"""
Phase 964 — ROS (Reactive Oxygen Species) burden prediction from drug structure.

Rule-based scoring of oxidative stress risk from physicochemical and structural features.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ROSBurdenResult", "predict_ros_burden", "screen_ros_risk"]

_VALID_IONIZATION = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class ROSBurdenResult:
    drug_name: str
    logp: float
    mw: float
    pka: float
    ionization_type: str
    has_quinone: bool
    has_nitro_group: bool
    ros_score: float
    ros_risk_category: str
    gsh_depletion_risk: str
    mitochondrial_toxicity_index: float
    oxidative_stress_mechanism: str
    dili_ros_risk: bool
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _compute_ros_score(
    has_quinone: bool,
    has_nitro_group: bool,
    logp: float,
    mw: float,
    ionization_type: str,
    pka: float,
) -> float:
    score = 10.0  # base

    if has_quinone:
        score += 35.0

    if has_nitro_group:
        score += 25.0

    if logp > 3.0 and mw < 400.0:
        score += 15.0

    if ionization_type == "acid" and pka < 4.0:
        score += 10.0

    if mw > 500.0:
        score -= 10.0

    if ionization_type == "base" and pka > 9.0:
        score += 10.0

    return _clamp(score, 0.0, 100.0)


def _risk_category(ros_score: float) -> str:
    if ros_score > 60.0:
        return "high"
    if ros_score > 30.0:
        return "moderate"
    return "low"


def _gsh_risk(ros_score: float) -> str:
    if ros_score > 60.0:
        return "high"
    if ros_score > 30.0:
        return "moderate"
    return "low"


def _mitochondrial_toxicity_index(ros_score: float, logp: float) -> float:
    mt = ros_score * (1.0 + 0.1 * abs(logp - 3.0))
    return _clamp(mt, 0.0, 100.0)


def _oxidative_mechanism(
    has_quinone: bool,
    has_nitro_group: bool,
    ros_score: float,
) -> str:
    if has_quinone:
        return "quinone_redox_cycling"
    if has_nitro_group:
        return "nitro_reduction_activation"
    if ros_score > 30.0:
        return "cyp_oxidative_metabolism"
    return "minimal_ros_generation"


def predict_ros_burden(
    drug_name: str,
    logp: float,
    mw: float,
    pka: float,
    ionization_type: str,
    has_quinone: bool = False,
    has_nitro_group: bool = False,
) -> ROSBurdenResult:
    """Predict ROS generation burden from structural and physicochemical descriptors."""
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got '{ionization_type}'"
        )

    ros_score = _compute_ros_score(
        has_quinone=has_quinone,
        has_nitro_group=has_nitro_group,
        logp=logp,
        mw=mw,
        ionization_type=ionization_type,
        pka=pka,
    )
    ros_cat = _risk_category(ros_score)
    gsh_risk = _gsh_risk(ros_score)
    mt_index = _mitochondrial_toxicity_index(ros_score, logp)
    mechanism = _oxidative_mechanism(has_quinone, has_nitro_group, ros_score)
    dili_risk = ros_score > 50.0

    alerts = []
    if has_quinone:
        alerts.append("quinone")
    if has_nitro_group:
        alerts.append("nitro")
    alert_str = ", ".join(alerts) if alerts else "none"

    notes = (
        f"ROS score={ros_score:.1f}, category={ros_cat}, "
        f"structural alerts=[{alert_str}], logP={logp:.2f}, MW={mw:.1f}, "
        f"ionization={ionization_type} (pKa={pka:.1f}). "
        f"Mitochondrial toxicity index={mt_index:.1f}. "
        f"DILI-via-ROS risk: {'YES' if dili_risk else 'NO'}."
    )

    return ROSBurdenResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        pka=pka,
        ionization_type=ionization_type,
        has_quinone=has_quinone,
        has_nitro_group=has_nitro_group,
        ros_score=ros_score,
        ros_risk_category=ros_cat,
        gsh_depletion_risk=gsh_risk,
        mitochondrial_toxicity_index=mt_index,
        oxidative_stress_mechanism=mechanism,
        dili_ros_risk=dili_risk,
        notes=notes,
    )


def screen_ros_risk(compounds: list) -> list:
    """Screen a list of compound dicts through predict_ros_burden.

    Each dict must have: drug_name, logp, mw, pka, ionization_type.
    Optional: has_quinone, has_nitro_group.

    Returns list of ROSBurdenResult sorted by ros_score descending.
    """
    results = []
    for cpd in compounds:
        result = predict_ros_burden(
            drug_name=cpd["drug_name"],
            logp=cpd["logp"],
            mw=cpd["mw"],
            pka=cpd["pka"],
            ionization_type=cpd["ionization_type"],
            has_quinone=cpd.get("has_quinone", False),
            has_nitro_group=cpd.get("has_nitro_group", False),
        )
        results.append(result)

    results.sort(key=lambda r: r.ros_score, reverse=True)
    return results
