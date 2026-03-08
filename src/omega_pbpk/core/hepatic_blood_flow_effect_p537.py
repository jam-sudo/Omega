"""
Phase 537 — Hepatic Blood Flow Effect on Drug Clearance

Analyzes how changes in hepatic blood flow affect drug clearance using
the well-stirred model, classifying drugs as flow-limited or enzyme-limited.
"""

from __future__ import annotations

from dataclasses import dataclass

# Hepatic blood flow conditions (L/h)
_Q_NORMAL = 90.0
_Q_HEART_FAILURE = 45.0
_Q_EXERCISE = 180.0
_Q_CIRRHOSIS = 120.0  # portal shunting increases effective flow

# Extraction ratio thresholds
_EH_HIGH = 0.7
_EH_LOW = 0.3


def _well_stirred_clh(q: float, fup: float, cl_int: float) -> float:
    """Compute hepatic clearance via well-stirred model."""
    return q * fup * cl_int / (q + fup * cl_int)


def _extraction_ratio(q: float, fup: float, cl_int: float) -> float:
    """Compute hepatic extraction ratio."""
    return fup * cl_int / (q + fup * cl_int)


def _classify_extraction(eh: float) -> str:
    if eh > _EH_HIGH:
        return "flow_limited"
    if eh < _EH_LOW:
        return "enzyme_limited"
    return "intermediate"


def _clinical_implication(extraction_class: str) -> str:
    if extraction_class == "flow_limited":
        return (
            "High-extraction drug: clearance is sensitive to hepatic blood flow. "
            "Conditions reducing cardiac output (heart failure, shock) significantly "
            "reduce clearance and increase drug exposure. Dose reduction may be needed."
        )
    if extraction_class == "enzyme_limited":
        return (
            "Low-extraction drug: clearance is determined by enzyme activity and "
            "protein binding, not blood flow. Changes in cardiac output have minimal "
            "effect on clearance. Monitor for enzyme inducers/inhibitors instead."
        )
    return (
        "Intermediate-extraction drug: clearance is moderately sensitive to both "
        "blood flow and enzyme activity. Both cardiac output and enzyme status "
        "should be considered when adjusting dosing."
    )


@dataclass(frozen=True)
class HepaticBloodFlowResult:
    """Result of hepatic blood flow effect analysis."""

    drug_name: str
    fup: float
    cl_int_L_per_h: float

    # Normal conditions
    eh_normal: float
    clh_normal_L_per_h: float
    fh_normal: float  # = 1 - eh_normal

    # Classification
    extraction_class: str  # "flow_limited" / "enzyme_limited" / "intermediate"

    # Pathological / altered conditions
    eh_heart_failure: float
    clh_heart_failure_L_per_h: float

    eh_exercise: float
    clh_exercise_L_per_h: float

    eh_cirrhosis: float
    clh_cirrhosis_L_per_h: float

    # Sensitivity metric
    flow_sensitivity_index: float  # (CLh_exercise - CLh_heart_failure) / CLh_normal

    clinical_implication: str
    notes: str


def analyze_hepatic_blood_flow_effect(
    drug_name: str,
    fup: float,
    cl_int_L_per_h: float,
    q_normal_L_per_h: float = 90.0,
) -> HepaticBloodFlowResult:
    """Analyze the effect of hepatic blood flow changes on drug clearance.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    fup:
        Fraction unbound in plasma (0 < fup <= 1).
    cl_int_L_per_h:
        Intrinsic hepatic clearance (L/h).
    q_normal_L_per_h:
        Normal hepatic blood flow (L/h); default 90.0.

    Returns
    -------
    HepaticBloodFlowResult
    """
    if not (0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1]; got {fup}")
    if cl_int_L_per_h <= 0:
        raise ValueError(f"cl_int_L_per_h must be > 0; got {cl_int_L_per_h}")
    if q_normal_L_per_h <= 0:
        raise ValueError(f"q_normal_L_per_h must be > 0; got {q_normal_L_per_h}")

    # Scale alternative Q values relative to any supplied q_normal
    scale = q_normal_L_per_h / _Q_NORMAL
    q_hf = _Q_HEART_FAILURE * scale
    q_ex = _Q_EXERCISE * scale
    q_ci = _Q_CIRRHOSIS * scale

    # Normal
    eh_norm = _extraction_ratio(q_normal_L_per_h, fup, cl_int_L_per_h)
    clh_norm = _well_stirred_clh(q_normal_L_per_h, fup, cl_int_L_per_h)
    fh_norm = 1.0 - eh_norm

    # Heart failure
    eh_hf = _extraction_ratio(q_hf, fup, cl_int_L_per_h)
    clh_hf = _well_stirred_clh(q_hf, fup, cl_int_L_per_h)

    # Exercise
    eh_ex = _extraction_ratio(q_ex, fup, cl_int_L_per_h)
    clh_ex = _well_stirred_clh(q_ex, fup, cl_int_L_per_h)

    # Cirrhosis
    eh_ci = _extraction_ratio(q_ci, fup, cl_int_L_per_h)
    clh_ci = _well_stirred_clh(q_ci, fup, cl_int_L_per_h)

    extraction_class = _classify_extraction(eh_norm)

    # Flow sensitivity: how much CLh varies across extreme flow conditions
    flow_sensitivity_index = (clh_ex - clh_hf) / clh_norm if clh_norm > 0 else 0.0

    notes = (
        f"Well-stirred model; Q_normal={q_normal_L_per_h:.1f} L/h, "
        f"Q_HF={q_hf:.1f} L/h, Q_exercise={q_ex:.1f} L/h, "
        f"Q_cirrhosis={q_ci:.1f} L/h."
    )

    return HepaticBloodFlowResult(
        drug_name=drug_name,
        fup=fup,
        cl_int_L_per_h=cl_int_L_per_h,
        eh_normal=eh_norm,
        clh_normal_L_per_h=clh_norm,
        fh_normal=fh_norm,
        extraction_class=extraction_class,
        eh_heart_failure=eh_hf,
        clh_heart_failure_L_per_h=clh_hf,
        eh_exercise=eh_ex,
        clh_exercise_L_per_h=clh_ex,
        eh_cirrhosis=eh_ci,
        clh_cirrhosis_L_per_h=clh_ci,
        flow_sensitivity_index=flow_sensitivity_index,
        clinical_implication=_clinical_implication(extraction_class),
        notes=notes,
    )


def compare_extraction_drugs(drugs: list) -> list:
    """Compare multiple drugs by their flow sensitivity.

    Parameters
    ----------
    drugs:
        List of dicts, each with keys:
        ``drug_name``, ``fup``, ``cl_int_L_per_h``.

    Returns
    -------
    list of HepaticBloodFlowResult sorted by flow_sensitivity_index descending.
    """
    results = []
    for d in drugs:
        result = analyze_hepatic_blood_flow_effect(
            drug_name=d["drug_name"],
            fup=d["fup"],
            cl_int_L_per_h=d["cl_int_L_per_h"],
        )
        results.append(result)
    results.sort(key=lambda r: r.flow_sensitivity_index, reverse=True)
    return results
