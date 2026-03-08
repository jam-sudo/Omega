"""
Phase 633 — Drug Myocardial Ischemia PK
Models drug PK changes during myocardial ischemia (reduced cardiac output, altered perfusion).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MyocardialIschemiaPKResult:
    """Result of myocardial ischemia PK prediction."""

    drug_name: str
    ischemia_severity: str
    cardiac_output_factor: float
    cl_normal_L_per_h: float
    cl_ischemia_L_per_h: float
    vd_normal_L: float
    vd_ischemia_L: float
    t_half_normal_h: float
    t_half_ischemia_h: float
    auc_ratio: float
    dose_adjustment_factor: float
    hepatic_perfusion_reduction: float
    renal_perfusion_reduction: float
    notes: str


_SEVERITY_CO_FACTOR = {
    "mild": 0.85,
    "moderate": 0.65,
    "severe": 0.40,
}

_SEVERITY_VD_FACTOR = {
    "mild": 1.0,
    "moderate": 1.1,
    "severe": 0.9,
}

_LN2 = math.log(2.0)


def _validate_inputs(
    drug_name: str,
    cl_normal_L_per_h: float,
    vd_normal_L: float,
    ischemia_severity: str,
    f_hepatic: float,
    f_renal: float,
) -> None:
    if ischemia_severity not in ("mild", "moderate", "severe"):
        raise ValueError(
            f"ischemia_severity must be 'mild', 'moderate', or 'severe'; got {ischemia_severity!r}"
        )
    if cl_normal_L_per_h <= 0:
        raise ValueError(f"cl_normal_L_per_h must be > 0; got {cl_normal_L_per_h}")
    if vd_normal_L <= 0:
        raise ValueError(f"vd_normal_L must be > 0; got {vd_normal_L}")
    if not (0.0 <= f_hepatic <= 1.0):
        raise ValueError(f"f_hepatic must be in [0, 1]; got {f_hepatic}")
    if not (0.0 <= f_renal <= 1.0):
        raise ValueError(f"f_renal must be in [0, 1]; got {f_renal}")
    if f_hepatic + f_renal > 1.0:
        raise ValueError(f"f_hepatic + f_renal must be <= 1.0; got {f_hepatic + f_renal}")


def predict_ischemia_pk(
    drug_name: str,
    cl_normal_L_per_h: float,
    vd_normal_L: float,
    ischemia_severity: str,
    f_hepatic: float = 0.6,
    f_renal: float = 0.3,
) -> MyocardialIschemiaPKResult:
    """
    Predict PK changes during myocardial ischemia.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    cl_normal_L_per_h : float
        Normal clearance (L/h).
    vd_normal_L : float
        Normal volume of distribution (L).
    ischemia_severity : str
        One of 'mild', 'moderate', 'severe'.
    f_hepatic : float
        Fraction of total CL that is hepatic (default 0.6).
    f_renal : float
        Fraction of total CL that is renal (default 0.3).

    Returns
    -------
    MyocardialIschemiaPKResult
    """
    _validate_inputs(
        drug_name, cl_normal_L_per_h, vd_normal_L, ischemia_severity, f_hepatic, f_renal
    )

    co_factor = _SEVERITY_CO_FACTOR[ischemia_severity]

    # Organ perfusion reductions
    hepatic_perf = co_factor  # proportional to CO
    renal_perf = co_factor * 0.8  # more sensitive
    # Clamp renal to [0.1, 1.0]
    renal_perf = max(0.1, min(1.0, renal_perf))

    # CL components
    cl_hepatic_new = cl_normal_L_per_h * f_hepatic * hepatic_perf
    cl_renal_new = cl_normal_L_per_h * f_renal * renal_perf
    cl_other = cl_normal_L_per_h * (1.0 - f_hepatic - f_renal)
    cl_ischemia = cl_hepatic_new + cl_renal_new + cl_other

    # Vd changes
    vd_factor = _SEVERITY_VD_FACTOR[ischemia_severity]
    vd_ischemia = vd_normal_L * vd_factor

    # Half-lives
    t_half_normal = _LN2 * vd_normal_L / cl_normal_L_per_h
    t_half_ischemia = _LN2 * vd_ischemia / cl_ischemia

    # AUC ratio: AUC ∝ 1/CL for same dose
    auc_ratio = cl_normal_L_per_h / cl_ischemia

    # Dose adjustment: reduce dose proportionally to CL reduction
    dose_adjustment_factor = cl_ischemia / cl_normal_L_per_h

    notes = (
        f"Ischemia ({ischemia_severity}): CO reduced to {co_factor * 100:.0f}% of normal. "
        f"Hepatic perfusion: {hepatic_perf * 100:.0f}%, renal perfusion: {renal_perf * 100:.0f}%. "
        f"CL reduced from {cl_normal_L_per_h:.2f} to {cl_ischemia:.2f} L/h "
        f"(dose reduction to {dose_adjustment_factor * 100:.0f}% of normal recommended)."
    )

    return MyocardialIschemiaPKResult(
        drug_name=drug_name,
        ischemia_severity=ischemia_severity,
        cardiac_output_factor=co_factor,
        cl_normal_L_per_h=cl_normal_L_per_h,
        cl_ischemia_L_per_h=cl_ischemia,
        vd_normal_L=vd_normal_L,
        vd_ischemia_L=vd_ischemia,
        t_half_normal_h=t_half_normal,
        t_half_ischemia_h=t_half_ischemia,
        auc_ratio=auc_ratio,
        dose_adjustment_factor=dose_adjustment_factor,
        hepatic_perfusion_reduction=hepatic_perf,
        renal_perfusion_reduction=renal_perf,
        notes=notes,
    )


def ischemia_severity_comparison(
    drug_name: str,
    cl_normal_L_per_h: float,
    vd_normal_L: float,
    f_hepatic: float = 0.6,
    f_renal: float = 0.3,
) -> list[MyocardialIschemiaPKResult]:
    """
    Return PK predictions for all three ischemia severities, sorted by cl_ischemia descending.

    Parameters
    ----------
    drug_name : str
    cl_normal_L_per_h : float
    vd_normal_L : float
    f_hepatic : float
    f_renal : float

    Returns
    -------
    list of MyocardialIschemiaPKResult
    """
    results = [
        predict_ischemia_pk(drug_name, cl_normal_L_per_h, vd_normal_L, sev, f_hepatic, f_renal)
        for sev in ("mild", "moderate", "severe")
    ]
    results.sort(key=lambda r: r.cl_ischemia_L_per_h, reverse=True)
    return results
