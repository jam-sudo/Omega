"""Drug-induced QT prolongation and TdP risk prediction.

Uses clinical TdP risk database approach (FDA/ICH E14 criteria).
Distinct from herg_qt.py which focuses on physicochemical hERG scoring;
this module takes hERG IC50 and plasma concentration as primary inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "QTProlongationResult",
    "predict_qt_prolongation",
    "screen_cardiac_safety",
]


@dataclass(frozen=True)
class QTProlongationResult:
    """Result of QT prolongation risk assessment."""

    drug_name: str
    dose_mg: float
    c_plasma_mg_L: float  # plasma concentration at which risk is assessed

    # hERG inhibition
    herg_ic50_uM: float
    herg_inhibition_pct: float  # % inhibition at therapeutic Cmax

    # QT prolongation metrics
    delta_qtc_ms: float  # estimated QTc prolongation (ms)
    baseline_qtc_ms: float  # reference baseline QTc
    predicted_qtc_ms: float  # baseline + delta

    # Risk classification (FDA/ICH E14 criteria)
    qtc_prolongation_category: str  # "none", "mild", "moderate", "severe"
    tdp_risk_category: str  # "low", "conditional_risk", "known_risk"

    # Risk factors
    risk_factors_count: int
    risk_factors: list[str]

    recommendation: str
    notes: str


def _estimate_cmax_mg_L(
    dose_mg: float,
    mw: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float,
) -> float:
    """Estimate peak plasma concentration (Cmax) from 1-compartment oral model.

    Parameters
    ----------
    dose_mg : float
        Oral dose (mg).
    mw : float
        Molecular weight (g/mol), used for unit conversion.
    cl_L_per_h : float
        Total clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    ka_per_h : float
        Absorption rate constant (1/h).
    f_oral : float
        Oral bioavailability fraction (0–1).

    Returns
    -------
    float
        Estimated Cmax in mg/L.
    """
    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # Guard: ka must differ from ke
    if abs(ka_per_h - ke) < 1e-6:
        ka_per_h = ka_per_h * 1.001

    tmax = math.log(ka_per_h / ke) / (ka_per_h - ke)
    tmax = max(tmax, 1e-6)

    # Cmax in mg/L (dose in mg, Vd in L)
    dose_available = dose_mg * f_oral  # mg
    cmax = (dose_available / vd_L) * (ka_per_h / (ka_per_h - ke)) * (
        math.exp(-ke * tmax) - math.exp(-ka_per_h * tmax)
    )
    return max(cmax, 0.0)


def _qtc_prolongation_category(delta_qtc_ms: float) -> str:
    """Classify QTc prolongation per ICH E14 guidance."""
    if delta_qtc_ms < 5.0:
        return "none"
    elif delta_qtc_ms < 10.0:
        return "mild"
    elif delta_qtc_ms < 20.0:
        return "moderate"
    else:
        return "severe"


def _tdp_risk_category(delta_qtc_ms: float, herg_inhibition_pct: float) -> str:
    """Classify TdP risk based on QTc prolongation and hERG inhibition."""
    if delta_qtc_ms > 20.0 or herg_inhibition_pct > 50.0:
        return "known_risk"
    elif delta_qtc_ms > 10.0 or herg_inhibition_pct > 20.0:
        return "conditional_risk"
    else:
        return "low"


def _recommendation(tdp_risk: str, qtc_category: str) -> str:
    """Return clinical recommendation based on risk categories."""
    if tdp_risk == "known_risk":
        return (
            "HIGH RISK: Drug carries known TdP risk. Avoid in patients with QTc >450 ms "
            "(women) or >440 ms (men), electrolyte imbalances, or concomitant QT-prolonging "
            "drugs. ECG monitoring required before and during therapy."
        )
    elif tdp_risk == "conditional_risk":
        return (
            "CONDITIONAL RISK: Drug may prolong QTc under certain conditions. Obtain baseline "
            "ECG and correct electrolyte abnormalities before use. Avoid combining with other "
            "QT-prolonging agents. Monitor QTc periodically."
        )
    else:
        return (
            "LOW RISK: Predicted QTc prolongation is minimal. Standard cardiac monitoring "
            "is sufficient. Re-evaluate if patient develops new risk factors."
        )


def predict_qt_prolongation(
    drug_name: str,
    dose_mg: float,
    mw: float,
    logP: float,
    herg_ic50_uM: float,
    c_plasma_mg_L: float | None = None,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    ka_per_h: float = 1.5,
    f_oral: float = 0.9,
    baseline_qtc_ms: float = 400.0,
    patient_female: bool = False,
    hypokalemia: bool = False,
    concomitant_qt_drug: bool = False,
) -> QTProlongationResult:
    """Predict drug-induced QT prolongation risk.

    Parameters
    ----------
    drug_name : str
        Name of the drug being assessed.
    dose_mg : float
        Dose administered (mg).
    mw : float
        Molecular weight (g/mol).
    logP : float
        Octanol-water partition coefficient (informational).
    herg_ic50_uM : float
        hERG channel IC50 (µM). Lower = more potent inhibitor.
    c_plasma_mg_L : float or None
        Plasma concentration (mg/L) at which to assess risk. If None,
        Cmax is estimated analytically from the 1-compartment model.
    cl_L_per_h : float
        Total clearance (L/h). Used only when c_plasma_mg_L is None.
    vd_L : float
        Volume of distribution (L). Used only when c_plasma_mg_L is None.
    ka_per_h : float
        Absorption rate constant (1/h). Used only when c_plasma_mg_L is None.
    f_oral : float
        Oral bioavailability (0–1). Used only when c_plasma_mg_L is None.
    baseline_qtc_ms : float
        Patient baseline QTc (ms). Default 400 ms.
    patient_female : bool
        Female sex increases baseline QT risk.
    hypokalemia : bool
        Hypokalemia is a major QT risk factor.
    concomitant_qt_drug : bool
        Concomitant use of another QT-prolonging drug.

    Returns
    -------
    QTProlongationResult
    """
    # Input validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")
    if herg_ic50_uM <= 0:
        raise ValueError(f"herg_ic50_uM must be positive, got {herg_ic50_uM}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")

    # Determine plasma concentration
    if c_plasma_mg_L is None:
        c_plasma_mg_L = _estimate_cmax_mg_L(dose_mg, mw, cl_L_per_h, vd_L, ka_per_h, f_oral)

    # Convert mg/L to µM: (mg/L * 1000 mg/g) / (mw g/mol) * (1 mmol/1000 µmol) ... simplified
    # c (mg/L) → c (µg/mL) → c (µM) = c_mg_L * 1000 / mw
    c_uM = (c_plasma_mg_L * 1000.0) / mw

    # hERG inhibition (Hill equation with Hill coefficient = 1)
    herg_inhibition_pct = c_uM / (c_uM + herg_ic50_uM) * 100.0

    # Base delta QTc from hERG inhibition (empirical linear: 50% inhibition → ~20 ms)
    delta_qtc_base = herg_inhibition_pct * 0.4

    # Collect risk factors and their additive/multiplicative contributions
    risk_factors: list[str] = []
    female_offset = 0.0
    hypo_offset = 0.0

    if patient_female:
        female_offset = 5.0
        risk_factors.append("Female sex (increased QT sensitivity, +5 ms)")

    if hypokalemia:
        hypo_offset = 10.0
        risk_factors.append("Hypokalemia (reduces repolarization reserve, +10 ms)")

    if concomitant_qt_drug:
        delta_qtc_base = delta_qtc_base * 1.5
        risk_factors.append("Concomitant QT-prolonging drug (×1.5 multiplier)")

    delta_qtc_ms = delta_qtc_base + female_offset + hypo_offset
    predicted_qtc_ms = baseline_qtc_ms + delta_qtc_ms

    qtc_cat = _qtc_prolongation_category(delta_qtc_ms)
    tdp_risk = _tdp_risk_category(delta_qtc_ms, herg_inhibition_pct)
    recommendation = _recommendation(tdp_risk, qtc_cat)

    notes_parts = [
        f"hERG IC50={herg_ic50_uM:.2f} µM; Cmax={c_plasma_mg_L:.4f} mg/L "
        f"({c_uM:.3f} µM); logP={logP:.1f}"
    ]
    if c_plasma_mg_L == 0.0:
        notes_parts.append("Warning: estimated Cmax is 0 — check PK parameters")

    return QTProlongationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        c_plasma_mg_L=c_plasma_mg_L,
        herg_ic50_uM=herg_ic50_uM,
        herg_inhibition_pct=round(herg_inhibition_pct, 4),
        delta_qtc_ms=round(delta_qtc_ms, 4),
        baseline_qtc_ms=baseline_qtc_ms,
        predicted_qtc_ms=round(predicted_qtc_ms, 4),
        qtc_prolongation_category=qtc_cat,
        tdp_risk_category=tdp_risk,
        risk_factors_count=len(risk_factors),
        risk_factors=risk_factors,
        recommendation=recommendation,
        notes="; ".join(notes_parts),
    )


def screen_cardiac_safety(
    compounds: list[dict],
) -> list[QTProlongationResult]:
    """Screen a library of compounds for cardiac QT safety.

    Parameters
    ----------
    compounds : list of dict
        Each dict must contain: name, dose_mg, mw, logP, herg_ic50_uM.
        Optional keys: c_plasma_mg_L, cl_L_per_h, vd_L, ka_per_h, f_oral,
        baseline_qtc_ms, patient_female, hypokalemia, concomitant_qt_drug.

    Returns
    -------
    list[QTProlongationResult]
        Sorted by delta_qtc_ms descending (highest risk first).
    """
    results: list[QTProlongationResult] = []
    for cmpd in compounds:
        result = predict_qt_prolongation(
            drug_name=cmpd["name"],
            dose_mg=cmpd["dose_mg"],
            mw=cmpd["mw"],
            logP=cmpd["logP"],
            herg_ic50_uM=cmpd["herg_ic50_uM"],
            c_plasma_mg_L=cmpd.get("c_plasma_mg_L"),
            cl_L_per_h=cmpd.get("cl_L_per_h", 10.0),
            vd_L=cmpd.get("vd_L", 50.0),
            ka_per_h=cmpd.get("ka_per_h", 1.5),
            f_oral=cmpd.get("f_oral", 0.9),
            baseline_qtc_ms=cmpd.get("baseline_qtc_ms", 400.0),
            patient_female=cmpd.get("patient_female", False),
            hypokalemia=cmpd.get("hypokalemia", False),
            concomitant_qt_drug=cmpd.get("concomitant_qt_drug", False),
        )
        results.append(result)

    results.sort(key=lambda r: r.delta_qtc_ms, reverse=True)
    return results
