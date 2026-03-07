"""
Phase 929: Hepatic drug concentration prediction from plasma data.
Supports Kp_liver estimation, hepatocellular ion trapping, hepatic extraction,
and DILI risk assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "HepaticDrugConcentrationResult",
    "predict_hepatic_concentration",
    "screen_hepatic_risk",
]


@dataclass(frozen=True)
class HepaticDrugConcentrationResult:
    """Result of hepatic drug concentration prediction."""

    drug_name: str
    logp: float
    fu_plasma: float
    pka: float
    ionization_type: str
    c_plasma_mg_L: float
    kp_liver: float
    c_liver_mg_L: float
    c_hepatocyte_mg_L: float
    e_hepatic: float
    liver_to_plasma_ratio: float
    clint_L_per_h: float
    dili_risk: str
    notes: str


def _compute_kp_liver(logp: float, fu_plasma: float) -> float:
    """
    Compute liver Kp using the practical formula:
        Kp_liver = exp(0.8 * logP - 0.5 * logfu)
    where logfu = log10(fu_plasma).
    Clamped to [0.1, 1000].
    """
    logfu = math.log10(fu_plasma)
    kp = math.exp(0.8 * logp - 0.5 * logfu)
    return max(0.1, min(1000.0, kp))


def _compute_kp_cell_plasma(pka: float, ionization_type: str) -> float:
    """
    Compute intracellular/plasma concentration ratio via pH-partition (ion trapping).
    pH_cell = 7.0, pH_plasma = 7.4.
    - base: Kp = (1 + 10^(pH_cell - pKa)) / (1 + 10^(pH_plasma - pKa))
    - acid: Kp = (1 + 10^(pKa - pH_cell)) / (1 + 10^(pKa - pH_plasma))
    - neutral: Kp = 1.0
    """
    pH_cell = 7.0
    pH_plasma = 7.4

    if ionization_type == "neutral":
        return 1.0
    elif ionization_type == "base":
        # Weak bases accumulate in acidic (lower pH) compartments via ion trapping
        # Kp = (1 + 10^(pKa - pH_cell)) / (1 + 10^(pKa - pH_plasma))
        # pH_cell(7.0) < pH_plasma(7.4) → more ionized in cell → Kp > 1
        numerator = 1.0 + 10 ** (pka - pH_cell)
        denominator = 1.0 + 10 ** (pka - pH_plasma)
        return numerator / denominator
    else:  # acid
        # Weak acids accumulate in more alkaline (higher pH) compartments
        # Kp = (1 + 10^(pH_cell - pKa)) / (1 + 10^(pH_plasma - pKa))
        # pH_cell(7.0) < pH_plasma(7.4) → less ionized in cell → Kp < 1
        numerator = 1.0 + 10 ** (pH_cell - pka)
        denominator = 1.0 + 10 ** (pH_plasma - pka)
        return numerator / denominator


def _classify_dili_risk(kp_liver: float, liver_to_plasma_ratio: float) -> str:
    """Classify DILI risk from Kp_liver and liver-to-plasma ratio."""
    if kp_liver > 100 or liver_to_plasma_ratio > 50:
        return "high"
    elif kp_liver > 10:
        return "moderate"
    else:
        return "low"


def predict_hepatic_concentration(
    drug_name: str,
    logp: float,
    fu_plasma: float,
    pka: float,
    ionization_type: str,
    c_plasma_mg_L: float,
    clint_L_per_h: float = 5.0,
    q_liver_L_per_h: float = 97.0,
) -> HepaticDrugConcentrationResult:
    """
    Predict liver and hepatocyte drug concentration from plasma concentration.

    Parameters
    ----------
    drug_name : str
    logp : float
        Octanol-water partition coefficient (log scale).
    fu_plasma : float
        Unbound fraction in plasma, in (0, 1].
    pka : float
        Acid/base dissociation constant.
    ionization_type : str
        One of "acid", "base", "neutral".
    c_plasma_mg_L : float
        Total plasma drug concentration (mg/L).
    clint_L_per_h : float
        Intrinsic clearance (L/h). Default 5.0.
    q_liver_L_per_h : float
        Hepatic blood flow (L/h). Default 97.0.

    Returns
    -------
    HepaticDrugConcentrationResult
    """
    # Validation
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if c_plasma_mg_L < 0:
        raise ValueError("c_plasma_mg_L must be >= 0")
    if clint_L_per_h < 0:
        raise ValueError("clint_L_per_h must be >= 0")
    if q_liver_L_per_h <= 0:
        raise ValueError("q_liver_L_per_h must be > 0")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be one of 'acid', 'base', 'neutral'; got '{ionization_type}'"
        )

    # Liver Kp
    kp_liver = _compute_kp_liver(logp, fu_plasma)

    # Liver concentration
    c_liver_mg_L = kp_liver * c_plasma_mg_L

    # Hepatic extraction ratio: E_h = (CLint * fu_p) / (Q + CLint * fu_p)
    clint_fu = clint_L_per_h * fu_plasma
    e_hepatic = clint_fu / (q_liver_L_per_h + clint_fu)
    # Clamp to [0, 1]
    e_hepatic = max(0.0, min(1.0, e_hepatic))

    # Hepatocellular concentration via ion trapping
    kp_cell_plasma = _compute_kp_cell_plasma(pka, ionization_type)
    c_hepatocyte_mg_L = c_liver_mg_L * kp_cell_plasma

    # Liver-to-plasma ratio == kp_liver (by definition of the model)
    liver_to_plasma_ratio = kp_liver

    # DILI risk classification
    dili_risk = _classify_dili_risk(kp_liver, liver_to_plasma_ratio)

    # Notes
    notes_parts = [
        f"Kp_liver={kp_liver:.2f} (logP={logp}, fu={fu_plasma})",
        f"E_h={e_hepatic:.3f}",
        f"Kp_cell_plasma={kp_cell_plasma:.3f} ({ionization_type}, pKa={pka})",
        f"DILI risk: {dili_risk}",
    ]
    notes = "; ".join(notes_parts)

    return HepaticDrugConcentrationResult(
        drug_name=drug_name,
        logp=logp,
        fu_plasma=fu_plasma,
        pka=pka,
        ionization_type=ionization_type,
        c_plasma_mg_L=c_plasma_mg_L,
        kp_liver=kp_liver,
        c_liver_mg_L=c_liver_mg_L,
        c_hepatocyte_mg_L=c_hepatocyte_mg_L,
        e_hepatic=e_hepatic,
        liver_to_plasma_ratio=liver_to_plasma_ratio,
        clint_L_per_h=clint_L_per_h,
        dili_risk=dili_risk,
        notes=notes,
    )


def screen_hepatic_risk(
    compounds: list,
) -> list:
    """
    Screen a list of compounds for hepatic risk.

    Each compound dict must include:
        drug_name, logp, fu_plasma, pka, ionization_type, c_plasma_mg_L
    Optional:
        clint_L_per_h (default 5.0), q_liver_L_per_h (default 97.0)

    Returns list of HepaticDrugConcentrationResult sorted by
    liver_to_plasma_ratio descending (highest risk first).
    """
    results = []
    for cmpd in compounds:
        result = predict_hepatic_concentration(
            drug_name=cmpd["drug_name"],
            logp=cmpd["logp"],
            fu_plasma=cmpd["fu_plasma"],
            pka=cmpd["pka"],
            ionization_type=cmpd["ionization_type"],
            c_plasma_mg_L=cmpd["c_plasma_mg_L"],
            clint_L_per_h=cmpd.get("clint_L_per_h", 5.0),
            q_liver_L_per_h=cmpd.get("q_liver_L_per_h", 97.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.liver_to_plasma_ratio, reverse=True)
    return results
