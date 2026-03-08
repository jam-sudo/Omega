"""Phase 499 — Saliva Drug Concentration Model.

Predict salivary drug concentrations for TDM and forensic applications
using the Henderson-Hasselbalch pH-partitioning model.
"""

from __future__ import annotations

from dataclasses import dataclass

_SALIVA_PH: float = 6.8
_PLASMA_PH: float = 7.4
_RATIO_MIN: float = 0.01
_RATIO_MAX: float = 10.0

_VALID_DRUG_TYPES = {"neutral", "acid", "base"}


@dataclass(frozen=True)
class SalivaDrugResult:
    """Result of saliva drug concentration prediction."""

    drug_name: str
    plasma_conc_mg_L: float
    fu_plasma: float
    pKa: float
    drug_type: str
    saliva_plasma_ratio: float
    saliva_conc_mg_L: float
    monitoring_utility: str
    notes: str


def _compute_ratio(fu_plasma: float, pKa: float, drug_type: str) -> float:
    """Compute saliva/plasma ratio via Henderson-Hasselbalch."""
    if drug_type == "neutral":
        ratio = fu_plasma
    elif drug_type == "acid":
        numerator = 1.0 + 10.0 ** (_SALIVA_PH - pKa)
        denominator = 1.0 + 10.0 ** (_PLASMA_PH - pKa)
        ratio = fu_plasma * numerator / denominator
    else:  # base
        numerator = 1.0 + 10.0 ** (pKa - _SALIVA_PH)
        denominator = 1.0 + 10.0 ** (pKa - _PLASMA_PH)
        ratio = fu_plasma * numerator / denominator

    return max(_RATIO_MIN, min(_RATIO_MAX, ratio))


def _classify_monitoring(ratio: float) -> str:
    """Classify monitoring utility based on saliva/plasma ratio."""
    if 0.8 <= ratio <= 1.2:
        return "excellent"
    if 0.5 <= ratio <= 1.5:
        return "good"
    return "poor"


def _build_notes(drug_type: str, ratio: float, pKa: float) -> str:
    """Build informative notes string."""
    parts: list[str] = []
    if drug_type == "neutral":
        parts.append("Neutral drug: ratio equals unbound plasma fraction.")
    elif drug_type == "acid":
        if ratio < 1.0:
            parts.append(
                "Acidic drug: lower saliva concentration due to greater ionization at lower pH."
            )
        else:
            parts.append("Acidic drug: ratio reflects pH-partitioning.")
    else:
        if ratio > 1.0:
            parts.append("Basic drug: ion trapping increases salivary concentration at lower pH.")
        else:
            parts.append("Basic drug: ratio reflects pH-partitioning.")
    if ratio <= _RATIO_MIN:
        parts.append("Ratio clamped at minimum 0.01.")
    if ratio >= _RATIO_MAX:
        parts.append("Ratio clamped at maximum 10.0.")
    return " ".join(parts)


def predict_saliva_concentration(
    drug_name: str,
    plasma_conc_mg_L: float,
    fu_plasma: float,
    pKa: float,
    drug_type: str,
) -> SalivaDrugResult:
    """Predict salivary drug concentration from plasma concentration.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    plasma_conc_mg_L:
        Total plasma concentration in mg/L (must be > 0).
    fu_plasma:
        Unbound fraction in plasma (0 < fu_plasma <= 1).
    pKa:
        Ionisation constant of the drug.
    drug_type:
        One of "neutral", "acid", or "base".

    Returns
    -------
    SalivaDrugResult
    """
    if plasma_conc_mg_L <= 0:
        raise ValueError(f"plasma_conc_mg_L must be > 0, got {plasma_conc_mg_L}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {_VALID_DRUG_TYPES}, got '{drug_type}'")

    ratio = _compute_ratio(fu_plasma, pKa, drug_type)
    saliva_conc = plasma_conc_mg_L * ratio
    utility = _classify_monitoring(ratio)
    notes = _build_notes(drug_type, ratio, pKa)

    return SalivaDrugResult(
        drug_name=drug_name,
        plasma_conc_mg_L=plasma_conc_mg_L,
        fu_plasma=fu_plasma,
        pKa=pKa,
        drug_type=drug_type,
        saliva_plasma_ratio=ratio,
        saliva_conc_mg_L=saliva_conc,
        monitoring_utility=utility,
        notes=notes,
    )


def compare_saliva_monitoring(compounds: list) -> list:
    """Batch-screen compounds, sorted ascending by saliva/plasma ratio.

    Parameters
    ----------
    compounds:
        List of dicts with keys: drug_name, plasma_conc_mg_L, fu_plasma,
        pKa, drug_type.

    Returns
    -------
    List of SalivaDrugResult sorted by saliva_plasma_ratio ascending.
    """
    results: list[SalivaDrugResult] = []
    for compound in compounds:
        result = predict_saliva_concentration(
            drug_name=compound["drug_name"],
            plasma_conc_mg_L=compound["plasma_conc_mg_L"],
            fu_plasma=compound["fu_plasma"],
            pKa=compound["pKa"],
            drug_type=compound["drug_type"],
        )
        results.append(result)

    results.sort(key=lambda r: r.saliva_plasma_ratio)
    return results
