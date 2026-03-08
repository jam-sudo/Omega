"""
Phase 508 — Drug Concentration in Tears

Predict drug concentration in lacrimal fluid (tears) for ophthalmic
and systemic drug monitoring applications.

Model:
  - Tear/plasma ratio based on ionization state and protein binding
  - Dilution factor for tear secretion
  - Monitoring application classification
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TearDrugConcentrationResult:
    """Result of tear drug concentration prediction."""

    drug_name: str
    plasma_conc_mg_L: float
    fu_plasma: float
    pKa: float
    drug_type: str  # "acid", "base", or "neutral"

    tear_plasma_ratio: float
    tear_conc_mg_L: float

    monitoring_application: str  # "TDM_feasible" or "research_only"
    notes: str


# Physiological constants
_PH_PLASMA: float = 7.4
_PH_TEAR: float = 7.4
_TEAR_DILUTION_FACTOR: float = 0.6
_MIN_RATIO: float = 0.01
_MAX_RATIO: float = 5.0

# TDM feasibility range
_TDM_RATIO_LOW: float = 0.5
_TDM_RATIO_HIGH: float = 2.0


def _compute_ionization_ratio(
    fu: float,
    pKa: float,
    drug_type: str,
    ph_plasma: float = _PH_PLASMA,
    ph_tear: float = _PH_TEAR,
) -> float:
    """
    Compute theoretical tear/plasma ratio based on ionization.

    For a base:
        R = fu * (1 + 10^(pKa - pH_plasma)) / (1 + 10^(pKa - pH_tear))
    For an acid:
        R = fu * (1 + 10^(pH_tear - pKa)) / (1 + 10^(pH_plasma - pKa))
    For neutral:
        R = fu

    Since pH_tear = pH_plasma = 7.4 at equilibrium, all forms reduce to R = fu,
    but the formula is kept general for non-physiological scenarios.
    """
    if drug_type == "base":
        numerator = 1.0 + 10.0 ** (pKa - ph_plasma)
        denominator = 1.0 + 10.0 ** (pKa - ph_tear)
        if denominator == 0:
            ratio = fu
        else:
            ratio = fu * numerator / denominator
    elif drug_type == "acid":
        numerator = 1.0 + 10.0 ** (ph_tear - pKa)
        denominator = 1.0 + 10.0 ** (ph_plasma - pKa)
        if denominator == 0:
            ratio = fu
        else:
            ratio = fu * numerator / denominator
    else:
        # neutral
        ratio = fu

    return ratio


def predict_tear_concentration(
    drug_name: str,
    plasma_conc_mg_L: float,
    fu_plasma: float,
    pKa: float,
    drug_type: str,
) -> TearDrugConcentrationResult:
    """
    Predict drug concentration in tears from plasma concentration.

    Parameters
    ----------
    drug_name : str
    plasma_conc_mg_L : float
        Plasma drug concentration (mg/L). Must be > 0.
    fu_plasma : float
        Unbound fraction in plasma (0, 1]. Must be in (0, 1].
    pKa : float
        Drug pKa value.
    drug_type : str
        "acid", "base", or "neutral".

    Returns
    -------
    TearDrugConcentrationResult
    """
    # Validation
    if plasma_conc_mg_L <= 0:
        raise ValueError("plasma_conc_mg_L must be positive")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if drug_type not in {"acid", "base", "neutral"}:
        raise ValueError("drug_type must be 'acid', 'base', or 'neutral'")

    # Compute theoretical ratio
    theoretical_ratio = _compute_ionization_ratio(
        fu=fu_plasma,
        pKa=pKa,
        drug_type=drug_type,
        ph_plasma=_PH_PLASMA,
        ph_tear=_PH_TEAR,
    )

    # Apply tear dilution factor
    effective_ratio = theoretical_ratio * _TEAR_DILUTION_FACTOR

    # Clamp to physiological range
    effective_ratio = max(_MIN_RATIO, min(_MAX_RATIO, effective_ratio))

    # Compute tear concentration
    tear_conc = plasma_conc_mg_L * effective_ratio

    # Monitoring application
    if _TDM_RATIO_LOW <= effective_ratio <= _TDM_RATIO_HIGH:
        monitoring_application = "TDM_feasible"
    else:
        monitoring_application = "research_only"

    notes = (
        f"Drug type: {drug_type}, pKa={pKa:.2f}, fu={fu_plasma:.3f}; "
        f"theoretical ratio={theoretical_ratio:.4f}, "
        f"effective ratio (×{_TEAR_DILUTION_FACTOR})={effective_ratio:.4f}; "
        f"pH_tear={_PH_TEAR}, pH_plasma={_PH_PLASMA}"
    )

    return TearDrugConcentrationResult(
        drug_name=drug_name,
        plasma_conc_mg_L=plasma_conc_mg_L,
        fu_plasma=fu_plasma,
        pKa=pKa,
        drug_type=drug_type,
        tear_plasma_ratio=effective_ratio,
        tear_conc_mg_L=tear_conc,
        monitoring_application=monitoring_application,
        notes=notes,
    )


def compare_tear_monitoring(compounds: list) -> list:
    """
    Compare tear monitoring feasibility across multiple compounds.

    Parameters
    ----------
    compounds : list of dict
        Each dict must have keys: drug_name, plasma_conc_mg_L, fu_plasma, pKa, drug_type.

    Returns
    -------
    list of TearDrugConcentrationResult
        Sorted by tear_plasma_ratio descending.
    """
    results = []
    for comp in compounds:
        r = predict_tear_concentration(
            drug_name=comp["drug_name"],
            plasma_conc_mg_L=comp["plasma_conc_mg_L"],
            fu_plasma=comp["fu_plasma"],
            pKa=comp["pKa"],
            drug_type=comp["drug_type"],
        )
        results.append(r)

    results.sort(key=lambda r: r.tear_plasma_ratio, reverse=True)
    return results
