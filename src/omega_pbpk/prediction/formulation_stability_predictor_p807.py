"""Formulation stability prediction — Phase 808.

Predicts chemical degradation and physical instability for drug formulations
based on structural alerts, formulation type, and storage conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FormulationStabilityResult:
    """Result of formulation stability prediction."""

    drug_name: str
    formulation_type: (
        str  # "tablet", "capsule", "solution", "suspension", "lyophilized", "emulsion"
    )
    storage_condition: str  # "25C_60RH", "40C_75RH", "2-8C", "frozen"
    t90_months: float  # time to 90% potency (ICH shelf life)
    degradation_rate_per_month: float  # first-order k (per month)
    predicted_2yr_potency_pct: float  # % potency at 24 months
    hydrolysis_risk: str  # "high", "moderate", "low"
    oxidation_risk: str  # "high", "moderate", "low"
    photodegradation_risk: str  # "high", "moderate", "low"
    physical_instability_risk: str  # "high", "moderate", "low"
    overall_stability_class: (
        str  # "excellent" (t90>24m), "good" (12-24m), "marginal" (6-12m), "poor" (<6m)
    )
    recommended_excipients: str
    notes: str


def _hydrolysis_risk(
    formulation_type: str,
    has_ester: bool,
    has_amide: bool,
    water_activity: float,
) -> str:
    """Determine hydrolysis risk from structural alerts and formulation."""
    if has_ester:
        # Esters are highly susceptible to hydrolysis in aqueous environments
        if formulation_type == "solution" or water_activity > 0.6:
            return "high"
        return "moderate"
    if has_amide:
        return "moderate"
    return "low"


def _oxidation_risk(
    formulation_type: str,
    has_double_bond: bool,
    has_phenol: bool,
) -> str:
    """Determine oxidation risk from structural alerts and formulation."""
    # O2-exposed formulations: tablet, capsule, solution
    o2_exposed = formulation_type in ("tablet", "capsule", "solution")
    if (has_double_bond or has_phenol) and o2_exposed:
        return "high"
    if has_double_bond or has_phenol:
        return "moderate"
    return "low"


def _photodegradation_risk(
    has_phenol: bool,
    has_double_bond: bool,
) -> str:
    """Determine photodegradation risk from structural alerts."""
    if has_double_bond:
        # Assumes aromatic double bond — high photodegradation risk
        return "high"
    if has_phenol:
        return "moderate"
    return "low"


def _physical_instability_risk(
    formulation_type: str,
    logp: float,
) -> str:
    """Determine physical instability risk."""
    if formulation_type == "emulsion":
        if logp < 0:
            return "high"
        return "moderate"
    if formulation_type == "suspension":
        return "moderate"
    return "low"


def _overall_stability_class(t90: float) -> str:
    """Classify overall stability based on t90."""
    if t90 > 24.0:
        return "excellent"
    if t90 >= 12.0:
        return "good"
    if t90 >= 6.0:
        return "marginal"
    return "poor"


def _recommended_excipients(
    hydrolysis_risk: str,
    oxidation_risk: str,
    photodegradation_risk: str,
    physical_instability_risk: str,
) -> str:
    """Generate recommended excipient suggestions."""
    excipients = []
    if hydrolysis_risk == "high":
        excipients.append("desiccant (silica gel)")
    if hydrolysis_risk in ("high", "moderate"):
        excipients.append("anhydrous excipients")
    if oxidation_risk == "high":
        excipients.append("antioxidants (BHT/BHA/alpha-tocopherol)")
    if oxidation_risk in ("high", "moderate"):
        excipients.append("nitrogen purge packaging")
    if photodegradation_risk in ("high", "moderate"):
        excipients.append("amber glass/opaque packaging")
    if physical_instability_risk == "high":
        excipients.append("emulsifiers (polysorbate 80)")
    if physical_instability_risk == "moderate" and not excipients:
        excipients.append("suspending agents (HPMC)")
    if not excipients:
        excipients.append("standard excipients acceptable")
    return "; ".join(excipients)


def _storage_modifier(storage_condition: str) -> float:
    """Return t90 multiplier based on storage condition."""
    modifiers = {
        "40C_75RH": 0.4,
        "25C_60RH": 1.0,
        "2-8C": 1.8,
        "frozen": 3.0,
    }
    return modifiers.get(storage_condition, 1.0)


def predict_formulation_stability(
    drug_name: str,
    formulation_type: str = "tablet",
    storage_condition: str = "25C_60RH",
    logp: float = 2.0,
    pka: float | None = None,
    has_ester: bool = False,
    has_amide: bool = False,
    has_aldehyde: bool = False,
    has_double_bond: bool = False,
    has_phenol: bool = False,
    mw_da: float = 300.0,
    water_activity: float = 0.5,  # for solid formulations
) -> FormulationStabilityResult:
    """Predict formulation stability from structural and environmental factors.

    Args:
        drug_name: Name of the drug.
        formulation_type: Type of formulation ("tablet", "capsule", "solution",
            "suspension", "lyophilized", "emulsion").
        storage_condition: Storage condition ("25C_60RH", "40C_75RH", "2-8C", "frozen").
        logp: Lipophilicity (log P).
        pka: Drug pKa (optional, currently informational).
        has_ester: Whether drug contains ester functional group.
        has_amide: Whether drug contains amide functional group.
        has_aldehyde: Whether drug contains aldehyde functional group.
        has_double_bond: Whether drug contains reactive double bond.
        has_phenol: Whether drug contains phenol group.
        mw_da: Molecular weight in Daltons.
        water_activity: Water activity of the formulation (0-1).

    Returns:
        FormulationStabilityResult with predicted stability metrics.

    Raises:
        ValueError: If water_activity is outside [0, 1] or mw_da <= 0.
    """
    if not 0.0 <= water_activity <= 1.0:
        raise ValueError(f"water_activity must be in [0, 1], got {water_activity}")
    if mw_da <= 0:
        raise ValueError(f"mw_da must be > 0, got {mw_da}")

    valid_formulations = {"tablet", "capsule", "solution", "suspension", "lyophilized", "emulsion"}
    if formulation_type not in valid_formulations:
        raise ValueError(
            f"formulation_type must be one of {valid_formulations}, got {formulation_type!r}"
        )

    valid_conditions = {"25C_60RH", "40C_75RH", "2-8C", "frozen"}
    if storage_condition not in valid_conditions:
        raise ValueError(
            f"storage_condition must be one of {valid_conditions}, got {storage_condition!r}"
        )

    # Compute individual risk levels
    hyd_risk = _hydrolysis_risk(formulation_type, has_ester, has_amide, water_activity)
    ox_risk = _oxidation_risk(formulation_type, has_double_bond, has_phenol)
    photo_risk = _photodegradation_risk(has_phenol, has_double_bond)
    phys_risk = _physical_instability_risk(formulation_type, logp)

    # Aldehyde: adds to hydrolysis and oxidation if not already high
    if has_aldehyde:
        risk_levels = {"low": 0, "moderate": 1, "high": 2}
        if risk_levels[hyd_risk] < 1:
            hyd_risk = "moderate"
        if risk_levels[ox_risk] < 2:
            ox_risk = "high"

    # Base t90 = 60 months; each high risk reduces by -12m, moderate by -6m
    t90_base = 60.0
    risk_deductions = {"high": -12.0, "moderate": -6.0, "low": 0.0}

    t90_base += risk_deductions[hyd_risk]
    t90_base += risk_deductions[ox_risk]
    t90_base += risk_deductions[photo_risk]
    t90_base += risk_deductions[phys_risk]

    # Ensure t90_base doesn't go below 1 month
    t90_base = max(t90_base, 1.0)

    # Apply storage condition modifier
    storage_mod = _storage_modifier(storage_condition)
    t90_months = t90_base * storage_mod
    t90_months = max(t90_months, 0.1)

    # First-order degradation rate: t90 = ln(10/9) / k ≈ 0.1054 / k
    # => k = 0.1054 / t90
    degradation_rate_per_month = 0.1054 / t90_months

    # Potency at 24 months: C(t) = C0 * exp(-k*t)
    predicted_2yr_potency_pct = 100.0 * math.exp(-degradation_rate_per_month * 24.0)
    predicted_2yr_potency_pct = max(0.0, min(100.0, predicted_2yr_potency_pct))

    # Overall stability class from t90
    stability_class = _overall_stability_class(t90_months)

    # Recommended excipients
    excipients = _recommended_excipients(hyd_risk, ox_risk, photo_risk, phys_risk)

    notes = (
        f"formulation={formulation_type}; storage={storage_condition}; "
        f"logP={logp}; water_activity={water_activity}; "
        f"MW={mw_da} Da; t90={t90_months:.1f} months"
    )

    return FormulationStabilityResult(
        drug_name=drug_name,
        formulation_type=formulation_type,
        storage_condition=storage_condition,
        t90_months=t90_months,
        degradation_rate_per_month=degradation_rate_per_month,
        predicted_2yr_potency_pct=predicted_2yr_potency_pct,
        hydrolysis_risk=hyd_risk,
        oxidation_risk=ox_risk,
        photodegradation_risk=photo_risk,
        physical_instability_risk=phys_risk,
        overall_stability_class=stability_class,
        recommended_excipients=excipients,
        notes=notes,
    )


def screen_formulation_stability(
    drug_name: str,
    conditions: list[dict],
) -> list[FormulationStabilityResult]:
    """Screen a drug across multiple storage conditions.

    Args:
        drug_name: Name of the drug.
        conditions: List of dicts with keyword arguments for
            predict_formulation_stability (excluding drug_name).

    Returns:
        List of FormulationStabilityResult sorted by t90 descending (most stable first).
    """
    results = []
    for cond in conditions:
        result = predict_formulation_stability(drug_name=drug_name, **cond)
        results.append(result)
    # Sort by t90 descending
    results.sort(key=lambda r: r.t90_months, reverse=True)
    return results
