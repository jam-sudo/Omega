"""Phase 452 — Drug Stability Screening.

High-throughput stability screening of multiple drug candidates using
functional group stability rules aligned with FDA/ICH Q1A guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any

# ---------------------------------------------------------------------------
# Functional group stability rules
# ---------------------------------------------------------------------------
FUNCTIONAL_GROUPS: dict[str, dict[str, Any]] = {
    "amine": {
        "hydrolysis": 0.1,
        "oxidation": 0.3,
        "photolysis": 0.0,
        "isomerization": 0.0,
        "k_base": 3e-4,
        "excipient_risk": "moderate",
    },
    "ester": {
        "hydrolysis": 0.8,
        "oxidation": 0.1,
        "photolysis": 0.0,
        "isomerization": 0.0,
        "k_base": 8e-4,
        "excipient_risk": "high",
    },
    "alcohol": {
        "hydrolysis": 0.0,
        "oxidation": 0.4,
        "photolysis": 0.0,
        "isomerization": 0.0,
        "k_base": 1e-4,
        "excipient_risk": "low",
    },
    "acid": {
        "hydrolysis": 0.2,
        "oxidation": 0.1,
        "photolysis": 0.0,
        "isomerization": 0.1,
        "k_base": 2e-4,
        "excipient_risk": "moderate",
    },
    "lactam": {
        "hydrolysis": 0.9,
        "oxidation": 0.0,
        "photolysis": 0.1,
        "isomerization": 0.0,
        "k_base": 5e-4,
        "excipient_risk": "high",
    },
    "none": {
        "hydrolysis": 0.0,
        "oxidation": 0.05,
        "photolysis": 0.0,
        "isomerization": 0.0,
        "k_base": 5e-5,
        "excipient_risk": "low",
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DrugStabilityResult:
    """Result of a drug stability screening calculation."""

    drug_name: str
    chemical_class: str  # "amine", "ester", "alcohol", "acid", "lactam", "none"
    degradation_pathways: (
        str  # comma-separated: "hydrolysis", "oxidation", "photolysis", "isomerization"
    )
    primary_degradation_rate: float  # k_deg in days^-1 at 25C/60%RH
    t90_days: float  # time to 90% remaining
    forced_degradation_score: float  # 0-100 (100 = most stable)
    storage_requirement: str  # "room_temp", "refrigerate", "freeze", "protect_light"
    packaging_recommendation: str  # "amber_glass", "HDPE", "blister", "aluminum_foil"
    excipient_compatibility_risk: str  # "low", "moderate", "high"
    shelf_life_estimate_months: float
    anda_stability_protocol: str  # "25C_60RH", "30C_65RH", "40C_75RH_accelerated"
    notes: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_degradation_pathways(fg_data: dict[str, Any]) -> str:
    """Return comma-separated degradation pathways with fraction > 0."""
    pathway_names = ["hydrolysis", "oxidation", "photolysis", "isomerization"]
    active = [p for p in pathway_names if fg_data[p] > 0]
    return ", ".join(active) if active else "none"


def _compute_storage_requirement(t90_days: float, fg_data: dict[str, Any], logp: float) -> str:
    """Determine storage requirement from t90 and photolysis risk."""
    if fg_data["photolysis"] > 0 and logp > 2:
        return "protect_light"
    if t90_days > 730:
        return "room_temp"
    elif t90_days >= 365:
        return "room_temp"
    elif t90_days >= 180:
        return "refrigerate"
    else:
        return "freeze"


def _compute_packaging(
    chemical_class: str, t90_days: float, fg_data: dict[str, Any], logp: float
) -> str:
    """Determine packaging recommendation."""
    if fg_data["photolysis"] > 0:
        return "amber_glass"
    if chemical_class in ("lactam", "ester"):
        return "aluminum_foil"
    if t90_days < 365:
        return "HDPE"
    return "blister"


def _compute_anda_protocol(t90_days: float) -> str:
    """Select ANDA stability protocol based on t90."""
    if t90_days > 365:
        return "25C_60RH"
    elif t90_days > 180:
        return "30C_65RH"
    else:
        return "40C_75RH_accelerated"


# ---------------------------------------------------------------------------
# Core screening function
# ---------------------------------------------------------------------------


def screen_drug_stability(
    drug_name: str,
    chemical_class: str,
    logp: float,
    mw_Da: float,
) -> DrugStabilityResult:
    """Predict stability profile of a single drug candidate.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    chemical_class:
        Functional group class: one of "amine", "ester", "alcohol", "acid", "lactam", "none".
    logp:
        Calculated or measured logP (octanol-water partition coefficient).
    mw_Da:
        Molecular weight in Daltons (must be > 0).

    Returns
    -------
    DrugStabilityResult
        Stability screening result with shelf-life estimate and storage guidance.

    Raises
    ------
    ValueError
        If chemical_class is not recognized or mw_Da <= 0.
    """
    if chemical_class not in FUNCTIONAL_GROUPS:
        raise ValueError(
            f"Unrecognized chemical_class '{chemical_class}'. "
            f"Must be one of: {list(FUNCTIONAL_GROUPS.keys())}"
        )
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be positive, got {mw_Da}")

    fg_data = FUNCTIONAL_GROUPS[chemical_class]
    k_base = fg_data["k_base"]

    # Primary degradation rate at 25C/60%RH
    # k_deg = k_base * (1 + logP * 0.1 + mw_Da / 1000 * 0.2)
    primary_degradation_rate = k_base * (1.0 + logp * 0.1 + mw_Da / 1000.0 * 0.2)

    # t90: time for 10% degradation, first-order kinetics: C = C0 * exp(-k*t)
    # At t90: C = 0.9 * C0 => exp(-k*t90) = 0.9 => t90 = -ln(0.9) / k
    t90_days = -log(0.9) / primary_degradation_rate

    # Shelf life estimate in months
    shelf_life_estimate_months = t90_days / 30.0

    # Forced degradation score: stability after 90 days of stress
    raw_score = 100.0 * exp(-primary_degradation_rate * 90.0)
    forced_degradation_score = max(0.0, min(100.0, raw_score))

    # Degradation pathways
    degradation_pathways = _compute_degradation_pathways(fg_data)

    # Storage requirement
    storage_requirement = _compute_storage_requirement(t90_days, fg_data, logp)

    # Packaging recommendation
    packaging_recommendation = _compute_packaging(chemical_class, t90_days, fg_data, logp)

    # Excipient compatibility risk from fg_data
    excipient_compatibility_risk = fg_data["excipient_risk"]

    # ANDA stability protocol
    anda_stability_protocol = _compute_anda_protocol(t90_days)

    # Notes
    notes = (
        f"{drug_name} is classified as a {chemical_class} compound. "
        f"Estimated t90 = {t90_days:.1f} days ({shelf_life_estimate_months:.1f} months). "
        f"Primary degradation pathway(s): {degradation_pathways}. "
        f"Recommended storage: {storage_requirement}."
    )

    return DrugStabilityResult(
        drug_name=drug_name,
        chemical_class=chemical_class,
        degradation_pathways=degradation_pathways,
        primary_degradation_rate=primary_degradation_rate,
        t90_days=t90_days,
        forced_degradation_score=forced_degradation_score,
        storage_requirement=storage_requirement,
        packaging_recommendation=packaging_recommendation,
        excipient_compatibility_risk=excipient_compatibility_risk,
        shelf_life_estimate_months=shelf_life_estimate_months,
        anda_stability_protocol=anda_stability_protocol,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison / screening function
# ---------------------------------------------------------------------------


def compare_drug_stability(compounds_list: list[dict[str, Any]]) -> list[DrugStabilityResult]:
    """Screen and compare stability of multiple drug candidates.

    Parameters
    ----------
    compounds_list:
        List of dicts, each with keys: drug_name, chemical_class, logp, mw_Da.

    Returns
    -------
    List of DrugStabilityResult sorted by forced_degradation_score descending
    (most stable first).
    """
    results = []
    for compound in compounds_list:
        result = screen_drug_stability(
            drug_name=compound["drug_name"],
            chemical_class=compound["chemical_class"],
            logp=compound["logp"],
            mw_Da=compound["mw_Da"],
        )
        results.append(result)

    results.sort(key=lambda r: r.forced_degradation_score, reverse=True)
    return results
