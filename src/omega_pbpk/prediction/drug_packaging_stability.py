"""Drug Packaging Stability — Phase 1042.

Predicts drug stability in different packaging materials and storage conditions
using first-order degradation kinetics with Arrhenius temperature acceleration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_PACKAGING = {"HDPE", "glass", "blister_PVC", "blister_PVDC", "aluminum_foil"}
_VALID_STORAGE = {"25C_60RH", "30C_65RH", "40C_75RH"}
_VALID_SUSCEPTIBILITY = {"low", "moderate", "high"}
_VALID_PATHWAYS = {"hydrolysis", "oxidation", "photodegradation", "combined"}

# Base degradation rate constants (/month) by susceptibility
_K_HYDROLYSIS = {"low": 0.001, "moderate": 0.005, "high": 0.02}
_K_OXIDATION = {"low": 0.002, "moderate": 0.008, "high": 0.03}

# Packaging transmission factors: (moisture_factor, O2_factor)
_PACKAGING_FACTORS: dict[str, tuple[float, float]] = {
    "aluminum_foil": (0.01, 0.01),
    "glass": (0.05, 0.1),
    "blister_PVDC": (0.1, 0.2),
    "HDPE": (0.3, 0.5),
    "blister_PVC": (0.8, 0.9),
}

# Arrhenius temperature acceleration factors
_TEMP_FACTORS = {
    "25C_60RH": 1.0,
    "30C_65RH": 2.0,
    "40C_75RH": 10.0,
}


@dataclass(frozen=True)
class PackagingStabilityResult:
    drug_name: str
    packaging: str
    storage_condition: str
    initial_purity_pct: float
    predicted_purity_at_12m: float
    predicted_purity_at_24m: float
    predicted_purity_at_36m: float
    shelf_life_months: float
    degradation_rate_per_month: float
    degradation_pathway: str
    ict_compliant: bool
    recommended_packaging: bool
    notes: str


def predict_packaging_stability(
    drug_name: str,
    mw_Da: float,
    logp: float,
    hydrolysis_susceptibility: str = "low",
    oxidation_susceptibility: str = "low",
    packaging: str = "HDPE",
    storage_condition: str = "25C_60RH",
    initial_purity_pct: float = 99.5,
) -> PackagingStabilityResult:
    """Predict drug stability in a given packaging under given storage conditions."""
    # Validation
    if initial_purity_pct <= 0:
        raise ValueError("initial_purity_pct must be > 0")
    if packaging not in _VALID_PACKAGING:
        raise ValueError(f"packaging must be one of: {sorted(_VALID_PACKAGING)}")
    if storage_condition not in _VALID_STORAGE:
        raise ValueError(f"storage_condition must be one of: {sorted(_VALID_STORAGE)}")
    if hydrolysis_susceptibility not in _VALID_SUSCEPTIBILITY:
        raise ValueError(
            f"hydrolysis_susceptibility must be one of: {sorted(_VALID_SUSCEPTIBILITY)}"
        )
    if oxidation_susceptibility not in _VALID_SUSCEPTIBILITY:
        raise ValueError(
            f"oxidation_susceptibility must be one of: {sorted(_VALID_SUSCEPTIBILITY)}"
        )

    # Base rates
    k_hyd_base = _K_HYDROLYSIS[hydrolysis_susceptibility]
    k_ox_base = _K_OXIDATION[oxidation_susceptibility]

    # Packaging factors
    moisture_factor, o2_factor = _PACKAGING_FACTORS[packaging]

    # Temperature factor
    temp_factor = _TEMP_FACTORS[storage_condition]

    # Actual rates
    k_hyd_actual = k_hyd_base * moisture_factor * temp_factor
    k_ox_actual = k_ox_base * o2_factor * temp_factor
    k_total = k_hyd_actual + k_ox_actual

    # Photodegradation contribution
    photo_active = logp > 3 and o2_factor > 0.5
    if photo_active:
        k_total *= 1.2

    # Determine pathway
    if k_hyd_actual > k_ox_actual * 2:
        pathway = "hydrolysis"
    elif k_ox_actual > k_hyd_actual * 2:
        pathway = "oxidation"
    else:
        pathway = "combined"

    # Override if photodegradation applies
    if photo_active:
        pathway = "combined"

    # Purity at time points
    def purity_at(t_months: float) -> float:
        return initial_purity_pct * math.exp(-k_total * t_months)

    p12 = purity_at(12.0)
    p24 = purity_at(24.0)
    p36 = purity_at(36.0)

    # Shelf life: time to 90% purity
    if k_total > 0:
        shelf_life = math.log(initial_purity_pct / 90.0) / k_total
    else:
        shelf_life = 120.0
    shelf_life = max(0.0, min(120.0, shelf_life))

    ict_compliant = shelf_life >= 24.0
    recommended_packaging = shelf_life >= 36.0

    notes = (
        f"Drug: {drug_name} (MW={mw_Da:.1f} Da, logP={logp:.2f}). "
        f"Packaging: {packaging}, Storage: {storage_condition}. "
        f"k_total={k_total:.5f}/month. "
        f"Shelf life: {shelf_life:.1f} months (ICH Q1E 90% criterion). "
        f"Pathway: {pathway}. "
        f"ICH compliant (>=24m): {ict_compliant}. "
        f"Recommended packaging (>=36m): {recommended_packaging}."
    )

    return PackagingStabilityResult(
        drug_name=drug_name,
        packaging=packaging,
        storage_condition=storage_condition,
        initial_purity_pct=initial_purity_pct,
        predicted_purity_at_12m=p12,
        predicted_purity_at_24m=p24,
        predicted_purity_at_36m=p36,
        shelf_life_months=shelf_life,
        degradation_rate_per_month=k_total,
        degradation_pathway=pathway,
        ict_compliant=ict_compliant,
        recommended_packaging=recommended_packaging,
        notes=notes,
    )


def compare_packaging_options(
    drug_name: str,
    mw_Da: float,
    logp: float,
    hydrolysis_susceptibility: str = "low",
    oxidation_susceptibility: str = "low",
    storage_condition: str = "25C_60RH",
    initial_purity_pct: float = 99.5,
) -> list:
    """Compare all 5 packaging types, sorted by shelf_life descending."""
    results = []
    for pkg in sorted(_VALID_PACKAGING):
        result = predict_packaging_stability(
            drug_name=drug_name,
            mw_Da=mw_Da,
            logp=logp,
            hydrolysis_susceptibility=hydrolysis_susceptibility,
            oxidation_susceptibility=oxidation_susceptibility,
            packaging=pkg,
            storage_condition=storage_condition,
            initial_purity_pct=initial_purity_pct,
        )
        results.append(result)
    results.sort(key=lambda r: r.shelf_life_months, reverse=True)
    return results
