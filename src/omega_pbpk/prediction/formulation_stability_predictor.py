"""Formulation stability prediction — Phase 488."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FormulationStabilityResult:
    """Result of formulation stability assessment."""

    drug_name: str
    ph: float
    temperature_C: float
    humidity_pct: float
    hydrolysis_risk: str
    oxidation_risk: str
    humidity_risk: str
    overall_stability: str
    predicted_shelf_life_months: float
    recommended_storage: str
    notes: str


def predict_hydrolysis_rate(
    drug_pka: float,
    ph: float,
    temperature_C: float = 25.0,
    activation_energy_kJ_mol: float = 80.0,
) -> float:
    """Predict hydrolysis rate (fraction/h).

    Uses Arrhenius temperature effect and pH-rate profile
    (rate increases at extreme pH: pH < 3 or pH > 9).

    Args:
        drug_pka: Drug pKa.
        ph: Formulation pH.
        temperature_C: Storage temperature (°C).
        activation_energy_kJ_mol: Activation energy for hydrolysis (kJ/mol).

    Returns:
        Hydrolysis rate in fraction/h.
    """
    if ph <= 0:
        raise ValueError("ph must be > 0")
    if temperature_C <= 0:
        raise ValueError("temperature_C must be > 0 (Kelvin-compatible)")

    R = 8.314e-3  # kJ/(mol·K)
    T_ref_K = 298.15  # 25°C reference
    T_K = temperature_C + 273.15

    # Arrhenius temperature factor relative to 25°C
    arrhenius_factor = math.exp((activation_energy_kJ_mol / R) * (1.0 / T_ref_K - 1.0 / T_K))

    # pH-rate profile: base rate at neutral pH, increases at extremes
    # Acid catalysis: significant below pH 3
    # Base catalysis: significant above pH 9
    base_rate = 1e-4  # fraction/h at 25°C, neutral pH

    if ph < 3.0:
        # Acid-catalyzed: rate increases as pH decreases
        ph_factor = 10.0 ** (3.0 - ph)
    elif ph > 9.0:
        # Base-catalyzed: rate increases as pH increases
        ph_factor = 10.0 ** (ph - 9.0)
    elif 4.0 <= ph <= 8.0:
        # Near-neutral: minimum hydrolysis
        ph_factor = 1.0
    else:
        # Transition zones: mild enhancement
        if ph < 4.0:
            ph_factor = 1.0 + (4.0 - ph) * 2.0
        else:
            ph_factor = 1.0 + (ph - 8.0) * 2.0

    # pKa proximity effect: ionizable drugs near pKa may be more susceptible
    pka_factor = 1.0
    if abs(drug_pka - ph) < 1.0:
        pka_factor = 1.2

    rate = base_rate * ph_factor * arrhenius_factor * pka_factor
    return rate


def predict_oxidation_risk(
    logP: float,
    n_double_bonds: int,
    has_sulfur: bool = False,
    n_phenol: int = 0,
) -> str:
    """Predict oxidation risk category.

    Args:
        logP: Lipophilicity.
        n_double_bonds: Number of oxidizable double bonds.
        has_sulfur: Whether the drug contains sulfur (thioethers, etc.).
        n_phenol: Number of phenolic groups.

    Returns:
        Risk category: 'low', 'moderate', or 'high'.
    """
    score = 0

    # Lipophilicity: more lipophilic → more susceptible to oxidation
    if logP >= 3.0:
        score += 2
    elif logP >= 1.0:
        score += 1

    # Unsaturation
    if n_double_bonds >= 3:
        score += 4
    elif n_double_bonds >= 1:
        score += 1

    # Sulfur: thioethers are readily oxidized
    if has_sulfur:
        score += 2

    # Phenol groups: auto-oxidation susceptibility
    if n_phenol >= 2:
        score += 2
    elif n_phenol == 1:
        score += 1

    if score >= 4:
        return "high"
    elif score >= 2:
        return "moderate"
    else:
        return "low"


def predict_photodegradation(
    chromophore_groups: int,
    uv_exposure_h: float = 100.0,
) -> float:
    """Predict fraction of drug photodegraded.

    Args:
        chromophore_groups: Number of UV-absorbing chromophore groups.
        uv_exposure_h: Duration of UV exposure (h).

    Returns:
        Fraction degraded (0-1).
    """
    if chromophore_groups < 0:
        raise ValueError("chromophore_groups must be >= 0")
    if uv_exposure_h < 0:
        raise ValueError("uv_exposure_h must be >= 0")

    if chromophore_groups == 0:
        return 0.0

    # Pseudo-first-order photodegradation rate (fraction/h)
    # More chromophores → faster photodegradation
    k_photo = chromophore_groups * 2e-4  # base rate per chromophore

    # First-order: fraction_degraded = 1 - exp(-k * t)
    fraction = 1.0 - math.exp(-k_photo * uv_exposure_h)
    return min(1.0, fraction)


def _classify_hydrolysis_risk(rate: float) -> str:
    """Classify hydrolysis rate into risk category."""
    if rate < 1e-3:
        return "low"
    elif rate < 1e-2:
        return "moderate"
    else:
        return "high"


def _classify_humidity_risk(humidity_pct: float) -> str:
    """Classify humidity into risk category."""
    if humidity_pct > 75.0:
        return "high"
    elif humidity_pct >= 40.0:
        return "moderate"
    else:
        return "low"


def _overall_stability(
    hydrolysis_risk: str,
    oxidation_risk: str,
    humidity_risk: str,
) -> str:
    """Determine overall stability from individual risk categories."""
    risk_score = {"low": 0, "moderate": 1, "high": 2}
    total = risk_score[hydrolysis_risk] + risk_score[oxidation_risk] + risk_score[humidity_risk]

    if total == 0:
        return "excellent"
    elif total <= 2:
        return "good"
    elif total <= 4:
        return "fair"
    else:
        return "poor"


def _shelf_life_months(
    hydrolysis_risk: str,
    oxidation_risk: str,
    humidity_risk: str,
    temperature_C: float,
) -> float:
    """Estimate predicted shelf life in months.

    Baseline 36 months at 25°C, reduced by risk factors and temperature.
    """
    baseline = 36.0  # months

    risk_multipliers = {"low": 1.0, "moderate": 0.6, "high": 0.3}
    h_mult = risk_multipliers[hydrolysis_risk]
    o_mult = risk_multipliers[oxidation_risk]
    u_mult = risk_multipliers[humidity_risk]

    # Temperature penalty: Arrhenius rule of thumb ~2x per 10°C above 25°C
    if temperature_C > 25.0:
        temp_factor = 0.5 ** ((temperature_C - 25.0) / 10.0)
    else:
        temp_factor = 1.0

    shelf_life = baseline * h_mult * o_mult * u_mult * temp_factor
    return max(1.0, round(shelf_life, 1))


def _recommended_storage(
    hydrolysis_risk: str,
    oxidation_risk: str,
    humidity_risk: str,
) -> str:
    """Generate storage recommendation."""
    recommendations = []

    if hydrolysis_risk == "high":
        recommendations.append("anhydrous conditions")
    if oxidation_risk in ("moderate", "high"):
        recommendations.append("nitrogen atmosphere or antioxidant")
    if humidity_risk == "high":
        recommendations.append("desiccant packaging")
    elif humidity_risk == "moderate":
        recommendations.append("controlled humidity (<60% RH)")

    if not recommendations:
        return "Standard storage (15-25°C, <75% RH) acceptable"

    return "Recommended: " + "; ".join(recommendations) + "; store at 2-8°C if needed"


def assess_formulation_stability(
    drug_name: str,
    ph: float,
    temperature_C: float,
    humidity_pct: float,
    pka: float = 7.0,
    logP: float = 2.0,
    n_double_bonds: int = 0,
    has_sulfur: bool = False,
) -> FormulationStabilityResult:
    """Assess overall formulation stability.

    Args:
        drug_name: Name of the drug.
        ph: Formulation pH.
        temperature_C: Storage temperature (°C).
        humidity_pct: Relative humidity (%).
        pka: Drug pKa.
        logP: Lipophilicity.
        n_double_bonds: Number of oxidizable double bonds.
        has_sulfur: Whether drug contains sulfur groups.

    Returns:
        FormulationStabilityResult with risk assessments and recommendations.
    """
    if ph <= 0:
        raise ValueError("ph must be > 0")
    if temperature_C <= 0:
        raise ValueError("temperature_C must be > 0")
    if not 0.0 <= humidity_pct <= 100.0:
        raise ValueError("humidity_pct must be between 0 and 100")

    hydrolysis_rate = predict_hydrolysis_rate(pka, ph, temperature_C)
    hydrolysis_risk = _classify_hydrolysis_risk(hydrolysis_rate)

    oxidation_risk = predict_oxidation_risk(logP, n_double_bonds, has_sulfur)

    humidity_risk = _classify_humidity_risk(humidity_pct)

    overall = _overall_stability(hydrolysis_risk, oxidation_risk, humidity_risk)

    shelf_life = _shelf_life_months(hydrolysis_risk, oxidation_risk, humidity_risk, temperature_C)

    storage = _recommended_storage(hydrolysis_risk, oxidation_risk, humidity_risk)

    notes = (
        f"Hydrolysis rate={hydrolysis_rate:.2e}/h; pH={ph}; T={temperature_C}°C; RH={humidity_pct}%"
    )

    return FormulationStabilityResult(
        drug_name=drug_name,
        ph=ph,
        temperature_C=temperature_C,
        humidity_pct=humidity_pct,
        hydrolysis_risk=hydrolysis_risk,
        oxidation_risk=oxidation_risk,
        humidity_risk=humidity_risk,
        overall_stability=overall,
        predicted_shelf_life_months=shelf_life,
        recommended_storage=storage,
        notes=notes,
    )
