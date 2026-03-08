"""
Phase 446 — Solution Viscosity Prediction for Liquid Formulations

Predicts viscosity of pharmaceutical liquid formulations based on excipient
type and concentration. Important for injectables, eye drops, and oral liquids.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Excipient database
# ---------------------------------------------------------------------------

EXCIPIENTS = {
    "water": {"base_visc_cP": 1.0, "conc_factor": 0.0, "behavior": "newtonian"},
    "PEG_400": {"base_visc_cP": 2.0, "conc_factor": 0.3, "behavior": "newtonian"},
    "glycerin": {"base_visc_cP": 3.0, "conc_factor": 0.5, "behavior": "newtonian"},
    "HPMC_0.5pct": {"base_visc_cP": 5.0, "conc_factor": 2.0, "behavior": "shear_thinning"},
    "HPMC_1pct": {"base_visc_cP": 15.0, "conc_factor": 5.0, "behavior": "shear_thinning"},
    "HPMC_2pct": {"base_visc_cP": 80.0, "conc_factor": 20.0, "behavior": "shear_thinning"},
    "carbomer_0.5pct": {"base_visc_cP": 500.0, "conc_factor": 100.0, "behavior": "shear_thinning"},
    "alginate_1pct": {"base_visc_cP": 100.0, "conc_factor": 30.0, "behavior": "shear_thinning"},
    "poloxamer_20pct": {"base_visc_cP": 200.0, "conc_factor": 50.0, "behavior": "shear_thinning"},
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViscosityResult:
    drug_name: str
    formulation_type: str  # "aqueous_solution", "suspension", "emulsion", "gel", "hydrogel"
    drug_concentration_mg_mL: float
    excipient_name: str  # main viscosity modifier
    excipient_concentration_pct: float
    temperature_C: float
    viscosity_cP: float  # centipoise (water = 1 cP at 20°C)
    viscosity_class: str  # "water_like"(<2), "low"(2-10), "moderate"(10-100), "high"(100-1000), "very_high"(>1000)
    syringeability_class: str  # "excellent"(<5), "good"(5-20), "acceptable"(20-50), "difficult"(50-200), "not_injectable"(>200)
    flow_behavior: str  # "newtonian", "shear_thinning", "shear_thickening"
    injectability_rating: float  # 0-100 (100=easiest)
    storage_temp_effect: str  # "refrigerator_increases_viscosity", "stable", "decreases_with_temp"
    notes: str


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _viscosity_class(visc_cP: float) -> str:
    if visc_cP < 2.0:
        return "water_like"
    elif visc_cP < 10.0:
        return "low"
    elif visc_cP < 100.0:
        return "moderate"
    elif visc_cP < 1000.0:
        return "high"
    else:
        return "very_high"


def _syringeability_class(visc_cP: float) -> str:
    if visc_cP < 5.0:
        return "excellent"
    elif visc_cP < 20.0:
        return "good"
    elif visc_cP < 50.0:
        return "acceptable"
    elif visc_cP < 200.0:
        return "difficult"
    else:
        return "not_injectable"


def _injectability_rating(visc_cP: float) -> float:
    """0-100 rating where 100 is easiest to inject."""
    rating = 100.0 / (1.0 + visc_cP / 20.0)
    return max(0.0, min(100.0, rating))


def _storage_temp_effect(flow_behavior: str, base_visc_cP: float) -> str:
    if flow_behavior == "shear_thinning":
        return "decreases_with_temp"
    elif base_visc_cP > 10.0:
        return "refrigerator_increases_viscosity"
    else:
        return "stable"


def _temperature_correction(raw_viscosity: float, temperature_C: float) -> float:
    """Approximate temperature correction: visc(T) = raw * (298.15 / (T+273.15))^3"""
    t_kelvin = temperature_C + 273.15
    correction = (298.15 / t_kelvin) ** 3
    return raw_viscosity * correction


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_formulation_viscosity(
    drug_name: str,
    formulation_type: str,
    drug_concentration_mg_mL: float,
    excipient_name: str,
    excipient_concentration_pct: float,
    temperature_C: float = 25.0,
    drug_logp: float = 0.0,
) -> ViscosityResult:
    """Predict solution viscosity for a liquid pharmaceutical formulation.

    Parameters
    ----------
    drug_name : str
    formulation_type : str
        One of: "aqueous_solution", "suspension", "emulsion", "gel", "hydrogel"
    drug_concentration_mg_mL : float
        Drug concentration in mg/mL (must be >= 0)
    excipient_name : str
        Excipient name (must be in EXCIPIENTS database)
    excipient_concentration_pct : float
        Excipient concentration in percent (must be >= 0)
    temperature_C : float
        Temperature in Celsius (must be 4 to 80)
    drug_logp : float
        Log P of drug for contribution calculation (default 0.0)

    Returns
    -------
    ViscosityResult
    """
    if excipient_name not in EXCIPIENTS:
        raise ValueError(
            f"excipient_name '{excipient_name}' not found in EXCIPIENTS database. "
            f"Valid options: {list(EXCIPIENTS.keys())}"
        )
    if drug_concentration_mg_mL < 0:
        raise ValueError(f"drug_concentration_mg_mL must be >= 0, got {drug_concentration_mg_mL}")
    if excipient_concentration_pct < 0:
        raise ValueError(
            f"excipient_concentration_pct must be >= 0, got {excipient_concentration_pct}"
        )
    if not (4.0 <= temperature_C <= 80.0):
        raise ValueError(f"temperature_C must be between 4 and 80, got {temperature_C}")

    exc_data = EXCIPIENTS[excipient_name]
    base_visc_cP = exc_data["base_visc_cP"]
    conc_factor = exc_data["conc_factor"]
    flow_behavior = exc_data["behavior"]

    # Drug contribution (small)
    drug_contribution = drug_concentration_mg_mL * 0.001 * (1.0 + drug_logp * 0.1)

    # Excipient contribution
    excipient_contribution = conc_factor * excipient_concentration_pct

    # Raw viscosity at 25°C reference
    raw_viscosity = base_visc_cP + drug_contribution + excipient_contribution

    # Temperature correction
    visc_at_temp = _temperature_correction(raw_viscosity, temperature_C)

    # Clamp to valid range
    visc_at_temp = max(0.5, min(100000.0, visc_at_temp))

    visc_class = _viscosity_class(visc_at_temp)
    syringe_class = _syringeability_class(visc_at_temp)
    inject_rating = _injectability_rating(visc_at_temp)
    storage_effect = _storage_temp_effect(flow_behavior, base_visc_cP)

    notes = (
        f"Viscosity: {visc_at_temp:.2f} cP ({visc_class}). "
        f"Injectability: {inject_rating:.1f}/100. "
        f"Flow behavior: {flow_behavior}."
    )

    return ViscosityResult(
        drug_name=drug_name,
        formulation_type=formulation_type,
        drug_concentration_mg_mL=drug_concentration_mg_mL,
        excipient_name=excipient_name,
        excipient_concentration_pct=excipient_concentration_pct,
        temperature_C=temperature_C,
        viscosity_cP=visc_at_temp,
        viscosity_class=visc_class,
        syringeability_class=syringe_class,
        flow_behavior=flow_behavior,
        injectability_rating=inject_rating,
        storage_temp_effect=storage_effect,
        notes=notes,
    )


def screen_excipients_viscosity(
    drug_name: str,
    drug_concentration_mg_mL: float,
    temperature_C: float = 25.0,
) -> list[ViscosityResult]:
    """Screen all excipients at 1% concentration, sorted by viscosity ascending.

    Parameters
    ----------
    drug_name : str
    drug_concentration_mg_mL : float
    temperature_C : float

    Returns
    -------
    List of ViscosityResult sorted by viscosity_cP ascending (thinnest first)
    """
    results = []
    for exc_name in EXCIPIENTS:
        result = predict_formulation_viscosity(
            drug_name=drug_name,
            formulation_type="aqueous_solution",
            drug_concentration_mg_mL=drug_concentration_mg_mL,
            excipient_name=exc_name,
            excipient_concentration_pct=1.0,
            temperature_C=temperature_C,
            drug_logp=0.0,
        )
        results.append(result)

    results.sort(key=lambda r: r.viscosity_cP)
    return results
