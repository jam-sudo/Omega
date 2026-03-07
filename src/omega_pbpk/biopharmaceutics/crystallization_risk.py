"""Drug Crystallization Risk Assessment — Phase 253.

Predicts drug crystallization risk in amorphous solid dispersions (ASD)
and after in vivo supersaturation. Crystallization reduces bioavailability
for BCS II/IV drugs.

Key concepts:
- Amorphous drug is supersaturated relative to crystalline form
- Crystallization rate depends on temperature, humidity, drug-polymer interactions
- Fragility index (m) predicts crystallization tendency
- Induction time before nucleation: t_ind = A / log(S)^2

References
----------
- Baird JA et al., J Pharm Sci. 2010;99(9):3787-806
- Murdande SB et al., J Pharm Sci. 2010;99(3):1254-64
- Bhugra C, Pikal MJ. J Pharm Sci. 2008;97(4):1329-49
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrystTendencyResult:
    """Result from crystallization tendency assessment.

    Attributes
    ----------
    logP : Octanol-water partition coefficient.
    mw : Molecular weight (g/mol).
    tm_C : Melting point (degrees Celsius).
    tg_C : Glass transition temperature (degrees Celsius).
    tg_tm_ratio : Tg/Tm in Kelvin — key predictor of crystallization.
    fragility_m : Fragility index (empirical, higher = more fragile glass).
    crystallization_tendency : Risk category ('low', 'moderate', 'high').
    notes : Interpretation notes.
    """

    logP: float
    mw: float
    tm_C: float
    tg_C: float
    tg_tm_ratio: float
    fragility_m: float
    crystallization_tendency: str
    notes: str


@dataclass(frozen=True)
class ASDStabilityResult:
    """Result from ASD stability assessment.

    Attributes
    ----------
    drug_name : Name of the drug.
    polymer_type : Polymer used in ASD ('HPMC', 'PVP', 'PVPVA', 'copovidone').
    humidity_pct : Relative humidity (%).
    temperature_C : Storage temperature (degrees Celsius).
    tg_effective_C : Effective Tg after polymer + humidity corrections (degrees C).
    tg_tm_ratio_effective : Effective Tg/Tm in Kelvin.
    crystallization_risk : Risk category ('low', 'moderate', 'high').
    estimated_shelf_life_months : Estimated shelf life (months).
    recommendation : Storage/formulation recommendation.
    notes : Interpretation notes.
    """

    drug_name: str
    polymer_type: str
    humidity_pct: float
    temperature_C: float
    tg_effective_C: float
    tg_tm_ratio_effective: float
    crystallization_risk: str
    estimated_shelf_life_months: float
    recommendation: str
    notes: str


# ---------------------------------------------------------------------------
# Polymer Tg elevation (degrees C)
# ---------------------------------------------------------------------------

_POLYMER_TG_ELEVATION: dict[str, float] = {
    "HPMC": 15.0,
    "PVP": 10.0,
    "PVPVA": 12.0,
    "copovidone": 12.0,
}

_VALID_POLYMERS = set(_POLYMER_TG_ELEVATION.keys())


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def crystallization_tendency(
    logP: float,
    mw: float,
    tm_C: float,
    tg_C: float,
) -> CrystTendencyResult:
    """Assess intrinsic crystallization tendency of an amorphous drug.

    Parameters
    ----------
    logP : Octanol-water partition coefficient.
    mw : Molecular weight (g/mol). Must be > 0.
    tm_C : Melting point (degrees Celsius). Must be > 0.
    tg_C : Glass transition temperature (degrees Celsius).
           Must be > 0 and < tm_C.

    Returns
    -------
    CrystTendencyResult
        Includes Tg/Tm ratio, fragility index, and risk category.

    Raises
    ------
    ValueError
        If input validation fails.
    """
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if tm_C <= 0:
        raise ValueError(f"tm_C must be > 0, got {tm_C}")
    if tg_C <= 0:
        raise ValueError(f"tg_C must be > 0, got {tg_C}")
    if tg_C >= tm_C:
        raise ValueError(f"tg_C ({tg_C}) must be < tm_C ({tm_C})")

    # Convert to Kelvin
    tm_K = tm_C + 273.15
    tg_K = tg_C + 273.15

    tg_tm_ratio = tg_K / tm_K

    # Fragility index: empirical formula from Baird et al. 2010
    fragility_m = 17.7 + 0.0017 * mw

    # Crystallization tendency category
    if tg_tm_ratio > 0.8:
        tendency = "low"
        tendency_note = (
            f"Tg/Tm={tg_tm_ratio:.3f} > 0.8: low crystallization tendency. "
            "Amorphous form likely stable without polymer."
        )
    elif tg_tm_ratio >= 0.7:
        tendency = "moderate"
        tendency_note = (
            f"Tg/Tm={tg_tm_ratio:.3f} in [0.7, 0.8]: moderate tendency. "
            "Polymer stabilizer recommended."
        )
    else:
        tendency = "high"
        tendency_note = (
            f"Tg/Tm={tg_tm_ratio:.3f} < 0.7: high crystallization tendency. "
            "Strong polymer interaction required; consider alternative formulation."
        )

    notes = (
        f"logP={logP:.2f}, MW={mw:.1f}, Tm={tm_C:.1f}C, Tg={tg_C:.1f}C. "
        f"Fragility m={fragility_m:.1f}. {tendency_note}"
    )

    return CrystTendencyResult(
        logP=logP,
        mw=mw,
        tm_C=tm_C,
        tg_C=tg_C,
        tg_tm_ratio=tg_tm_ratio,
        fragility_m=fragility_m,
        crystallization_tendency=tendency,
        notes=notes,
    )


def induction_time(
    solubility_crystal_mg_mL: float,
    c_supersaturated_mg_mL: float,
    temperature_C: float = 37.0,
) -> float:
    """Estimate nucleation induction time under supersaturation.

    Uses classical nucleation theory approximation:
        t_ind (h) = A / log(S)^2
    where A = 10 h (calibrated empirical constant) and S is the
    supersaturation ratio.

    Parameters
    ----------
    solubility_crystal_mg_mL : Equilibrium crystalline solubility (mg/mL).
    c_supersaturated_mg_mL : Supersaturated drug concentration (mg/mL).
    temperature_C : Temperature (degrees Celsius). Not currently used in
                    the simplified model but included for API consistency.

    Returns
    -------
    float
        Estimated induction time in hours until nucleation begins.
        Returns a very large value (1e6 h) if S <= 1 (no supersaturation).

    Raises
    ------
    ValueError
        If solubility or concentration are non-positive.
    """
    if solubility_crystal_mg_mL <= 0:
        raise ValueError(f"solubility_crystal_mg_mL must be > 0, got {solubility_crystal_mg_mL}")
    if c_supersaturated_mg_mL <= 0:
        raise ValueError(f"c_supersaturated_mg_mL must be > 0, got {c_supersaturated_mg_mL}")

    S = c_supersaturated_mg_mL / solubility_crystal_mg_mL

    if S <= 1.0:
        # No supersaturation — effectively infinite induction time
        return 1e6

    A = 10.0  # hours (empirical calibration constant)
    log_S = math.log(S)
    t_ind = A / (log_S**2)
    return t_ind


def assess_asd_stability(
    drug_name: str,
    logP: float,
    mw: float,
    tm_C: float,
    tg_C: float,
    polymer_type: str = "HPMC",
    humidity_pct: float = 60.0,
    temperature_C: float = 25.0,
) -> ASDStabilityResult:
    """Assess stability of an amorphous solid dispersion (ASD).

    Accounts for polymer-induced Tg elevation and humidity plasticization
    to estimate effective Tg and crystallization risk under storage conditions.

    Parameters
    ----------
    drug_name : Name of the drug.
    logP : Octanol-water partition coefficient.
    mw : Molecular weight (g/mol). Must be > 0.
    tm_C : Melting point (degrees Celsius). Must be > 0.
    tg_C : Drug glass transition temperature (degrees Celsius).
           Must be > 0 and < tm_C.
    polymer_type : Polymer in ASD. One of 'HPMC', 'PVP', 'PVPVA', 'copovidone'.
    humidity_pct : Relative humidity (%). Must be in [0, 100].
    temperature_C : Storage temperature (degrees Celsius).

    Returns
    -------
    ASDStabilityResult
        Effective Tg, risk category, shelf-life estimate, and recommendations.

    Raises
    ------
    ValueError
        If any input validation fails.
    """
    # Validate shared params
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if tm_C <= 0:
        raise ValueError(f"tm_C must be > 0, got {tm_C}")
    if tg_C <= 0:
        raise ValueError(f"tg_C must be > 0, got {tg_C}")
    if tg_C >= tm_C:
        raise ValueError(f"tg_C ({tg_C}) must be < tm_C ({tm_C})")
    if not (0.0 <= humidity_pct <= 100.0):
        raise ValueError(f"humidity_pct must be in [0, 100], got {humidity_pct}")
    if polymer_type not in _VALID_POLYMERS:
        raise ValueError(
            f"polymer_type must be one of {sorted(_VALID_POLYMERS)}, got {polymer_type!r}"
        )

    # Polymer Tg elevation
    polymer_tg_boost = _POLYMER_TG_ELEVATION[polymer_type]

    # Humidity plasticization: -2°C per 10% RH above 40%
    excess_rh = max(0.0, humidity_pct - 40.0)
    humidity_tg_penalty = (excess_rh / 10.0) * 2.0

    # Effective Tg
    tg_effective_C = tg_C + polymer_tg_boost - humidity_tg_penalty

    # Convert to Kelvin for ratio
    tm_K = tm_C + 273.15
    tg_eff_K = tg_effective_C + 273.15
    tg_tm_ratio_eff = tg_eff_K / tm_K

    # Risk category based on effective Tg/Tm
    if tg_tm_ratio_eff > 0.8:
        risk = "low"
    elif tg_tm_ratio_eff >= 0.7:
        risk = "moderate"
    else:
        risk = "high"

    # Estimated shelf life (months) based on risk and Tg margin above storage temp
    tg_margin = tg_effective_C - temperature_C  # how far above storage temp
    if tg_margin >= 50:
        shelf_life_months = 36.0
    elif tg_margin >= 30:
        shelf_life_months = 24.0
    elif tg_margin >= 15:
        shelf_life_months = 18.0
    elif tg_margin >= 5:
        shelf_life_months = 12.0
    elif tg_margin >= 0:
        shelf_life_months = 6.0
    else:
        # Below Tg — rapid crystallization expected
        shelf_life_months = max(1.0, 3.0 + tg_margin * 0.1)

    # Recommendation
    if risk == "low":
        recommendation = (
            f"ASD with {polymer_type} expected to be physically stable. "
            "Standard packaging (moisture barrier) sufficient."
        )
    elif risk == "moderate":
        recommendation = (
            f"ASD with {polymer_type} has moderate stability. "
            "Use desiccant in packaging; monitor at ICH conditions."
        )
    else:
        recommendation = (
            f"ASD with {polymer_type} has high crystallization risk. "
            "Increase polymer loading, use a more hygroscopic-resistant polymer, "
            "or consider alternative formulation strategy."
        )

    notes = (
        f"{drug_name}: drug Tg={tg_C:.1f}C, polymer boost={polymer_tg_boost:.1f}C, "
        f"humidity penalty={humidity_tg_penalty:.1f}C, "
        f"effective Tg={tg_effective_C:.1f}C, Tg/Tm={tg_tm_ratio_eff:.3f}. "
        f"Storage: {temperature_C:.1f}C/{humidity_pct:.0f}%RH. "
        f"Tg margin above storage={tg_margin:.1f}C."
    )

    return ASDStabilityResult(
        drug_name=drug_name,
        polymer_type=polymer_type,
        humidity_pct=humidity_pct,
        temperature_C=temperature_C,
        tg_effective_C=tg_effective_C,
        tg_tm_ratio_effective=tg_tm_ratio_eff,
        crystallization_risk=risk,
        estimated_shelf_life_months=shelf_life_months,
        recommendation=recommendation,
        notes=notes,
    )
