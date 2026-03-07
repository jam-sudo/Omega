"""Amorphous Solid Dispersion (ASD) Physical Stability Predictor — Phase 814.

Provides:
- AmorphousStabilityResult dataclass
- predict_amorphous_stability() using Gordon-Taylor Tg blending and
  mobility-based crystallization kinetics
- optimize_drug_loading() for screening drug loading percentages
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Polymer library: name → (Tg_celsius, miscibility_logP_range, description)
# ---------------------------------------------------------------------------
_POLYMER_DB: dict[str, dict] = {
    "HPMC-AS": {
        "tg_celsius": 120.0,
        "miscible_logp_min": 0.0,
        "miscible_logp_max": 4.0,
        "description": "Hydroxypropyl methylcellulose acetate succinate",
    },
    "PVP": {
        "tg_celsius": 175.0,
        "miscible_logp_min": -2.0,
        "miscible_logp_max": 2.5,
        "description": "Polyvinylpyrrolidone K30",
    },
    "PVP-VA": {
        "tg_celsius": 110.0,
        "miscible_logp_min": -1.0,
        "miscible_logp_max": 3.0,
        "description": "Polyvinylpyrrolidone-vinyl acetate",
    },
    "HPMC": {
        "tg_celsius": 170.0,
        "miscible_logp_min": -1.0,
        "miscible_logp_max": 2.0,
        "description": "Hydroxypropyl methylcellulose E5",
    },
    "Soluplus": {
        "tg_celsius": 70.0,
        "miscible_logp_min": 0.5,
        "miscible_logp_max": 5.0,
        "description": "Polyvinyl caprolactam-polyvinyl acetate-polyethylene glycol graft copolymer",
    },
    "EUDRAGIT_L100": {
        "tg_celsius": 130.0,
        "miscible_logp_min": 1.0,
        "miscible_logp_max": 4.5,
        "description": "Methacrylic acid copolymer type L",
    },
}


def _gordon_taylor_tg(
    tg1_k: float,
    tg2_k: float,
    w1: float,
    k_gt: float,
) -> float:
    """Gordon-Taylor equation for blend Tg (all in Kelvin).

    Tg_blend = (w1 * Tg1 + K * w2 * Tg2) / (w1 + K * w2)
    """
    w2 = 1.0 - w1
    numerator = w1 * tg1_k + k_gt * w2 * tg2_k
    denominator = w1 + k_gt * w2
    if denominator == 0.0:
        return tg1_k
    return numerator / denominator


def _sigmoid(x: float) -> float:
    """Logistic sigmoid clamped to [0, 1]."""
    try:
        val = 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        val = 0.0 if x < 0 else 1.0
    return max(0.0, min(1.0, val))


def _miscibility(logp: float, polymer_name: str) -> str:
    """Determine drug-polymer miscibility based on logP and polymer type."""
    polymer = _POLYMER_DB.get(polymer_name)
    if polymer is None:
        # Unknown polymer — uncertain
        return "uncertain"
    lo = polymer["miscible_logp_min"]
    hi = polymer["miscible_logp_max"]
    margin = (hi - lo) * 0.15  # 15% margin for uncertain zone
    if lo <= logp <= hi:
        return "miscible"
    elif logp < lo - margin or logp > hi + margin:
        return "immiscible"
    else:
        return "uncertain"


def _recommend_loading(logp: float, t_above_tg: float) -> float:
    """Recommend optimal drug loading percentage.

    Rules:
    - If t_above_tg > 20: low loading (10-20%)
    - If 0 < t_above_tg <= 20: moderate loading (20-30%)
    - If t_above_tg <= 0: higher loading (30-40%)
    - LogP adjustment: high logP allows slightly higher loading
    """
    if t_above_tg > 20.0:
        base = 15.0
    elif t_above_tg > 0.0:
        base = 25.0
    else:
        base = 35.0

    # logP bonus: +2% per logP unit above 2, capped at +10%
    logp_bonus = max(0.0, min(10.0, (logp - 2.0) * 2.0))
    loading = base + logp_bonus
    # Clamp to [10, 50]
    return max(10.0, min(50.0, loading))


def _excipient_recommendation(
    polymer_name: str,
    logp: float,
    tg_blend_celsius: float,
    t_above_tg: float,
) -> str:
    """Generate excipient recommendation string."""
    recs = []
    if t_above_tg > 10.0:
        recs.append("Add surfactant (e.g., SDS 1-5%) to improve miscibility and inhibit nucleation")
    if logp > 4.0:
        recs.append("Consider Soluplus or EUDRAGIT_L100 for high-logP drugs")
    if tg_blend_celsius < 50.0:
        recs.append("Add silica (Aerosil 200, 0.5-1%) to prevent caking")
    if not recs:
        recs.append(f"{polymer_name} alone should provide adequate stability")
    return "; ".join(recs)


@dataclass(frozen=True)
class AmorphousStabilityResult:
    compound_name: str
    polymer_name: str
    drug_loading_pct: float
    tg_blend_celsius: float
    t_above_tg_celsius: float
    crystallization_tendency: str
    molecular_mobility_factor: float
    predicted_shelf_life_months: float
    amorphous_fraction_2yr: float
    miscibility: str
    recommended_drug_loading_pct: float
    excipient_recommendation: str
    notes: str


def predict_amorphous_stability(
    compound_name: str,
    logp: float,
    mw_da: float,
    tg_drug_celsius: float = 100.0,
    polymer_name: str = "HPMC-AS",
    tg_polymer_celsius: float = 120.0,
    drug_loading_pct: float = 30.0,
    storage_temp_celsius: float = 25.0,
    water_activity: float = 0.3,
) -> AmorphousStabilityResult:
    """Predict amorphous solid dispersion physical stability.

    Parameters
    ----------
    compound_name : str
    logp : float
        Octanol-water partition coefficient.
    mw_da : float
        Molecular weight in Daltons.
    tg_drug_celsius : float
        Glass transition temperature of pure amorphous drug (°C).
    polymer_name : str
        Polymer name (e.g., "HPMC-AS", "PVP", "PVP-VA").
    tg_polymer_celsius : float
        Glass transition temperature of polymer (°C). Overridden by DB if
        polymer is in the database.
    drug_loading_pct : float
        Drug loading percentage (must be in (0, 100)).
    storage_temp_celsius : float
        Storage temperature (°C).
    water_activity : float
        Relative humidity / 100 at storage conditions (0-1).

    Returns
    -------
    AmorphousStabilityResult
    """
    if not (0.0 < drug_loading_pct < 100.0):
        raise ValueError(f"drug_loading_pct must be in (0, 100), got {drug_loading_pct}.")

    # Use polymer DB for Tg if available
    polymer_info = _POLYMER_DB.get(polymer_name)
    if polymer_info is not None:
        tg_polymer_celsius_eff = polymer_info["tg_celsius"]
    else:
        tg_polymer_celsius_eff = tg_polymer_celsius

    # Convert to Kelvin for Gordon-Taylor
    tg_drug_k = tg_drug_celsius + 273.15
    tg_poly_k = tg_polymer_celsius_eff + 273.15

    w1 = drug_loading_pct / 100.0  # drug weight fraction

    # Gordon-Taylor constant K = Tg_drug / Tg_polymer (simplified)
    k_gt = tg_drug_k / tg_poly_k

    tg_blend_k = _gordon_taylor_tg(tg_drug_k, tg_poly_k, w1, k_gt)
    tg_blend_celsius = tg_blend_k - 273.15

    # Water plasticization: Tg drops by 40 * water_activity
    water_activity_clamped = max(0.0, min(1.0, water_activity))
    tg_blend_eff_celsius = tg_blend_celsius - 40.0 * water_activity_clamped

    t_above_tg = storage_temp_celsius - tg_blend_eff_celsius

    # Crystallization tendency
    if t_above_tg > 20.0:
        crystallization_tendency = "high"
    elif t_above_tg > 0.0:
        crystallization_tendency = "moderate"
    else:
        crystallization_tendency = "low"

    # Molecular mobility factor (logistic)
    molecular_mobility_factor = _sigmoid(0.1 * (t_above_tg - 10.0))

    # Shelf life in months — base by tendency, modified by logP
    # Higher logP = more hydrophobic = slower water absorption = more stable
    if crystallization_tendency == "low":
        base_shelf_life = 36.0
    elif crystallization_tendency == "moderate":
        base_shelf_life = 18.0
    else:
        base_shelf_life = 6.0

    # LogP modifier: 0.5 per unit above 2, capped at +12 months
    logp_bonus = max(0.0, min(12.0, (logp - 2.0) * 0.5))
    predicted_shelf_life_months = base_shelf_life + logp_bonus

    # Amorphous fraction after 2 years (first-order decay)
    # A(t) = exp(-ln(2)/t_half * t) where t_half = predicted_shelf_life_months
    t_2yr_months = 24.0
    if predicted_shelf_life_months > 0.0:
        k_cryst = math.log(2.0) / predicted_shelf_life_months
        amorphous_fraction_2yr = math.exp(-k_cryst * t_2yr_months)
    else:
        amorphous_fraction_2yr = 0.0
    amorphous_fraction_2yr = max(0.0, min(1.0, amorphous_fraction_2yr))

    # Miscibility
    miscibility = _miscibility(logp, polymer_name)

    # Recommended drug loading
    recommended_loading = _recommend_loading(logp, t_above_tg)

    # Excipient recommendation
    excipient_rec = _excipient_recommendation(polymer_name, logp, tg_blend_celsius, t_above_tg)

    # Notes
    notes_parts = []
    if tg_blend_eff_celsius < storage_temp_celsius and crystallization_tendency == "high":
        notes_parts.append("Tg is well below storage temperature; crystallization risk is high.")
    if water_activity_clamped > 0.5:
        notes_parts.append("High water activity significantly lowers effective Tg.")
    if miscibility == "immiscible":
        notes_parts.append(
            f"Drug-polymer ({polymer_name}) miscibility is poor; consider alternative polymer."
        )
    notes = "; ".join(notes_parts) if notes_parts else "OK"

    return AmorphousStabilityResult(
        compound_name=compound_name,
        polymer_name=polymer_name,
        drug_loading_pct=drug_loading_pct,
        tg_blend_celsius=tg_blend_celsius,
        t_above_tg_celsius=t_above_tg,
        crystallization_tendency=crystallization_tendency,
        molecular_mobility_factor=molecular_mobility_factor,
        predicted_shelf_life_months=predicted_shelf_life_months,
        amorphous_fraction_2yr=amorphous_fraction_2yr,
        miscibility=miscibility,
        recommended_drug_loading_pct=recommended_loading,
        excipient_recommendation=excipient_rec,
        notes=notes,
    )


def optimize_drug_loading(
    compound_name: str,
    logp: float,
    mw_da: float,
    tg_drug_celsius: float = 100.0,
    polymer_name: str = "HPMC-AS",
    loadings: list[float] | None = None,
) -> list[AmorphousStabilityResult]:
    """Screen multiple drug loadings and return results sorted by shelf life.

    Parameters
    ----------
    compound_name : str
    logp : float
    mw_da : float
    tg_drug_celsius : float
    polymer_name : str
    loadings : list[float] or None
        Drug loading percentages to test. Defaults to [10, 20, 30, 40, 50].

    Returns
    -------
    list[AmorphousStabilityResult]
        Sorted descending by predicted_shelf_life_months.
    """
    if loadings is None:
        loadings = [10.0, 20.0, 30.0, 40.0, 50.0]

    results = []
    for loading in loadings:
        result = predict_amorphous_stability(
            compound_name=compound_name,
            logp=logp,
            mw_da=mw_da,
            tg_drug_celsius=tg_drug_celsius,
            polymer_name=polymer_name,
            drug_loading_pct=loading,
        )
        results.append(result)

    # Sort by predicted_shelf_life_months descending
    results.sort(key=lambda r: r.predicted_shelf_life_months, reverse=True)
    return results
