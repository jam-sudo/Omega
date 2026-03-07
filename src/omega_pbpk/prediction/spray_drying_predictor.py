"""Phase 351 — Spray Drying Process Parameter Predictor.

Predicts spray drying process parameters for amorphous solid dispersion (ASD) formulation
using Gordon-Taylor glass transition equation and empirical correlations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Polymer glass transition temperatures (Celsius)
_POLYMER_TG: dict[str, float] = {
    "HPMC-AS": 120.0,
    "PVP-VA": 106.0,
    "HPMC": 155.0,
    "Eudragit": 130.0,
}

_VALID_SOLVENTS = {"acetone", "ethanol", "DCM", "water"}
_VALID_POLYMERS = set(_POLYMER_TG.keys())

# Gordon-Taylor constant (typical value)
_K_GT = 0.23


@dataclass(frozen=True)
class SprayDryingPredictionResult:
    """Result of spray drying process parameter prediction."""

    drug_name: str
    drug_load_pct: float
    polymer_type: str
    tg_asd_C: float
    outlet_temp_C: float
    stability_margin_C: float
    residual_solvent_ppm: float
    particle_d50_um: float
    process_yield_pct: float
    crystallization_risk: str
    ice_formation_risk: bool
    formulation_recommendation: str
    notes: str


def _gordon_taylor_tg(
    tg_drug_C: float,
    tg_polymer_C: float,
    w1: float,
) -> float:
    """Compute glass transition of ASD using Gordon-Taylor equation.

    Args:
        tg_drug_C: Glass transition of drug (Celsius).
        tg_polymer_C: Glass transition of polymer (Celsius).
        w1: Weight fraction of drug (0 to 1).

    Returns:
        Tg of ASD in Celsius.
    """
    w2 = 1.0 - w1
    numerator = w1 * tg_drug_C + _K_GT * w2 * tg_polymer_C
    denominator = w1 + _K_GT * w2
    return numerator / denominator


def _crystallization_risk(stability_margin_C: float) -> str:
    """Classify crystallization risk from stability margin."""
    if stability_margin_C > 20.0:
        return "low"
    if stability_margin_C >= 10.0:
        return "moderate"
    return "high"


def _formulation_recommendation(
    crystallization_risk: str,
    drug_load_pct: float,
    tg_asd_C: float,
    outlet_temp_C: float,
) -> str:
    """Generate formulation recommendation based on predictions."""
    if crystallization_risk == "high":
        return (
            f"High crystallization risk (margin {tg_asd_C:.1f} - {outlet_temp_C:.1f} = "
            f"{tg_asd_C - outlet_temp_C:.1f} C). "
            "Reduce drug load or increase polymer ratio. Consider lower inlet temperature."
        )
    if crystallization_risk == "moderate":
        return (
            f"Moderate crystallization risk. Drug load {drug_load_pct:.1f}% is acceptable "
            "but monitor for physical stability. Store below 25 C / 60% RH."
        )
    return (
        f"ASD with {drug_load_pct:.1f}% drug load appears stable. "
        "Stability margin exceeds 20 C threshold. Standard storage conditions apply."
    )


def _build_notes(
    residual_solvent_ppm: float,
    particle_d50_um: float,
    process_yield_pct: float,
    solvent: str,
) -> str:
    """Build notes string summarising process parameters."""
    parts = [
        f"Solvent: {solvent}",
        f"Residual solvent: {residual_solvent_ppm:.0f} ppm",
        f"Particle d50: {particle_d50_um:.2f} um",
        f"Process yield: {process_yield_pct:.1f}%",
    ]
    if residual_solvent_ppm > 3000:
        parts.append(
            "WARNING: residual solvent exceeds typical ICH Q3C limit — increase inlet temp."
        )
    return "; ".join(parts)


def predict_spray_drying(
    drug_name: str,
    drug_load_pct: float,
    polymer_type: str,
    tg_drug_C: float,
    inlet_temp_C: float,
    feed_conc_mg_mL: float,
    solvent: str,
    drug_logP: float,
) -> SprayDryingPredictionResult:
    """Predict spray drying process parameters for ASD formulation.

    Args:
        drug_name: Name of the drug compound.
        drug_load_pct: Drug loading percentage (0 < drug_load_pct <= 80).
        polymer_type: Polymer carrier ("HPMC-AS", "PVP-VA", "HPMC", "Eudragit").
        tg_drug_C: Glass transition temperature of the drug (Celsius).
        inlet_temp_C: Spray dryer inlet temperature (Celsius, must be > 0).
        feed_conc_mg_mL: Drug concentration in feed solution (mg/mL, must be > 0).
        solvent: Spray drying solvent ("acetone", "ethanol", "DCM", "water").
        drug_logP: Partition coefficient (logP) of the drug.

    Returns:
        SprayDryingPredictionResult with all predicted parameters.

    Raises:
        ValueError: On invalid input parameters.
    """
    if not (0 < drug_load_pct <= 80):
        raise ValueError(f"drug_load_pct must be in (0, 80]; got {drug_load_pct}")
    if polymer_type not in _VALID_POLYMERS:
        raise ValueError(
            f"polymer_type must be one of {sorted(_VALID_POLYMERS)}; got {polymer_type!r}"
        )
    if inlet_temp_C <= 0:
        raise ValueError(f"inlet_temp_C must be > 0; got {inlet_temp_C}")
    if feed_conc_mg_mL <= 0:
        raise ValueError(f"feed_conc_mg_mL must be > 0; got {feed_conc_mg_mL}")
    if solvent not in _VALID_SOLVENTS:
        raise ValueError(f"solvent must be one of {sorted(_VALID_SOLVENTS)}; got {solvent!r}")

    # Gordon-Taylor glass transition of ASD
    w1 = drug_load_pct / 100.0
    tg_polymer = _POLYMER_TG[polymer_type]
    tg_asd = _gordon_taylor_tg(tg_drug_C, tg_polymer, w1)

    # Outlet temperature (empirical)
    outlet_temp = inlet_temp_C * 0.6 - 10.0

    # Residual solvent (ppm)
    residual_solvent = max(100.0, 5000.0 * math.exp(-0.02 * (inlet_temp_C - 60.0)))

    # Particle size d50 (um)
    particle_d50 = 2.0 + 10.0 * feed_conc_mg_mL / 100.0

    # Process yield (%)
    process_yield = min(90.0, 40.0 + inlet_temp_C * 0.4)

    # Stability margin
    margin = tg_asd - outlet_temp

    # Risk classification
    cryst_risk = _crystallization_risk(margin)

    # Ice formation (practically never for spray drying, but flagged if outlet < 0)
    ice_risk = outlet_temp < 0.0

    recommendation = _formulation_recommendation(cryst_risk, drug_load_pct, tg_asd, outlet_temp)
    notes = _build_notes(residual_solvent, particle_d50, process_yield, solvent)

    return SprayDryingPredictionResult(
        drug_name=drug_name,
        drug_load_pct=drug_load_pct,
        polymer_type=polymer_type,
        tg_asd_C=round(tg_asd, 4),
        outlet_temp_C=round(outlet_temp, 4),
        stability_margin_C=round(margin, 4),
        residual_solvent_ppm=round(residual_solvent, 4),
        particle_d50_um=round(particle_d50, 4),
        process_yield_pct=round(process_yield, 4),
        crystallization_risk=cryst_risk,
        ice_formation_risk=ice_risk,
        formulation_recommendation=recommendation,
        notes=notes,
    )


def optimize_drug_load(
    drug_name: str,
    polymer_type: str,
    tg_drug_C: float,
    inlet_temp_C: float,
    feed_conc: float,
    solvent: str,
    logP: float,
    drug_loads: list[float],
) -> list[SprayDryingPredictionResult]:
    """Evaluate a set of drug loads and return results sorted by stability margin (descending).

    Args:
        drug_name: Name of the drug compound.
        polymer_type: Polymer carrier type.
        tg_drug_C: Glass transition temperature of the drug (Celsius).
        inlet_temp_C: Spray dryer inlet temperature (Celsius).
        feed_conc: Drug concentration in feed solution (mg/mL).
        solvent: Spray drying solvent.
        logP: Partition coefficient of the drug.
        drug_loads: List of drug load percentages to evaluate.

    Returns:
        List of SprayDryingPredictionResult sorted by stability_margin_C descending.
    """
    results: list[SprayDryingPredictionResult] = []
    for load in drug_loads:
        result = predict_spray_drying(
            drug_name=drug_name,
            drug_load_pct=load,
            polymer_type=polymer_type,
            tg_drug_C=tg_drug_C,
            inlet_temp_C=inlet_temp_C,
            feed_conc_mg_mL=feed_conc,
            solvent=solvent,
            drug_logP=logP,
        )
        results.append(result)
    results.sort(key=lambda r: r.stability_margin_C, reverse=True)
    return results
