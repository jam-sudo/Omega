"""Phase 263 -- Salt Bridge Stability Predictor.

Predict stability of salt bridges and ionic interactions in protein-drug
complexes.  Pure Python, no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fabs

# ---------------------------------------------------------------------------
# Salt bridge type database
# ---------------------------------------------------------------------------

_SALT_BRIDGE_DB: dict[str, dict] = {
    "Lys_Asp": {
        "strength_kcal_mol": 3.5,
        "pH_sensitivity": 0.8,
        "ionic_strength_sensitivity": 0.9,
    },
    "Arg_Glu": {
        "strength_kcal_mol": 4.0,
        "pH_sensitivity": 0.7,
        "ionic_strength_sensitivity": 0.85,
    },
    "Lys_Glu": {
        "strength_kcal_mol": 3.0,
        "pH_sensitivity": 0.85,
        "ionic_strength_sensitivity": 0.9,
    },
    "Arg_Asp": {
        "strength_kcal_mol": 3.8,
        "pH_sensitivity": 0.75,
        "ionic_strength_sensitivity": 0.88,
    },
    "His_Asp": {
        "strength_kcal_mol": 2.5,
        "pH_sensitivity": 1.0,
        "ionic_strength_sensitivity": 0.7,
    },
}

VALID_SALT_BRIDGE_TYPES = list(_SALT_BRIDGE_DB.keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SaltBridgeStabilityResult:
    """Result of salt bridge stability prediction."""

    drug_name: str
    salt_bridge_type: str
    physiological_ph: float
    ionic_strength_mM: float
    interaction_strength_kcal_mol: float
    stability_score: float
    probability_intact: float
    half_life_in_plasma_h: float
    disruption_risk: str
    optimal_ph_for_stability: float
    notes: str


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------


def _compute_effective_strength(
    base_strength: float,
    pH_sensitivity: float,
    ionic_strength_sensitivity: float,
    ph: float,
    ionic_strength_mM: float,
) -> float:
    """Compute effective interaction strength at given conditions."""
    ph_factor = 1.0 - pH_sensitivity * fabs(ph - 7.4) / 7.4
    is_factor = 1.0 - ionic_strength_sensitivity * ionic_strength_mM / 500.0
    return base_strength * ph_factor * is_factor


def _disruption_risk(probability_intact: float) -> str:
    if probability_intact > 0.7:
        return "low"
    if probability_intact >= 0.4:
        return "moderate"
    return "high"


def _build_notes(
    salt_bridge_type: str,
    stability_score: float,
    disruption_risk: str,
    ionic_strength_mM: float,
    ph: float,
) -> str:
    parts = []
    if disruption_risk == "high":
        parts.append(f"{salt_bridge_type} salt bridge has high disruption risk at pH {ph:.1f}.")
    elif disruption_risk == "moderate":
        parts.append(f"{salt_bridge_type} salt bridge shows moderate stability at pH {ph:.1f}.")
    else:
        parts.append(f"{salt_bridge_type} salt bridge is stable at physiological conditions.")
    if ionic_strength_mM > 300:
        parts.append("Elevated ionic strength reduces electrostatic interactions.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_salt_bridge_stability(
    drug_name: str,
    salt_bridge_type: str,
    physiological_ph: float = 7.4,
    ionic_strength_mM: float = 150.0,
) -> SaltBridgeStabilityResult:
    """Predict stability of a specific salt bridge type.

    Parameters
    ----------
    drug_name:
        Name of the drug molecule.
    salt_bridge_type:
        One of the supported types (e.g. "Lys_Asp", "Arg_Glu", ...).
    physiological_ph:
        pH of the environment (must be in [4, 10]).
    ionic_strength_mM:
        Ionic strength in mM (must be >= 0).

    Returns
    -------
    SaltBridgeStabilityResult
    """
    if salt_bridge_type not in _SALT_BRIDGE_DB:
        raise ValueError(
            f"Unknown salt_bridge_type '{salt_bridge_type}'. Valid types: {VALID_SALT_BRIDGE_TYPES}"
        )
    if not (4.0 <= physiological_ph <= 10.0):
        raise ValueError("physiological_ph must be in [4, 10]")
    if ionic_strength_mM < 0:
        raise ValueError("ionic_strength_mM must be >= 0")

    entry = _SALT_BRIDGE_DB[salt_bridge_type]
    base_strength = entry["strength_kcal_mol"]
    pH_sensitivity = entry["pH_sensitivity"]
    is_sensitivity = entry["ionic_strength_sensitivity"]

    effective_strength = _compute_effective_strength(
        base_strength=base_strength,
        pH_sensitivity=pH_sensitivity,
        ionic_strength_sensitivity=is_sensitivity,
        ph=physiological_ph,
        ionic_strength_mM=ionic_strength_mM,
    )

    # Clamp effective_strength at 0 (conditions can theoretically produce negative)
    effective_strength = max(0.0, effective_strength)

    stability_score = min(100.0, max(0.0, effective_strength / 5.0 * 100.0))
    probability_intact = min(1.0, effective_strength / 4.0)
    half_life_in_plasma_h = probability_intact * 24.0
    risk = _disruption_risk(probability_intact)
    optimal_ph = 7.4

    notes = _build_notes(
        salt_bridge_type=salt_bridge_type,
        stability_score=stability_score,
        disruption_risk=risk,
        ionic_strength_mM=ionic_strength_mM,
        ph=physiological_ph,
    )

    return SaltBridgeStabilityResult(
        drug_name=drug_name,
        salt_bridge_type=salt_bridge_type,
        physiological_ph=physiological_ph,
        ionic_strength_mM=ionic_strength_mM,
        interaction_strength_kcal_mol=effective_strength,
        stability_score=stability_score,
        probability_intact=probability_intact,
        half_life_in_plasma_h=half_life_in_plasma_h,
        disruption_risk=risk,
        optimal_ph_for_stability=optimal_ph,
        notes=notes,
    )


def screen_salt_bridges(
    drug_name: str,
    physiological_ph: float = 7.4,
    ionic_strength_mM: float = 150.0,
) -> list:
    """Screen all supported salt bridge types and sort by stability_score descending.

    Parameters
    ----------
    drug_name:
        Name of the drug molecule.
    physiological_ph:
        pH of the environment.
    ionic_strength_mM:
        Ionic strength in mM.

    Returns
    -------
    list[SaltBridgeStabilityResult] sorted by stability_score descending.
    """
    results = [
        predict_salt_bridge_stability(
            drug_name=drug_name,
            salt_bridge_type=sb_type,
            physiological_ph=physiological_ph,
            ionic_strength_mM=ionic_strength_mM,
        )
        for sb_type in _SALT_BRIDGE_DB
    ]
    results.sort(key=lambda r: r.stability_score, reverse=True)
    return results
