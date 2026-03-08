"""Phase 426 — Amorphous Solid Dispersion (ASD) physical stability prediction.

Predicts recrystallization onset and shelf life for amorphous solid dispersions
using the Gordon-Taylor equation and WLF-inspired mobility analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Polymer database
POLYMERS: dict = {
    "HPMC-AS": {"tg_C": 120, "solubility_param": 22.0, "hygroscopicity": "low"},
    "PVP-VA": {"tg_C": 105, "solubility_param": 22.5, "hygroscopicity": "moderate"},
    "HPMC": {"tg_C": 150, "solubility_param": 21.5, "hygroscopicity": "low"},
    "Eudragit_L": {"tg_C": 130, "solubility_param": 20.0, "hygroscopicity": "low"},
    "PVP": {"tg_C": 168, "solubility_param": 23.0, "hygroscopicity": "high"},
}


@dataclass(frozen=True)
class AmorphousStabilityResult:
    drug_name: str
    polymer_name: str
    drug_tg_C: float
    polymer_tg_C: float
    composite_tg_C: float
    drug_loading_pct: float
    storage_temp_C: float
    t_onset_recrystallization_months: float
    shelf_life_months: float
    stability_class: str
    mobility_parameter: float
    recommended_storage: str
    notes: str


def _gordon_taylor(drug_tg_C: float, polymer_tg_C: float, drug_loading_pct: float) -> float:
    """Compute composite Tg using Gordon-Taylor equation (K=0.5)."""
    K = 0.5
    w_drug = drug_loading_pct / 100.0
    w_polymer = 1.0 - w_drug
    composite_tg_C = (w_drug * drug_tg_C + K * w_polymer * polymer_tg_C) / (w_drug + K * w_polymer)
    return composite_tg_C


def _compute_mobility(composite_tg_C: float, storage_temp_C: float) -> float:
    """WLF-inspired molecular mobility parameter."""
    Tg_K = composite_tg_C + 273.15
    T_K = storage_temp_C + 273.15
    delta_T = T_K - Tg_K
    if delta_T <= 0:
        # Glassy state — very stable
        return 1e-6
    wlf_exponent = 17.44 * delta_T / (51.6 + delta_T)
    mobility = math.exp(wlf_exponent) * 0.001
    # Clamp
    mobility = max(1e-8, min(1.0, mobility))
    return mobility


def _stability_class(shelf_life_months: float) -> str:
    if shelf_life_months > 24.0:
        return "excellent"
    elif shelf_life_months >= 12.0:
        return "good"
    elif shelf_life_months >= 6.0:
        return "moderate"
    else:
        return "poor"


def _recommended_storage(composite_tg_C: float, storage_temp_C: float) -> str:
    delta = composite_tg_C - storage_temp_C
    if delta > 50.0:
        return "room_temperature"
    elif delta > 25.0:
        return "refrigerator"
    else:
        return "freezer"


def predict_amorphous_stability(
    drug_name: str,
    polymer_name: str,
    drug_tg_C: float,
    drug_loading_pct: float,
    storage_temp_C: float = 25.0,
) -> AmorphousStabilityResult:
    """Predict ASD physical stability.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    polymer_name:
        Matrix polymer — must be one of POLYMERS.
    drug_tg_C:
        Drug glass transition temperature (degrees C).
    drug_loading_pct:
        Drug loading in the ASD (%).  Must be in (0, 100).
    storage_temp_C:
        Storage temperature (degrees C).  Must be in [-30, 60].

    Returns
    -------
    AmorphousStabilityResult
    """
    if polymer_name not in POLYMERS:
        raise ValueError(
            f"polymer_name must be one of {list(POLYMERS.keys())}, got {polymer_name!r}"
        )
    if not (0 < drug_loading_pct < 100):
        raise ValueError(f"drug_loading_pct must be in (0, 100), got {drug_loading_pct}")
    if not (-30 <= storage_temp_C <= 60):
        raise ValueError(f"storage_temp_C must be in [-30, 60], got {storage_temp_C}")

    polymer_tg_C = float(POLYMERS[polymer_name]["tg_C"])
    composite_tg_C = _gordon_taylor(drug_tg_C, polymer_tg_C, drug_loading_pct)

    mobility = _compute_mobility(composite_tg_C, storage_temp_C)

    if mobility > 0:
        t_onset_raw = 1.0 / mobility
    else:
        t_onset_raw = 1000.0
    t_onset = min(120.0, t_onset_raw)

    shelf_life = t_onset * 0.8

    stab_class = _stability_class(shelf_life)
    rec_storage = _recommended_storage(composite_tg_C, storage_temp_C)

    notes = (
        f"{drug_name} / {polymer_name} ASD: composite Tg = {composite_tg_C:.1f} C "
        f"(drug loading {drug_loading_pct:.1f}%). "
        f"Predicted shelf life: {shelf_life:.1f} months ({stab_class}). "
        f"Recommended storage: {rec_storage}."
    )

    return AmorphousStabilityResult(
        drug_name=drug_name,
        polymer_name=polymer_name,
        drug_tg_C=drug_tg_C,
        polymer_tg_C=polymer_tg_C,
        composite_tg_C=composite_tg_C,
        drug_loading_pct=drug_loading_pct,
        storage_temp_C=storage_temp_C,
        t_onset_recrystallization_months=t_onset,
        shelf_life_months=shelf_life,
        stability_class=stab_class,
        mobility_parameter=mobility,
        recommended_storage=rec_storage,
        notes=notes,
    )


def screen_polymers_stability(
    drug_name: str,
    drug_tg_C: float,
    drug_loading_pct: float,
    storage_temp_C: float = 25.0,
) -> list:
    """Screen all available polymers and rank by shelf life (best first).

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    drug_tg_C:
        Drug glass transition temperature (degrees C).
    drug_loading_pct:
        Drug loading in the ASD (%).
    storage_temp_C:
        Storage temperature (degrees C).

    Returns
    -------
    List of AmorphousStabilityResult sorted by shelf_life_months descending.
    """
    results = []
    for polymer_name in POLYMERS:
        result = predict_amorphous_stability(
            drug_name=drug_name,
            polymer_name=polymer_name,
            drug_tg_C=drug_tg_C,
            drug_loading_pct=drug_loading_pct,
            storage_temp_C=storage_temp_C,
        )
        results.append(result)
    results.sort(key=lambda r: r.shelf_life_months, reverse=True)
    return results
