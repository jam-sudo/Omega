"""Phase 259: Drug Crystal Growth Predictor.

Predict crystal growth kinetics and impact on dissolution using
Ostwald ripening model with temperature and humidity effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Crystal habit lookup table
# ---------------------------------------------------------------------------

CRYSTAL_HABITS: dict[str, dict[str, float]] = {
    "needle": {
        "aspect_ratio": 10.0,
        "growth_rate_modifier": 1.5,
        "dissolution_rate_modifier": 0.7,
    },
    "plate": {
        "aspect_ratio": 5.0,
        "growth_rate_modifier": 1.2,
        "dissolution_rate_modifier": 0.9,
    },
    "cubic": {
        "aspect_ratio": 1.0,
        "growth_rate_modifier": 1.0,
        "dissolution_rate_modifier": 1.0,
    },
    "dendrite": {
        "aspect_ratio": 15.0,
        "growth_rate_modifier": 2.0,
        "dissolution_rate_modifier": 0.5,
    },
    "spherulite": {
        "aspect_ratio": 3.0,
        "growth_rate_modifier": 0.8,
        "dissolution_rate_modifier": 1.2,
    },
}

# Storage conditions for comparison (temp_C, rh_pct)
_STORAGE_CONDITIONS: list[tuple[float, float]] = [
    (25.0, 60.0),
    (40.0, 75.0),
    (5.0, 30.0),
    (-20.0, 10.0),
]

_BASE_GROWTH_RATE_UM_PER_DAY = 0.05  # Ostwald ripening constant


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrystalGrowthResult:
    """Result of crystal growth prediction over 30 days storage."""

    drug_name: str
    crystal_habit: str
    storage_temp_C: float
    storage_rh_pct: float
    initial_size_um: float
    final_size_um: float
    growth_rate_um_per_day: float
    size_increase_pct: float
    dissolution_time_change_pct: float
    ostwald_ripening_risk: str
    recommended_stabilizer: str
    notes: str


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_crystal_growth(
    drug_name: str,
    initial_size_um: float,
    crystal_habit: str,
    storage_temp_C: float = 25.0,
    storage_rh_pct: float = 60.0,
) -> CrystalGrowthResult:
    """Predict crystal growth over 30 days for a given storage condition.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    initial_size_um:
        Initial crystal size in micrometres (must be > 0).
    crystal_habit:
        Crystal morphology; one of needle, plate, cubic, dendrite, spherulite.
    storage_temp_C:
        Storage temperature in degrees Celsius (must be in (-80, 60)).
    storage_rh_pct:
        Relative humidity in percent (0-100).

    Returns
    -------
    CrystalGrowthResult
        Frozen dataclass with all growth and dissolution predictions.
    """
    # --- Validation ---
    if initial_size_um <= 0:
        raise ValueError(f"initial_size_um must be > 0, got {initial_size_um}")
    if not (-80.0 < storage_temp_C < 60.0):
        raise ValueError(f"storage_temp_C must be in (-80, 60), got {storage_temp_C}")
    if crystal_habit not in CRYSTAL_HABITS:
        raise ValueError(
            f"crystal_habit must be one of {sorted(CRYSTAL_HABITS)}, got '{crystal_habit}'"
        )

    habit = CRYSTAL_HABITS[crystal_habit]
    growth_rate_modifier = habit["growth_rate_modifier"]
    dissolution_rate_modifier = habit["dissolution_rate_modifier"]

    # --- Growth rate calculation ---
    growth_rate = _BASE_GROWTH_RATE_UM_PER_DAY
    # Temperature effect (Arrhenius-simplified)
    growth_rate *= math.exp(0.05 * (storage_temp_C - 25.0))
    # Humidity effect
    growth_rate *= 1.0 + 0.005 * storage_rh_pct
    # Crystal habit modifier
    growth_rate *= growth_rate_modifier

    # --- Size after 30 days ---
    final_size_um = initial_size_um * (1.0 + growth_rate * 30.0 / initial_size_um)
    size_increase_pct = (final_size_um - initial_size_um) / initial_size_um * 100.0

    # --- Dissolution time change ---
    # Larger crystals dissolve more slowly; habit affects rate
    dissolution_time_change_pct = size_increase_pct * (1.0 / dissolution_rate_modifier - 1.0) * 0.5

    # --- Ostwald ripening risk ---
    if size_increase_pct < 20.0:
        ostwald_ripening_risk = "low"
        recommended_stabilizer = "none"
    elif size_increase_pct <= 50.0:
        ostwald_ripening_risk = "moderate"
        recommended_stabilizer = "PVP"
    else:
        ostwald_ripening_risk = "high"
        recommended_stabilizer = "HPC"

    notes = (
        f"30-day Ostwald ripening simulation for {crystal_habit} habit crystals "
        f"at {storage_temp_C}C / {storage_rh_pct}% RH. "
        f"Growth rate modifier: {growth_rate_modifier:.2f}. "
        f"Dissolution rate modifier: {dissolution_rate_modifier:.2f}."
    )

    return CrystalGrowthResult(
        drug_name=drug_name,
        crystal_habit=crystal_habit,
        storage_temp_C=storage_temp_C,
        storage_rh_pct=storage_rh_pct,
        initial_size_um=initial_size_um,
        final_size_um=final_size_um,
        growth_rate_um_per_day=growth_rate,
        size_increase_pct=size_increase_pct,
        dissolution_time_change_pct=dissolution_time_change_pct,
        ostwald_ripening_risk=ostwald_ripening_risk,
        recommended_stabilizer=recommended_stabilizer,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Storage condition comparison
# ---------------------------------------------------------------------------


def compare_storage_conditions(
    drug_name: str,
    initial_size_um: float,
    crystal_habit: str,
) -> list[CrystalGrowthResult]:
    """Compare four standard storage conditions and sort best-to-worst.

    Conditions compared: 25C/60RH, 40C/75RH, 5C/30RH, -20C/10RH.

    Returns a list sorted by size_increase_pct ascending (best storage first).
    """
    results = [
        predict_crystal_growth(
            drug_name=drug_name,
            initial_size_um=initial_size_um,
            crystal_habit=crystal_habit,
            storage_temp_C=temp_c,
            storage_rh_pct=rh_pct,
        )
        for temp_c, rh_pct in _STORAGE_CONDITIONS
    ]
    return sorted(results, key=lambda r: r.size_increase_pct)
