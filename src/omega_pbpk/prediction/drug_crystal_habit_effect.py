"""Phase 997 — Crystal habit effect on drug dissolution and absorption."""

from __future__ import annotations

from dataclasses import dataclass

# Crystal habit properties: (surface_area_factor, wettability_factor)
_HABIT_PROPERTIES: dict[str, tuple[float, float]] = {
    "sphere": (1.0, 0.95),
    "cube": (0.83, 0.90),
    "plate": (1.3, 0.70),
    "needle": (1.8, 0.50),  # 0.50 → product 0.90 < sphere 0.95 (poor wetting dominates)
    "rod": (1.4, 0.75),
    "irregular": (2.0, 0.60),
}

_SPHERE_DISSOLUTION_FACTOR = _HABIT_PROPERTIES["sphere"][0] * _HABIT_PROPERTIES["sphere"][1]


@dataclass(frozen=True)
class CrystalHabitEffectResult:
    """Result of crystal habit effect prediction."""

    drug_name: str
    crystal_habit: str
    logp: float
    intrinsic_solubility_mg_mL: float
    surface_area_factor: float
    wettability_factor: float
    dissolution_rate_factor: float
    effective_dissolution_rate_mg_min: float
    estimated_tmax_min: float
    absorption_enhancement: str
    recommendation: str
    notes: str


def predict_crystal_habit_effect(
    drug_name: str,
    crystal_habit: str,
    logp: float = 2.0,
    mw: float = 300.0,
    particle_size_um: float = 50.0,
    dose_mg: float = 100.0,
) -> CrystalHabitEffectResult:
    """Predict the effect of crystal habit on dissolution and absorption.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    crystal_habit:
        One of: sphere, cube, plate, needle, rod, irregular.
    logp:
        Log octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol). Must be > 0.
    particle_size_um:
        Particle size in micrometres. Must be > 0.
    dose_mg:
        Dose in milligrams. Must be > 0.

    Returns
    -------
    CrystalHabitEffectResult
    """
    allowed = set(_HABIT_PROPERTIES)
    if crystal_habit not in allowed:
        raise ValueError(f"crystal_habit must be one of {sorted(allowed)}, got {crystal_habit!r}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if particle_size_um <= 0:
        raise ValueError(f"particle_size_um must be > 0, got {particle_size_um}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")

    surface_area_factor, wettability_factor = _HABIT_PROPERTIES[crystal_habit]

    # Yalkowsky intrinsic solubility
    intrinsic_solubility = 10 ** (-0.7 * logp + 0.44)

    dissolution_rate_factor = surface_area_factor * wettability_factor

    # Noyes-Whitney base rate (mg/min), normalised to 50 µm reference
    rate_base = 0.1 * intrinsic_solubility * (50.0 / particle_size_um)
    effective_dissolution_rate = rate_base * dissolution_rate_factor

    # Time to dissolve 90% of dose
    estimated_tmax_min = 0.9 * dose_mg / effective_dissolution_rate

    # Absorption enhancement vs sphere (factor ~0.95)
    if dissolution_rate_factor > 1.3:
        absorption_enhancement = "significant"
    elif dissolution_rate_factor > _SPHERE_DISSOLUTION_FACTOR:
        absorption_enhancement = "moderate"
    else:
        absorption_enhancement = "none"

    # Recommendation text
    if crystal_habit in {"needle", "irregular"}:
        recommendation = (
            f"{crystal_habit.capitalize()} habit shows poor wettability "
            "(contact angle effect) despite high geometric surface area. "
            "Consider milling to sphere/cube or surface treatment to improve wetting."
        )
    elif crystal_habit in {"sphere", "cube"}:
        recommendation = (
            f"{crystal_habit.capitalize()} habit provides good wettability "
            "and predictable dissolution. Recommended for BCS II/IV compounds."
        )
    else:
        recommendation = (
            f"{crystal_habit.capitalize()} habit offers moderate surface area "
            "with acceptable wettability. Monitor lot-to-lot variability."
        )

    notes = (
        f"Intrinsic solubility: {intrinsic_solubility:.4f} mg/mL (Yalkowsky). "
        f"Dissolution rate factor relative to sphere: "
        f"{dissolution_rate_factor / _SPHERE_DISSOLUTION_FACTOR:.3f}. "
        f"Particle size: {particle_size_um} µm."
    )

    return CrystalHabitEffectResult(
        drug_name=drug_name,
        crystal_habit=crystal_habit,
        logp=logp,
        intrinsic_solubility_mg_mL=intrinsic_solubility,
        surface_area_factor=surface_area_factor,
        wettability_factor=wettability_factor,
        dissolution_rate_factor=dissolution_rate_factor,
        effective_dissolution_rate_mg_min=effective_dissolution_rate,
        estimated_tmax_min=estimated_tmax_min,
        absorption_enhancement=absorption_enhancement,
        recommendation=recommendation,
        notes=notes,
    )


def compare_crystal_habits(
    drug_name: str,
    logp: float = 2.0,
    mw: float = 300.0,
    particle_size_um: float = 50.0,
    dose_mg: float = 100.0,
) -> list[CrystalHabitEffectResult]:
    """Compare all 6 crystal habits, sorted by dissolution_rate_factor descending.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logp:
        Log octanol-water partition coefficient.
    mw:
        Molecular weight (g/mol).
    particle_size_um:
        Particle size in micrometres.
    dose_mg:
        Dose in milligrams.

    Returns
    -------
    List of CrystalHabitEffectResult sorted by dissolution_rate_factor descending.
    """
    results = [
        predict_crystal_habit_effect(
            drug_name=drug_name,
            crystal_habit=habit,
            logp=logp,
            mw=mw,
            particle_size_um=particle_size_um,
            dose_mg=dose_mg,
        )
        for habit in _HABIT_PROPERTIES
    ]
    results.sort(key=lambda r: r.dissolution_rate_factor, reverse=True)
    return results


__all__ = [
    "CrystalHabitEffectResult",
    "predict_crystal_habit_effect",
    "compare_crystal_habits",
]
