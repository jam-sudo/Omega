"""Crystal habit dissolution prediction — Phase 422.

Predicts how crystal morphology (needle, plate, cube, sphere, irregular)
affects dissolution rate via Noyes-Whitney surface area relationship.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrystalHabitResult:
    """Result of crystal habit dissolution prediction."""

    drug_name: str
    crystal_habit: str
    particle_diameter_um: float
    aspect_ratio: float
    surface_area_factor: float
    effective_surface_area_cm2_per_g: float
    dissolution_rate_relative: float
    t50_relative: float
    solubility_effect: str
    process_sensitivity: str
    formulation_recommendation: str
    notes: str


_VALID_HABITS = {"needle", "plate", "cube", "sphere", "irregular"}


def _compute_surface_area_factor(crystal_habit: str, aspect_ratio: float) -> float:
    """Compute surface area factor relative to sphere."""
    if crystal_habit == "sphere":
        return 1.0
    elif crystal_habit == "cube":
        return 1.24
    elif crystal_habit == "plate":
        return 1.5 * (aspect_ratio**0.3)
    elif crystal_habit == "needle":
        return 2.0 * (aspect_ratio**0.4)
    elif crystal_habit == "irregular":
        return 1.8
    else:
        raise ValueError(f"Unknown crystal habit: {crystal_habit}")


def _solubility_effect(crystal_habit: str) -> str:
    """Return solubility effect string for given habit."""
    if crystal_habit in ("sphere", "cube"):
        return "none"
    elif crystal_habit in ("needle", "plate"):
        return "slight_increase"
    elif crystal_habit == "irregular":
        return "moderate_increase"
    else:
        return "none"


def _process_sensitivity(crystal_habit: str) -> str:
    """Return process sensitivity string for given habit."""
    if crystal_habit == "needle":
        return "high"
    elif crystal_habit == "plate":
        return "moderate"
    elif crystal_habit in ("cube", "sphere"):
        return "low"
    elif crystal_habit == "irregular":
        return "high"
    else:
        return "moderate"


def _formulation_recommendation(crystal_habit: str) -> str:
    """Return formulation recommendation for given habit."""
    if crystal_habit == "needle":
        return (
            "Use gentle milling to avoid habit change; consider wet granulation "
            "to preserve high surface area; monitor crystal form post-processing."
        )
    elif crystal_habit == "plate":
        return (
            "Platelets may compact well; monitor dissolution after compression; "
            "consider direct compression or roller compaction."
        )
    elif crystal_habit == "cube":
        return (
            "Cubes provide good flow and compressibility; standard milling "
            "and direct compression suitable."
        )
    elif crystal_habit == "sphere":
        return (
            "Spherical particles offer excellent flow; spray drying or "
            "spheronization recommended; lowest dissolution rate among habits."
        )
    elif crystal_habit == "irregular":
        return (
            "Irregular morphology may cause batch-to-batch variability; "
            "consider controlled crystallization or milling to standardize."
        )
    else:
        return "Evaluate dissolution and processability experimentally."


def predict_crystal_habit_dissolution(
    drug_name: str,
    crystal_habit: str,
    particle_diameter_um: float,
    drug_density_g_cm3: float = 1.2,
    aspect_ratio: float = 1.0,
    drug_solubility_mg_mL: float = 0.1,
) -> CrystalHabitResult:
    """Predict dissolution rate based on crystal habit.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    crystal_habit:
        Crystal morphology: "needle", "plate", "cube", "sphere", or "irregular".
    particle_diameter_um:
        Equivalent sphere diameter in micrometers.
    drug_density_g_cm3:
        Drug crystal density in g/cm3.
    aspect_ratio:
        Length-to-width ratio; must be >= 1.0.
    drug_solubility_mg_mL:
        Drug aqueous solubility in mg/mL.

    Returns
    -------
    CrystalHabitResult
    """
    if crystal_habit not in _VALID_HABITS:
        raise ValueError(
            f"crystal_habit must be one of {sorted(_VALID_HABITS)}, got '{crystal_habit}'"
        )
    if particle_diameter_um <= 0:
        raise ValueError(f"particle_diameter_um must be > 0, got {particle_diameter_um}")
    if drug_density_g_cm3 <= 0:
        raise ValueError(f"drug_density_g_cm3 must be > 0, got {drug_density_g_cm3}")
    if aspect_ratio < 1.0:
        raise ValueError(f"aspect_ratio must be >= 1.0, got {aspect_ratio}")

    surface_area_factor = _compute_surface_area_factor(crystal_habit, aspect_ratio)

    # Effective surface area per gram
    # For a sphere: SA/V = 6/d; SA per gram = (6/d_cm) / density
    d_cm = particle_diameter_um * 1e-4
    sa_sphere_per_g = (6.0 / d_cm) / drug_density_g_cm3  # cm^2/g
    effective_sa = sa_sphere_per_g * surface_area_factor

    # Noyes-Whitney: dissolution rate proportional to surface area
    dissolution_rate_relative = surface_area_factor
    t50_relative = 1.0 / surface_area_factor

    sol_effect = _solubility_effect(crystal_habit)
    proc_sens = _process_sensitivity(crystal_habit)
    formulation_rec = _formulation_recommendation(crystal_habit)

    notes = (
        f"{drug_name} ({crystal_habit}): surface area factor = {surface_area_factor:.3f} "
        f"relative to sphere. Dissolution rate improved by {(surface_area_factor - 1.0) * 100:.1f}% "
        f"vs sphere. Effective SA = {effective_sa:.1f} cm2/g."
    )

    return CrystalHabitResult(
        drug_name=drug_name,
        crystal_habit=crystal_habit,
        particle_diameter_um=particle_diameter_um,
        aspect_ratio=aspect_ratio,
        surface_area_factor=surface_area_factor,
        effective_surface_area_cm2_per_g=effective_sa,
        dissolution_rate_relative=dissolution_rate_relative,
        t50_relative=t50_relative,
        solubility_effect=sol_effect,
        process_sensitivity=proc_sens,
        formulation_recommendation=formulation_rec,
        notes=notes,
    )


def compare_crystal_habits(
    drug_name: str,
    particle_diameter_um: float,
    drug_solubility_mg_mL: float,
    habits_list: list = None,
) -> list:
    """Compare dissolution across crystal habits.

    Parameters
    ----------
    drug_name:
        Drug name.
    particle_diameter_um:
        Equivalent sphere diameter in micrometers.
    drug_solubility_mg_mL:
        Drug solubility in mg/mL.
    habits_list:
        List of habit names to compare. Defaults to all 5 habits.

    Returns
    -------
    list of CrystalHabitResult sorted by dissolution_rate_relative descending.
    """
    if habits_list is None:
        habits_list = ["sphere", "cube", "plate", "needle", "irregular"]

    results = []
    for habit in habits_list:
        # Use aspect_ratio=3.0 for needle and plate by default
        aspect_ratio = 3.0 if habit in ("needle", "plate") else 1.0
        result = predict_crystal_habit_dissolution(
            drug_name=drug_name,
            crystal_habit=habit,
            particle_diameter_um=particle_diameter_um,
            aspect_ratio=aspect_ratio,
            drug_solubility_mg_mL=drug_solubility_mg_mL,
        )
        results.append(result)

    results.sort(key=lambda r: r.dissolution_rate_relative, reverse=True)
    return results
