"""Phase 339 — Drug Crystal Habit Effects.

Models how crystal habit (morphology) affects drug dissolution and bioavailability.
Different crystal habits (needles, plates, cubes, spheres) have different surface
areas and dissolution rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VALID_HABITS = {"needle", "plate", "cube", "sphere"}

# SSA multiplier relative to sphere of same volume
_SSA_FACTORS: dict[str, float] = {
    "sphere": 1.0,
    "cube": 0.806,
    "plate": 2.5,
    "needle": 3.0,
}

# Empirical bioavailability factors relative to cube
_BA_FACTORS: dict[str, float] = {
    "needle": 1.3,
    "plate": 1.2,
    "sphere": 1.1,
    "cube": 1.0,
}

# Drug density g/cm³
_DRUG_DENSITY_G_CM3 = 1.2

# Base dissolution constant (dimensionless/h)
_K_DISS_BASE = 1e-3


@dataclass(frozen=True)
class CrystalHabitResult:
    """Results of crystal habit dissolution simulation."""

    drug_name: str
    habit: str  # "needle", "plate", "cube", "sphere"
    particle_size_um: float
    dose_mg: float

    specific_surface_area_cm2_per_g: float  # SSA from geometry
    dissolution_rate_mg_per_h: float  # initial rate
    t90_dissolution_h: float  # time to 90% dissolved
    f_dissolved_2h: float  # fraction dissolved at 2 h
    f_dissolved_4h: float  # fraction dissolved at 4 h

    bioavailability_factor: float  # relative BA (cube=1.0 reference)
    notes: str


def _compute_ssa(habit: str, particle_size_um: float) -> float:
    """Compute specific surface area (cm²/g) for a given habit and particle size."""
    d_cm = particle_size_um * 1e-4
    ssa_sphere = 6.0 / (_DRUG_DENSITY_G_CM3 * d_cm)
    return ssa_sphere * _SSA_FACTORS[habit]


def calculate_crystal_habit_dissolution(
    drug_name: str,
    habit: str,
    particle_size_um: float,
    dose_mg: float,
    solubility_mg_mL: float = 1.0,
    dose_volume_mL: float = 250.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
) -> CrystalHabitResult:
    """Simulate dissolution and absorption for a given crystal habit.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    habit:
        Crystal habit: "needle", "plate", "cube", or "sphere".
    particle_size_um:
        Mean particle diameter in micrometres (> 0).
    dose_mg:
        Dose in mg (> 0).
    solubility_mg_mL:
        Aqueous solubility in mg/mL (> 0).
    dose_volume_mL:
        Volume of fluid in GI lumen (mL).
    cl_L_per_h:
        Systemic clearance (L/h), currently reserved for future PK.
    vd_L:
        Volume of distribution (L), reserved for future PK.

    Returns
    -------
    CrystalHabitResult
    """
    # --- Input validation ---
    if habit not in VALID_HABITS:
        raise ValueError(f"habit must be one of {VALID_HABITS}, got {habit!r}")
    if particle_size_um <= 0:
        raise ValueError(f"particle_size_um must be > 0, got {particle_size_um}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")

    ssa = _compute_ssa(habit, particle_size_um)

    # Initial dissolution rate (mg/h) before any dissolved drug builds up
    dissolution_rate_init = _K_DISS_BASE * ssa * dose_mg

    # --- Forward Euler simulation over 0–12 h ---
    dt = 0.1  # h
    t_end = 12.0
    n_steps = int(round(t_end / dt)) + 1
    times = np.linspace(0.0, t_end, n_steps)

    m_undissolved = dose_mg
    c_dissolved = 0.0  # mg/mL in dose_volume

    ka = 1.5  # absorption rate constant /h

    t90_h = t_end  # default if never reaches 90%
    f_dissolved_2h = 0.0
    f_dissolved_4h = 0.0

    for i, _t in enumerate(times[:-1]):
        # Dissolution rate (mg/h) – Noyes-Whitney saturated limit
        saturation_factor = max(0.0, 1.0 - c_dissolved / solubility_mg_mL)
        rate = 0.0
        if m_undissolved > 0.0:
            rate = _K_DISS_BASE * ssa * m_undissolved * saturation_factor
            rate = min(rate, m_undissolved / dt)  # can't dissolve more than available

        dm_undissolved = -rate * dt
        dc_dissolved = (rate / dose_volume_mL - ka * c_dissolved) * dt

        m_undissolved = max(0.0, m_undissolved + dm_undissolved)
        c_dissolved = max(0.0, c_dissolved + dc_dissolved)

        t_next = times[i + 1]

        # Capture t90
        if t90_h >= t_end and m_undissolved <= 0.1 * dose_mg:
            t90_h = t_next

        # Capture f_dissolved at 2 h and 4 h
        if abs(t_next - 2.0) < dt / 2.0:
            f_dissolved_2h = (dose_mg - m_undissolved) / dose_mg
        if abs(t_next - 4.0) < dt / 2.0:
            f_dissolved_4h = (dose_mg - m_undissolved) / dose_mg

    # Clamp fractions to [0, 1]
    f_dissolved_2h = float(np.clip(f_dissolved_2h, 0.0, 1.0))
    f_dissolved_4h = float(np.clip(f_dissolved_4h, 0.0, 1.0))

    ba_factor = _BA_FACTORS[habit]

    notes = (
        f"{drug_name} ({habit}): SSA={ssa:.1f} cm²/g, "
        f"t90={t90_h:.2f}h, f4h={f_dissolved_4h:.3f}, "
        f"BA_factor={ba_factor:.2f}"
    )

    return CrystalHabitResult(
        drug_name=drug_name,
        habit=habit,
        particle_size_um=particle_size_um,
        dose_mg=dose_mg,
        specific_surface_area_cm2_per_g=float(ssa),
        dissolution_rate_mg_per_h=float(dissolution_rate_init),
        t90_dissolution_h=float(t90_h),
        f_dissolved_2h=f_dissolved_2h,
        f_dissolved_4h=f_dissolved_4h,
        bioavailability_factor=ba_factor,
        notes=notes,
    )


def compare_crystal_habits(
    drug_name: str,
    particle_size_um: float,
    dose_mg: float,
    solubility_mg_mL: float = 1.0,
    **kwargs,
) -> dict:
    """Compare dissolution across all four crystal habits.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    particle_size_um:
        Mean particle diameter in micrometres.
    dose_mg:
        Dose in mg.
    solubility_mg_mL:
        Aqueous solubility in mg/mL.
    **kwargs:
        Additional keyword arguments forwarded to
        ``calculate_crystal_habit_dissolution``.

    Returns
    -------
    dict
        Keys are habit names plus ``"fastest_dissolving"``.
    """
    results: dict[str, CrystalHabitResult] = {}
    for habit in VALID_HABITS:
        results[habit] = calculate_crystal_habit_dissolution(
            drug_name=drug_name,
            habit=habit,
            particle_size_um=particle_size_um,
            dose_mg=dose_mg,
            solubility_mg_mL=solubility_mg_mL,
            **kwargs,
        )

    fastest = min(results, key=lambda h: results[h].t90_dissolution_h)
    return {**results, "fastest_dissolving": fastest}
