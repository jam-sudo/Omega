"""Dissolution media effect on drug dissolution rate (biorelevant media, Phase 389)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Solubility enhancement factor relative to pure water for each media type
_MEDIA_FACTORS: dict[str, float] = {
    "water": 1.0,
    "HCl_pH_1_2": 0.5,
    "PBS_pH_6_8": 1.0,
    "FaSSIF": 2.0,
    "FeSSIF": 4.0,
}

# Diffusion layer thickness (µm) — fixed per Noyes-Whitney
_H_UM: float = 50.0

# Particle density (mg/mm³ = g/cm³)
_RHO_MG_MM3: float = 1.2  # mg/mm³

VALID_MEDIA: frozenset[str] = frozenset(_MEDIA_FACTORS)


@dataclass
class DissolutionMediaResult:
    """Result of a biorelevant dissolution media simulation."""

    drug_name: str
    dose_mg: float
    media_type: str
    solubility_effective_mg_mL: float
    times_min: list[float] = field(default_factory=list)
    pct_dissolved: list[float] = field(default_factory=list)
    t50_min: float | None = None
    t80_min: float | None = None
    t_complete_min: float | None = None  # None if not reached
    notes: str = ""


def simulate_dissolution_media(
    drug_name: str,
    dose_mg: float,
    media_type: str,
    solubility_fasted_mg_mL: float,
    particle_radius_um: float,
    diffusion_coeff_cm2_s: float,
    t_end_min: float = 120.0,
    dt_min: float = 0.1,
) -> DissolutionMediaResult:
    """Simulate drug dissolution in a specific biorelevant dissolution medium.

    Uses the Noyes-Whitney equation with shrinking sphere (cube-root law):
        dm/dt = D * A * (Cs_media - C) / h
    where sink conditions (C << Cs) are assumed, so C ≈ 0.

    Parameters
    ----------
    drug_name:
        Name of the drug being simulated.
    dose_mg:
        Total drug dose in milligrams.
    media_type:
        One of "water", "HCl_pH_1_2", "PBS_pH_6_8", "FaSSIF", "FeSSIF".
    solubility_fasted_mg_mL:
        Drug solubility measured in fasted state (reference for scaling).
    particle_radius_um:
        Initial particle radius in micrometres.
    diffusion_coeff_cm2_s:
        Diffusion coefficient in cm²/s.
    t_end_min:
        Simulation end time in minutes.
    dt_min:
        Integration time step in minutes.

    Returns
    -------
    DissolutionMediaResult
    """
    if not drug_name:
        raise ValueError("drug_name must not be empty")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if media_type not in VALID_MEDIA:
        raise ValueError(f"media_type must be one of {sorted(VALID_MEDIA)}, got '{media_type}'")
    if solubility_fasted_mg_mL <= 0:
        raise ValueError("solubility_fasted_mg_mL must be > 0")
    if particle_radius_um <= 0:
        raise ValueError("particle_radius_um must be > 0")
    if diffusion_coeff_cm2_s <= 0:
        raise ValueError("diffusion_coeff_cm2_s must be > 0")
    if t_end_min <= 0:
        raise ValueError("t_end_min must be > 0")
    if dt_min <= 0:
        raise ValueError("dt_min must be > 0")

    media_factor = _MEDIA_FACTORS[media_type]
    cs_effective = solubility_fasted_mg_mL * media_factor  # mg/mL

    # Unit conversions — work in µm and minutes for convenience:
    # D [cm²/s] → [µm²/min]: D * 1e8 [µm²/cm²] * 60 [s/min]
    D_um2_min = diffusion_coeff_cm2_s * 1.0e8 * 60.0

    # Cs [mg/mL = mg/cm³] → [mg/µm³]: divide by 1e12 (cm³/µm³)
    cs_mg_um3 = cs_effective / 1.0e12

    # Density [mg/mm³] → [mg/µm³]: divide by 1e9 (mm³/µm³)
    rho_mg_um3 = _RHO_MG_MM3 / 1.0e9

    # Number of particles (conserved throughout simulation)
    vol0_um3 = (4.0 / 3.0) * math.pi * particle_radius_um**3
    mass_per_particle_mg = vol0_um3 * rho_mg_um3
    if mass_per_particle_mg <= 0.0:
        mass_per_particle_mg = 1.0e-20
    n_particles = dose_mg / mass_per_particle_mg

    h_um = _H_UM  # fixed diffusion layer thickness

    n_steps = max(int(t_end_min / dt_min), 1)
    times_min: list[float] = []
    pct_dissolved: list[float] = []

    mass_undissolved = dose_mg
    mass_dissolved = 0.0

    times_min.append(0.0)
    pct_dissolved.append(0.0)

    t50: float | None = None
    t80: float | None = None
    t_complete: float | None = None

    for i in range(n_steps):
        t_cur = (i + 1) * dt_min

        if mass_undissolved <= 1.0e-9 * dose_mg:
            mass_undissolved = 0.0
            mass_dissolved = dose_mg
        else:
            # Current radius from remaining mass per particle (cube-root law)
            mass_pp = mass_undissolved / n_particles
            r_cur = (mass_pp / rho_mg_um3 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

            # Total particle surface area
            area_um2 = n_particles * 4.0 * math.pi * r_cur**2

            # Noyes-Whitney rate (sink, C=0)
            rate_mg_min = D_um2_min * area_um2 * cs_mg_um3 / h_um

            dm = rate_mg_min * dt_min
            dm = min(dm, mass_undissolved)
            mass_undissolved -= dm
            mass_dissolved += dm

        pct = min(100.0 * mass_dissolved / dose_mg, 100.0)
        times_min.append(t_cur)
        pct_dissolved.append(pct)

        if t50 is None and pct >= 50.0:
            t50 = t_cur
        if t80 is None and pct >= 80.0:
            t80 = t_cur
        if t_complete is None and pct >= 99.9:
            t_complete = t_cur

    notes_parts: list[str] = [
        f"Media factor vs water: {media_factor:.1f}x",
        f"Effective Cs: {cs_effective:.4f} mg/mL",
    ]
    if t_complete is None:
        notes_parts.append("Drug not fully dissolved within simulation window.")

    return DissolutionMediaResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        media_type=media_type,
        solubility_effective_mg_mL=cs_effective,
        times_min=times_min,
        pct_dissolved=pct_dissolved,
        t50_min=t50,
        t80_min=t80,
        t_complete_min=t_complete,
        notes="; ".join(notes_parts),
    )


def compare_media(
    drug_name: str,
    dose_mg: float,
    solubility_fasted_mg_mL: float,
    particle_radius_um: float,
    diffusion_coeff_cm2_s: float,
    t_end_min: float = 120.0,
    dt_min: float = 0.1,
) -> dict[str, DissolutionMediaResult]:
    """Compare dissolution across all 5 biorelevant media types.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg.
    solubility_fasted_mg_mL:
        Fasted-state solubility (mg/mL) used as reference.
    particle_radius_um:
        Initial particle radius in µm.
    diffusion_coeff_cm2_s:
        Diffusion coefficient in cm²/s.
    t_end_min:
        Simulation end time (minutes).
    dt_min:
        ODE time step (minutes).

    Returns
    -------
    dict mapping media_type → DissolutionMediaResult
    """
    return {
        media: simulate_dissolution_media(
            drug_name=drug_name,
            dose_mg=dose_mg,
            media_type=media,
            solubility_fasted_mg_mL=solubility_fasted_mg_mL,
            particle_radius_um=particle_radius_um,
            diffusion_coeff_cm2_s=diffusion_coeff_cm2_s,
            t_end_min=t_end_min,
            dt_min=dt_min,
        )
        for media in sorted(VALID_MEDIA)
    }


__all__ = [
    "DissolutionMediaResult",
    "VALID_MEDIA",
    "simulate_dissolution_media",
    "compare_media",
]
