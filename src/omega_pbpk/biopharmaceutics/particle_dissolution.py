"""Particle size-dependent dissolution — Noyes-Whitney with shrinking sphere."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParticleResult:
    drug_name: str
    particle_radius_um: float
    dissolved_fraction: list[float]
    times_h: list[float]
    t63_h: float
    diffusion_layer_thickness_um: float
    dissolution_rate_mg_h: float
    f_dissolved_1h: float
    f_dissolved_4h: float


def _diffusion_layer_thickness(r_um: float) -> float:
    """Higuchi model: h = r/3 for small particles, 10 um for large particles."""
    return float(min(max(r_um / 3.0 if r_um < 30.0 else 10.0, 0.1), 30.0))


def simulate_particle_dissolution(
    drug_name: str,
    dose_mg: float,
    particle_radius_um: float,
    solubility_mg_mL: float,
    diffusivity_cm2_s: float = 5e-6,
    density_g_cm3: float = 1.2,
    t_end_h: float = 6.0,
    dt_h: float = 0.01,
) -> ParticleResult:
    """Simulate Noyes-Whitney dissolution with shrinking sphere.

    dM/dt = D * A * (Cs - C) / h
    where A = 4*pi*r^2 * n_particles, r shrinks as drug dissolves.
    Sink conditions assumed (C << Cs), so C ~ 0.
    """
    if particle_radius_um <= 0:
        raise ValueError("particle_radius_um must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")

    # Convert units: D in cm²/s → um²/h, r in um, h in um
    # D [cm²/s] * 1e8 [um²/cm²] * 3600 [s/h] = D_um2_h
    D_um2_h = diffusivity_cm2_s * 1e8 * 3600.0

    # Cs in mg/mL = mg/cm³ = mg/(1e12 um³) -- need consistent units
    # Use mg/um³: Cs_mg_um3 = solubility_mg_mL / 1e12
    Cs_mg_um3 = solubility_mg_mL / 1e12

    # density in mg/um³: density_g_cm3 * 1e-3 [g/mg] / (1e12 [um³/cm³])^-1
    # density [g/cm³] = density [g/cm³] * 1000 [mg/g] / (1e12 [um³/cm³])
    rho_mg_um3 = density_g_cm3 * 1000.0 / 1e12

    # Volume of one particle
    r0 = particle_radius_um
    vol_particle_um3 = (4.0 / 3.0) * math.pi * r0**3
    mass_per_particle_mg = vol_particle_um3 * rho_mg_um3

    if mass_per_particle_mg <= 0:
        mass_per_particle_mg = 1e-12

    n_particles = dose_mg / mass_per_particle_mg

    h = _diffusion_layer_thickness(r0)

    # Initial dissolution rate (sink conditions): dM/dt = D*n*4pi*r²*Cs/h
    init_area_um2 = n_particles * 4.0 * math.pi * r0**2
    init_rate_mg_h = D_um2_h * init_area_um2 * Cs_mg_um3 / h

    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)
    dissolved = np.zeros(n_steps + 1)

    mass_undissolved = dose_mg  # mg remaining as solid
    mass_dissolved = 0.0

    for i in range(n_steps):
        if mass_undissolved <= 1e-9:
            dissolved[i + 1] = 1.0
            continue
        # Current radius from remaining mass
        mass_per_particle_cur = mass_undissolved / n_particles
        if mass_per_particle_cur <= 0:
            dissolved[i + 1] = 1.0
            continue
        r_cur = (mass_per_particle_cur / rho_mg_um3 * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        h_cur = _diffusion_layer_thickness(r_cur)
        area_cur = n_particles * 4.0 * math.pi * r_cur**2
        rate = D_um2_h * area_cur * Cs_mg_um3 / max(h_cur, 1e-3)
        dm = rate * dt_h
        dm = min(dm, mass_undissolved)
        mass_undissolved -= dm
        mass_dissolved += dm
        dissolved[i + 1] = mass_dissolved / dose_mg

    # Find t63 (63% dissolved)
    t63 = t_end_h
    for i, frac in enumerate(dissolved):
        if frac >= 0.63:
            t63 = float(times[i])
            break

    # Interpolate f at 1h and 4h
    def _f_at(t_target: float) -> float:
        if t_target >= t_end_h:
            return float(dissolved[-1])
        idx = int(t_target / dt_h)
        return float(dissolved[min(idx, n_steps)])

    f1h = _f_at(1.0)
    f4h = _f_at(4.0)

    return ParticleResult(
        drug_name=drug_name,
        particle_radius_um=particle_radius_um,
        dissolved_fraction=dissolved.tolist(),
        times_h=times.tolist(),
        t63_h=t63,
        diffusion_layer_thickness_um=h,
        dissolution_rate_mg_h=init_rate_mg_h,
        f_dissolved_1h=f1h,
        f_dissolved_4h=f4h,
    )


def compare_particle_sizes(
    drug_name: str,
    dose_mg: float,
    radii_um: list[float],
    solubility_mg_mL: float,
) -> list[ParticleResult]:
    """Simulate dissolution for each particle radius."""
    return [
        simulate_particle_dissolution(drug_name, dose_mg, r, solubility_mg_mL) for r in radii_um
    ]


__all__ = [
    "ParticleResult",
    "simulate_particle_dissolution",
    "compare_particle_sizes",
    "_diffusion_layer_thickness",
]
