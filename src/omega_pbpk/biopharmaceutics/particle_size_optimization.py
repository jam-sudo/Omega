"""Particle size optimization for dissolution — Phase 450.

Uses Noyes-Whitney shrinking sphere model to simulate dissolution kinetics
at multiple particle sizes and identify the optimal size meeting a target
dissolution requirement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "ParticleSizeResult",
    "ParticleSizeOptResult",
    "dissolution_rate",
    "time_to_dissolve",
    "dissolution_profile",
    "optimize_particle_size",
    "biopharmaceutics_classification",
]


# ---------------------------------------------------------------------------
# Phase 450 new API
# ---------------------------------------------------------------------------


@dataclass
class ParticleSizeResult:
    """Result from optimize_particle_size (Phase 450 API)."""

    optimal_radius_um: float
    predicted_dissolution_time_h: float
    target_dissolution_time_h: float
    solubility_mg_mL: float
    notes: str


def dissolution_rate(
    radius_um: float,
    drug_solubility_mg_mL: float,
    diffusion_coeff_cm2_s: float = 1e-6,
    boundary_layer_um: float = 10.0,
    density_g_cm3: float = 1.2,
) -> float:
    """Noyes-Whitney dissolution rate (fraction per hour) for a spherical particle.

    Uses dR/dt = -D * Cs / (rho * h) where Cs is solubility and h is boundary layer.
    Rate = fraction of radius lost per hour, approximated as surface area weighted.

    Parameters
    ----------
    radius_um:
        Particle radius in micrometres. Must be > 0.
    drug_solubility_mg_mL:
        Drug solubility (mg/mL). Must be > 0.
    diffusion_coeff_cm2_s:
        Diffusion coefficient (cm^2/s). Must be > 0.
    boundary_layer_um:
        Diffusion boundary layer thickness (um). Must be > 0.
    density_g_cm3:
        Particle density (g/cm^3). Must be > 0.

    Returns
    -------
    Fractional dissolution rate per hour (positive, dimensionless/h).
    """
    if radius_um <= 0:
        raise ValueError("radius_um must be > 0")
    if drug_solubility_mg_mL <= 0:
        raise ValueError("drug_solubility_mg_mL must be > 0")
    if diffusion_coeff_cm2_s <= 0:
        raise ValueError("diffusion_coeff_cm2_s must be > 0")
    if boundary_layer_um <= 0:
        raise ValueError("boundary_layer_um must be > 0")
    if density_g_cm3 <= 0:
        raise ValueError("density_g_cm3 must be > 0")

    # Convert units: D in cm^2/h, h in cm, rho in mg/cm^3, Cs in mg/cm^3
    D_cm2_h = diffusion_coeff_cm2_s * 3600.0
    h_cm = boundary_layer_um * 1e-4
    rho_mg_cm3 = density_g_cm3 * 1000.0
    Cs_mg_cm3 = drug_solubility_mg_mL  # mg/mL == mg/cm^3

    # dR/dt = -D * Cs / (rho * h)  [cm/h]
    dr_dt_cm_h = D_cm2_h * Cs_mg_cm3 / (rho_mg_cm3 * h_cm)

    # Fraction of radius dissolved per hour = dr_dt / R
    R_cm = radius_um * 1e-4
    return dr_dt_cm_h / R_cm


def time_to_dissolve(
    radius_um: float,
    drug_solubility_mg_mL: float,
    diffusion_coeff_cm2_s: float = 1e-6,
    boundary_layer_um: float = 10.0,
    density_g_cm3: float = 1.2,
) -> float:
    """Estimate time to fully dissolve a spherical particle (Noyes-Whitney).

    t_dissolve = R / (dR/dt) = R * rho * h / (D * Cs)

    Parameters
    ----------
    radius_um:
        Particle radius in micrometres. Must be > 0.
    drug_solubility_mg_mL:
        Drug solubility (mg/mL). Must be > 0.
    diffusion_coeff_cm2_s:
        Diffusion coefficient (cm^2/s). Must be > 0.
    boundary_layer_um:
        Diffusion boundary layer thickness (um). Must be > 0.
    density_g_cm3:
        Particle density (g/cm^3). Must be > 0.

    Returns
    -------
    Dissolution time in hours.
    """
    if radius_um <= 0:
        raise ValueError("radius_um must be > 0")
    if drug_solubility_mg_mL <= 0:
        raise ValueError("drug_solubility_mg_mL must be > 0")
    if diffusion_coeff_cm2_s <= 0:
        raise ValueError("diffusion_coeff_cm2_s must be > 0")
    if boundary_layer_um <= 0:
        raise ValueError("boundary_layer_um must be > 0")
    if density_g_cm3 <= 0:
        raise ValueError("density_g_cm3 must be > 0")

    D_cm2_h = diffusion_coeff_cm2_s * 3600.0
    # boundary_layer_um not used in Higuchi approximation
    rho_mg_cm3 = density_g_cm3 * 1000.0
    Cs_mg_cm3 = drug_solubility_mg_mL
    R_cm = radius_um * 1e-4

    # t = R^2 * rho / (2 * D * Cs)  [Higuchi shrinking sphere]
    return (R_cm ** 2 * rho_mg_cm3) / (2.0 * D_cm2_h * Cs_mg_cm3)


def dissolution_profile(
    radius_um: float,
    drug_solubility_mg_mL: float,
    diffusion_coeff_cm2_s: float = 1e-6,
    boundary_layer_um: float = 10.0,
    density_g_cm3: float = 1.2,
    t_end_h: float = 2.0,
    dt_h: float = 0.05,
) -> dict[str, list[float]]:
    """Simulate dissolution fraction over time for a single radius.

    Uses shrinking sphere model with forward Euler integration.

    Parameters
    ----------
    radius_um:
        Particle radius (um). Must be > 0.
    drug_solubility_mg_mL:
        Drug solubility (mg/mL). Must be > 0.
    diffusion_coeff_cm2_s:
        Diffusion coefficient (cm^2/s). Must be > 0.
    boundary_layer_um:
        Boundary layer thickness (um). Must be > 0.
    density_g_cm3:
        Particle density (g/cm^3). Must be > 0.
    t_end_h:
        Simulation end time (h). Must be > 0.
    dt_h:
        Time step (h). Must be > 0.

    Returns
    -------
    dict with 'times_h' and 'fraction_dissolved' (both lists of float).
    """
    if radius_um <= 0:
        raise ValueError("radius_um must be > 0")
    if drug_solubility_mg_mL <= 0:
        raise ValueError("drug_solubility_mg_mL must be > 0")
    if diffusion_coeff_cm2_s <= 0:
        raise ValueError("diffusion_coeff_cm2_s must be > 0")
    if boundary_layer_um <= 0:
        raise ValueError("boundary_layer_um must be > 0")
    if density_g_cm3 <= 0:
        raise ValueError("density_g_cm3 must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    D_cm2_h = diffusion_coeff_cm2_s * 3600.0
    h_cm = boundary_layer_um * 1e-4
    rho_mg_cm3 = density_g_cm3 * 1000.0
    Cs_mg_cm3 = drug_solubility_mg_mL

    # dR/dt = -D * Cs / (rho * h)
    dr_dt = D_cm2_h * Cs_mg_cm3 / (rho_mg_cm3 * h_cm)

    R = radius_um * 1e-4  # cm
    R0 = R
    n_steps = max(int(round(t_end_h / dt_h)), 1)
    times_h: list[float] = [0.0]
    fracs: list[float] = [0.0]

    for i in range(n_steps):
        t = (i + 1) * dt_h
        R = max(R - dr_dt * dt_h, 0.0)
        if R0 > 0:
            frac = 1.0 - (R / R0) ** 3
        else:
            frac = 1.0
        frac = min(frac, 1.0)
        times_h.append(t)
        fracs.append(frac)
        if R <= 0:
            # Particle fully dissolved — fill remaining steps
            for j in range(i + 1, n_steps):
                times_h.append((j + 1) * dt_h)
                fracs.append(1.0)
            break

    return {"times_h": times_h, "fraction_dissolved": fracs}


def _optimize_particle_size_phase450(
    target_dissolution_time_h: float,
    drug_solubility_mg_mL: float,
    radius_range_um: tuple[float, float] = (0.1, 500.0),
    diffusion_coeff_cm2_s: float = 1e-6,
    boundary_layer_um: float = 10.0,
    density_g_cm3: float = 1.2,
) -> ParticleSizeResult:
    """Find optimal particle radius for target dissolution time (Phase 450 API).

    Inverts the Higuchi shrinking sphere formula:
        t = R^2 * rho / (2 * D * Cs)  =>  R = sqrt(2 * D * Cs * t / rho)

    Parameters
    ----------
    target_dissolution_time_h:
        Desired dissolution time (h). Must be > 0.
    drug_solubility_mg_mL:
        Drug solubility (mg/mL). Must be > 0.
    radius_range_um:
        (min, max) radius search range in um.
    diffusion_coeff_cm2_s:
        Diffusion coefficient (cm^2/s). Must be > 0.
    boundary_layer_um:
        Boundary layer thickness (um). Must be > 0.
    density_g_cm3:
        Particle density (g/cm^3). Must be > 0.

    Returns
    -------
    ParticleSizeResult
    """
    if target_dissolution_time_h <= 0:
        raise ValueError("target_dissolution_time_h must be > 0")
    if drug_solubility_mg_mL <= 0:
        raise ValueError("drug_solubility_mg_mL must be > 0")
    if diffusion_coeff_cm2_s <= 0:
        raise ValueError("diffusion_coeff_cm2_s must be > 0")
    if boundary_layer_um <= 0:
        raise ValueError("boundary_layer_um must be > 0")
    if density_g_cm3 <= 0:
        raise ValueError("density_g_cm3 must be > 0")
    if radius_range_um[0] <= 0 or radius_range_um[1] <= radius_range_um[0]:
        raise ValueError("radius_range_um must have 0 < min < max")

    D_cm2_h = diffusion_coeff_cm2_s * 3600.0
    rho_mg_cm3 = density_g_cm3 * 1000.0
    Cs_mg_cm3 = drug_solubility_mg_mL

    # Analytical inversion of t = R^2 * rho / (2 * D * Cs)
    R_cm = math.sqrt(2.0 * D_cm2_h * Cs_mg_cm3 * target_dissolution_time_h / rho_mg_cm3)
    R_um = R_cm / 1e-4  # cm -> um

    notes_parts = []
    r_min, r_max = radius_range_um
    if R_um < r_min:
        notes_parts.append(
            f"Optimal radius {R_um:.3f} um is below range minimum {r_min} um; clamped."
        )
        R_um = r_min
    elif R_um > r_max:
        notes_parts.append(
            f"Optimal radius {R_um:.3f} um exceeds range maximum {r_max} um; clamped."
        )
        R_um = r_max

    predicted_t = time_to_dissolve(
        radius_um=R_um,
        drug_solubility_mg_mL=drug_solubility_mg_mL,
        diffusion_coeff_cm2_s=diffusion_coeff_cm2_s,
        boundary_layer_um=boundary_layer_um,
        density_g_cm3=density_g_cm3,
    )

    return ParticleSizeResult(
        optimal_radius_um=R_um,
        predicted_dissolution_time_h=predicted_t,
        target_dissolution_time_h=target_dissolution_time_h,
        solubility_mg_mL=drug_solubility_mg_mL,
        notes="; ".join(notes_parts) if notes_parts else "optimal radius computed analytically",
    )


# ---------------------------------------------------------------------------
# Result dataclass (frozen=False because lists are mutable fields)
# ---------------------------------------------------------------------------


@dataclass
class ParticleSizeOptResult:
    """Results from particle size optimization."""

    drug_name: str
    target_dissolved_pct_at_60min: float
    solubility_mg_mL: float
    dose_mg: float
    particle_sizes_um: list[float] = field(default_factory=list)
    dissolved_pct_at_60min: list[float] = field(default_factory=list)
    optimal_size_um: float = 0.0
    optimal_dissolved_pct: float = 0.0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal dissolution simulation
# ---------------------------------------------------------------------------


def _simulate_dissolution(
    dose_mg: float,
    particle_size_um: float,
    solubility_mg_mL: float,
    diffusion_coeff_cm2_s: float,
    density_g_cm3: float,
    media_volume_mL: float,
    t_end_min: float = 120.0,
    dt_min: float = 0.5,
) -> list[tuple[float, float]]:
    """Simulate Noyes-Whitney dissolution using shrinking sphere (forward Euler).

    Model:
        dm/dt = -D * A * (Cs - C) / h
        A = 4 * pi * r^2 * N   (total particle surface area)
        h = 50e-4 cm (diffusion layer thickness = 50 um)
        r shrinks as particles dissolve:  r(t) = r0 * (m(t)/m0)^(1/3)
        C_media = mass_dissolved / media_volume

    Parameters
    ----------
    dose_mg:
        Total drug mass to dissolve (mg).
    particle_size_um:
        Particle diameter in um. Radius = particle_size_um / 2.
    solubility_mg_mL:
        Saturated solubility in mg/mL (= mg/cm^3).
    diffusion_coeff_cm2_s:
        Diffusion coefficient in cm^2/s.
    density_g_cm3:
        Particle density (g/cm^3).
    media_volume_mL:
        Volume of dissolution media (mL = cm^3).
    t_end_min:
        End of simulation (minutes).
    dt_min:
        Time step (minutes).

    Returns
    -------
    List of (time_min, dissolved_fraction) tuples.
    """
    r0_cm = (particle_size_um / 2.0) * 1e-4  # um -> cm

    if r0_cm <= 0:
        return [(t, 1.0) for t in [i * dt_min for i in range(int(t_end_min / dt_min) + 1)]]

    density_mg_cm3 = density_g_cm3 * 1000.0  # g/cm^3 -> mg/cm^3

    vol_particle_cm3 = (4.0 / 3.0) * math.pi * r0_cm**3
    mass_per_particle_mg = vol_particle_cm3 * density_mg_cm3

    if mass_per_particle_mg <= 0.0:
        mass_per_particle_mg = 1e-15

    n_particles = dose_mg / mass_per_particle_mg

    # Diffusion layer thickness: 50 um = 50e-4 cm
    h_cm = 50e-4

    # Convert D from cm^2/s to cm^2/min
    D_cm2_min = diffusion_coeff_cm2_s * 60.0

    n_steps = max(int(t_end_min / dt_min), 1)
    results: list[tuple[float, float]] = [(0.0, 0.0)]

    mass_undissolved = dose_mg
    mass_dissolved = 0.0

    for step in range(n_steps):
        if mass_undissolved <= 1e-12 * dose_mg:
            t = (step + 1) * dt_min
            results.append((t, 1.0))
            continue

        # Current radius
        fraction_remaining = mass_undissolved / dose_mg
        r_cur_cm = r0_cm * (fraction_remaining ** (1.0 / 3.0))

        # Surface area of all particles
        area_cm2 = n_particles * 4.0 * math.pi * r_cur_cm**2

        # Bulk concentration in media
        c_media = mass_dissolved / media_volume_mL  # mg/mL = mg/cm^3

        # Driving force: Cs - C (cannot exceed solubility)
        driving = max(solubility_mg_mL - c_media, 0.0)

        # Dissolution rate (mg/min)
        rate_mg_min = D_cm2_min * area_cm2 * driving / h_cm

        # Forward Euler step
        dm = rate_mg_min * dt_min
        dm = min(dm, mass_undissolved)

        mass_undissolved -= dm
        mass_dissolved += dm

        t = (step + 1) * dt_min
        results.append((t, mass_dissolved / dose_mg))

    return results


def _dissolved_pct_at_time(
    dissolution_profile: list[tuple[float, float]],
    target_min: float,
) -> float:
    """Interpolate dissolved percentage at a given time (minutes)."""
    if not dissolution_profile:
        return 0.0

    # Find bracketing points
    for i in range(1, len(dissolution_profile)):
        t_prev, f_prev = dissolution_profile[i - 1]
        t_curr, f_curr = dissolution_profile[i]
        if t_curr >= target_min:
            if t_curr == t_prev:
                return f_curr * 100.0
            # Linear interpolation
            alpha = (target_min - t_prev) / (t_curr - t_prev)
            return (f_prev + alpha * (f_curr - f_prev)) * 100.0

    # Beyond simulation end: return last value
    return dissolution_profile[-1][1] * 100.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_particle_size(
    drug_name: str,
    target_dissolved_pct_at_60min: float,
    solubility_mg_mL: float,
    diffusion_coeff_cm2_s: float,
    dose_mg: float,
    density_g_cm3: float,
    particle_sizes_um: list[float],
    media_volume_mL: float = 900.0,
) -> ParticleSizeOptResult:
    """Optimize particle size to achieve target dissolution at 60 minutes.

    For each candidate particle size, simulates Noyes-Whitney dissolution
    (shrinking sphere, forward Euler, accounting for bulk concentration
    build-up in finite media volume). Identifies the largest particle size
    that achieves at least the target dissolved percentage at 60 minutes,
    giving a conservative optimal choice.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    target_dissolved_pct_at_60min:
        Target percent dissolved at 60 minutes (0–100).
    solubility_mg_mL:
        Aqueous solubility (mg/mL).
    diffusion_coeff_cm2_s:
        Diffusion coefficient (cm^2/s). Typical ~5e-6 cm^2/s.
    dose_mg:
        Dose mass in mg.
    density_g_cm3:
        Particle density (g/cm^3).
    particle_sizes_um:
        List of candidate particle diameters in um (must be > 0).
    media_volume_mL:
        Volume of dissolution media in mL (default 900 mL, USP apparatus II).

    Returns
    -------
    ParticleSizeOptResult
    """
    if not drug_name:
        raise ValueError("drug_name must not be empty")
    if not (0.0 <= target_dissolved_pct_at_60min <= 100.0):
        raise ValueError("target_dissolved_pct_at_60min must be in [0, 100]")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if diffusion_coeff_cm2_s <= 0:
        raise ValueError("diffusion_coeff_cm2_s must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if density_g_cm3 <= 0:
        raise ValueError("density_g_cm3 must be > 0")
    if not particle_sizes_um:
        raise ValueError("particle_sizes_um must not be empty")
    if any(s <= 0 for s in particle_sizes_um):
        raise ValueError("All particle_sizes_um must be > 0")
    if media_volume_mL <= 0:
        raise ValueError("media_volume_mL must be > 0")

    sizes_sorted = sorted(particle_sizes_um)
    dissolved_pcts: list[float] = []

    for size_um in sizes_sorted:
        profile = _simulate_dissolution(
            dose_mg=dose_mg,
            particle_size_um=size_um,
            solubility_mg_mL=solubility_mg_mL,
            diffusion_coeff_cm2_s=diffusion_coeff_cm2_s,
            density_g_cm3=density_g_cm3,
            media_volume_mL=media_volume_mL,
        )
        pct = _dissolved_pct_at_time(profile, target_min=60.0)
        dissolved_pcts.append(round(pct, 4))

    # Find optimal: largest particle size that meets target
    optimal_size_um = sizes_sorted[0]
    optimal_pct = dissolved_pcts[0]

    for size, pct in zip(sizes_sorted, dissolved_pcts, strict=True):
        if pct >= target_dissolved_pct_at_60min:
            optimal_size_um = size
            optimal_pct = pct

    notes: list[str] = []

    # Check if any size meets the target
    max_pct = max(dissolved_pcts)
    if max_pct < target_dissolved_pct_at_60min:
        best_sz = sizes_sorted[dissolved_pcts.index(max_pct)]
        notes.append(
            f"No particle size in the given range achieves target "
            f"{target_dissolved_pct_at_60min:.1f}% at 60 min. "
            f"Maximum achieved: {max_pct:.1f}% at {best_sz} um."
        )
        # Fallback: use size with highest dissolution
        best_idx = dissolved_pcts.index(max_pct)
        optimal_size_um = sizes_sorted[best_idx]
        optimal_pct = max_pct

    # Dose number assessment
    dose_number = dose_mg / (solubility_mg_mL * 250.0)  # 250 mL standard volume
    if dose_number > 1.0:
        notes.append(
            f"High dose number ({dose_number:.1f}): solubility-limited dissolution likely. "
            "Consider solubility enhancement strategies."
        )

    if optimal_pct >= 85.0:
        notes.append(
            f"Optimal size {optimal_size_um} um achieves rapid dissolution (>85% at 60 min)."
        )

    return ParticleSizeOptResult(
        drug_name=drug_name,
        target_dissolved_pct_at_60min=target_dissolved_pct_at_60min,
        solubility_mg_mL=solubility_mg_mL,
        dose_mg=dose_mg,
        particle_sizes_um=sizes_sorted,
        dissolved_pct_at_60min=dissolved_pcts,
        optimal_size_um=optimal_size_um,
        optimal_dissolved_pct=optimal_pct,
        notes=notes,
    )


def biopharmaceutics_classification(
    dissolved_pct: float,
    fa_from_permeability: float,
) -> str:
    """Predict BCS class from dissolution and permeability data.

    Parameters
    ----------
    dissolved_pct:
        Percent dissolved at 60 minutes (0–100).
    fa_from_permeability:
        Fraction absorbed from permeability data (0–1).
        High permeability: fa >= 0.85 (BCS high Peff criterion).

    Returns
    -------
    BCS class string: "BCS Class I", "BCS Class II", "BCS Class III",
    or "BCS Class IV".

    Notes
    -----
    - dissolved_pct > 85% at 60 min → high dissolution
    - fa_from_permeability >= 0.85 → high permeability
    """
    if not (0.0 <= dissolved_pct <= 100.0):
        raise ValueError("dissolved_pct must be in [0, 100]")
    if not (0.0 <= fa_from_permeability <= 1.0):
        raise ValueError("fa_from_permeability must be in [0, 1]")

    high_dissolution = dissolved_pct > 85.0
    high_permeability = fa_from_permeability >= 0.85

    if high_dissolution and high_permeability:
        return "BCS Class I"
    elif not high_dissolution and high_permeability:
        return "BCS Class II"
    elif high_dissolution and not high_permeability:
        return "BCS Class III"
    else:
        return "BCS Class IV"
