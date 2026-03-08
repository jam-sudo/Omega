"""Phase 600 — Drug Nanocrystal Dissolution.

Model enhanced dissolution of drug nanocrystals using Ostwald-Freundlich
size-dependent solubility and Noyes-Whitney shrinking sphere kinetics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class NanocrystalDissolutionResult:
    """Result of nanocrystal dissolution simulation."""

    drug_name: str
    initial_particle_radius_nm: float
    times_min: list  # time points (minutes)
    particle_radius_nm: list  # particle radius over time (nm)
    c_dissolved_mg_mL: list  # dissolved drug concentration (mg/mL)
    c_supersaturation: list  # C_dissolved / bulk_solubility
    fraction_dissolved: list  # fraction of dose dissolved [0, 1]
    t_50pct_dissolved_min: float
    t_90pct_dissolved_min: float
    dissolution_rate_enhancement: float  # ratio vs bulk (100 µm) t_90pct
    peak_supersaturation: float
    auc_dissolved_mg_h_per_mL: float  # AUC in mg·h/mL (converted from min)
    notes: str


def _trapz_min_to_h(times_min: list, y: list) -> float:
    """Trapezoidal AUC over time in minutes, converted to h units (mg·h/mL)."""
    auc_mg_min_per_mL = sum(
        0.5 * (y[i] + y[i - 1]) * (times_min[i] - times_min[i - 1])
        for i in range(1, len(times_min))
    )
    return auc_mg_min_per_mL / 60.0


def _run_dissolution(
    dose_mg: float,
    initial_radius_nm: float,
    bulk_solubility_mg_mL: float,
    drug_density_g_cm3: float,
    diffusion_coeff_cm2_s: float,
    volume_fluid_mL: float,
    t_end_min: float,
    dt_min: float,
) -> tuple[list, list, list, list, list]:
    """Core forward Euler dissolution simulation.

    Returns (times_min, radius_nm, c_dissolved, c_supersaturation, fraction_dissolved).
    """
    # Ostwald-Freundlich constants
    gamma = 25e-3  # N/m  surface tension
    Vm = 200e-6  # m³/mol  molar volume
    R_gas = 8.314  # J/(mol·K)
    T = 310.0  # K

    # k_diss has units of 1/min: Noyes-Whitney normalized rate constant
    # k_diss = 6 * D [cm²/s] * 60 [s/min] / (r0 [cm] * rho [g/cm³] * 1000 [mg/g])
    r0_cm = initial_radius_nm * 1e-7  # nm → cm
    k_diss = 6.0 * diffusion_coeff_cm2_s * 60.0 / (r0_cm * drug_density_g_cm3 * 1000.0)

    n = max(int(round(t_end_min / dt_min)), 1)
    times = [i * dt_min for i in range(n + 1)]

    r_nm = initial_radius_nm
    c = 0.0  # mg/mL

    radius_arr = [0.0] * (n + 1)
    c_arr = [0.0] * (n + 1)
    c_super = [0.0] * (n + 1)
    frac_arr = [0.0] * (n + 1)

    radius_arr[0] = r_nm
    c_arr[0] = c
    c_super[0] = c / bulk_solubility_mg_mL if bulk_solubility_mg_mL > 0 else 0.0
    frac_arr[0] = 0.0

    max_conc = dose_mg / volume_fluid_mL  # can't exceed dose/vol

    for i in range(n):
        if r_nm <= 0.0:
            # Fully dissolved
            c = min(max_conc, c)
            radius_arr[i + 1] = 0.0
            c_arr[i + 1] = c
            c_super[i + 1] = c / bulk_solubility_mg_mL
            frac_arr[i + 1] = min(c * volume_fluid_mL / dose_mg, 1.0)
            continue

        # Size-dependent solubility (Ostwald-Freundlich)
        r_m = r_nm * 1e-9  # nm → m
        exp_arg = min(2.0 * gamma * Vm / (R_gas * T * r_m), 20.0)
        s_nano = bulk_solubility_mg_mL * math.exp(exp_arg)

        # Driving force: cannot dissolve more than available in solid phase
        c_drive = max(s_nano - c, 0.0)

        # Relative surface area factor (r/r0)^2
        r_ratio = r_nm / initial_radius_nm
        surf_factor = r_ratio * r_ratio

        # dC/dt = k_diss * (S_nano - C) * (r/r0)^2  [mg/mL/min]
        dc = k_diss * c_drive * surf_factor

        # dr/dt = -k_diss * r0 * (S_nano - C) / (6 * S_bulk)  [nm/min]
        dr = -k_diss * initial_radius_nm * c_drive / (6.0 * bulk_solubility_mg_mL)

        c_new = c + dc * dt_min
        r_new = r_nm + dr * dt_min

        # Clamp: can't exceed max concentration or current size-dependent solubility
        c_new = min(c_new, s_nano)
        c_new = min(c_new, max_conc)
        c_new = max(c_new, 0.0)
        r_new = max(r_new, 0.0)

        # If radius hits 0, fully dissolve
        if r_new <= 0.0:
            c_new = min(max_conc, c_new)
            r_new = 0.0

        c = c_new
        r_nm = r_new

        radius_arr[i + 1] = r_nm
        c_arr[i + 1] = c
        c_super[i + 1] = c / bulk_solubility_mg_mL
        frac_arr[i + 1] = min(c * volume_fluid_mL / dose_mg, 1.0)

    return times, radius_arr, c_arr, c_super, frac_arr


def simulate_nanocrystal_dissolution(
    drug_name: str,
    dose_mg: float,
    initial_radius_nm: float,
    bulk_solubility_mg_mL: float,
    drug_density_g_cm3: float = 1.2,
    diffusion_coeff_cm2_s: float = 1e-5,
    volume_fluid_mL: float = 250.0,
    t_end_min: float = 120.0,
    dt_min: float = 0.5,
) -> NanocrystalDissolutionResult:
    """Simulate dissolution of drug nanocrystals.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        Total drug dose (mg).
    initial_radius_nm : float
        Initial nanoparticle radius (nm).
    bulk_solubility_mg_mL : float
        Bulk (thermodynamic) solubility (mg/mL).
    drug_density_g_cm3 : float
        Particle density (g/cm³).
    diffusion_coeff_cm2_s : float
        Diffusion coefficient (cm²/s).
    volume_fluid_mL : float
        Volume of dissolution medium (mL).
    t_end_min : float
        Simulation end time (min).
    dt_min : float
        Forward Euler step size (min).
    """
    # --- Validation ---
    if initial_radius_nm <= 0:
        raise ValueError("initial_radius_nm must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if bulk_solubility_mg_mL <= 0:
        raise ValueError("bulk_solubility_mg_mL must be > 0")
    if volume_fluid_mL <= 0:
        raise ValueError("volume_fluid_mL must be > 0")

    times, radius_arr, c_arr, c_super, frac_arr = _run_dissolution(
        dose_mg=dose_mg,
        initial_radius_nm=initial_radius_nm,
        bulk_solubility_mg_mL=bulk_solubility_mg_mL,
        drug_density_g_cm3=drug_density_g_cm3,
        diffusion_coeff_cm2_s=diffusion_coeff_cm2_s,
        volume_fluid_mL=volume_fluid_mL,
        t_end_min=t_end_min,
        dt_min=dt_min,
    )

    # t_50, t_90
    t_50 = t_end_min
    t_90 = t_end_min
    for i, f in enumerate(frac_arr):
        if f >= 0.5 and t_50 == t_end_min:
            t_50 = times[i]
        if f >= 0.9 and t_90 == t_end_min:
            t_90 = times[i]

    # Dissolution rate enhancement vs bulk (100 µm = 100_000 nm)
    bulk_radius_nm = 100_000.0
    times_b, _, _, _, frac_b = _run_dissolution(
        dose_mg=dose_mg,
        initial_radius_nm=bulk_radius_nm,
        bulk_solubility_mg_mL=bulk_solubility_mg_mL,
        drug_density_g_cm3=drug_density_g_cm3,
        diffusion_coeff_cm2_s=diffusion_coeff_cm2_s,
        volume_fluid_mL=volume_fluid_mL,
        t_end_min=t_end_min,
        dt_min=dt_min,
    )
    t_90_bulk = t_end_min
    for i, f in enumerate(frac_b):
        if f >= 0.9:
            t_90_bulk = times_b[i]
            break

    if t_90 > 0 and t_90_bulk < t_end_min:
        enhancement = t_90_bulk / t_90
    elif t_90 < t_end_min and t_90_bulk == t_end_min:
        # nano reaches 90%, bulk does not → use ratio of t_end to t_90_nano
        enhancement = t_end_min / t_90
    else:
        enhancement = 1.0

    peak_supersaturation = max(c_super) if c_super else 0.0
    auc = _trapz_min_to_h(times, c_arr)

    notes = (
        f"Initial radius: {initial_radius_nm} nm. "
        f"Bulk solubility: {bulk_solubility_mg_mL} mg/mL. "
        f"t50={t_50:.1f} min, t90={t_90:.1f} min. "
        f"Enhancement vs 100 µm bulk: {enhancement:.2f}×."
    )

    return NanocrystalDissolutionResult(
        drug_name=drug_name,
        initial_particle_radius_nm=initial_radius_nm,
        times_min=times,
        particle_radius_nm=radius_arr,
        c_dissolved_mg_mL=c_arr,
        c_supersaturation=c_super,
        fraction_dissolved=frac_arr,
        t_50pct_dissolved_min=t_50,
        t_90pct_dissolved_min=t_90,
        dissolution_rate_enhancement=enhancement,
        peak_supersaturation=peak_supersaturation,
        auc_dissolved_mg_h_per_mL=auc,
        notes=notes,
    )


__all__ = [
    "NanocrystalDissolutionResult",
    "simulate_nanocrystal_dissolution",
]
