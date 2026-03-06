"""In-vitro dissolution modeling for biopharmaceutics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "DissolutionResult",
    "simulate_noyes_whitney",
    "simulate_weibull",
    "compare_dissolution_models",
]


@dataclass
class DissolutionResult:
    drug_name: str
    dissolution_model: str
    model_params: dict
    times_h: list
    pct_dissolved: list
    t80_h: Optional[float]
    t90_h: Optional[float]
    dissolution_rate_pct_per_h: float
    sink_conditions: bool


def _noyes_whitney(
    cs_mg_mL: float,
    c_t_mg_mL: float,
    surface_area_cm2: float,
    h_cm: float,
    D_cm2_per_s: float,
    V_mL: float,
) -> float:
    """Dissolution rate dM/dt = D*A*(Cs-C)/h (mg/s)."""
    driving_force = cs_mg_mL - c_t_mg_mL
    if driving_force <= 0.0:
        return 0.0
    # D [cm2/s] * A [cm2] * (Cs-C) [mg/mL = mg/cm3] / h [cm] -> mg/s
    return D_cm2_per_s * surface_area_cm2 * driving_force / h_cm


def _interpolate_time_at_pct(times: np.ndarray, pcts: np.ndarray, target: float) -> Optional[float]:
    """Linear interpolation to find time when pct_dissolved crosses target."""
    for i in range(1, len(pcts)):
        if pcts[i - 1] <= target <= pcts[i]:
            if pcts[i] == pcts[i - 1]:
                return float(times[i - 1])
            frac = (target - pcts[i - 1]) / (pcts[i] - pcts[i - 1])
            return float(times[i - 1] + frac * (times[i] - times[i - 1]))
    return None


def simulate_noyes_whitney(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    particle_radius_cm: float = 0.005,
    density_g_cm3: float = 1.2,
    diffusivity_cm2_per_s: float = 1e-5,
    viscosity_poise: float = 0.01,
    dissolution_volume_mL: float = 900.0,
    t_end_h: float = 2.0,
    dt_s: float = 1.0,
) -> DissolutionResult:
    """Simulate dissolution of spherical particles using Noyes-Whitney / Euler method."""
    # Diffusion layer thickness: h = D / k_convection, k_convection = D / (r * 0.01)
    k_convection = diffusivity_cm2_per_s / (particle_radius_cm * 0.01)
    h_cm = diffusivity_cm2_per_s / k_convection  # = particle_radius_cm * 0.01

    # Initial particle properties
    # Volume of one sphere: (4/3)*pi*r^3 cm^3
    # density in g/cm^3 -> mass per particle in g -> convert to mg
    volume_per_particle_cm3 = (4.0 / 3.0) * math.pi * particle_radius_cm ** 3
    mass_per_particle_mg = volume_per_particle_cm3 * density_g_cm3 * 1000.0  # g->mg

    # Number of particles
    n_particles = dose_mg / mass_per_particle_mg

    t_end_s = t_end_h * 3600.0
    n_steps = int(t_end_s / dt_s)

    times_s = []
    pcts = []

    remaining_mass_mg = dose_mg
    dissolved_mass_mg = 0.0
    sink = solubility_mg_mL * dissolution_volume_mL  # total sink capacity mg

    initial_rate_sum = 0.0
    initial_rate_count = 0

    for step in range(n_steps + 1):
        t_s = step * dt_s
        times_s.append(t_s)
        pct = dissolved_mass_mg / dose_mg * 100.0
        pcts.append(pct)

        if dissolved_mass_mg >= 0.99 * dose_mg:
            break

        if remaining_mass_mg <= 0.0:
            break

        # Current concentration in dissolution medium
        c_t = dissolved_mass_mg / dissolution_volume_mL  # mg/mL

        # Sink condition: concentration << solubility (< 10% of Cs)
        sink_cond = c_t < 0.1 * solubility_mg_mL

        # Current radius from remaining mass
        # remaining mass = n_particles * density * (4/3)*pi*r^3 * 1000
        r_current_cm3 = remaining_mass_mg / (n_particles * density_g_cm3 * 1000.0)
        r_current_cm = (r_current_cm3 / ((4.0 / 3.0) * math.pi)) ** (1.0 / 3.0)

        # Surface area of all particles
        surface_area_cm2 = n_particles * 4.0 * math.pi * r_current_cm ** 2

        # Dissolution rate
        rate_mg_per_s = _noyes_whitney(
            cs_mg_mL=solubility_mg_mL,
            c_t_mg_mL=c_t,
            surface_area_cm2=surface_area_cm2,
            h_cm=h_cm,
            D_cm2_per_s=diffusivity_cm2_per_s,
            V_mL=dissolution_volume_mL,
        )

        # Euler step
        delta_mg = rate_mg_per_s * dt_s
        delta_mg = min(delta_mg, remaining_mass_mg)
        dissolved_mass_mg += delta_mg
        remaining_mass_mg -= delta_mg

        # Track initial rate (first 10 steps)
        if step < 10:
            rate_pct_per_s = (delta_mg / dose_mg * 100.0) / dt_s
            initial_rate_sum += rate_pct_per_s
            initial_rate_count += 1

    times_h_arr = np.array(times_s) / 3600.0
    pcts_arr = np.array(pcts)

    t80 = _interpolate_time_at_pct(times_h_arr, pcts_arr, 80.0)
    t90 = _interpolate_time_at_pct(times_h_arr, pcts_arr, 90.0)

    initial_rate_pct_per_s = initial_rate_sum / max(initial_rate_count, 1)
    initial_rate_pct_per_h = initial_rate_pct_per_s * 3600.0

    # Sink condition based on final concentration
    final_c = (dose_mg - remaining_mass_mg) / dissolution_volume_mL
    sink_conditions = final_c < 0.3 * solubility_mg_mL

    return DissolutionResult(
        drug_name=drug_name,
        dissolution_model="noyes_whitney",
        model_params={
            "solubility_mg_mL": solubility_mg_mL,
            "particle_radius_cm": particle_radius_cm,
            "density_g_cm3": density_g_cm3,
            "diffusivity_cm2_per_s": diffusivity_cm2_per_s,
            "h_cm": h_cm,
            "dissolution_volume_mL": dissolution_volume_mL,
        },
        times_h=times_h_arr.tolist(),
        pct_dissolved=pcts_arr.tolist(),
        t80_h=t80,
        t90_h=t90,
        dissolution_rate_pct_per_h=initial_rate_pct_per_h,
        sink_conditions=sink_conditions,
    )


def simulate_weibull(
    drug_name: str,
    dose_mg: float,
    td_h: float = 0.5,
    beta: float = 1.5,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> DissolutionResult:
    """Simulate dissolution using Weibull model: F(t) = 100*(1 - exp(-(t/td)^beta))."""
    times_h = np.arange(0.0, t_end_h + dt_h, dt_h)
    pcts = 100.0 * (1.0 - np.exp(-((times_h / td_h) ** beta)))
    pcts = np.clip(pcts, 0.0, 100.0)

    # Analytical t80 and t90
    # F(t) = 80 -> (t/td)^beta = -ln(0.2) -> t = td * (-ln(0.2))^(1/beta)
    t80 = td_h * ((-math.log(0.2)) ** (1.0 / beta))
    t90 = td_h * ((-math.log(0.1)) ** (1.0 / beta))

    # Only report if within simulation range
    t80_out: Optional[float] = t80 if t80 <= t_end_h else None
    t90_out: Optional[float] = t90 if t90 <= t_end_h else None

    # Initial rate: derivative at t->0+ for beta>=1 or numeric for beta<1
    # d(F)/dt = 100 * (beta/td) * (t/td)^(beta-1) * exp(-(t/td)^beta)
    # For beta >= 1, rate at t=0 is 0 (or infinite for beta<1); use first two steps numerically
    if len(pcts) >= 2:
        initial_rate = (pcts[1] - pcts[0]) / dt_h
    else:
        initial_rate = 0.0

    return DissolutionResult(
        drug_name=drug_name,
        dissolution_model="weibull",
        model_params={"td_h": td_h, "beta": beta, "dose_mg": dose_mg},
        times_h=times_h.tolist(),
        pct_dissolved=pcts.tolist(),
        t80_h=t80_out,
        t90_h=t90_out,
        dissolution_rate_pct_per_h=float(initial_rate),
        sink_conditions=True,
    )


def compare_dissolution_models(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float = 0.1,
) -> dict:
    """Compare Noyes-Whitney and Weibull dissolution models.

    Returns
    -------
    dict[str, DissolutionResult]
        Keys: 'noyes_whitney', 'weibull_fast', 'weibull_slow'
    """
    nw = simulate_noyes_whitney(
        drug_name=drug_name,
        dose_mg=dose_mg,
        solubility_mg_mL=solubility_mg_mL,
    )
    weibull_fast = simulate_weibull(
        drug_name=drug_name,
        dose_mg=dose_mg,
        td_h=0.5,
        beta=2.0,
    )
    weibull_slow = simulate_weibull(
        drug_name=drug_name,
        dose_mg=dose_mg,
        td_h=0.5,
        beta=0.8,
    )
    return {
        "noyes_whitney": nw,
        "weibull_fast": weibull_fast,
        "weibull_slow": weibull_slow,
    }
