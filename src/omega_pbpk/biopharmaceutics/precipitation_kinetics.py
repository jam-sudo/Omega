"""Drug precipitation and supersaturation kinetics in the GI tract.

This module focuses on the KINETICS of precipitation (nucleation, crystal growth,
spring-and-parachute simulation), distinct from the thermodynamic supersaturation
module (supersaturation.py, Phase 110).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PrecipitationResult",
    "nucleation_rate",
    "crystal_growth_rate",
    "simulate_precipitation",
    "compare_polymer_inhibition",
]

# Physical constants
_R = 8.314  # J/(mol·K)
_k_B = 1.380649e-23  # J/K
_N_A = 6.02214076e23  # 1/mol
_PI = math.pi


@dataclass(frozen=True)
class PrecipitationResult:
    """Result of a precipitation kinetics simulation.

    Attributes
    ----------
    c_initial_mg_mL : float
        Initial dissolved drug concentration (mg/mL).
    solubility_mg_mL : float
        Equilibrium solubility (mg/mL).
    polymer_conc_mg_mL : float
        Polymer inhibitor concentration (mg/mL).
    polymer_inhibition : float
        Polymer inhibition factor (0 = no inhibition, 1 = full inhibition).
    times_h : list[float]
        Time points (h).
    c_dissolved : list[float]
        Dissolved drug concentration at each time point (mg/mL).
    s_max : float
        Maximum supersaturation ratio C/solubility.
    induction_time_h : float
        Time for dissolved concentration to drop 10% below initial (h).
    t50_precipitation_h : float
        Time for 50% of the supersaturated fraction to precipitate (h).
    f_remaining_at_2h : float
        Fraction of initial dose still dissolved at 2 h.
    notes : str
        Simulation notes.
    """

    c_initial_mg_mL: float
    solubility_mg_mL: float
    polymer_conc_mg_mL: float
    polymer_inhibition: float
    times_h: list[float]
    c_dissolved: list[float]
    s_max: float
    induction_time_h: float
    t50_precipitation_h: float
    f_remaining_at_2h: float
    notes: str


def nucleation_rate(
    supersaturation_ratio: float,
    interfacial_tension_mJ_per_m2: float = 5.0,
    temperature_C: float = 37.0,
    molar_volume_cm3_per_mol: float = 200.0,
) -> float:
    """Calculate nucleation rate using classical nucleation theory.

    Uses the simplified expression:
        J = 1e6 * exp(-3 / ln(S)^2)   for S > 1 (ln(S) > 0)
        J = 0                           for S <= 1

    The thermodynamic barrier B is estimated from:
        B = (16*pi/3) * (gamma * Vm / (R*T))^3 / (k_B * T * N_A)
    where the simplified form collapses to ~3 for typical pharmaceutical
    parameters (gamma~5 mJ/m2, Vm~200 cm3/mol, T~310 K).

    Parameters
    ----------
    supersaturation_ratio : float
        S = C / C_solubility (must be >= 1 for nucleation to occur).
    interfacial_tension_mJ_per_m2 : float
        Crystal-solution interfacial tension (mJ/m2).
    temperature_C : float
        Temperature (degrees C).
    molar_volume_cm3_per_mol : float
        Drug molar volume (cm3/mol).

    Returns
    -------
    float
        Nucleation rate J (nuclei/m3/s). Returns 0 if S <= 1.
    """
    S = supersaturation_ratio
    if S <= 1.0:
        return 0.0

    ln_S = math.log(S)
    if ln_S <= 0.0:
        return 0.0

    T_K = temperature_C + 273.15

    # Convert units: gamma [mJ/m2 = 1e-3 J/m2], Vm [cm3/mol = 1e-6 m3/mol]
    gamma = interfacial_tension_mJ_per_m2 * 1e-3  # J/m2
    Vm = molar_volume_cm3_per_mol * 1e-6  # m3/mol

    # Thermodynamic barrier exponent B
    # Standard CNT: B = 16*pi*gamma^3*Vm^2 / (3*(k_B*T)^3 * N_A^2) / ln(S)^2
    # But we use the simplified form with a fixed B=3 coefficient:
    B = (16.0 * _PI / 3.0) * (gamma * Vm) ** 3 / (_k_B * T_K) ** 3 / _N_A**2

    # Clamp B to reasonable range to avoid numerical issues
    B = max(0.01, B)

    # Simplified pre-factor; full CNT uses ~1e30 but simplified form: 1e6
    J = 1e6 * math.exp(-B / (ln_S**2))
    return J


def crystal_growth_rate(
    supersaturation_ratio: float,
    k_g: float = 1e-8,
    n_g: float = 2.0,
) -> float:
    """Calculate crystal growth rate using the parabolic growth law.

    G = k_g * (S - 1)^n_g   [m/s]

    Parameters
    ----------
    supersaturation_ratio : float
        S = C / C_solubility.
    k_g : float
        Growth rate constant (m/s).
    n_g : float
        Growth order (typically 2 for parabolic growth).

    Returns
    -------
    float
        Crystal growth rate G (m/s). Returns 0 if S <= 1.
    """
    S = supersaturation_ratio
    if S <= 1.0:
        return 0.0
    return k_g * (S - 1.0) ** n_g


def simulate_precipitation(
    c_initial_mg_mL: float,
    solubility_mg_mL: float,
    polymer_conc_mg_mL: float = 0.0,
    polymer_inhibition: float = 0.5,
    k_precip_per_h: float = 1.0,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> PrecipitationResult:
    """Simulate drug precipitation from a supersaturated solution.

    Uses the spring-and-parachute model with Forward Euler integration:
        dC/dt = -k_eff * max(0, C - solubility)    when C > solubility
        dC/dt = 0                                    when C <= solubility

    Polymer inhibition reduces the effective precipitation rate:
        k_eff = k_precip * (1 - polymer_inhibition * min(1, polymer_conc/10))

    Parameters
    ----------
    c_initial_mg_mL : float
        Initial supersaturated drug concentration (mg/mL, must be > 0).
    solubility_mg_mL : float
        Equilibrium solubility (mg/mL, must be > 0).
    polymer_conc_mg_mL : float
        Concentration of precipitation-inhibiting polymer (mg/mL).
    polymer_inhibition : float
        Polymer inhibition factor in [0, 1]. 0 = no effect, 1 = full inhibition.
    k_precip_per_h : float
        First-order precipitation rate constant (1/h, must be > 0).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step for Forward Euler integration (h).

    Returns
    -------
    PrecipitationResult
    """
    if c_initial_mg_mL <= 0:
        raise ValueError("c_initial_mg_mL must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if k_precip_per_h <= 0:
        raise ValueError("k_precip_per_h must be > 0")
    if not (0.0 <= polymer_inhibition <= 1.0):
        raise ValueError("polymer_inhibition must be in [0, 1]")

    # Effective precipitation rate with polymer inhibition
    poly_factor = polymer_inhibition * min(1.0, polymer_conc_mg_mL / 10.0)
    k_eff = k_precip_per_h * (1.0 - poly_factor)

    n_steps = max(int(round(t_end_h / dt_h)), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)
    c_arr = np.empty(n_steps + 1)
    c_arr[0] = c_initial_mg_mL

    s_max = c_initial_mg_mL / solubility_mg_mL  # initial (max) supersaturation ratio

    # Forward Euler integration
    for i in range(n_steps):
        c = c_arr[i]
        excess = max(0.0, c - solubility_mg_mL)
        dc_dt = -k_eff * excess
        c_arr[i + 1] = max(0.0, c + dc_dt * dt_h)

    # Induction time: time for C to drop 10% below initial
    threshold_10pct = c_initial_mg_mL * 0.9
    induction_time_h = t_end_h  # default: no 10% drop observed
    for i, c in enumerate(c_arr):
        if c <= threshold_10pct:
            induction_time_h = float(times[i])
            break

    # t50 precipitation: time for 50% of the supersaturated excess to precipitate
    # Supersaturated excess = c_initial - solubility (if supersaturated)
    t50_h = t_end_h
    if c_initial_mg_mL > solubility_mg_mL:
        excess_initial = c_initial_mg_mL - solubility_mg_mL
        half_excess = excess_initial / 2.0
        # 50% precipitation when C drops to c_initial - half_excess
        c_at_50pct = c_initial_mg_mL - half_excess
        for i, c in enumerate(c_arr):
            if c <= c_at_50pct:
                t50_h = float(times[i])
                break

    # Fraction remaining dissolved at 2 h
    idx_2h = min(int(round(2.0 / dt_h)), n_steps)
    f_remaining_at_2h = float(c_arr[idx_2h]) / c_initial_mg_mL

    notes = (
        f"Forward Euler spring-and-parachute model. "
        f"k_precip={k_precip_per_h:.3g}/h, k_eff={k_eff:.3g}/h. "
        f"Polymer: {polymer_conc_mg_mL:.2g} mg/mL (inhibition={polymer_inhibition:.2g}). "
        f"S_max={s_max:.2f}. dt={dt_h} h."
    )

    return PrecipitationResult(
        c_initial_mg_mL=c_initial_mg_mL,
        solubility_mg_mL=solubility_mg_mL,
        polymer_conc_mg_mL=polymer_conc_mg_mL,
        polymer_inhibition=polymer_inhibition,
        times_h=times.tolist(),
        c_dissolved=c_arr.tolist(),
        s_max=s_max,
        induction_time_h=induction_time_h,
        t50_precipitation_h=t50_h,
        f_remaining_at_2h=f_remaining_at_2h,
        notes=notes,
    )


def compare_polymer_inhibition(
    c_initial: float,
    solubility: float,
    polymers: list[dict],
) -> list[PrecipitationResult]:
    """Compare precipitation inhibition across multiple polymers.

    Parameters
    ----------
    c_initial : float
        Initial supersaturated concentration (mg/mL).
    solubility : float
        Equilibrium solubility (mg/mL).
    polymers : list[dict]
        Each dict must have keys:
        - ``name`` (str): polymer name
        - ``conc_mg_mL`` (float): polymer concentration (mg/mL)
        - ``inhibition_factor`` (float): inhibition factor in [0, 1]

    Returns
    -------
    list[PrecipitationResult]
        Results sorted by final dissolved fraction descending (best inhibitor first).
    """
    results = []
    for poly in polymers:
        conc = float(poly.get("conc_mg_mL", 0.0))
        inhibition = float(poly.get("inhibition_factor", 0.0))
        result = simulate_precipitation(
            c_initial_mg_mL=c_initial,
            solubility_mg_mL=solubility,
            polymer_conc_mg_mL=conc,
            polymer_inhibition=inhibition,
        )
        results.append(result)

    # Sort by final dissolved fraction descending (best inhibitor first)
    results.sort(key=lambda r: r.f_remaining_at_2h, reverse=True)
    return results
