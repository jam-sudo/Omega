"""Drug crystallization kinetics in supersaturated solutions — Phase 477."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "CrystallizationResult",
    "nucleation_rate",
    "crystal_growth_rate",
    "simulate_crystallization",
    "induction_time_estimate",
    "estimate_precipitation_risk",
]

# Physical constants
_R = 8.314  # J/(mol·K)
_k_B = 1.380649e-23  # J/K
_N_A = 6.02214076e23  # 1/mol
_PI = math.pi

# Molar volume of typical small drug (cm^3/mol)
_V_M_CM3 = 200.0


@dataclass
class CrystallizationResult:
    """Result of a crystallization kinetics simulation.

    Attributes
    ----------
    times_h : list[float]
        Simulation time points (h).
    conc_dissolved_mg_mL : list[float]
        Dissolved drug concentration at each time point (mg/mL).
    conc_crystallized_mg_mL : list[float]
        Crystallized (solid) drug concentration at each time point (mg/mL).
    supersaturation : list[float]
        Supersaturation ratio S = C_dissolved / C_equilibrium at each time point.
    induction_time_h : float
        Estimated induction time before significant nucleation (h). NaN if no
        crystallization occurs.
    t90_crystallized_h : float
        Time to 90% crystallization of the supersaturated fraction (h). NaN if not
        reached.
    notes : str
        Summary notes about the simulation outcome.
    """

    times_h: list[float] = field(default_factory=list)
    conc_dissolved_mg_mL: list[float] = field(default_factory=list)
    conc_crystallized_mg_mL: list[float] = field(default_factory=list)
    supersaturation: list[float] = field(default_factory=list)
    induction_time_h: float = float("nan")
    t90_crystallized_h: float = float("nan")
    notes: str = ""


def nucleation_rate(
    supersaturation_ratio: float,
    surface_energy_mJ_m2: float = 5.0,
    temperature_C: float = 37.0,
) -> float:
    """Classical nucleation theory: rate of nucleus formation.

    Uses CNT with a spherical nucleus assumption:
        J = A * exp(-B / ln(S)^2)
    where B is proportional to surface_energy^3 / T^3.

    Parameters
    ----------
    supersaturation_ratio : float
        S = C / C_eq  (must be > 0; S <= 1 gives zero rate).
    surface_energy_mJ_m2 : float
        Solid-liquid interfacial energy (mJ/m²). Higher values → higher barrier.
    temperature_C : float
        Temperature (°C).

    Returns
    -------
    float
        Nucleation rate (nuclei/mL/h). Returns 0 when S <= 1.
    """
    if supersaturation_ratio <= 0:
        raise ValueError("supersaturation_ratio must be > 0")
    if surface_energy_mJ_m2 < 0:
        raise ValueError("surface_energy_mJ_m2 must be >= 0")

    if supersaturation_ratio <= 1.0:
        return 0.0

    T_K = temperature_C + 273.15
    # Convert mJ/m² → J/m²
    gamma = surface_energy_mJ_m2 * 1e-3  # J/m²
    # V_m in m³/mol
    V_m = _V_M_CM3 * 1e-6  # m³/mol

    ln_S = math.log(supersaturation_ratio)

    # Classical nucleation energy barrier (dimensionless)
    # ΔG* / kT = 16π γ³ V_m² N_A / (3 (_R T)³ (ln S)²)
    # Re-expressed using molecular volume v_m = V_m / N_A:
    # ΔG* / kT = 16π γ³ v_m² / (3 (k_B T)³ (ln S)²)
    v_m = V_m / _N_A  # m³/molecule
    numerator = 16.0 * _PI * (gamma**3) * (v_m**2)
    denominator = 3.0 * (_k_B * T_K) ** 3 * (ln_S**2)
    delta_g_star_kT = numerator / denominator

    # Pre-exponential (empirical, scaled to nuclei/mL/h)
    A = 1.0e10  # nuclei/mL/h (order of magnitude from literature)
    rate = A * math.exp(-min(delta_g_star_kT, 700.0))  # clamp to avoid underflow
    return rate


def crystal_growth_rate(
    supersaturation_ratio: float,
    kg: float = 0.1,
    g_exponent: float = 2.0,
) -> float:
    """Power-law crystal growth rate model.

    RG = kg * (S - 1)^g  for S > 1, else 0.

    Parameters
    ----------
    supersaturation_ratio : float
        S = C / C_eq (must be > 0).
    kg : float
        Growth rate constant (mg/mL/h per unit driving force^g).
    g_exponent : float
        Growth rate exponent (typically 1–3).

    Returns
    -------
    float
        Crystal growth rate (mg/mL/h). Returns 0 when S <= 1.
    """
    if supersaturation_ratio <= 0:
        raise ValueError("supersaturation_ratio must be > 0")
    if kg < 0:
        raise ValueError("kg must be >= 0")
    if g_exponent < 0:
        raise ValueError("g_exponent must be >= 0")

    if supersaturation_ratio <= 1.0:
        return 0.0

    driving_force = supersaturation_ratio - 1.0
    return kg * (driving_force**g_exponent)


def simulate_crystallization(
    initial_conc_mg_mL: float,
    equilibrium_solubility_mg_mL: float,
    t_end_h: float = 2.0,
    dt_h: float = 0.01,
    surface_energy_mJ_m2: float = 5.0,
    kg: float = 0.1,
    temperature_C: float = 37.0,
) -> CrystallizationResult:
    """Simulate drug crystallization from a supersaturated solution.

    Uses Forward Euler ODE integration of:
        dC_dissolved/dt = -(J_nuc * seed_factor + RG)  [when S > 1]
        dC_crystallized/dt = -dC_dissolved/dt

    where J_nuc is nucleation rate (nuclei/mL/h) converted to mass flux via an
    empirical seed mass factor, and RG is crystal growth rate.

    Parameters
    ----------
    initial_conc_mg_mL : float
        Initial dissolved drug concentration (mg/mL). Must be > 0.
    equilibrium_solubility_mg_mL : float
        Equilibrium (crystalline) solubility (mg/mL). Must be > 0.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration time step (h).
    surface_energy_mJ_m2 : float
        Interfacial energy (mJ/m²) for nucleation rate calculation.
    kg : float
        Crystal growth rate constant (mg/mL/h).
    temperature_C : float
        Temperature (°C).

    Returns
    -------
    CrystallizationResult
        Simulation result dataclass.
    """
    if initial_conc_mg_mL <= 0:
        raise ValueError("initial_conc_mg_mL must be > 0")
    if equilibrium_solubility_mg_mL <= 0:
        raise ValueError("equilibrium_solubility_mg_mL must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if surface_energy_mJ_m2 < 0:
        raise ValueError("surface_energy_mJ_m2 must be >= 0")
    if kg < 0:
        raise ValueError("kg must be >= 0")

    C_eq = equilibrium_solubility_mg_mL

    # If already at or below equilibrium — no crystallization
    if initial_conc_mg_mL <= C_eq:
        n_steps = max(int(t_end_h / dt_h), 1)
        times = [i * dt_h for i in range(n_steps + 1)]
        s_vals = [initial_conc_mg_mL / C_eq] * len(times)
        return CrystallizationResult(
            times_h=times,
            conc_dissolved_mg_mL=[initial_conc_mg_mL] * len(times),
            conc_crystallized_mg_mL=[0.0] * len(times),
            supersaturation=s_vals,
            induction_time_h=float("nan"),
            t90_crystallized_h=float("nan"),
            notes="S <= 1: no crystallization occurs.",
        )

    # Empirical: nucleus seed mass (mg) to convert nuclei/mL/h → mg/mL/h
    # Each nucleus ~ 4/3 π r³ ρ; with r~100 nm, ρ~1200 kg/m³ → ~5e-12 mg
    _NUCLEUS_MASS_MG = 5e-12

    times_h: list[float] = []
    conc_d: list[float] = []
    conc_c: list[float] = []
    s_list: list[float] = []

    C_d = initial_conc_mg_mL  # dissolved
    C_cr = 0.0  # crystallized

    t = 0.0
    induction_time_h = float("nan")
    induction_found = False
    n_steps = max(int(round(t_end_h / dt_h)), 1)

    # threshold for "significant" crystallization start: 1% of supersaturated excess
    excess = initial_conc_mg_mL - C_eq
    induction_threshold = 0.01 * excess if excess > 0 else 1e-9

    for _ in range(n_steps + 1):
        times_h.append(t)
        conc_d.append(C_d)
        conc_c.append(C_cr)
        S = C_d / C_eq
        s_list.append(S)

        # Detect induction time (first time crystallized > threshold)
        if not induction_found and C_cr >= induction_threshold:
            induction_time_h = t
            induction_found = True

        if t >= t_end_h:
            break

        # Rates (only when supersaturated)
        if C_d > C_eq:
            J_nuc = nucleation_rate(S, surface_energy_mJ_m2, temperature_C)
            nuc_mass_flux = J_nuc * _NUCLEUS_MASS_MG  # mg/mL/h
            R_growth = crystal_growth_rate(S, kg)
            total_rate = nuc_mass_flux + R_growth
        else:
            total_rate = 0.0

        # Forward Euler
        dC = -total_rate * dt_h
        # Clamp: cannot go below equilibrium solubility or below 0
        if C_d + dC < C_eq:
            dC = max(C_eq - C_d, -C_d)  # stop at equilibrium or 0

        C_d_new = C_d + dC
        C_cr_new = C_cr - dC  # conservation

        C_d = C_d_new
        C_cr = C_cr_new
        t = round(t + dt_h, 10)

    # Compute t90: time when 90% of supersaturated excess has crystallized
    t90 = float("nan")
    target_crystallized = 0.9 * excess
    for i, cc in enumerate(conc_c):
        if cc >= target_crystallized:
            t90 = times_h[i]
            break

    # Build notes
    final_s = conc_d[-1] / C_eq
    if final_s <= 1.05:
        notes = "Crystallization complete: dissolved concentration reached equilibrium."
    elif math.isnan(induction_time_h):
        notes = (
            "Crystallization initiated but induction threshold not clearly detected; "
            "increase simulation time or check parameters."
        )
    else:
        notes = (
            f"Induction time {induction_time_h:.3f} h; "
            f"final S={final_s:.2f} (partial crystallization)."
        )

    return CrystallizationResult(
        times_h=times_h,
        conc_dissolved_mg_mL=conc_d,
        conc_crystallized_mg_mL=conc_c,
        supersaturation=s_list,
        induction_time_h=induction_time_h,
        t90_crystallized_h=t90,
        notes=notes,
    )


def induction_time_estimate(
    supersaturation_ratio: float,
    surface_energy_mJ_m2: float = 5.0,
    temperature_C: float = 37.0,
) -> float:
    """Estimate nucleation induction time from classical nucleation theory.

    Induction time ∝ 1 / J_nuc (nucleation rate).
    Returns a practical estimate in hours.

    Parameters
    ----------
    supersaturation_ratio : float
        S = C / C_eq (must be > 1 for nucleation to occur).
    surface_energy_mJ_m2 : float
        Interfacial energy (mJ/m²).
    temperature_C : float
        Temperature (°C).

    Returns
    -------
    float
        Estimated induction time (h). Returns a large number (1e6 h) when S <= 1
        (no nucleation expected within practical time).
    """
    if supersaturation_ratio <= 0:
        raise ValueError("supersaturation_ratio must be > 0")
    if surface_energy_mJ_m2 < 0:
        raise ValueError("surface_energy_mJ_m2 must be >= 0")

    if supersaturation_ratio <= 1.0:
        return 1.0e6  # practically infinite

    J = nucleation_rate(supersaturation_ratio, surface_energy_mJ_m2, temperature_C)
    if J <= 0:
        return 1.0e6

    # Induction time ~ 1/J, scaled so that J=1e10 → t_ind ~ 1e-10 h (very fast)
    # Reference: a practical threshold of 1 nucleus/mL → t_ind = 1/J h
    t_ind = 1.0 / J  # h
    # Cap at 1e6 h to avoid numerical extremes
    return min(t_ind, 1.0e6)


def estimate_precipitation_risk(
    auc_supersaturated: float,
    equilibrium_solubility_mg_mL: float,
    t_window_h: float = 1.0,
) -> str:
    """Classify precipitation risk from AUC of supersaturated concentration.

    Risk index = (AUC_supersaturated / equilibrium_solubility) / t_window

    This represents the mean supersaturation ratio integrated over time.

    Risk thresholds:
        - low      : mean_S < 2
        - moderate : 2 <= mean_S < 5
        - high     : mean_S >= 5

    Parameters
    ----------
    auc_supersaturated : float
        AUC of dissolved concentration while supersaturated (mg/mL · h). Must be >= 0.
    equilibrium_solubility_mg_mL : float
        Equilibrium solubility (mg/mL). Must be > 0.
    t_window_h : float
        Time window over which AUC was integrated (h). Must be > 0.

    Returns
    -------
    str
        'low', 'moderate', or 'high'.
    """
    if auc_supersaturated < 0:
        raise ValueError("auc_supersaturated must be >= 0")
    if equilibrium_solubility_mg_mL <= 0:
        raise ValueError("equilibrium_solubility_mg_mL must be > 0")
    if t_window_h <= 0:
        raise ValueError("t_window_h must be > 0")

    mean_s = (auc_supersaturated / t_window_h) / equilibrium_solubility_mg_mL

    if mean_s < 2.0:
        return "low"
    if mean_s < 5.0:
        return "moderate"
    return "high"
