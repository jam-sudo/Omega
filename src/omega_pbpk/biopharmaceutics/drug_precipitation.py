"""Phase 438 — Drug Precipitation Kinetics.

Model drug precipitation from supersaturated solutions using a 3-state ODE:
  - Supersaturated dissolved drug (C_super)
  - Crystalline precipitate (C_cryst)
  - Cumulatively absorbed drug (C_absorbed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PrecipitationResult:
    """Time-course result for a drug precipitation simulation."""

    drug_name: str
    initial_conc_mg_mL: float
    equilibrium_sol_mg_mL: float
    k_precip_per_h: float
    times_h: list[float] = field(default_factory=list)
    c_super_mg_mL: list[float] = field(default_factory=list)
    c_cryst_mg_mL: list[float] = field(default_factory=list)
    c_absorbed_mg_mL: list[float] = field(default_factory=list)
    f_absorbed: float = 0.0
    f_precipitated: float = 0.0
    t50_h: float = float("inf")
    notes: str = ""


def supersaturation_index(
    actual_conc_mg_mL: float,
    equilibrium_solubility_mg_mL: float,
) -> float:
    """Return supersaturation ratio C/Cs.

    Parameters
    ----------
    actual_conc_mg_mL : float
        Actual drug concentration (mg/mL).
    equilibrium_solubility_mg_mL : float
        Equilibrium solubility (mg/mL, must be > 0).

    Returns
    -------
    float
        C/Cs ratio (1.0 = saturated, <1 = unsaturated, >1 = supersaturated).
    """
    if equilibrium_solubility_mg_mL <= 0:
        raise ValueError("equilibrium_solubility_mg_mL must be > 0")
    return actual_conc_mg_mL / equilibrium_solubility_mg_mL


def simulate_precipitation(
    drug_name: str,
    initial_conc_mg_mL: float,
    equilibrium_sol_mg_mL: float,
    k_precip_per_h: float,
    k_dissolve_per_h: float,
    absorption_rate_per_h: float,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> PrecipitationResult:
    """Simulate 3-state precipitation ODE with forward Euler integration.

    States
    ------
    C_super : supersaturated dissolved drug (mg/mL)
    C_cryst : crystalline precipitate (mg/mL-equivalent)
    C_absorbed : cumulative absorbed drug (mg/mL-equivalent)

    ODEs
    ----
    dC_super/dt   = k_dis * C_cryst - k_precip * max(0, C_super - C_eq) - k_abs * C_super
    dC_cryst/dt   = k_precip * max(0, C_super - C_eq) - k_dis * C_cryst
    dC_absorbed/dt = k_abs * C_super

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    initial_conc_mg_mL : float
        Initial supersaturated concentration (mg/mL, must be >= 0).
    equilibrium_sol_mg_mL : float
        Equilibrium solubility / C_eq (mg/mL, must be > 0).
    k_precip_per_h : float
        First-order precipitation rate constant (1/h, must be >= 0).
    k_dissolve_per_h : float
        First-order re-dissolution rate constant (1/h, must be >= 0).
    absorption_rate_per_h : float
        First-order intestinal absorption rate (1/h, must be > 0).
    t_end_h : float
        Simulation end time (h, must be > 0).
    dt_h : float
        Forward Euler time step (h, must be > 0).

    Returns
    -------
    PrecipitationResult
    """
    if initial_conc_mg_mL < 0:
        raise ValueError("initial_conc_mg_mL must be >= 0")
    if equilibrium_sol_mg_mL <= 0:
        raise ValueError("equilibrium_sol_mg_mL must be > 0")
    if k_precip_per_h < 0:
        raise ValueError("k_precip_per_h must be >= 0")
    if k_dissolve_per_h < 0:
        raise ValueError("k_dissolve_per_h must be >= 0")
    if absorption_rate_per_h <= 0:
        raise ValueError("absorption_rate_per_h must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    n_steps = max(int(t_end_h / dt_h), 1)
    actual_dt = t_end_h / n_steps

    c_eq = equilibrium_sol_mg_mL
    k_p = k_precip_per_h
    k_d = k_dissolve_per_h
    k_abs = absorption_rate_per_h

    # Initial conditions
    c_super = initial_conc_mg_mL
    c_cryst = 0.0
    c_absorbed = 0.0

    times: list[float] = [0.0]
    c_super_ts: list[float] = [c_super]
    c_cryst_ts: list[float] = [c_cryst]
    c_absorbed_ts: list[float] = [c_absorbed]

    t50_h = float("inf")
    # Track 50% precipitation: when C_cryst first reaches 50% of total initial
    # (i.e., 50% of initial_conc has precipitated as crystals)
    target_cryst_for_t50 = 0.5 * initial_conc_mg_mL

    for i in range(n_steps):
        excess = max(0.0, c_super - c_eq)

        d_super = k_d * c_cryst - k_p * excess - k_abs * c_super
        d_cryst = k_p * excess - k_d * c_cryst
        d_absorbed = k_abs * c_super

        c_super_new = max(0.0, c_super + d_super * actual_dt)
        c_cryst_new = max(0.0, c_cryst + d_cryst * actual_dt)
        c_absorbed_new = c_absorbed + d_absorbed * actual_dt

        t_now = (i + 1) * actual_dt
        times.append(t_now)
        c_super_ts.append(c_super_new)
        c_cryst_ts.append(c_cryst_new)
        c_absorbed_ts.append(c_absorbed_new)

        # Detect t50 (time to 50% precipitation)
        if math.isinf(t50_h) and c_cryst_new >= target_cryst_for_t50 and initial_conc_mg_mL > 0:
            t50_h = t_now

        c_super = c_super_new
        c_cryst = c_cryst_new
        c_absorbed = c_absorbed_new

    total_mass = initial_conc_mg_mL
    if total_mass > 0:
        f_absorbed = min(1.0, c_absorbed / total_mass)
        f_precipitated = min(1.0, c_cryst / total_mass)
    else:
        f_absorbed = 0.0
        f_precipitated = 0.0

    notes = (
        f"3-state forward Euler ODE; dt={actual_dt:.4f} h, t_end={t_end_h} h. "
        f"C_eq={c_eq} mg/mL, k_precip={k_p}/h, k_dis={k_d}/h, k_abs={k_abs}/h. "
        f"Final: C_super={c_super:.4g}, C_cryst={c_cryst:.4g}, C_absorbed={c_absorbed:.4g} mg/mL."
    )

    return PrecipitationResult(
        drug_name=drug_name,
        initial_conc_mg_mL=initial_conc_mg_mL,
        equilibrium_sol_mg_mL=c_eq,
        k_precip_per_h=k_p,
        times_h=times,
        c_super_mg_mL=c_super_ts,
        c_cryst_mg_mL=c_cryst_ts,
        c_absorbed_mg_mL=c_absorbed_ts,
        f_absorbed=f_absorbed,
        f_precipitated=f_precipitated,
        t50_h=t50_h,
        notes=notes,
    )


def supersaturation_stability(
    drug_name: str,
    solubility_mg_mL: float,
    supersaturation_ratios: list[float],
    k_precip_base_per_h: float,
) -> list[dict]:
    """Assess supersaturation stability across a range of supersaturation ratios.

    Nucleation rate scales with SR^2: k_precip = k_precip_base * SR^2.
    For each SR, compute t50 = time to lose 50% of supersaturation via precipitation.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    solubility_mg_mL : float
        Equilibrium solubility (mg/mL, must be > 0).
    supersaturation_ratios : list[float]
        List of supersaturation ratios (C/Cs, each must be >= 1).
    k_precip_base_per_h : float
        Base precipitation rate constant at SR=1 (1/h, must be > 0).

    Returns
    -------
    list[dict]
        One dict per SR with keys: sr, initial_conc_mg_mL, k_precip_effective,
        t50_h, stable (bool, t50 > 4 h).
    """
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if not supersaturation_ratios:
        raise ValueError("supersaturation_ratios must not be empty")
    if k_precip_base_per_h <= 0:
        raise ValueError("k_precip_base_per_h must be > 0")
    for sr in supersaturation_ratios:
        if sr < 1.0:
            raise ValueError(f"All supersaturation_ratios must be >= 1, got {sr}")

    results: list[dict] = []

    for sr in supersaturation_ratios:
        k_eff = k_precip_base_per_h * sr ** 2
        initial_conc = sr * solubility_mg_mL

        res = simulate_precipitation(
            drug_name=drug_name,
            initial_conc_mg_mL=initial_conc,
            equilibrium_sol_mg_mL=solubility_mg_mL,
            k_precip_per_h=k_eff,
            k_dissolve_per_h=0.0,
            absorption_rate_per_h=0.001,  # negligible absorption for stability test
            t_end_h=24.0,
            dt_h=0.05,
        )

        results.append(
            {
                "sr": sr,
                "initial_conc_mg_mL": initial_conc,
                "k_precip_effective": k_eff,
                "t50_h": res.t50_h,
                "stable": res.t50_h > 4.0,
            }
        )

    return results


