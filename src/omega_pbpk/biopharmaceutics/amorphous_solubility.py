"""Amorphous solubility advantage and spring-and-parachute supersaturation simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "AmorphousSolubilityResult",
    "SPSimResult",
    "amorphous_solubility_advantage",
    "spring_and_parachute_sim",
]

# Universal gas constant [J/(mol·K)]
_R = 8.314


@dataclass(frozen=True)
class AmorphousSolubilityResult:
    """Result from amorphous solubility advantage calculation."""

    drug_name: str
    tm_C: float
    tg_C: float
    temperature_C: float
    delta_hfus_J_per_mol: float
    sa_sc_ratio: float
    notes: str


@dataclass
class SPSimResult:
    """Result from spring-and-parachute ODE simulation."""

    drug_name: str
    times_h: list[float]
    c_supersaturated: list[float]
    c_cryst_sed: list[float]
    cmax: float
    auc: float
    f_precipitated: float
    notes: str


def amorphous_solubility_advantage(
    drug_name: str,
    tm_C: float,
    tg_C: float,
    logP: float,  # noqa: N803 — kept for API clarity
    temperature_C: float = 37.0,
) -> AmorphousSolubilityResult:
    """Calculate amorphous vs crystalline solubility ratio (Murdande 2010 approach).

    Uses the simplified Murdande equation:
        ln(Sa/Sc) = ΔHfus * (Tm - T) / (R * T * Tm)

    where ΔHfus is estimated from the melting point as:
        ΔHfus ≈ 50 * Tm  [J/mol]  (typical pharmaceutical solids)

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    tm_C:
        Melting temperature in °C. Must be > 0 °C (i.e. solid at room temp).
    tg_C:
        Glass transition temperature in °C. Used for notes/classification only.
    logP:
        Octanol-water partition coefficient. Retained for API signature
        (influences amorphous behaviour in practice, but Murdande equation
        does not use it directly).
    temperature_C:
        Simulation temperature in °C. Defaults to body temperature (37 °C).

    Returns
    -------
    AmorphousSolubilityResult
        Contains the solubility advantage ratio Sa/Sc and supporting data.

    Raises
    ------
    ValueError
        If any thermodynamic temperature would be non-positive, or if
        temperature_C >= tm_C (drug would be molten).
    """
    if tm_C <= 0:
        raise ValueError(f"tm_C must be > 0 °C, got {tm_C}")
    if temperature_C <= 0:
        raise ValueError(f"temperature_C must be > 0 °C, got {temperature_C}")
    if temperature_C >= tm_C:
        raise ValueError(
            f"temperature_C ({temperature_C} °C) must be less than tm_C ({tm_C} °C); "
            "drug would be above its melting point."
        )

    T = temperature_C + 273.15  # K
    Tm = tm_C + 273.15  # K

    # ΔHfus estimation [J/mol] — Walden's rule approximation for pharma solids
    delta_hfus = 50.0 * Tm  # J/mol

    # Simplified Murdande (2010): ln(Sa/Sc) = ΔHfus*(Tm-T)/(R*T*Tm)
    ln_ratio = delta_hfus * (Tm - T) / (_R * T * Tm)
    sa_sc_ratio = math.exp(ln_ratio)

    # Build informative notes
    tg_margin = tm_C - tg_C
    stability_note = (
        "Good glass-forming stability (Tg/Tm > 0.7)"
        if (tg_C + 273.15) / Tm >= 0.7
        else "Moderate glass-forming stability"
    )
    notes = (
        f"ΔHfus={delta_hfus:.0f} J/mol; ln(Sa/Sc)={ln_ratio:.3f}; "
        f"Tg/Tm={((tg_C+273.15)/Tm):.3f}; Tm-Tg={tg_margin:.1f}°C; "
        f"logP={logP:.2f}; {stability_note}"
    )

    return AmorphousSolubilityResult(
        drug_name=drug_name,
        tm_C=tm_C,
        tg_C=tg_C,
        temperature_C=temperature_C,
        delta_hfus_J_per_mol=delta_hfus,
        sa_sc_ratio=sa_sc_ratio,
        notes=notes,
    )


def spring_and_parachute_sim(
    drug_name: str,
    crystalline_sol_mg_mL: float,
    amorphous_sol_mg_mL: float,
    k_precipitation_per_h: float,
    k_dissolution_per_h: float,
    dose_mg: float,
    volume_GI_mL: float,  # noqa: N803
    t_end_h: float = 6.0,
    dt_h: float = 0.01,
) -> SPSimResult:
    """Simulate spring-and-parachute supersaturation/precipitation kinetics.

    ODE system (forward Euler):
        dC_super/dt = k_dis * C_cryst_sed / V
                      - k_precip * (C_super - C_crystal) * [C_super > C_crystal]
        dC_cryst_sed/dt = -k_dis * C_cryst_sed / V
                          + k_precip * (C_super - C_crystal) * [C_super > C_crystal]

    Initial conditions:
        C_super[0]    = min(dose_mg / volume_GI_mL, amorphous_sol_mg_mL)
        C_cryst_sed[0] = dose_mg / volume_GI_mL - C_super[0]   (≥ 0)

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    crystalline_sol_mg_mL:
        Equilibrium crystalline solubility [mg/mL].
    amorphous_sol_mg_mL:
        Amorphous (supersaturated) solubility limit [mg/mL].
    k_precipitation_per_h:
        First-order precipitation rate constant [1/h].
    k_dissolution_per_h:
        First-order re-dissolution rate constant [1/h] (from sediment).
    dose_mg:
        Administered dose [mg].
    volume_GI_mL:
        Volume of GI fluid [mL].
    t_end_h:
        Simulation end time [h].
    dt_h:
        Forward Euler time step [h].

    Returns
    -------
    SPSimResult
        Time-course, Cmax, AUC, fraction precipitated, and notes.

    Raises
    ------
    ValueError
        On non-positive inputs or logical inconsistencies.
    """
    if crystalline_sol_mg_mL <= 0:
        raise ValueError(f"crystalline_sol_mg_mL must be > 0, got {crystalline_sol_mg_mL}")
    if amorphous_sol_mg_mL <= 0:
        raise ValueError(f"amorphous_sol_mg_mL must be > 0, got {amorphous_sol_mg_mL}")
    if amorphous_sol_mg_mL < crystalline_sol_mg_mL:
        raise ValueError(
            "amorphous_sol_mg_mL must be >= crystalline_sol_mg_mL; "
            f"got {amorphous_sol_mg_mL} < {crystalline_sol_mg_mL}"
        )
    if k_precipitation_per_h < 0:
        raise ValueError(f"k_precipitation_per_h must be >= 0, got {k_precipitation_per_h}")
    if k_dissolution_per_h < 0:
        raise ValueError(f"k_dissolution_per_h must be >= 0, got {k_dissolution_per_h}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if volume_GI_mL <= 0:
        raise ValueError(f"volume_GI_mL must be > 0, got {volume_GI_mL}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    C_crystal = crystalline_sol_mg_mL
    V = volume_GI_mL
    kp = k_precipitation_per_h
    kd = k_dissolution_per_h

    # Initial conditions
    C_super_0 = min(dose_mg / V, amorphous_sol_mg_mL)
    C_sed_0 = max(dose_mg / V - C_super_0, 0.0)

    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)
    times: list[float] = [0.0]
    c_sup: list[float] = [C_super_0]
    c_sed: list[float] = [C_sed_0]

    c_s = C_super_0
    c_d = C_sed_0

    for i in range(n_steps):
        t_cur = (i + 1) * dt_h

        # Precipitation flux (only when supersaturated above crystalline)
        precip_flux = kp * (c_s - C_crystal) if c_s > C_crystal else 0.0

        # Re-dissolution flux from sediment
        dis_flux = kd * c_d / V

        dc_s = dis_flux - precip_flux
        dc_d = precip_flux - dis_flux  # note: c_d in mg/mL units of sediment conc equiv

        c_s_new = max(c_s + dc_s * dt_h, 0.0)
        c_d_new = max(c_d + dc_d * dt_h, 0.0)

        c_s = c_s_new
        c_d = c_d_new

        times.append(min(t_cur, t_end_h))
        c_sup.append(c_s)
        c_sed.append(c_d)

    cmax = max(c_sup)

    # AUC via trapezoidal rule (manual)
    auc = 0.0
    for i in range(len(times) - 1):
        auc += 0.5 * (c_sup[i] + c_sup[i + 1]) * (times[i + 1] - times[i])

    # Fraction precipitated = (mass in sediment at end) / (dose / V in solution equiv)
    # Total mass conservation: C_super + C_sed ≈ initial total conc
    total_initial = C_super_0 + C_sed_0  # mg/mL equiv
    f_precip = c_sed[-1] / total_initial if total_initial > 0 else 0.0
    f_precip = min(max(f_precip, 0.0), 1.0)

    notes = (
        f"C_super_0={C_super_0:.3f} mg/mL; C_sed_0={C_sed_0:.3f} mg/mL; "
        f"kp={kp}/h; kd={kd}/h; Ccryst={C_crystal} mg/mL"
    )

    return SPSimResult(
        drug_name=drug_name,
        times_h=times,
        c_supersaturated=c_sup,
        c_cryst_sed=c_sed,
        cmax=cmax,
        auc=auc,
        f_precipitated=f_precip,
        notes=notes,
    )
