"""Michaelis-Menten enzyme kinetics and saturation at therapeutic concentrations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MMPKResult:
    """Result from Michaelis-Menten PK simulation."""

    drug_name: str
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    cmax: float = 0.0
    tmax_h: float = 0.0
    auc: float = 0.0
    t_half_initial_h: float = 0.0
    saturation_at_cmax: float = 0.0
    notes: str = ""


def mm_velocity(
    substrate_conc_uM: float,
    vmax_nmol_min: float,
    km_uM: float,
) -> float:
    """Michaelis-Menten metabolic velocity.

    v = Vmax * S / (Km + S) [nmol/min]

    Args:
        substrate_conc_uM: Substrate concentration [uM]
        vmax_nmol_min: Maximum velocity [nmol/min]
        km_uM: Michaelis constant [uM]

    Returns:
        Metabolic rate [nmol/min]
    """
    if km_uM <= 0:
        raise ValueError("km_uM must be > 0")
    if vmax_nmol_min <= 0:
        raise ValueError("vmax_nmol_min must be > 0")
    if substrate_conc_uM < 0:
        raise ValueError("substrate_conc_uM must be >= 0")

    return vmax_nmol_min * substrate_conc_uM / (km_uM + substrate_conc_uM)


def saturation_fraction(
    substrate_conc_uM: float,
    km_uM: float,
) -> float:
    """Fraction of maximum velocity achieved.

    f_sat = S / (Km + S)

    Args:
        substrate_conc_uM: Substrate concentration [uM]
        km_uM: Michaelis constant [uM]

    Returns:
        Saturation fraction (0-1)
    """
    if km_uM <= 0:
        raise ValueError("km_uM must be > 0")
    if substrate_conc_uM < 0:
        raise ValueError("substrate_conc_uM must be >= 0")

    return substrate_conc_uM / (km_uM + substrate_conc_uM)


def apparent_km_with_inhibitor(
    km_uM: float,
    ki_uM: float,
    inhibitor_conc_uM: float,
    inhibition_type: str = "competitive",
) -> float:
    """Apparent Km in presence of inhibitor.

    competitive: Km_app = Km * (1 + I/Ki)
    uncompetitive: Km_app = Km / (1 + I/Ki)
    mixed: Km_app = Km * (1 + I/Ki) / (1 + I/(alpha*Ki)) where alpha=2

    Args:
        km_uM: Intrinsic Michaelis constant [uM]
        ki_uM: Inhibitor constant [uM]
        inhibitor_conc_uM: Inhibitor concentration [uM]
        inhibition_type: 'competitive', 'uncompetitive', or 'mixed'

    Returns:
        Apparent Km [uM]
    """
    if km_uM <= 0:
        raise ValueError("km_uM must be > 0")
    if ki_uM <= 0:
        raise ValueError("ki_uM must be > 0")
    if inhibitor_conc_uM < 0:
        raise ValueError("inhibitor_conc_uM must be >= 0")

    ratio = inhibitor_conc_uM / ki_uM

    if inhibition_type == "competitive":
        return km_uM * (1.0 + ratio)
    elif inhibition_type == "uncompetitive":
        return km_uM / (1.0 + ratio)
    elif inhibition_type == "mixed":
        alpha = 2.0
        return km_uM * (1.0 + ratio) / (1.0 + ratio / alpha)
    else:
        raise ValueError(
            f"inhibition_type must be 'competitive', 'uncompetitive', or 'mixed', got '{inhibition_type}'"
        )


def dose_proportionality_mm(
    doses_mg: list[float],
    vd_L: float = 50.0,
    km_mg_L: float = 10.0,
    vmax_mg_h: float = 100.0,
    f_oral: float = 1.0,
) -> list[dict]:
    """Assess dose proportionality under Michaelis-Menten elimination.

    For each dose:
        Cmax ≈ dose * f_oral / Vd
        CL_eff = Vmax / (Km + Cmax)  [L/h]
        AUC_ss ≈ dose / CL_eff  (simplified)
        dose_normalized_auc = AUC / dose

    Args:
        doses_mg: List of doses to evaluate [mg]
        vd_L: Volume of distribution [L]
        km_mg_L: Michaelis constant [mg/L]
        vmax_mg_h: Maximum elimination rate [mg/h]
        f_oral: Oral bioavailability fraction

    Returns:
        List of dicts: {dose_mg, cmax_mg_L, cl_eff_L_per_h, auc_mg_h_L, dose_normalized_auc}
    """
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be > 0")
    if vmax_mg_h <= 0:
        raise ValueError("vmax_mg_h must be > 0")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")

    results = []
    for dose in doses_mg:
        if dose <= 0:
            raise ValueError("All doses must be > 0")
        cmax = dose * f_oral / vd_L
        cl_eff = vmax_mg_h / (km_mg_L + cmax)
        auc = dose / cl_eff
        dose_normalized_auc = auc / dose
        results.append(
            {
                "dose_mg": dose,
                "cmax_mg_L": cmax,
                "cl_eff_L_per_h": cl_eff,
                "auc_mg_h_L": auc,
                "dose_normalized_auc": dose_normalized_auc,
            }
        )
    return results


def simulate_mm_pk(
    drug_name: str,
    dose_mg: float,
    vd_L: float,
    km_mg_L: float,
    vmax_mg_h: float,
    ka_per_h: float = 1.0,
    route: str = "oral",
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> MMPKResult:
    """Simulate 1-compartment PK with Michaelis-Menten elimination.

    ODEs:
        oral: dA_gut/dt = -ka * A_gut
              dCp/dt = ka * A_gut / Vd - Vmax * Cp / (Km + Cp) / Vd
        iv:   dCp/dt = -Vmax * Cp / (Km + Cp) / Vd

    Args:
        drug_name: Drug identifier
        dose_mg: Dose [mg]
        vd_L: Volume of distribution [L]
        km_mg_L: Michaelis constant [mg/L]
        vmax_mg_h: Maximum elimination rate [mg/h]
        ka_per_h: Absorption rate constant [1/h] (oral only)
        route: 'oral' or 'iv'
        t_end_h: Simulation end time [h]
        dt_h: Time step [h]

    Returns:
        MMPKResult with PK metrics
    """
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be > 0")
    if vmax_mg_h <= 0:
        raise ValueError("vmax_mg_h must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")

    n_steps = int(t_end_h / dt_h) + 1

    if route == "iv":
        cp = dose_mg / vd_L
        a_gut = 0.0
    else:
        cp = 0.0
        a_gut = dose_mg

    times_h = []
    c_plasma = []

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)
        c_plasma.append(cp)

        # MM elimination rate [mg/h/L] * Vd → but we track Cp directly
        elim_rate_mg_h = vmax_mg_h * cp / (km_mg_L + cp)

        if route == "oral":
            absorption_mg_h = ka_per_h * a_gut
            dcp_dt = absorption_mg_h / vd_L - elim_rate_mg_h / vd_L
            da_gut_dt = -ka_per_h * a_gut
            a_gut = max(0.0, a_gut + da_gut_dt * dt_h)
        else:
            dcp_dt = -elim_rate_mg_h / vd_L

        cp = max(0.0, cp + dcp_dt * dt_h)

    # PK metrics
    cmax = max(c_plasma)
    tmax_h = times_h[c_plasma.index(cmax)]

    # AUC by trapezoidal
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma[i] + c_plasma[i - 1]) * (times_h[i] - times_h[i - 1])

    # Initial apparent half-life: at early time, Cp ≈ Cmax, CL_eff = Vmax/(Km+Cmax)
    cl_eff_initial = vmax_mg_h / (km_mg_L + cmax)
    t_half_initial = math.log(2) * vd_L / cl_eff_initial if cl_eff_initial > 0 else float("inf")

    saturation = saturation_fraction(cmax, km_mg_L) if km_mg_L > 0 else 0.0

    notes = (
        f"MM-PK: Km={km_mg_L:.1f} mg/L, Vmax={vmax_mg_h:.1f} mg/h. "
        f"Saturation at Cmax: {saturation:.1%}. "
        f"Initial t1/2: {t_half_initial:.2f} h"
    )

    return MMPKResult(
        drug_name=drug_name,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        t_half_initial_h=t_half_initial,
        saturation_at_cmax=saturation,
        notes=notes,
    )
