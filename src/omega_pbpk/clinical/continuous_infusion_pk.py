"""Continuous IV infusion PK model.

Simulates loading dose + maintenance infusion, steady-state concentration,
and infusion rate optimization.

ODE: dC/dt = R/Vd - (CL/Vd)*C
Analytical CSS = R / CL
Time to 95% SS: t = -ln(0.05) * Vd/CL
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class InfusionResult:
    drug_name: str
    target_css_mg_L: float
    infusion_rate_mg_per_h: float
    loading_dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    css_achieved_mg_L: float = 0.0
    time_to_95_pct_ss_h: float = 0.0
    fluctuation_pct: float = 0.0
    notes: str = ""


def simulate_continuous_infusion(
    drug_name: str,
    infusion_rate_mg_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    loading_dose_mg: float = 0.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> InfusionResult:
    """Simulate plasma concentration profile during continuous IV infusion.

    Uses forward Euler integration of dC/dt = R/Vd - (CL/Vd)*C.
    """
    if infusion_rate_mg_per_h <= 0:
        raise ValueError(f"infusion_rate_mg_per_h must be > 0, got {infusion_rate_mg_per_h}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)
    c = loading_dose_mg / vd_L if loading_dose_mg > 0 else 0.0

    times_h: list[float] = []
    c_plasma: list[float] = []

    t = 0.0
    while t <= t_end_h + 1e-9:
        times_h.append(round(t, 6))
        c_plasma.append(c)
        dc_dt = infusion_rate_mg_per_h / vd_L - ke * c
        c = c + dc_dt * dt_h
        if c < 0:
            c = 0.0
        t += dt_h

    # Analytical CSS and time to 95% SS
    css_analytical = infusion_rate_mg_per_h / cl_L_per_h
    time_to_95_ss = -math.log(0.05) / ke  # 3 * t½

    css_achieved = c_plasma[-1] if c_plasma else 0.0

    notes = (
        f"Analytical CSS={css_analytical:.3f} mg/L; "
        f"ke={ke:.4f} 1/h; "
        f"t½={math.log(2) / ke:.2f} h; "
        f"t95%SS(analytical)={time_to_95_ss:.2f} h"
    )

    return InfusionResult(
        drug_name=drug_name,
        target_css_mg_L=css_analytical,
        infusion_rate_mg_per_h=infusion_rate_mg_per_h,
        loading_dose_mg=loading_dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        css_achieved_mg_L=css_achieved,
        time_to_95_pct_ss_h=time_to_95_ss,
        fluctuation_pct=0.0,
        notes=notes,
    )


def calculate_infusion_rate(
    target_css_mg_L: float,
    cl_L_per_h: float,
) -> float:
    """Infusion rate = Css * CL  (steady-state condition)."""
    if target_css_mg_L <= 0:
        raise ValueError(f"target_css_mg_L must be > 0, got {target_css_mg_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    return target_css_mg_L * cl_L_per_h


def calculate_loading_dose(
    target_css_mg_L: float,
    vd_L: float,
    cl_L_per_h: float | None = None,
) -> float:
    """Loading dose = Css * Vd  (to immediately achieve target concentration)."""
    if target_css_mg_L <= 0:
        raise ValueError(f"target_css_mg_L must be > 0, got {target_css_mg_L}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    return target_css_mg_L * vd_L


def optimize_infusion_regimen(
    drug_name: str,
    target_css_mg_L: float,
    cl_L_per_h: float,
    vd_L: float,
    target_time_to_ss_h: float = 2.0,
    t_end_h: float = 12.0,
) -> InfusionResult:
    """Compute optimal infusion rate and loading dose to reach target CSS quickly.

    Loading dose = target_css * Vd (achieves target immediately).
    Infusion rate = target_css * CL (maintains target at steady state).
    """
    if target_css_mg_L <= 0:
        raise ValueError(f"target_css_mg_L must be > 0, got {target_css_mg_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    infusion_rate = calculate_infusion_rate(target_css_mg_L, cl_L_per_h)
    loading_dose = calculate_loading_dose(target_css_mg_L, vd_L)

    result = simulate_continuous_infusion(
        drug_name=drug_name,
        infusion_rate_mg_per_h=infusion_rate,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        loading_dose_mg=loading_dose,
        t_end_h=t_end_h,
        dt_h=0.05,
    )

    # Override target_css to explicitly reflect what was requested
    return InfusionResult(
        drug_name=result.drug_name,
        target_css_mg_L=target_css_mg_L,
        infusion_rate_mg_per_h=result.infusion_rate_mg_per_h,
        loading_dose_mg=result.loading_dose_mg,
        times_h=result.times_h,
        c_plasma_mg_L=result.c_plasma_mg_L,
        css_achieved_mg_L=result.css_achieved_mg_L,
        time_to_95_pct_ss_h=result.time_to_95_pct_ss_h,
        fluctuation_pct=result.fluctuation_pct,
        notes=result.notes + f"; optimized for target={target_css_mg_L} mg/L",
    )
