"""Phase 473: Lymphocyte PK Model.

Models drug concentration in lymphocytes (immune cells) using a 2-compartment
plasma-to-lymphocyte intracellular model. Key for immunosuppressants and
antiretrovirals that concentrate in lymphocytes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LymphocytePKResult:
    """Result of lymphocyte PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    p_gp_substrate: bool
    times_h: list
    c_plasma_mg_L: list
    c_lymph_mg_L: list
    cmax_lymph_mg_L: float
    tmax_lymph_h: float
    auc_lymph_mg_h_per_L: float
    auc_plasma_mg_h_per_L: float
    car: float
    lymph_to_plasma_ratio: float
    notes: str


def _compute_car(logP: float, p_gp_substrate: bool) -> float:
    """Compute cellular accumulation ratio (CAR)."""
    if logP > 0:
        car = 10 ** (0.5 * logP)
    else:
        car = 1.0
    # Clamp to [1, 1000]
    car = max(1.0, min(1000.0, car))
    if p_gp_substrate:
        car = car / 3.0
    return car


def simulate_lymphocyte_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 5.0,
    vd_lymph_L: float = 40.0,
    p_gp_substrate: bool = False,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> LymphocytePKResult:
    """Simulate drug concentration in plasma and lymphocytes.

    Uses a 2-compartment model with forward Euler integration:
      dC_plasma/dt = -Q_lymph/vd_plasma*(C_plasma - C_lymph/CAR) - k_el*C_plasma
      dC_lymph/dt = Q_lymph/vd_lymph*(C_plasma - C_lymph/CAR)

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    logP:
        Lipophilicity (log octanol-water partition coefficient).
    cl_sys_L_per_h:
        Systemic clearance in L/h.
    vd_plasma_L:
        Plasma volume of distribution in L (default 5.0).
    vd_lymph_L:
        Lymphoid tissue volume in L (default 40.0).
    p_gp_substrate:
        Whether the drug is a P-gp substrate (reduces CAR by 3x).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Integration time step in hours.

    Returns
    -------
    LymphocytePKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if vd_lymph_L <= 0:
        raise ValueError("vd_lymph_L must be > 0")

    # Parameters
    q_lymph = 1.5  # L/h — lymph flow rate
    k_el = cl_sys_L_per_h / vd_plasma_L  # elimination rate constant
    car = _compute_car(logP, p_gp_substrate)

    # Initial conditions: IV bolus → instantaneous distribution into plasma
    c_plasma = dose_mg / vd_plasma_L
    c_lymph = 0.0

    times = [0.0]
    c_plasma_list = [c_plasma]
    c_lymph_list = [c_lymph]

    n_steps = int(round(t_end_h / dt_h))
    t = 0.0

    for _ in range(n_steps):
        driving_force = c_plasma - c_lymph / car
        dc_plasma = -q_lymph / vd_plasma_L * driving_force - k_el * c_plasma
        dc_lymph = q_lymph / vd_lymph_L * driving_force

        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        c_lymph = max(0.0, c_lymph + dc_lymph * dt_h)
        t += dt_h

        times.append(t)
        c_plasma_list.append(c_plasma)
        c_lymph_list.append(c_lymph)

    # Compute metrics
    cmax_lymph = max(c_lymph_list)
    tmax_lymph = times[c_lymph_list.index(cmax_lymph)]

    # AUC via manual trapezoidal rule
    auc_lymph = sum(
        0.5 * (c_lymph_list[i] + c_lymph_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # Lymph-to-plasma ratio at t_end
    c_plasma_end = c_plasma_list[-1]
    c_lymph_end = c_lymph_list[-1]
    if c_plasma_end > 0:
        lymph_to_plasma_ratio = c_lymph_end / c_plasma_end
    else:
        lymph_to_plasma_ratio = 0.0

    pgp_note = " P-gp efflux reduces accumulation (CAR/3)." if p_gp_substrate else ""
    notes = f"CAR={car:.2f}; Q_lymph={q_lymph} L/h; k_el={k_el:.4f}/h.{pgp_note}"

    return LymphocytePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        p_gp_substrate=p_gp_substrate,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_lymph_mg_L=c_lymph_list,
        cmax_lymph_mg_L=cmax_lymph,
        tmax_lymph_h=tmax_lymph,
        auc_lymph_mg_h_per_L=auc_lymph,
        auc_plasma_mg_h_per_L=auc_plasma,
        car=car,
        lymph_to_plasma_ratio=lymph_to_plasma_ratio,
        notes=notes,
    )


def compare_p_gp_effect(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
) -> list:
    """Compare P-gp substrate vs non-substrate lymphocyte PK.

    Returns a list of two LymphocytePKResult objects sorted by CAR descending
    (highest accumulation first, i.e., non-P-gp substrate first).
    """
    non_pgp = simulate_lymphocyte_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        cl_sys_L_per_h=cl_sys_L_per_h,
        p_gp_substrate=False,
    )
    pgp = simulate_lymphocyte_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        cl_sys_L_per_h=cl_sys_L_per_h,
        p_gp_substrate=True,
    )
    return sorted([non_pgp, pgp], key=lambda r: r.car, reverse=True)
