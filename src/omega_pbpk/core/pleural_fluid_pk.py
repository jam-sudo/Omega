"""Phase 518 — Drug Distribution in Pleural Fluid.

2-compartment plasma/pleural fluid PK model for thoracic drug distribution.
IV bolus administration with bidirectional diffusion between plasma and
pleural space plus lymphatic drainage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PleuralFluidPKResult:
    """Result of pleural fluid PK simulation."""

    drug_name: str
    dose_mg: float
    ps_pleural_L_per_h: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_pleural_mg_L: list = field(default_factory=list)
    cmax_plasma: float = 0.0
    tmax_plasma_h: float = 0.0
    auc_plasma: float = 0.0
    cmax_pleural: float = 0.0
    tmax_pleural_h: float = 0.0
    auc_pleural: float = 0.0
    pleural_plasma_auc_ratio: float = 0.0
    pleural_penetration_pct: float = 0.0
    t_half_plasma_h: float = 0.0
    notes: str = ""


def simulate_pleural_fluid_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_plasma_L: float,
    ps_pleural_L_per_h: float = 0.005,
    v_pleural_L: float = 0.02,
    q_pleural_L_per_h: float = 0.012,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PleuralFluidPKResult:
    """Simulate drug distribution into pleural fluid after IV bolus.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    cl_L_per_h:
        Plasma clearance (L/h).
    vd_plasma_L:
        Plasma volume of distribution (L).
    ps_pleural_L_per_h:
        Pleural permeability-surface area product (L/h).
    v_pleural_L:
        Pleural fluid volume (L); default 0.02 L (20 mL).
    q_pleural_L_per_h:
        Lymphatic drainage flow (L/h); default 0.012 L/h.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step (h).

    Returns
    -------
    PleuralFluidPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if ps_pleural_L_per_h <= 0:
        raise ValueError("ps_pleural_L_per_h must be > 0")

    # Elimination rate constant from plasma
    k_el = cl_L_per_h / vd_plasma_L

    # Transfer rate constants
    # Plasma → pleural (per unit plasma concentration)
    k_p_to_pl = ps_pleural_L_per_h / vd_plasma_L
    # Pleural → plasma (per unit pleural concentration, symmetric diffusion flux back)
    k_pl_to_p = ps_pleural_L_per_h / vd_plasma_L

    # Pleural compartment: inflow from plasma, outflow by diffusion back + lymph drainage
    k_in = ps_pleural_L_per_h / v_pleural_L  # 1/h — plasma conc drives inflow
    k_out = ps_pleural_L_per_h / v_pleural_L + q_pleural_L_per_h / v_pleural_L

    # Initial conditions: IV bolus → all drug in plasma
    C_plasma = dose_mg / vd_plasma_L
    C_pleural = 0.0

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_pleural_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times.append(t)
        c_plasma_list.append(C_plasma)
        c_pleural_list.append(C_pleural)

        # Forward Euler derivatives
        # Plasma: loses drug to elimination and to pleural, gains back from pleural
        dC_plasma = -k_el * C_plasma - k_p_to_pl * C_plasma + k_pl_to_p * C_pleural
        # Pleural: driven by plasma concentration, loses by diffusion back + lymph
        dC_pleural = k_in * C_plasma - k_out * C_pleural

        C_plasma += dC_plasma * dt_h
        C_pleural += dC_pleural * dt_h

        # Clamp to non-negative
        if C_plasma < 0.0:
            C_plasma = 0.0
        if C_pleural < 0.0:
            C_pleural = 0.0

        t += dt_h

    # Plasma PK metrics
    cmax_plasma = c_plasma_list[0]  # IV bolus peak is at t=0
    tmax_plasma_h = times[c_plasma_list.index(cmax_plasma)]
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # Pleural PK metrics
    cmax_pleural = max(c_pleural_list)
    tmax_pleural_h = times[c_pleural_list.index(cmax_pleural)]
    auc_pleural = sum(
        0.5 * (c_pleural_list[i] + c_pleural_list[i - 1]) * (times[i] - times[i - 1])
        for i in range(1, len(times))
    )

    # Ratios and penetration
    pleural_plasma_auc_ratio = auc_pleural / auc_plasma if auc_plasma > 0.0 else 0.0
    pleural_penetration_pct = cmax_pleural / cmax_plasma * 100.0 if cmax_plasma > 0.0 else 0.0

    # Terminal half-life (plasma)
    t_half_plasma = math.log(2.0) / k_el if k_el > 0 else float("inf")

    notes = (
        f"k_el={k_el:.4f}/h, k_in={k_in:.4f}/h, k_out={k_out:.4f}/h, "
        f"pleural/plasma AUC ratio={pleural_plasma_auc_ratio:.4f}"
    )

    return PleuralFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ps_pleural_L_per_h=ps_pleural_L_per_h,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_pleural_mg_L=c_pleural_list,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        cmax_pleural=cmax_pleural,
        tmax_pleural_h=tmax_pleural_h,
        auc_pleural=auc_pleural,
        pleural_plasma_auc_ratio=pleural_plasma_auc_ratio,
        pleural_penetration_pct=pleural_penetration_pct,
        t_half_plasma_h=t_half_plasma,
        notes=notes,
    )


def compare_pleural_ps(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_plasma_L: float,
    ps_values: list,
) -> list:
    """Compare pleural fluid penetration across different PS values.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    cl_L_per_h:
        Plasma clearance (L/h).
    vd_plasma_L:
        Plasma volume of distribution (L).
    ps_values:
        List of PS (permeability-surface area) values (L/h) to compare.

    Returns
    -------
    list of PleuralFluidPKResult, sorted by pleural_plasma_auc_ratio descending.
    """
    results = [
        simulate_pleural_fluid_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_plasma_L=vd_plasma_L,
            ps_pleural_L_per_h=ps,
        )
        for ps in ps_values
    ]
    results.sort(key=lambda r: r.pleural_plasma_auc_ratio, reverse=True)
    return results
