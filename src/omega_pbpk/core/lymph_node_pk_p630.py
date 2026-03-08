"""Drug Lymph Node Concentration Modeling — Phase 630.

Models drug distribution into lymph nodes, important for immunosuppressants,
lymphoma drugs, and biologics. Lymphatic absorption is particularly relevant
for large, lipophilic molecules.

References:
    - Trevaskis NL et al. Nat Rev Drug Discov 2008.
    - Porter CJH & Charman WN. Adv Drug Deliv Rev 2001.
    - Charman WN et al. J Pharm Sci 1992.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LymphNodePKResult:
    """Result of lymph node PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_lymph_mg_g: list
    c_lymph_fluid_mg_L: list
    cmax_plasma_mg_L: float
    cmax_lymph_mg_g: float
    auc_plasma_mg_h_per_L: float
    auc_lymph_mg_h_per_g: float
    lymph_to_plasma_ratio: float
    lymphatic_absorption_pct: float
    t_half_lymph_h: float
    notes: str


def _trapezoidal_auc(times: list, conc: list) -> float:
    """Manual trapezoidal AUC calculation."""
    return sum(
        0.5 * (conc[i] + conc[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def _f_lymph(logP: float, mw_Da: float) -> float:
    """Fraction absorbed via lymphatics.

    Small/hydrophilic molecules: negligible lymphatic absorption.
    Large/lipophilic molecules: up to 30% via lymphatics.
    """
    if mw_Da < 500.0 and logP < 2.0:
        return 0.0
    return min(0.3, logP * 0.05 + mw_Da / 10000.0)


def _kp_lymph(logP: float, mw_Da: float) -> float:
    """Lymph node partition coefficient (dimensionless)."""
    raw = (logP * 0.8 + 1.0) / max(1.0, mw_Da / 300.0)
    return max(0.1, min(15.0, raw))


def _validate_inputs(
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    lymph_node_weight_g: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if lymph_node_weight_g <= 0:
        raise ValueError(f"lymph_node_weight_g must be > 0, got {lymph_node_weight_g}")


def simulate_lymph_node_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    lymph_node_weight_g: float = 30.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> LymphNodePKResult:
    """Simulate drug distribution into lymph nodes via forward Euler ODE.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose in mg.
    logP : float
        Octanol-water partition coefficient.
    mw_Da : float
        Molecular weight in Daltons.
    cl_sys_L_per_h : float
        Systemic clearance in L/h.
    vd_sys_L : float
        Systemic volume of distribution in L.
    lymph_node_weight_g : float, optional
        Total lymph node tissue weight in grams. Default 30.0.
    t_end_h : float, optional
        Simulation end time in hours. Default 72.0.
    dt_h : float, optional
        Time step in hours. Default 0.1.

    Returns
    -------
    LymphNodePKResult
        Dataclass with time series and summary PK metrics.
    """
    _validate_inputs(dose_mg, cl_sys_L_per_h, vd_sys_L, lymph_node_weight_g)

    fl = _f_lymph(logP, mw_Da)
    kp_ln = _kp_lymph(logP, mw_Da)

    Q_lymph = 0.12  # L/h total lymph flow
    vd_lymph_L = lymph_node_weight_g * kp_ln / 1000.0

    # Rate constants
    ka = 1.2  # /h oral absorption rate
    k_elim = cl_sys_L_per_h / vd_sys_L  # /h elimination from plasma
    k_lymph_in = Q_lymph / vd_sys_L  # /h plasma→lymph transfer
    k_lymph_out = Q_lymph / vd_lymph_L  # /h lymph→plasma transfer

    t_half_lymph_h = math.log(2.0) / k_lymph_out

    # Initial conditions
    A_gut = dose_mg  # amount in gut lumen (mg)
    C_plasma = 0.0  # concentration in plasma (mg/L)
    C_lymph = 0.0  # concentration in lymph volume (mg/L)

    # Output arrays
    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_plasma_arr = []
    c_lymph_mg_g_arr = []
    c_lymph_fluid_arr = []

    t = 0.0
    for _ in range(n_steps):
        times_h.append(t)
        c_plasma_arr.append(C_plasma)
        c_lymph_mg_g_arr.append(C_lymph * vd_lymph_L * 1000.0 / lymph_node_weight_g)
        c_lymph_fluid_arr.append(C_lymph)

        # ODEs
        dA_gut = -ka * A_gut

        dC_plasma = (
            ka * A_gut * (1.0 - fl) / vd_sys_L
            - k_elim * C_plasma
            - k_lymph_in * C_plasma
            + k_lymph_out * C_lymph * (vd_lymph_L / vd_sys_L)
        )

        dC_lymph = (
            k_lymph_in * C_plasma * (vd_sys_L / vd_lymph_L)
            + ka * A_gut * fl / vd_lymph_L
            - k_lymph_out * C_lymph
        )

        # Forward Euler update
        A_gut = max(0.0, A_gut + dt_h * dA_gut)
        C_plasma = max(0.0, C_plasma + dt_h * dC_plasma)
        C_lymph = max(0.0, C_lymph + dt_h * dC_lymph)

        t += dt_h

    # Summary metrics
    cmax_plasma = max(c_plasma_arr)
    cmax_lymph = max(c_lymph_mg_g_arr)
    auc_plasma = _trapezoidal_auc(times_h, c_plasma_arr)
    auc_lymph = _trapezoidal_auc(times_h, c_lymph_mg_g_arr)

    if auc_plasma > 0.0:
        lymph_to_plasma_ratio = auc_lymph / auc_plasma
    else:
        lymph_to_plasma_ratio = 0.0

    notes = (
        f"logP={logP:.2f}, MW={mw_Da:.0f} Da. "
        f"f_lymph={fl * 100:.1f}%, kp_lymph={kp_ln:.2f}. "
        f"Vd_lymph={vd_lymph_L * 1000:.1f} mL. "
        f"Lymph t½={t_half_lymph_h:.2f} h."
    )

    return LymphNodePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_arr,
        c_lymph_mg_g=c_lymph_mg_g_arr,
        c_lymph_fluid_mg_L=c_lymph_fluid_arr,
        cmax_plasma_mg_L=round(cmax_plasma, 6),
        cmax_lymph_mg_g=round(cmax_lymph, 6),
        auc_plasma_mg_h_per_L=round(auc_plasma, 4),
        auc_lymph_mg_h_per_g=round(auc_lymph, 4),
        lymph_to_plasma_ratio=round(lymph_to_plasma_ratio, 4),
        lymphatic_absorption_pct=round(fl * 100.0, 4),
        t_half_lymph_h=round(t_half_lymph_h, 4),
        notes=notes,
    )
