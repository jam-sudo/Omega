"""Tumor PK/PD model with solid tumor compartment, drug penetration, and cell kill.

Two-region tumor model:
  1. Vascular region (~30% of tumor): exchanges with plasma
  2. Avascular/necrotic core (~70%): receives drug by diffusion only
  3. Plasma (systemic 1-compartment source with CL)

PD models: Emax, log-kill, and direct (linear) cell kill.

References
----------
- Tredan O et al., JNCI. 2007;99(19):1441-54 (drug penetration in solid tumors)
- Minchinton AI & Tannock IF, Nat Rev Cancer. 2006;6(8):583-92 (hypoxia, penetration)
- Friberg LE et al., J Clin Oncol. 2002;20(24):4713-21 (myelosuppression PD model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

_VALID_PD_MODELS = {"emax", "log_kill", "direct"}


@dataclass(frozen=True)
class TumorPKResult:
    """Results from tumor PK/PD simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    tumor_weight_g : Tumor weight (g).
    dose_mg : Administered dose (mg).
    route : Administration route ('iv' or 'oral').
    times_h : Simulation time points (h).
    c_sys_mg_L : Systemic plasma concentration (mg/L).
    c_vasc_mg_L : Tumor vascular compartment concentration (mg/L).
    c_avsc_mg_L : Tumor avascular/core concentration (mg/L).
    tumor_cells_norm : Normalized tumor cell population (N/N0).
    log_cell_kill : log10(1/N) at each time point.
    cmax_sys : Peak systemic concentration (mg/L).
    cmax_tumor_vasc : Peak tumor vascular concentration (mg/L).
    cmax_tumor_avsc : Peak tumor avascular concentration (mg/L).
    auc_sys : Systemic AUC 0->t (mg*h/L).
    auc_tumor_vasc : Tumor vascular AUC 0->t (mg*h/L).
    tumor_eradication : True if N < 0.001 at any point.
    max_log_kill : Peak log10 cell kill.
    pd_model : PD model used.
    notes : Summary text.
    """

    drug_name: str
    tumor_weight_g: float
    dose_mg: float
    route: str
    times_h: list[float]
    c_sys_mg_L: list[float]
    c_vasc_mg_L: list[float]
    c_avsc_mg_L: list[float]
    tumor_cells_norm: list[float]
    log_cell_kill: list[float]
    cmax_sys: float
    cmax_tumor_vasc: float
    cmax_tumor_avsc: float
    auc_sys: float
    auc_tumor_vasc: float
    tumor_eradication: bool
    max_log_kill: float
    pd_model: str
    notes: str


def _kill_rate(c: float, pd_model: str, emax: float, ec50: float, hill_n: float) -> float:
    """Compute instantaneous kill rate from drug concentration."""
    if c <= 0.0:
        return 0.0
    if pd_model == "emax":
        return emax * c**hill_n / (ec50**hill_n + c**hill_n)
    if pd_model == "log_kill":
        return emax * math.log(1.0 + c / ec50)
    # direct
    return emax * c


def simulate_tumor_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    tumor_weight_g: float = 5.0,
    tumor_blood_flow_L_per_h: float = 0.01,
    f_vascular: float = 0.3,
    k_penetration_per_h: float = 0.1,
    drug_permeability: float = 1.0,
    pd_model: str = "emax",
    ec50_mg_L: float = 1.0,
    emax: float = 1.0,
    hill_n: float = 1.0,
    tumor_doubling_time_h: float = 168.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.1,
) -> TumorPKResult:
    """Simulate tumor PK/PD with vascular/avascular compartments and cell kill.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg). Must be > 0.
    cl_L_per_h : Systemic clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    route : 'iv' or 'oral'.
    ka_per_h : Oral absorption rate (h^-1).
    f_oral : Oral bioavailability (0-1).
    tumor_weight_g : Tumor weight (g). Must be > 0.
    tumor_blood_flow_L_per_h : Tumor blood flow (L/h).
    f_vascular : Fraction of tumor that is vascular (0-1).
    k_penetration_per_h : Drug diffusion into avascular core (h^-1).
    drug_permeability : Relative permeability (0-1).
    pd_model : PD model ('emax', 'log_kill', 'direct').
    ec50_mg_L : EC50 for PD effect (mg/L). Must be > 0.
    emax : Maximum kill rate (h^-1).
    hill_n : Hill coefficient.
    tumor_doubling_time_h : Tumor doubling time (h).
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    TumorPKResult

    Raises
    ------
    ValueError : If any parameter is invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if tumor_weight_g <= 0:
        raise ValueError("tumor_weight_g must be > 0")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")
    if pd_model not in _VALID_PD_MODELS:
        raise ValueError(f"Unknown pd_model '{pd_model}'. Choose from: {sorted(_VALID_PD_MODELS)}")

    # Volumes
    v_vasc = tumor_weight_g * f_vascular * 1e-3  # L
    v_avsc = tumor_weight_g * (1.0 - f_vascular) * 1e-3  # L
    v_sys = vd_L

    ke = cl_L_per_h / v_sys
    k_pen = k_penetration_per_h * drug_permeability
    k_back = k_pen * 0.1
    k_elim_avsc = 0.01
    kg = math.log(2.0) / tumor_doubling_time_h

    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    # State arrays
    c_sys = np.zeros(n_steps + 1)
    a_vasc = np.zeros(n_steps + 1)  # mg in tumor vascular
    a_avsc = np.zeros(n_steps + 1)  # mg in tumor avascular
    n_cells = np.zeros(n_steps + 1)  # normalized tumor cell count
    n_cells[0] = 1.0

    # Oral depot
    a_depot = 0.0
    if route == "iv":
        c_sys[0] = dose_mg / v_sys
    elif route == "oral":
        a_depot = dose_mg * f_oral
    else:
        raise ValueError(f"Unknown route '{route}'. Choose 'iv' or 'oral'.")

    for i in range(n_steps):
        c_vasc = a_vasc[i] / v_vasc if v_vasc > 0 else 0.0
        # Systemic PK
        if route == "oral":
            absorption = ka_per_h * a_depot
            dc_sys = (absorption / v_sys - ke * c_sys[i]) * dt_h
            a_depot = max(0.0, a_depot - absorption * dt_h)
        else:
            dc_sys = -ke * c_sys[i] * dt_h

        # Tumor vascular compartment
        da_vasc = (
            tumor_blood_flow_L_per_h * (c_sys[i] - c_vasc) - k_pen * a_vasc[i] + k_back * a_avsc[i]
        ) * dt_h

        # Tumor avascular compartment
        da_avsc = (k_pen * a_vasc[i] - k_back * a_avsc[i] - k_elim_avsc * a_avsc[i]) * dt_h

        # PD: cell kill
        kill = _kill_rate(c_vasc, pd_model, emax, ec50_mg_L, hill_n)
        dn = (kg - kill) * n_cells[i] * dt_h

        c_sys[i + 1] = max(0.0, c_sys[i] + dc_sys)
        a_vasc[i + 1] = max(0.0, a_vasc[i] + da_vasc)
        a_avsc[i + 1] = max(0.0, a_avsc[i] + da_avsc)
        n_cells[i + 1] = max(0.0, n_cells[i] + dn)

    # Concentrations
    c_vasc_arr = a_vasc / v_vasc if v_vasc > 0 else np.zeros_like(a_vasc)
    c_avsc_arr = a_avsc / v_avsc if v_avsc > 0 else np.zeros_like(a_avsc)

    # Log cell kill: log10(1/N) — avoid log(0)
    log_kill = np.where(n_cells > 0, -np.log10(np.maximum(n_cells, 1e-30)), 30.0)

    # Metrics
    cmax_sys = float(np.max(c_sys))
    cmax_vasc = float(np.max(c_vasc_arr))
    cmax_avsc = float(np.max(c_avsc_arr))
    auc_sys = float(np_trapz(c_sys, t))
    auc_vasc = float(np_trapz(c_vasc_arr, t))
    eradicated = bool(np.any(n_cells < 0.001))
    max_lk = float(np.max(log_kill))

    notes = (
        f"Tumor PK/PD: {drug_name}, {dose_mg}mg {route}. "
        f"Tumor={tumor_weight_g}g, pd_model={pd_model}. "
        f"Cmax_sys={cmax_sys:.4f} mg/L, Cmax_vasc={cmax_vasc:.4f} mg/L. "
        f"Max log-kill={max_lk:.2f}. Eradication={'yes' if eradicated else 'no'}."
    )

    return TumorPKResult(
        drug_name=drug_name,
        tumor_weight_g=tumor_weight_g,
        dose_mg=dose_mg,
        route=route,
        times_h=t.tolist(),
        c_sys_mg_L=c_sys.tolist(),
        c_vasc_mg_L=c_vasc_arr.tolist(),
        c_avsc_mg_L=c_avsc_arr.tolist(),
        tumor_cells_norm=n_cells.tolist(),
        log_cell_kill=log_kill.tolist(),
        cmax_sys=cmax_sys,
        cmax_tumor_vasc=cmax_vasc,
        cmax_tumor_avsc=cmax_avsc,
        auc_sys=auc_sys,
        auc_tumor_vasc=auc_vasc,
        tumor_eradication=eradicated,
        max_log_kill=max_lk,
        pd_model=pd_model,
        notes=notes,
    )


def dose_response_tumor(
    drug_name: str,
    doses_mg: list[float],
    cl_L_per_h: float,
    vd_L: float,
    **kwargs,
) -> list[TumorPKResult]:
    """Run tumor PK/PD simulations across a range of doses.

    Parameters
    ----------
    drug_name : Drug name.
    doses_mg : List of doses (mg).
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).
    **kwargs : Additional parameters passed to simulate_tumor_pk.

    Returns
    -------
    List of TumorPKResult, one per dose.
    """
    return [simulate_tumor_pk(drug_name, dose, cl_L_per_h, vd_L, **kwargs) for dose in doses_mg]


__all__ = [
    "TumorPKResult",
    "simulate_tumor_pk",
    "dose_response_tumor",
]
