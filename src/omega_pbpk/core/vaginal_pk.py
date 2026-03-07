"""Phase 216: Vaginal drug delivery PK model (gel, ring, tablet)."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Vaginal fluid volume (mL)
_V_VAGINAL_ML = 5.0

# Release rate constants (1/h) by formulation type
_K_RELEASE = {
    "gel": 0.3,
    "ring": 0.02,
    "tablet": 0.5,
}

# Local elimination rate constant (1/h)
_K_LOCAL_ELIM = 0.05

# Base local absorption rate constant (1/h)
_K_LOCAL_ABS_BASE = 0.15


@dataclass
class VaginalPKResult:
    """Result of vaginal PK simulation."""

    drug_name: str
    dose_mg: float
    formulation_type: str
    logP: float
    times_h: list[float]
    c_local_mg_mL: list[float]
    c_systemic_mg_L: list[float]
    cmax_local_mg_mL: float
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    auc_systemic_mg_h_per_L: float
    f_systemic_absorbed: float
    notes: str


def _absorption_factor(logP: float) -> float:
    """Logistic absorption factor: 1/(1+exp(-logP+1))."""
    return 1.0 / (1.0 + math.exp(-logP + 1.0))


def _trapz_auc(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i - 1] + values[i]) * dt
    return auc


def simulate_vaginal_pk(
    drug_name: str,
    dose_mg: float,
    formulation_type: str,
    logP: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> VaginalPKResult:
    """Simulate vaginal drug delivery PK using a 3-compartment Forward Euler ODE.

    Compartments:
      - A_vaginal (mg): drug in vaginal formulation
      - C_local (mg/mL): drug in vaginal fluid/tissue
      - C_systemic (mg/L): systemic plasma concentration

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg.
    formulation_type:
        One of "gel", "ring", "tablet".
    logP:
        Octanol-water partition coefficient.
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_L:
        Volume of distribution in L.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler step size in hours.

    Returns
    -------
    VaginalPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if formulation_type not in _K_RELEASE:
        raise ValueError(
            f"Invalid formulation_type '{formulation_type}'. "
            f"Must be one of {list(_K_RELEASE.keys())}"
        )
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")

    k_release = _K_RELEASE[formulation_type]
    k_local_abs = _K_LOCAL_ABS_BASE * _absorption_factor(logP)
    k_local_elim = _K_LOCAL_ELIM
    ke = cl_L_per_h / vd_L

    # Initial conditions
    A_vaginal = dose_mg
    C_local = 0.0
    C_systemic = 0.0

    times: list[float] = [0.0]
    c_local_list: list[float] = [C_local]
    c_systemic_list: list[float] = [C_systemic]

    n_steps = int(round(t_end_h / dt_h))
    t = 0.0

    for _ in range(n_steps):
        # ODEs
        dA_vaginal = -k_release * A_vaginal
        dC_local = (
            k_release * A_vaginal / _V_VAGINAL_ML - k_local_abs * C_local - k_local_elim * C_local
        )
        dC_systemic = k_local_abs * C_local * _V_VAGINAL_ML / vd_L - ke * C_systemic

        # Forward Euler update
        A_vaginal += dA_vaginal * dt_h
        C_local += dC_local * dt_h
        C_systemic += dC_systemic * dt_h

        # Clamp non-negative
        A_vaginal = max(0.0, A_vaginal)
        C_local = max(0.0, C_local)
        C_systemic = max(0.0, C_systemic)

        t += dt_h
        times.append(t)
        c_local_list.append(C_local)
        c_systemic_list.append(C_systemic)

    # Derived PK metrics
    cmax_local = max(c_local_list)
    cmax_systemic = max(c_systemic_list)

    # tmax_systemic: time of max systemic concentration
    idx_max_sys = c_systemic_list.index(cmax_systemic)
    tmax_systemic = times[idx_max_sys]

    auc_systemic = _trapz_auc(times, c_systemic_list)

    # f_systemic_absorbed: fraction of dose that reached systemic
    # estimated as AUC * CL / dose (in consistent units)
    # AUC (mg*h/L) * CL (L/h) = mg absorbed systemically
    amount_systemic_mg = auc_systemic * cl_L_per_h
    f_systemic_absorbed = amount_systemic_mg / dose_mg
    f_systemic_absorbed = min(1.0, f_systemic_absorbed)

    notes = (
        f"formulation={formulation_type}, k_release={k_release}/h, "
        f"k_local_abs={k_local_abs:.4f}/h, logP={logP}"
    )

    return VaginalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        logP=logP,
        times_h=times,
        c_local_mg_mL=c_local_list,
        c_systemic_mg_L=c_systemic_list,
        cmax_local_mg_mL=cmax_local,
        cmax_systemic_mg_L=cmax_systemic,
        tmax_systemic_h=tmax_systemic,
        auc_systemic_mg_h_per_L=auc_systemic,
        f_systemic_absorbed=f_systemic_absorbed,
        notes=notes,
    )


def compare_formulations(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> list[VaginalPKResult]:
    """Compare gel, ring, and tablet formulations, sorted by systemic AUC ascending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg.
    logP:
        Octanol-water partition coefficient.
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_L:
        Volume of distribution in L.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler step size in hours.

    Returns
    -------
    List of VaginalPKResult for [gel, ring, tablet] sorted by systemic AUC ascending.
    """
    results = []
    for formulation in ("gel", "ring", "tablet"):
        result = simulate_vaginal_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation_type=formulation,
            logP=logP,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_systemic_mg_h_per_L)
    return results
