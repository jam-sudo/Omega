"""Dermal PBPK Model — Phase 1070.

4-compartment skin PBPK for topical/transdermal drug delivery.

Compartments (amounts in mg, except C_plasma in mg/L):
    Vehicle        → A_vehicle (depletes first-order)
    Stratum corneum → A_sc
    Viable epidermis → A_ve
    Dermis          → A_d
    Systemic plasma  → C_plasma

ODEs (Forward Euler):
    dA_vehicle/dt = -k_vehicle_sc * A_vehicle
    dA_sc/dt      = k_vehicle_sc * A_vehicle - k_sc_ve * A_sc - k_sc_elim * A_sc
    dA_ve/dt      = k_sc_ve * A_sc - k_ve_d * A_ve - k_ve_elim * A_ve
    dA_d/dt       = k_ve_d * A_ve - k_d_sys * A_d - k_d_elim * A_d
    dC_plasma/dt  = k_d_sys * A_d / vd_sys - ke_sys * C_plasma

where ke_sys = cl_sys / vd_sys (h^-1).

Lipophilicity effect on k_vehicle_sc:
    effective_k_vehicle_sc = k_vehicle_sc * (1 + 0.1 * logP)
    logP is clamped to [-2, 6] before applying the correction.

Initial conditions:
    A_vehicle(0) = dose_mg; all others = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DermalPBPKResult",
    "simulate_dermal_pk",
    "compare_logp_dermal",
]

# ---------------------------------------------------------------------------
# Default kinetic parameters (physiologically realistic dermal values)
# ---------------------------------------------------------------------------

_DEFAULT_K_VEHICLE_SC: float = 0.10  # h^-1
_DEFAULT_K_SC_VE: float = 0.05  # h^-1
_DEFAULT_K_VE_D: float = 0.10  # h^-1
_DEFAULT_K_D_SYS: float = 0.05  # h^-1
_DEFAULT_K_SC_ELIM: float = 0.02  # h^-1
_DEFAULT_K_VE_ELIM: float = 0.01  # h^-1
_DEFAULT_K_D_ELIM: float = 0.005  # h^-1
_DEFAULT_VD_SYS: float = 50.0  # L
_DEFAULT_CL_SYS: float = 5.0  # L/h

# logP bounds for the lipophilicity correction
_LOGP_CORR_LOW: float = -2.0
_LOGP_CORR_HIGH: float = 6.0


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(len(times) - 1):
        total += 0.5 * (values[i] + values[i + 1]) * (times[i + 1] - times[i])
    return total


@dataclass
class DermalPBPKResult:
    """Result of a 4-compartment dermal PBPK simulation (Phase 1070).

    Attributes
    ----------
    drug_name : str
    dose_mg : float
    logP : float
    times_h : list
        Simulation time points (h).
    c_plasma_mg_L : list
        Systemic plasma concentration (mg/L) at each time point.
    a_sc_mg : list
        Amount in stratum corneum (mg).
    a_ve_mg : list
        Amount in viable epidermis (mg).
    a_dermis_mg : list
        Amount in dermis (mg).
    cmax_plasma : float
        Peak plasma concentration (mg/L).
    tmax_plasma_h : float
        Time of peak plasma concentration (h).
    auc_plasma : float
        AUC(0-t_end) for plasma (mg·h/L), trapezoidal.
    auc_dermis : float
        AUC(0-t_end) for dermis amount (mg·h), trapezoidal.
    f_systemic : float
        Fraction of dose reaching systemic circulation (cumulative amount
        entering plasma / dose_mg), clamped to [0, 1].
    notes : str
        Human-readable summary.
    """

    drug_name: str
    dose_mg: float
    logP: float
    times_h: list
    c_plasma_mg_L: list
    a_sc_mg: list
    a_ve_mg: list
    a_dermis_mg: list
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    auc_dermis: float
    f_systemic: float
    notes: str


def simulate_dermal_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    route: str = "topical",
    k_vehicle_sc: float = _DEFAULT_K_VEHICLE_SC,
    k_sc_ve: float = _DEFAULT_K_SC_VE,
    k_ve_d: float = _DEFAULT_K_VE_D,
    k_d_sys: float = _DEFAULT_K_D_SYS,
    k_sc_elim: float = _DEFAULT_K_SC_ELIM,
    k_ve_elim: float = _DEFAULT_K_VE_ELIM,
    k_d_elim: float = _DEFAULT_K_D_ELIM,
    vd_sys: float = _DEFAULT_VD_SYS,
    cl_sys: float = _DEFAULT_CL_SYS,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> DermalPBPKResult:
    """Simulate topical/transdermal drug delivery using a 4-compartment PBPK model.

    Parameters
    ----------
    drug_name:
        Name or identifier of the drug.
    dose_mg:
        Dose applied to the vehicle (mg). Must be > 0.
    logP:
        Octanol-water log partition coefficient. Must be in [-5, 10].
    route:
        Administration route label (e.g. ``"topical"``, ``"transdermal"``).
        Currently informational only.
    k_vehicle_sc:
        First-order rate constant vehicle → stratum corneum (h^-1). Default 0.1.
    k_sc_ve:
        First-order rate constant SC → viable epidermis (h^-1). Default 0.05.
    k_ve_d:
        First-order rate constant viable epidermis → dermis (h^-1). Default 0.1.
    k_d_sys:
        First-order rate constant dermis → systemic plasma (h^-1). Default 0.05.
    k_sc_elim:
        Elimination from SC (h^-1). Default 0.02.
    k_ve_elim:
        Elimination from viable epidermis (h^-1). Default 0.01.
    k_d_elim:
        Elimination from dermis (h^-1). Default 0.005.
    vd_sys:
        Systemic volume of distribution (L). Must be > 0. Default 50.
    cl_sys:
        Systemic clearance (L/h). Must be > 0. Default 5.
    t_end_h:
        Simulation end time (h). Default 24.
    dt_h:
        Forward Euler time step (h). Default 0.05.

    Returns
    -------
    DermalPBPKResult

    Raises
    ------
    ValueError
        If *dose_mg* <= 0, *vd_sys* <= 0, *cl_sys* <= 0, or
        *logP* outside [-5, 10].
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vd_sys <= 0.0:
        raise ValueError(f"vd_sys must be > 0, got {vd_sys}")
    if cl_sys <= 0.0:
        raise ValueError(f"cl_sys must be > 0, got {cl_sys}")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")

    # Lipophilicity correction on vehicle → SC transfer
    logp_clamped = max(_LOGP_CORR_LOW, min(_LOGP_CORR_HIGH, logP))
    effective_k_vehicle_sc = k_vehicle_sc * (1.0 + 0.1 * logp_clamped)

    ke_sys = cl_sys / vd_sys  # elimination rate constant (h^-1)

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    a_vehicle = dose_mg
    a_sc = 0.0
    a_ve = 0.0
    a_d = 0.0
    c_plasma = 0.0

    n_steps = max(int(math.ceil(t_end_h / dt_h)), 1)

    times: list = []
    c_plasma_list: list = []
    a_sc_list: list = []
    a_ve_list: list = []
    a_d_list: list = []

    cumulative_systemic = 0.0  # total amount that entered plasma (mg)

    t = 0.0
    for i in range(n_steps + 1):
        times.append(t)
        c_plasma_list.append(c_plasma)
        a_sc_list.append(a_sc)
        a_ve_list.append(a_ve)
        a_d_list.append(a_d)

        if i < n_steps:
            # Rate calculations
            rate_v_sc = effective_k_vehicle_sc * a_vehicle
            rate_sc_ve = k_sc_ve * a_sc
            rate_ve_d = k_ve_d * a_ve
            rate_d_sys = k_d_sys * a_d
            rate_sc_elim = k_sc_elim * a_sc
            rate_ve_elim = k_ve_elim * a_ve
            rate_d_elim = k_d_elim * a_d

            # ODE right-hand sides
            d_vehicle = -rate_v_sc
            d_sc = rate_v_sc - rate_sc_ve - rate_sc_elim
            d_ve = rate_sc_ve - rate_ve_d - rate_ve_elim
            d_d = rate_ve_d - rate_d_sys - rate_d_elim
            d_c_plasma = rate_d_sys / vd_sys - ke_sys * c_plasma

            # Forward Euler update (floor at 0 for amounts)
            a_vehicle = max(a_vehicle + d_vehicle * dt_h, 0.0)
            a_sc = max(a_sc + d_sc * dt_h, 0.0)
            a_ve = max(a_ve + d_ve * dt_h, 0.0)
            a_d = max(a_d + d_d * dt_h, 0.0)
            c_plasma = max(c_plasma + d_c_plasma * dt_h, 0.0)

            cumulative_systemic += rate_d_sys * dt_h
            t = min(t + dt_h, t_end_h)

    # ------------------------------------------------------------------
    # PK metrics
    # ------------------------------------------------------------------
    auc_plasma = _trapz(times, c_plasma_list)
    auc_dermis = _trapz(times, a_d_list)

    cmax_plasma = max(c_plasma_list)
    tmax_plasma_h = times[c_plasma_list.index(cmax_plasma)]

    f_systemic = min(cumulative_systemic / dose_mg, 1.0)

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    notes = (
        f"Dermal PBPK (Phase 1070): route={route}, logP={logP:.2f}, "
        f"dose={dose_mg:.2f} mg; effective_k_vehicle_sc={effective_k_vehicle_sc:.4f} h^-1; "
        f"Cmax={cmax_plasma:.4f} mg/L, Tmax={tmax_plasma_h:.2f} h, "
        f"f_systemic={f_systemic:.3f}"
    )

    return DermalPBPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        a_sc_mg=a_sc_list,
        a_ve_mg=a_ve_list,
        a_dermis_mg=a_d_list,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        auc_dermis=auc_dermis,
        f_systemic=f_systemic,
        notes=notes,
    )


def compare_logp_dermal(
    drug_name: str,
    dose_mg: float,
    logp_values: list,
    **kwargs,
) -> list:
    """Simulate dermal PK for multiple logP values and return results sorted by
    *auc_dermis* descending.

    Parameters
    ----------
    drug_name:
        Name or identifier of the drug.
    dose_mg:
        Applied dose (mg). Must be > 0.
    logp_values:
        List of logP values to evaluate.
    **kwargs:
        Additional keyword arguments forwarded to :func:`simulate_dermal_pk`.

    Returns
    -------
    list[DermalPBPKResult]
        One result per logP value, sorted by *auc_dermis* descending.
    """
    results: list[DermalPBPKResult] = []
    for lp in logp_values:
        result = simulate_dermal_pk(drug_name=drug_name, dose_mg=dose_mg, logP=lp, **kwargs)
        results.append(result)
    results.sort(key=lambda r: r.auc_dermis, reverse=True)
    return results
