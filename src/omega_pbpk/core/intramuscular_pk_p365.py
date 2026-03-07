"""Intramuscular (IM) Drug Absorption Model — Phase 365.

3-compartment Forward Euler model:
    injection_depot -> muscle_tissue -> systemic

Two release models:
    - first_order : k_release * A_depot  (aqueous formulation)
    - zero_order  : constant release_rate_mg_per_h  (oil depot)

References
----------
- Toutain PL & Bousquet-Melou A, J Vet Pharmacol Ther. 2004;27(6):427-39.
- Zuidema J et al., J Pharm Sci. 1994;83(8):1151-6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["IntramuscularPKResult365", "simulate_im_pk_p365"]

_VALID_RELEASE_MODELS = {"first_order", "zero_order"}


@dataclass
class IntramuscularPKResult365:
    """Results from the Phase 365 IM pharmacokinetic simulation.

    Attributes
    ----------
    times_h : Simulation time points (h).
    c_depot_mg_mL : Depot concentration over time (mg/mL).
    c_muscle_mg_mL : Muscle tissue concentration over time (mg/mL).
    c_systemic_mg_L : Systemic plasma concentration over time (mg/L).
    auc_systemic_mg_h_per_L : Systemic AUC0-t (mg*h/L), trapezoidal.
    cmax_systemic_mg_L : Peak systemic concentration (mg/L).
    tmax_systemic_h : Time of peak systemic concentration (h).
    bioavailability_pct : AUC_im / AUC_iv * 100 (%).
    flip_flop_pk : True when tmax > 2 * systemic t_half.
    release_model : Release model used ('first_order' or 'zero_order').
    notes : Summary string.
    """

    times_h: list = field(default_factory=list)
    c_depot_mg_mL: list = field(default_factory=list)
    c_muscle_mg_mL: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)
    auc_systemic_mg_h_per_L: float = 0.0
    cmax_systemic_mg_L: float = 0.0
    tmax_systemic_h: float = 0.0
    bioavailability_pct: float = 0.0
    flip_flop_pk: bool = False
    release_model: str = ""
    notes: str = ""


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_im_pk_p365(
    drug_name: str,
    dose_mg: float,
    release_model: str = "first_order",
    k_release_per_h: float = 0.5,
    release_rate_mg_per_h: float = 1.0,
    k_abs_per_h: float = 1.0,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 50.0,
    v_depot_mL: float = 2.0,
    v_muscle_mL: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> IntramuscularPKResult365:
    """Simulate IM PK with a 3-compartment Forward Euler model.

    Parameters
    ----------
    drug_name : Drug identifier string.
    dose_mg : Administered dose (mg). Must be > 0.
    release_model : 'first_order' or 'zero_order'.
    k_release_per_h : First-order release rate constant (1/h).
    release_rate_mg_per_h : Zero-order release rate (mg/h).
    k_abs_per_h : Muscle-to-systemic absorption rate constant (1/h).
    cl_sys_L_per_h : Systemic clearance (L/h). Must be > 0.
    vd_sys_L : Volume of distribution (L). Must be > 0.
    v_depot_mL : Volume of injection depot (mL). Default 2.0.
    v_muscle_mL : Volume of muscle compartment (mL). Default 50.0.
    t_end_h : Simulation end time (h).
    dt_h : Forward Euler step size (h).

    Returns
    -------
    IntramuscularPKResult365
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if release_model not in _VALID_RELEASE_MODELS:
        raise ValueError(
            f"release_model must be one of {_VALID_RELEASE_MODELS}, got '{release_model}'"
        )

    ke = cl_sys_L_per_h / vd_sys_L
    t_half_sys = math.log(2.0) / ke

    n_steps = max(int(round(t_end_h / dt_h)), 1)

    # State variables
    a_depot = dose_mg  # mg in injection depot
    a_muscle = 0.0  # mg in muscle tissue
    c_sys = 0.0  # mg/L systemic

    times: list[float] = []
    c_depot_list: list[float] = []
    c_muscle_list: list[float] = []
    c_sys_list: list[float] = []

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        c_depot_list.append(a_depot / v_depot_mL)
        c_muscle_list.append(a_muscle / v_muscle_mL)
        c_sys_list.append(c_sys)

        if i < n_steps:
            # Release flux from depot into muscle
            if release_model == "first_order":
                release_flux = k_release_per_h * a_depot
            else:
                # zero_order: limited by remaining amount
                release_flux = min(release_rate_mg_per_h, a_depot / dt_h)

            # Absorption flux from muscle to systemic
            abs_flux = k_abs_per_h * a_muscle

            # Forward Euler updates
            da_depot = -release_flux
            da_muscle = release_flux - abs_flux
            dc_sys = abs_flux / vd_sys_L - ke * c_sys

            a_depot = max(a_depot + da_depot * dt_h, 0.0)
            a_muscle = max(a_muscle + da_muscle * dt_h, 0.0)
            c_sys = max(c_sys + dc_sys * dt_h, 0.0)

    # --- Derived PK metrics ---
    auc = _trapezoidal_auc(times, c_sys_list)
    cmax = max(c_sys_list)
    tmax = times[c_sys_list.index(cmax)]

    # Bioavailability: AUC_im / AUC_iv  (AUC_iv = dose / CL)
    auc_iv = dose_mg / cl_sys_L_per_h
    bioavailability_pct = (auc / auc_iv * 100.0) if auc_iv > 0 else 0.0
    bioavailability_pct = min(bioavailability_pct, 100.0)

    # Flip-flop PK: absorption-rate limited when tmax > 2 * t_half
    flip_flop = tmax > 2.0 * t_half_sys

    notes = (
        f"IM {release_model} simulation for {drug_name}. "
        f"dose={dose_mg}mg, ke={ke:.4f}/h, t_half_sys={t_half_sys:.2f}h. "
        f"Cmax={cmax:.4f}mg/L at Tmax={tmax:.2f}h. "
        f"F={bioavailability_pct:.1f}%. "
        f"Flip-flop PK: {flip_flop}."
    )

    return IntramuscularPKResult365(
        times_h=times,
        c_depot_mg_mL=c_depot_list,
        c_muscle_mg_mL=c_muscle_list,
        c_systemic_mg_L=c_sys_list,
        auc_systemic_mg_h_per_L=auc,
        cmax_systemic_mg_L=cmax,
        tmax_systemic_h=tmax,
        bioavailability_pct=bioavailability_pct,
        flip_flop_pk=flip_flop,
        release_model=release_model,
        notes=notes,
    )
