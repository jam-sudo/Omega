"""Dynamic drug-drug interaction PK simulation — Phase 275.

Simulates the time-course of a victim drug's PK when co-administered
with an inhibitor (competitive or irreversible/MBI). Uses a two-drug
1-compartment model with CYP inhibition.

References
----------
- Rowland M & Matin SB, J Pharmacokinet Biopharm. 1973
- Grimm SW et al., Drug Metab Dispos. 2009;37(7):1355-70
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class DDIPKResult:
    """Result from dynamic DDI PK simulation."""

    victim_name: str
    inhibitor_name: str
    inhibition_type: str  # "competitive", "mbi"
    dose_victim_mg: float
    dose_inhibitor_mg: float
    times_h: list[float]
    c_victim_mg_L: list[float]  # Victim drug concentration
    c_inhibitor_mg_L: list[float]  # Inhibitor concentration
    cl_inhibited: list[float]  # Time-varying effective CL of victim
    auc_victim_control: float  # AUC without inhibitor
    auc_victim_ddi: float  # AUC with inhibitor
    aucr: float  # AUC ratio (DDI / control)
    cmax_victim: float
    notes: str


def simulate_ddi_pk(
    victim_name: str,
    dose_victim_mg: float,
    cl_victim_L_per_h: float,
    vd_victim_L: float,
    inhibitor_name: str,
    dose_inhibitor_mg: float,
    cl_inhibitor_L_per_h: float,
    vd_inhibitor_L: float,
    ki_mg_L: float,
    kinact_per_h: float = 0.0,  # MBI: inactivation rate constant
    kdeg_per_h: float = 0.05,  # CYP degradation rate (for MBI recovery)
    fm_cyp: float = 1.0,  # Fraction of victim metabolized by affected CYP
    inhibition_type: str = "competitive",
    route: str = "oral",
    ka_victim_per_h: float = 1.0,
    ka_inhib_per_h: float = 1.0,
    f_victim: float = 0.8,
    f_inhib: float = 0.8,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> DDIPKResult:
    """Simulate victim drug PK with dynamic DDI from competitive or MBI inhibitor.

    Parameters
    ----------
    victim_name : Victim drug name.
    dose_victim_mg : Victim dose (mg). Must be > 0.
    cl_victim_L_per_h : Victim clearance without inhibitor (L/h). Must be > 0.
    vd_victim_L : Victim Vd (L). Must be > 0.
    inhibitor_name : Inhibitor drug name.
    dose_inhibitor_mg : Inhibitor dose (mg). Must be > 0.
    cl_inhibitor_L_per_h : Inhibitor CL (L/h). Must be > 0.
    vd_inhibitor_L : Inhibitor Vd (L). Must be > 0.
    ki_mg_L : Inhibition constant (Ki) in mg/L. Must be > 0.
    kinact_per_h : MBI inactivation rate (h^-1). 0 for competitive.
    kdeg_per_h : CYP enzyme degradation rate (h^-1). Must be > 0.
    fm_cyp : Fraction of victim metabolized by inhibited CYP. Must be in (0, 1].
    inhibition_type : 'competitive' or 'mbi'.
    route : 'oral' or 'iv'.
    ka_victim_per_h, ka_inhib_per_h : Absorption rates.
    f_victim, f_inhib : Oral bioavailabilities.
    t_end_h : Simulation end time (h).
    dt_h : Time step (h).

    Returns
    -------
    DDIPKResult
    """
    if dose_victim_mg <= 0:
        raise ValueError("dose_victim_mg must be > 0")
    if cl_victim_L_per_h <= 0:
        raise ValueError("cl_victim_L_per_h must be > 0")
    if vd_victim_L <= 0:
        raise ValueError("vd_victim_L must be > 0")
    if dose_inhibitor_mg <= 0:
        raise ValueError("dose_inhibitor_mg must be > 0")
    if cl_inhibitor_L_per_h <= 0:
        raise ValueError("cl_inhibitor_L_per_h must be > 0")
    if vd_inhibitor_L <= 0:
        raise ValueError("vd_inhibitor_L must be > 0")
    if ki_mg_L <= 0:
        raise ValueError("ki_mg_L must be > 0")
    if kdeg_per_h <= 0:
        raise ValueError("kdeg_per_h must be > 0")
    if not (0 < fm_cyp <= 1):
        raise ValueError("fm_cyp must be in (0, 1]")
    if inhibition_type not in ("competitive", "mbi"):
        raise ValueError("inhibition_type must be 'competitive' or 'mbi'")
    if route not in ("oral", "iv"):
        raise ValueError("route must be 'oral' or 'iv'")

    ke_i = cl_inhibitor_L_per_h / vd_inhibitor_L

    n_steps = int(t_end_h / dt_h) + 1
    t = np.linspace(0.0, t_end_h, n_steps)

    c_v = np.zeros(n_steps)
    c_i = np.zeros(n_steps)
    enzyme_frac = np.ones(n_steps)  # For MBI: fraction of active enzyme remaining
    cl_eff_arr = np.zeros(n_steps)

    a_gut_v = 0.0
    a_gut_i = 0.0

    if route == "iv":
        c_v[0] = dose_victim_mg / vd_victim_L
        c_i[0] = dose_inhibitor_mg / vd_inhibitor_L
    else:
        a_gut_v = dose_victim_mg * f_victim
        a_gut_i = dose_inhibitor_mg * f_inhib

    for i in range(1, n_steps):
        cp_v = c_v[i - 1]
        cp_i = c_i[i - 1]
        enz = enzyme_frac[i - 1]

        # Inhibitor PK
        if route == "oral":
            abs_i = ka_inhib_per_h * a_gut_i
            a_gut_i = max(0.0, a_gut_i - abs_i * dt_h)
            dc_i = (abs_i - ke_i * cp_i * vd_inhibitor_L) / vd_inhibitor_L * dt_h
        else:
            dc_i = -ke_i * cp_i * dt_h

        c_i[i] = max(0.0, cp_i + dc_i)

        # Compute CL inhibition
        if inhibition_type == "competitive":
            r = 1.0 + cp_i / ki_mg_L  # CL/CL0 = fm/(1+I/Ki) + (1-fm)
            cl_frac = fm_cyp / r + (1.0 - fm_cyp)
        else:  # MBI
            # CYP fraction reduced by inactivation
            enzyme_inact = kinact_per_h * cp_i / (ki_mg_L + cp_i) * enz * dt_h
            enz_recovery = kdeg_per_h * (1.0 - enz) * dt_h
            new_enz = max(0.0, min(1.0, enz - enzyme_inact + enz_recovery))
            enzyme_frac[i] = new_enz
            cl_frac = fm_cyp * new_enz + (1.0 - fm_cyp)

        cl_eff = cl_victim_L_per_h * cl_frac
        cl_eff_arr[i] = cl_eff

        # Victim PK with inhibited CL
        ke_v_eff = cl_eff / vd_victim_L
        if route == "oral":
            abs_v = ka_victim_per_h * a_gut_v
            a_gut_v = max(0.0, a_gut_v - abs_v * dt_h)
            dc_v = (abs_v - ke_v_eff * cp_v * vd_victim_L) / vd_victim_L * dt_h
        else:
            dc_v = -ke_v_eff * cp_v * dt_h

        c_v[i] = max(0.0, cp_v + dc_v)

    # AUC with DDI
    auc_ddi = float(np_trapz(c_v, t))
    cmax_v = float(np.max(c_v))

    # AUC without inhibitor (analytical approximation)
    if route == "oral":
        auc_control = f_victim * dose_victim_mg / cl_victim_L_per_h
    else:
        auc_control = dose_victim_mg / cl_victim_L_per_h

    aucr = auc_ddi / auc_control if auc_control > 0 else 1.0

    # Fill CL at t=0
    cl_eff_arr[0] = cl_victim_L_per_h

    notes = (
        f"DDI: {inhibitor_name} ({inhibition_type}) inhibits {victim_name}. "
        f"Ki={ki_mg_L:.3f} mg/L. AUCR={aucr:.2f}x. "
        f"AUC_control={auc_control:.1f}, AUC_DDI={auc_ddi:.1f} mg*h/L."
    )

    return DDIPKResult(
        victim_name=victim_name,
        inhibitor_name=inhibitor_name,
        inhibition_type=inhibition_type,
        dose_victim_mg=dose_victim_mg,
        dose_inhibitor_mg=dose_inhibitor_mg,
        times_h=t.tolist(),
        c_victim_mg_L=c_v.tolist(),
        c_inhibitor_mg_L=c_i.tolist(),
        cl_inhibited=cl_eff_arr.tolist(),
        auc_victim_control=auc_control,
        auc_victim_ddi=auc_ddi,
        aucr=aucr,
        cmax_victim=cmax_v,
        notes=notes,
    )


__all__ = ["DDIPKResult", "simulate_ddi_pk"]
