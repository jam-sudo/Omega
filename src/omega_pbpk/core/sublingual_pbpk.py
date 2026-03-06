"""Sublingual/buccal mucosal absorption PBPK model.

Models dual-route absorption: direct mucosal uptake (bypassing hepatic
first-pass) and swallowed fraction (subject to oral first-pass metabolism).

Model structure
---------------
Two absorption routes:

  1. Sublingual/buccal mucosa -> direct systemic (bypass liver)
  2. Saliva swallowing -> oral depot -> first-pass -> systemic

Phase 1 (t < t_contact_h):
  Drug on mucosa absorbs at ka_mucosal_per_h.

Phase 2 (t = t_contact_h):
  Remaining mucosal drug split: f_swallowed -> oral depot, rest continues
  mucosal absorption.

References
----------
- Shojaei AH, J Pharm Pharm Sci. 1998;1(1):15-30 (buccal drug delivery)
- Zhang H et al., Clin Pharmacokinet. 2002;41(9):661-80 (sublingual PK)
- Rathbone MJ et al., Adv Drug Deliv Rev. 1994;13(1-2):1-22 (oral mucosal)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class SublingualPKResult:
    """Results from sublingual/buccal PK simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    route : Administration route ('sublingual' or 'buccal').
    dose_mg : Administered dose (mg).
    times_h : Simulation time points (h).
    cp_mg_L : Plasma concentration time-course (mg/L).
    a_mucosa_mg : Drug mass on mucosa over time (mg).
    cmax_mg_L : Peak plasma concentration (mg/L).
    tmax_h : Time of Cmax (h).
    auc_0_t : AUC from 0 to t_end (mg*h/L).
    bioavailability : Fraction of dose reaching systemic circulation.
    mucosal_fraction_absorbed : Fraction absorbed directly via mucosa.
    first_pass_avoided_mg : Drug mass that bypassed hepatic first-pass (mg).
    notes : Summary text.
    """

    drug_name: str
    route: str
    dose_mg: float
    times_h: list[float]
    cp_mg_L: list[float]
    a_mucosa_mg: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_0_t: float
    bioavailability: float
    mucosal_fraction_absorbed: float
    first_pass_avoided_mg: float
    notes: str


def simulate_sublingual_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "sublingual",
    ka_mucosal_per_h: float = 3.0,
    ka_swallowed_per_h: float = 1.0,
    f_swallowed: float = 0.3,
    f_hepatic_first_pass: float = 0.5,
    cl_L_per_h: float = 1.0,
    vd_L: float = 30.0,
    t_contact_h: float = 0.25,
    t_end_h: float = 12.0,
    dt_h: float = 0.02,
) -> SublingualPKResult:
    """Simulate sublingual or buccal drug PK with dual absorption routes.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose administered (mg).
    route : 'sublingual' or 'buccal'.
    ka_mucosal_per_h : Mucosal absorption rate constant (h^-1).
    ka_swallowed_per_h : Oral absorption rate of swallowed portion (h^-1).
    f_swallowed : Fraction of remaining mucosal drug swallowed at t_contact.
    f_hepatic_first_pass : Hepatic extraction ratio for swallowed drug.
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).
    t_contact_h : Contact time before swallowing event (h).
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    SublingualPKResult with full time-course and PK summary.

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not 0 <= f_swallowed <= 1:
        raise ValueError("f_swallowed must be in [0, 1]")
    if t_contact_h <= 0:
        raise ValueError("t_contact_h must be > 0")

    ke = cl_L_per_h / vd_L
    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    cp = np.zeros(n_steps + 1)
    a_mucosa = np.zeros(n_steps + 1)
    a_oral = np.zeros(n_steps + 1)

    a_mucosa[0] = dose_mg

    total_mucosal_absorbed = 0.0
    total_oral_absorbed = 0.0
    swallowing_done = False

    for i in range(n_steps):
        ti = t[i]

        # Swallowing event at t_contact_h
        if not swallowing_done and ti >= t_contact_h:
            swallowed = f_swallowed * a_mucosa[i]
            a_oral[i] += swallowed
            a_mucosa[i] = a_mucosa[i] * (1.0 - f_swallowed)
            swallowing_done = True

        # Mucosal absorption (direct to systemic)
        abs_mucosal = ka_mucosal_per_h * a_mucosa[i]
        # Oral absorption (subject to first-pass)
        abs_oral = ka_swallowed_per_h * a_oral[i]
        oral_to_systemic = abs_oral * (1.0 - f_hepatic_first_pass)

        # ODE steps
        da_mucosa = -abs_mucosal * dt_h
        da_oral = -abs_oral * dt_h
        dcp = ((abs_mucosal + oral_to_systemic) / vd_L - ke * cp[i]) * dt_h

        a_mucosa[i + 1] = max(0.0, a_mucosa[i] + da_mucosa)
        a_oral[i + 1] = max(0.0, a_oral[i] + da_oral)
        cp[i + 1] = max(0.0, cp[i] + dcp)

        total_mucosal_absorbed += abs_mucosal * dt_h
        total_oral_absorbed += oral_to_systemic * dt_h

    cmax = float(np.max(cp))
    tmax = float(t[int(np.argmax(cp))])
    auc = float(np_trapz(cp, t))
    total_systemic = total_mucosal_absorbed + total_oral_absorbed
    bioavailability = total_systemic / dose_mg
    mucosal_frac = total_mucosal_absorbed / dose_mg
    first_pass_avoided = total_mucosal_absorbed

    notes = (
        f"Sublingual PK: {drug_name}, {dose_mg}mg via {route}. "
        f"Bioavailability={bioavailability:.1%}, mucosal fraction={mucosal_frac:.1%}. "
        f"Cmax={cmax:.4f} mg/L at {tmax:.2f} h. AUC={auc:.3f} mg*h/L."
    )

    return SublingualPKResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        times_h=t.tolist(),
        cp_mg_L=cp.tolist(),
        a_mucosa_mg=a_mucosa.tolist(),
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_0_t=auc,
        bioavailability=float(np.clip(bioavailability, 0.0, 1.0)),
        mucosal_fraction_absorbed=float(np.clip(mucosal_frac, 0.0, 1.0)),
        first_pass_avoided_mg=float(first_pass_avoided),
        notes=notes,
    )


def compare_sublingual_oral(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 1.0,
    vd_L: float = 30.0,
    f_oral_hepatic: float = 0.5,
    **kwargs,
) -> tuple[SublingualPKResult, SublingualPKResult]:
    """Compare sublingual vs. pure oral administration.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg).
    cl_L_per_h : Clearance (L/h).
    vd_L : Volume of distribution (L).
    f_oral_hepatic : Hepatic first-pass fraction.
    **kwargs : Additional parameters for simulate_sublingual_pk.

    Returns
    -------
    Tuple of (sublingual_result, oral_result).
    The oral result simulates f_swallowed=1.0 (all drug swallowed).
    """
    sublingual = simulate_sublingual_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_hepatic_first_pass=f_oral_hepatic,
        **kwargs,
    )
    oral = simulate_sublingual_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="oral",
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_swallowed=1.0,
        f_hepatic_first_pass=f_oral_hepatic,
        **kwargs,
    )
    return sublingual, oral


__all__ = [
    "SublingualPKResult",
    "simulate_sublingual_pk",
    "compare_sublingual_oral",
]
