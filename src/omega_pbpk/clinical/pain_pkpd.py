"""Opioid analgesia PK/PD model with tolerance — Phase 267.

Models mu-opioid receptor (MOR) occupancy, analgesic effect,
and tolerance development for opioids such as morphine.

References
----------
- Bouillon T et al., Anesthesiology. 2004;100(2):240-50
- Dagenais J et al., J Pharmacokinet Pharmacodyn. 2014
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class PainPKPDResult:
    """Result from opioid PK/PD simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_effect_mg_L: list[float]
    receptor_occupancy_pct: list[float]  # MOR occupancy (%)
    analgesic_effect: list[float]  # 0-1 normalized effect
    tolerance_factor: list[float]  # EC50 multiplier over time (starts at 1.0)
    cmax_plasma: float
    auc_plasma: float
    peak_analgesia: float  # Max analgesic effect (0-1)
    time_peak_analgesia_h: float
    duration_effective_h: float  # Time analgesic effect > 0.5
    notes: str


def simulate_pain_pkpd(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ke0_per_h: float = 0.5,
    ec50_mg_L: float = 0.05,
    emax: float = 1.0,
    hill_n: float = 1.5,
    tolerance_k_per_h: float = 0.02,
    tolerance_max: float = 3.0,
    route: str = "iv",
    ka_per_h: float = 1.5,
    f_oral: float = 0.35,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> PainPKPDResult:
    """Simulate opioid analgesia with tolerance development.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg). Must be > 0.
    cl_L_per_h : Systemic clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    ke0_per_h : Effect compartment equilibration rate (h^-1).
    ec50_mg_L : Baseline EC50 for analgesia (mg/L). Must be > 0.
    emax : Maximum analgesic effect (0-1). Must be in (0, 1].
    hill_n : Hill coefficient.
    tolerance_k_per_h : Tolerance build-up rate constant (h^-1). Must be >= 0.
    tolerance_max : Maximum EC50 fold-increase from tolerance. Must be >= 1.
    route : 'iv' or 'oral'.
    ka_per_h : Oral absorption rate (h^-1).
    f_oral : Oral bioavailability (0, 1].
    t_end_h : Simulation end time (h).
    dt_h : Time step (h).

    Returns
    -------
    PainPKPDResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")
    if not (0 < emax <= 1):
        raise ValueError("emax must be in (0, 1]")
    if tolerance_k_per_h < 0:
        raise ValueError("tolerance_k_per_h must be >= 0")
    if tolerance_max < 1:
        raise ValueError("tolerance_max must be >= 1")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")

    ke = cl_L_per_h / vd_L
    n_steps = int(t_end_h / dt_h) + 1
    t = np.linspace(0.0, t_end_h, n_steps)

    c_plasma = np.zeros(n_steps)
    c_effect = np.zeros(n_steps)
    tol = np.ones(n_steps)  # tolerance factor (EC50 multiplier)
    a_gut = 0.0

    if route == "iv":
        c_plasma[0] = dose_mg / vd_L
    else:
        a_gut = dose_mg * f_oral

    for i in range(1, n_steps):
        cp = c_plasma[i - 1]
        ce = c_effect[i - 1]
        tol_prev = tol[i - 1]

        if route == "oral":
            abs_r = ka_per_h * a_gut
            a_gut = max(0.0, a_gut - abs_r * dt_h)
        else:
            abs_r = 0.0

        dc_p = (abs_r - ke * cp * vd_L) / vd_L * dt_h
        dc_e = ke0_per_h * (cp - ce) * dt_h

        # Tolerance: EC50 increases toward tolerance_max driven by effect compartment occupancy
        ec50_eff = ec50_mg_L * tol_prev
        occ = ce / (ec50_eff + ce) if ce > 0 else 0.0
        dtol = tolerance_k_per_h * occ * (tolerance_max - tol_prev) * dt_h

        c_plasma[i] = max(0.0, cp + dc_p)
        c_effect[i] = max(0.0, ce + dc_e)
        tol[i] = min(tolerance_max, tol_prev + dtol)

    # Derived quantities
    ec50_arr = ec50_mg_L * tol
    occupancy = np.where(
        c_effect > 0,
        c_effect**hill_n / (ec50_arr**hill_n + c_effect**hill_n) * 100.0,
        0.0,
    )
    effect = emax * (occupancy / 100.0)

    cmax_p = float(np.max(c_plasma))
    auc_p = float(np_trapz(c_plasma, t))
    peak_analg = float(np.max(effect))
    tpeak = float(t[np.argmax(effect)])
    # Duration: time when effect > 0.5 * emax
    threshold = 0.5 * emax
    duration_eff = float(np.sum(effect > threshold) * dt_h)

    notes = (
        f"{drug_name} {dose_mg}mg {route}: Cmax={cmax_p:.3f} mg/L, "
        f"peak analgesia={peak_analg:.2f} at {tpeak:.1f}h, "
        f"duration>{threshold:.2f}={duration_eff:.1f}h."
    )

    return PainPKPDResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=t.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        c_effect_mg_L=c_effect.tolist(),
        receptor_occupancy_pct=occupancy.tolist(),
        analgesic_effect=effect.tolist(),
        tolerance_factor=tol.tolist(),
        cmax_plasma=cmax_p,
        auc_plasma=auc_p,
        peak_analgesia=peak_analg,
        time_peak_analgesia_h=tpeak,
        duration_effective_h=duration_eff,
        notes=notes,
    )


__all__ = ["PainPKPDResult", "simulate_pain_pkpd"]
