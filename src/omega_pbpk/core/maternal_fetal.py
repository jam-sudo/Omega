"""Maternal-fetal PBPK model for placental drug transfer.

A simplified two-compartment maternal-fetal model with passive placental
transfer. The placenta is modeled as a membrane with a permeability-surface
area product (PS) for maternofetal and fetoplacental fluxes.

Model structure
---------------
Maternal plasma (Cm):
  dCm/dt = -CL_m/Vd_m * Cm - PS/Vd_m * (Cm - Cf/Kp_pla) + Rate_in(t)

Fetal plasma (Cf):
  dCf/dt = PS/Vd_f * (Cm - Cf/Kp_pla) - CL_f/Vd_f * Cf

where:
  CL_m    = maternal systemic clearance (L/h)
  Vd_m    = maternal volume of distribution (L)
  PS      = placental permeability-surface product (L/h)
  Kp_pla  = fetal-to-maternal partition coefficient (unitless)
  Vd_f    = fetal volume of distribution (L)
  CL_f    = fetal clearance (L/h), usually small

References
----------
- Ke AB et al., Clin Pharmacol Ther. 2012;91(6):1017–31 (mechanistic placental model)
- De Sousa Mendes M et al., Br J Clin Pharmacol. 2015;80(6):1340–50 (pregnancy PBPK)
- Abduljalil K et al., Clin Pharmacokinet. 2012;51(6):365–96 (physiological changes)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class MaternalFetalResult:
    """Results from maternal-fetal PBPK simulation.

    Attributes
    ----------
    drug_name       : Name of the drug.
    gestational_age : Gestational age (weeks) used for parameterisation.
    times_h         : Simulation time points (h).
    cm_mg_L         : Maternal plasma concentration profile (mg/L).
    cf_mg_L         : Fetal plasma concentration profile (mg/L).
    cmax_maternal   : Maternal peak concentration (mg/L).
    cmax_fetal      : Fetal peak concentration (mg/L).
    auc_maternal    : Maternal AUC 0→t (mg·h/L).
    auc_fetal       : Fetal AUC 0→t (mg·h/L).
    fetal_exposure_ratio : AUC_fetal / AUC_maternal (0–1 range for most drugs).
    tmax_fetal_h    : Time of fetal Cmax (h).
    kp_placenta     : Placental partition coefficient used.
    ps_L_per_h      : Placental PS used (L/h).
    notes           : Human-readable summary.
    """

    drug_name: str
    gestational_age: float
    times_h: list[float]
    cm_mg_L: list[float]
    cf_mg_L: list[float]
    cmax_maternal: float
    cmax_fetal: float
    auc_maternal: float
    auc_fetal: float
    fetal_exposure_ratio: float
    tmax_fetal_h: float
    kp_placenta: float
    ps_L_per_h: float
    notes: str


def _placental_ps(gestational_age_weeks: float, fup: float, logP: float = 1.0) -> float:
    """Estimate placental PS (L/h) as a function of gestational age and fup.

    Placental blood flow increases throughout pregnancy. PS is approximated as:
        PS ≈ 0.5 × Q_pla × fup × (1 + logP_factor)

    where Q_pla (placental blood flow) scales from ~0.5 L/h at 12 weeks to
    ~3.5 L/h at term (40 weeks), based on Abduljalil 2012.

    Parameters
    ----------
    gestational_age_weeks : Gestational age in weeks (12–40).
    fup : Unbound fraction in maternal plasma.
    logP : Log octanol-water coefficient (lipophilic drugs transfer faster).

    Returns
    -------
    PS in L/h.
    """
    # Clamp gestational age
    ga = max(12.0, min(40.0, gestational_age_weeks))
    # Placental blood flow: linear interpolation 0.5 → 3.5 L/h from 12 to 40 wk
    q_pla = 0.5 + (3.5 - 0.5) * (ga - 12.0) / (40.0 - 12.0)
    # Lipophilic factor: high logP increases membrane permeability (bounded)
    logP_factor = min(2.0, max(0.5, 1.0 + 0.2 * logP))
    return 0.5 * q_pla * fup * logP_factor


def simulate_maternal_fetal(
    drug_name: str,
    dose_mg: float,
    cl_maternal_L_per_h: float,
    vd_maternal_L: float,
    gestational_age_weeks: float = 28.0,
    fup: float = 0.1,
    logP: float = 1.0,
    kp_placenta: float = 0.5,
    vd_fetal_L: float | None = None,
    cl_fetal_L_per_h: float = 0.0,
    ps_L_per_h: float | None = None,
    route: str = "iv",
    ka_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> MaternalFetalResult:
    """Simulate maternal and fetal drug concentrations with placental transfer.

    Parameters
    ----------
    drug_name : Name of the drug.
    dose_mg : Maternal dose (mg).
    cl_maternal_L_per_h : Maternal systemic clearance (L/h).
    vd_maternal_L : Maternal volume of distribution (L).
    gestational_age_weeks : Gestational age in weeks (12–40).
    fup : Unbound fraction in maternal plasma (0–1).
    logP : Log octanol-water coefficient.
    kp_placenta : Fetal-to-maternal equilibrium partition coefficient.
    vd_fetal_L : Fetal Vd (L). Defaults to 0.15 × Vd_maternal (Ke 2012).
    cl_fetal_L_per_h : Fetal clearance (L/h). Usually negligible.
    ps_L_per_h : Override for placental PS (L/h). Auto-estimated if None.
    route : 'iv' (bolus) or 'oral' (first-order absorption with ka_per_h).
    ka_per_h : First-order oral absorption rate constant (h⁻¹).
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    MaternalFetalResult with full time-course and summary metrics.

    Raises
    ------
    ValueError : If any required parameter is non-positive.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_maternal_L_per_h <= 0:
        raise ValueError("cl_maternal_L_per_h must be > 0")
    if vd_maternal_L <= 0:
        raise ValueError("vd_maternal_L must be > 0")
    if not (0 < fup <= 1):
        raise ValueError("fup must be in (0, 1]")
    if not (12.0 <= gestational_age_weeks <= 42.0):
        raise ValueError("gestational_age_weeks must be 12–42")

    # Fetal Vd defaults to ~15% of maternal Vd (Ke 2012)
    if vd_fetal_L is None:
        vd_fetal_L = 0.15 * vd_maternal_L

    # Placental PS
    if ps_L_per_h is None:
        ps_L_per_h = _placental_ps(gestational_age_weeks, fup, logP)

    ke_m = cl_maternal_L_per_h / vd_maternal_L  # maternal elimination rate
    ke_f = cl_fetal_L_per_h / vd_fetal_L if vd_fetal_L > 0 else 0.0  # fetal elim rate

    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    cm = np.zeros(n_steps + 1)  # maternal plasma concentration (mg/L)
    cf = np.zeros(n_steps + 1)  # fetal plasma concentration (mg/L)

    # Initial conditions
    if route == "iv":
        cm[0] = dose_mg / vd_maternal_L
        depot = 0.0
    else:
        cm[0] = 0.0
        depot = dose_mg  # oral dose in depot (mg)

    # Forward Euler ODE integration
    ps_over_vm = ps_L_per_h / vd_maternal_L
    ps_over_vf = ps_L_per_h / vd_fetal_L

    for i in range(n_steps):
        # Oral absorption flux
        if route == "oral":
            abs_rate = ka_per_h * depot  # mg/h
            ddepot = -abs_rate * dt_h
            depot += ddepot
            depot = max(0.0, depot)
            absorbed_conc = abs_rate / vd_maternal_L  # → mg/L/h → added to maternal
        else:
            absorbed_conc = 0.0

        # Placental flux: net from maternal → fetal
        # Driving force: Cm - Cf/Kp_pla (equilibrium at Cf = Kp_pla * Cm)
        flux_term = cm[i] - cf[i] / kp_placenta  # mg/L

        dcm = (-ke_m * cm[i] - ps_over_vm * flux_term + absorbed_conc) * dt_h
        dcf = (ps_over_vf * flux_term - ke_f * cf[i]) * dt_h

        cm[i + 1] = max(0.0, cm[i] + dcm)
        cf[i + 1] = max(0.0, cf[i] + dcf)

    # Summary metrics
    cmax_m = float(np.max(cm))
    cmax_f = float(np.max(cf))
    auc_m = float(np_trapz(cm, t))
    auc_f = float(np_trapz(cf, t))
    fer = auc_f / auc_m if auc_m > 1e-12 else 0.0
    tmax_f_h = float(t[int(np.argmax(cf))])

    notes = (
        f"Maternal-fetal simulation: {drug_name}, GA={gestational_age_weeks}wk, "
        f"dose={dose_mg}mg {route}. "
        f"Fetal exposure ratio (AUC_f/AUC_m) = {fer:.3f}. "
        f"Kp_placenta = {kp_placenta}, PS = {ps_L_per_h:.3f} L/h."
    )

    return MaternalFetalResult(
        drug_name=drug_name,
        gestational_age=gestational_age_weeks,
        times_h=t.tolist(),
        cm_mg_L=cm.tolist(),
        cf_mg_L=cf.tolist(),
        cmax_maternal=cmax_m,
        cmax_fetal=cmax_f,
        auc_maternal=auc_m,
        auc_fetal=auc_f,
        fetal_exposure_ratio=fer,
        tmax_fetal_h=tmax_f_h,
        kp_placenta=kp_placenta,
        ps_L_per_h=ps_L_per_h,
        notes=notes,
    )


def compare_gestational_ages(
    drug_name: str,
    dose_mg: float,
    cl_maternal_L_per_h: float,
    vd_maternal_L: float,
    ages_weeks: list[float] | None = None,
    **kwargs,
) -> list[MaternalFetalResult]:
    """Compare maternal-fetal PK across gestational ages.

    Parameters
    ----------
    ages_weeks : List of gestational ages to simulate (default: [12, 20, 28, 36, 40]).

    Returns
    -------
    List of MaternalFetalResult in the same order as ages_weeks.
    """
    if ages_weeks is None:
        ages_weeks = [12.0, 20.0, 28.0, 36.0, 40.0]
    return [
        simulate_maternal_fetal(
            drug_name,
            dose_mg,
            cl_maternal_L_per_h,
            vd_maternal_L,
            gestational_age_weeks=ga,
            **kwargs,
        )
        for ga in ages_weeks
    ]


__all__ = [
    "MaternalFetalResult",
    "simulate_maternal_fetal",
    "compare_gestational_ages",
]
