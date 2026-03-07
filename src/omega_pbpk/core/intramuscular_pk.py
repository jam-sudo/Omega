"""Intramuscular (IM) drug absorption pharmacokinetics.

Models drug release from an IM depot into systemic circulation using a
3-compartment forward Euler ODE:

    IM depot (muscle) -> Plasma -> (elimination)
                         |
                  shallow tissue redistribution (implicit in Vd)

Two release profiles are supported:

    1. First-order (aqueous solution): simple exponential depot release.
    2. Biphasic (suspension / depot formulation): fast + slow phase.

Muscle blood flow scales the absorption rate constant.

References
----------
- Toutain PL & Bousquet-Melou A, J Vet Pharmacol Ther. 2004;27(6):427-39.
- Zuidema J et al., J Pharm Sci. 1994;83(8):1151-6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["IntramuscularPKResult", "simulate_im_pk", "compare_im_formulations"]

_VALID_FORMULATIONS = {"solution", "depot"}


@dataclass
class IntramuscularPKResult:
    """Results from an IM pharmacokinetic simulation.

    Attributes
    ----------
    drug_name : Drug identifier.
    dose_mg : Administered dose (mg).
    formulation : 'solution' or 'depot'.
    times_h : Simulation time points (h).
    a_im_mg : Drug mass remaining in IM depot over time (mg).
    c_plasma_mg_L : Plasma concentration over time (mg/L).
    cmax_mg_L : Peak plasma concentration (mg/L).
    tmax_h : Time of Cmax (h).
    auc_mg_h_per_L : AUC from 0 to t_end (mg*h/L), trapezoidal.
    t_half_h : Plasma elimination half-life (h).
    lag_time_h : Time to reach 5% of Cmax (h).
    f_absorbed : Fraction of dose absorbed into plasma by t_end.
    notes : Summary text.
    """

    drug_name: str
    dose_mg: float
    formulation: str
    times_h: list = field(default_factory=list)
    a_im_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t_half_h: float = 0.0
    lag_time_h: float = 0.0
    f_absorbed: float = 0.0
    notes: str = ""


def _adjust_ka_for_blood_flow(ka_per_h: float, blood_flow_mL_per_min: float) -> float:
    """Scale ka by muscle blood flow relative to 5 mL/min reference."""
    if blood_flow_mL_per_min <= 0:
        return ka_per_h
    return ka_per_h * math.sqrt(blood_flow_mL_per_min / 5.0)


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def simulate_im_pk(
    drug_name: str,
    dose_mg: float,
    formulation: str = "solution",
    ka_im_per_h: float = 1.0,
    f_fast: float = 0.7,
    ka_fast_per_h: float = 2.0,
    ka_slow_per_h: float = 0.05,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    muscle_flow_mL_per_min: float = 5.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> IntramuscularPKResult:
    """Simulate intramuscular PK using forward Euler integration.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Administered dose (mg). Must be > 0.
    formulation : 'solution' (first-order release) or 'depot' (biphasic).
    ka_im_per_h : First-order absorption rate constant (1/h) for solution.
    f_fast : Fast-phase fraction for depot formulation (0 < f_fast <= 1).
    ka_fast_per_h : Fast-phase rate constant (1/h) for depot.
    ka_slow_per_h : Slow-phase rate constant (1/h) for depot.
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).
    muscle_flow_mL_per_min : Muscle blood flow (mL/min); scales ka.
    t_end_h : Simulation end time (h).
    dt_h : Forward Euler step size (h).

    Returns
    -------
    IntramuscularPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(f"formulation must be one of {_VALID_FORMULATIONS}")
    if not (0 < f_fast <= 1):
        raise ValueError("f_fast must be in (0, 1]")
    if ka_im_per_h <= 0 or ka_fast_per_h <= 0 or ka_slow_per_h <= 0:
        raise ValueError("all ka values must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    ke = cl_L_per_h / vd_L
    t_half = math.log(2.0) / ke

    n_steps = max(int(t_end_h / dt_h), 1)

    times: list[float] = []
    a_im_list: list[float] = []
    c_plasma_list: list[float] = []

    if formulation == "solution":
        # First-order release; single depot compartment
        ka_adj = _adjust_ka_for_blood_flow(ka_im_per_h, muscle_flow_mL_per_min)
        a_im = dose_mg
        c_plasma = 0.0

        for i in range(n_steps + 1):
            t = i * dt_h
            times.append(t)
            a_im_list.append(a_im)
            c_plasma_list.append(c_plasma)
            if i < n_steps:
                # Forward Euler step
                da_im = -ka_adj * a_im
                dc_plasma = (ka_adj * a_im) / vd_L - ke * c_plasma
                a_im = max(a_im + da_im * dt_h, 0.0)
                c_plasma = max(c_plasma + dc_plasma * dt_h, 0.0)
    else:
        # Biphasic depot: fast + slow sub-compartments
        ka_fast_adj = _adjust_ka_for_blood_flow(ka_fast_per_h, muscle_flow_mL_per_min)
        ka_slow_adj = _adjust_ka_for_blood_flow(ka_slow_per_h, muscle_flow_mL_per_min)
        a_fast = f_fast * dose_mg
        a_slow = (1.0 - f_fast) * dose_mg
        c_plasma = 0.0

        for i in range(n_steps + 1):
            t = i * dt_h
            a_im_total = a_fast + a_slow
            times.append(t)
            a_im_list.append(a_im_total)
            c_plasma_list.append(c_plasma)
            if i < n_steps:
                da_fast = -ka_fast_adj * a_fast
                da_slow = -ka_slow_adj * a_slow
                input_rate = ka_fast_adj * a_fast + ka_slow_adj * a_slow
                dc_plasma = input_rate / vd_L - ke * c_plasma
                a_fast = max(a_fast + da_fast * dt_h, 0.0)
                a_slow = max(a_slow + da_slow * dt_h, 0.0)
                c_plasma = max(c_plasma + dc_plasma * dt_h, 0.0)

    # Derived PK parameters
    cmax = max(c_plasma_list)
    tmax = times[c_plasma_list.index(cmax)]
    auc = _trapezoidal_auc(times, c_plasma_list)

    # Lag time: time to reach 5% of Cmax
    threshold_5pct = 0.05 * cmax
    lag_time = 0.0
    for t_val, c_val in zip(times, c_plasma_list):
        if c_val >= threshold_5pct:
            lag_time = t_val
            break

    # Fraction absorbed = 1 - (remaining depot / dose)
    f_absorbed = 1.0 - a_im_list[-1] / dose_mg if dose_mg > 0 else 0.0
    f_absorbed = max(0.0, min(1.0, f_absorbed))

    notes = (
        f"IM {formulation} simulation for {drug_name}. "
        f"ke={ke:.3f}/h, t_half={t_half:.2f}h, "
        f"muscle_flow={muscle_flow_mL_per_min}mL/min. "
        f"Cmax={cmax:.4f}mg/L at Tmax={tmax:.2f}h."
    )

    return IntramuscularPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation=formulation,
        times_h=times,
        a_im_mg=a_im_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_h=t_half,
        lag_time_h=lag_time,
        f_absorbed=f_absorbed,
        notes=notes,
    )


def compare_im_formulations(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
) -> dict:
    """Compare solution vs depot IM formulations for a given drug.

    Solution: first-order release with ka_im=1.0/h.
    Depot: biphasic with slow dominant release (f_fast=0.3, ka_fast=1.0/h, ka_slow=0.05/h)
    ensuring depot Tmax > solution Tmax.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Dose in mg. Must be > 0.
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).

    Returns
    -------
    dict with keys: solution, depot, tmax_ratio, auc_ratio, notes.
    """
    sol = simulate_im_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation="solution",
        ka_im_per_h=1.0,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=72.0,
    )
    dep = simulate_im_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation="depot",
        f_fast=0.3,
        ka_fast_per_h=1.0,
        ka_slow_per_h=0.05,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_end_h=72.0,
    )

    tmax_ratio = dep.tmax_h / sol.tmax_h if sol.tmax_h > 0 else float("inf")
    auc_ratio = dep.auc_mg_h_per_L / sol.auc_mg_h_per_L if sol.auc_mg_h_per_L > 0 else float("inf")

    notes = (
        f"Depot tmax ({dep.tmax_h:.2f}h) vs solution tmax ({sol.tmax_h:.2f}h). "
        f"Tmax ratio={tmax_ratio:.2f}. AUC ratio={auc_ratio:.3f}. "
        "Depot provides prolonged release; solution has faster Cmax."
    )

    return {
        "solution": sol,
        "depot": dep,
        "tmax_ratio": tmax_ratio,
        "auc_ratio": auc_ratio,
        "notes": notes,
    }
