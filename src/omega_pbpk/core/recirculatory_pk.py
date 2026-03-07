"""Recirculatory pharmacokinetic model — Phase 264.

A 4-compartment model separating:
  1. Central (blood/plasma) — IV input, rapid equilibration
  2. Peripheral fast (well-perfused tissues: liver, kidneys, lungs)
  3. Peripheral slow (poorly-perfused tissues: muscle, fat)
  4. Effect compartment (optional ke0 link model)

References
----------
- Upton RN, Anesth Analg. 1999;89(4):942-50
- Krejcie TC & Avram MJ, Anesthesiology. 1999;91(1):148-57
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class RecirculatoryPKResult:
    """Result from recirculatory PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float]
    c_central_mg_L: list[float]
    c_fast_mg_L: list[float]
    c_slow_mg_L: list[float]
    c_effect_mg_L: list[float]
    cmax_central: float
    auc_central: float
    t_half_central_h: float
    effect_cmax: float
    effect_tmax_h: float
    notes: str


def simulate_recirculatory_pk(
    drug_name: str,
    dose_mg: float,
    cl_central_L_per_h: float,
    v_central_L: float,
    q_fast_L_per_h: float,
    v_fast_L: float,
    q_slow_L_per_h: float,
    v_slow_L: float,
    ke0_per_h: float = 0.5,
    v_effect_L: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> RecirculatoryPKResult:
    """Simulate recirculatory PK model (IV bolus).

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : IV bolus dose (mg). Must be > 0.
    cl_central_L_per_h : Clearance from central compartment (L/h). Must be > 0.
    v_central_L : Volume of central compartment (L). Must be > 0.
    q_fast_L_per_h : Inter-compartmental clearance to fast periphery (L/h). Must be > 0.
    v_fast_L : Volume of fast peripheral compartment (L). Must be > 0.
    q_slow_L_per_h : Inter-compartmental clearance to slow periphery (L/h). Must be > 0.
    v_slow_L : Volume of slow peripheral compartment (L). Must be > 0.
    ke0_per_h : Effect compartment equilibration rate (h^-1). Must be > 0.
    v_effect_L : Effect compartment volume (L, small). Must be > 0.
    t_end_h : Simulation end time (h).
    dt_h : Time step (h).

    Returns
    -------
    RecirculatoryPKResult
    """
    for name, val in [
        ("dose_mg", dose_mg),
        ("cl_central_L_per_h", cl_central_L_per_h),
        ("v_central_L", v_central_L),
        ("q_fast_L_per_h", q_fast_L_per_h),
        ("v_fast_L", v_fast_L),
        ("q_slow_L_per_h", q_slow_L_per_h),
        ("v_slow_L", v_slow_L),
        ("ke0_per_h", ke0_per_h),
        ("v_effect_L", v_effect_L),
    ]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0")

    n_steps = int(t_end_h / dt_h) + 1
    t = np.linspace(0.0, t_end_h, n_steps)

    c_cen = np.zeros(n_steps)
    c_fast = np.zeros(n_steps)
    c_slow = np.zeros(n_steps)
    c_eff = np.zeros(n_steps)

    c_cen[0] = dose_mg / v_central_L

    ke_cen = cl_central_L_per_h / v_central_L
    k12 = q_fast_L_per_h / v_central_L
    k21 = q_fast_L_per_h / v_fast_L
    k13 = q_slow_L_per_h / v_central_L
    k31 = q_slow_L_per_h / v_slow_L
    k1e = ke0_per_h  # from central to effect (negligible volume)

    for i in range(1, n_steps):
        cc = c_cen[i - 1]
        cf = c_fast[i - 1]
        cs = c_slow[i - 1]
        ce = c_eff[i - 1]

        dc_cen = (-ke_cen * cc - k12 * cc + k21 * cf - k13 * cc + k31 * cs - k1e * cc) * dt_h
        dc_fast = (k12 * cc - k21 * cf) * dt_h
        dc_slow = (k13 * cc - k31 * cs) * dt_h
        dc_eff = k1e * (cc - ce) * dt_h

        c_cen[i] = max(0.0, cc + dc_cen)
        c_fast[i] = max(0.0, cf + dc_fast)
        c_slow[i] = max(0.0, cs + dc_slow)
        c_eff[i] = max(0.0, ce + dc_eff)

    cmax = float(np.max(c_cen))
    auc = float(np_trapz(c_cen, t))

    # Terminal t1/2: approximate from last 30% of profile
    start_idx = int(n_steps * 0.7)
    end_idx = n_steps - 1
    if c_cen[start_idx] > 0 and c_cen[end_idx] > 0:
        dt_interval = t[end_idx] - t[start_idx]
        slope = (math.log(c_cen[end_idx]) - math.log(c_cen[start_idx])) / dt_interval
        t_half = -math.log(2) / slope if slope < 0 else t_end_h
    else:
        t_half = t_end_h

    effect_cmax = float(np.max(c_eff))
    effect_tmax = float(t[np.argmax(c_eff)])

    notes = (
        f"{drug_name}: IV {dose_mg} mg. Cmax={cmax:.3f} mg/L, AUC={auc:.2f} mg*h/L, "
        f"t\u00bd\u2248{t_half:.2f}h. Effect Cmax={effect_cmax:.3f} mg/L at {effect_tmax:.1f}h."
    )

    return RecirculatoryPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=t.tolist(),
        c_central_mg_L=c_cen.tolist(),
        c_fast_mg_L=c_fast.tolist(),
        c_slow_mg_L=c_slow.tolist(),
        c_effect_mg_L=c_eff.tolist(),
        cmax_central=cmax,
        auc_central=auc,
        t_half_central_h=t_half,
        effect_cmax=effect_cmax,
        effect_tmax_h=effect_tmax,
        notes=notes,
    )


__all__ = ["RecirculatoryPKResult", "simulate_recirculatory_pk"]
