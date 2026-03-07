"""
Phase 709 — Gastric emptying-controlled drug absorption model.

Models stomach → small intestine → plasma transit with food-state effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["GastricEmptyingResult", "simulate_gastric_emptying_pk", "compare_fed_fasted"]


@dataclass
class GastricEmptyingResult:
    """Result of gastric emptying PK simulation."""

    drug_name: str
    dose_mg: float
    food_state: str
    kge_per_h: float

    times_h: list = field(default_factory=list)
    a_stomach_mg: list = field(default_factory=list)
    a_intestine_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)

    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t_half_h: float = 0.0

    lag_time_h: float = 0.0
    tmax_without_lag_h: float = 0.0
    food_delay_ratio: float = 0.0

    notes: str = ""


def _validate_inputs(
    dose_mg: float,
    food_state: str,
    f_abs: float,
    ka_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    kge: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if food_state not in {"fasted", "fed"}:
        raise ValueError(f"food_state must be 'fasted' or 'fed', got '{food_state}'")
    if not (0 < f_abs <= 1):
        raise ValueError(f"f_abs must be in (0, 1], got {f_abs}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if kge <= 0:
        raise ValueError(f"kge must be > 0, got {kge}")


def simulate_gastric_emptying_pk(
    drug_name: str,
    dose_mg: float,
    food_state: str = "fasted",
    ka_per_h: float = 1.5,
    f_abs: float = 0.9,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    kge_fasted_per_h: float = 2.0,
    kge_fed_per_h: float = 0.5,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> GastricEmptyingResult:
    """Simulate gastric emptying-controlled drug absorption.

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose in mg.
        food_state: 'fasted' or 'fed'.
        ka_per_h: Intestinal absorption rate constant (1/h).
        f_abs: Fraction absorbed from intestine.
        cl_L_per_h: Systemic clearance (L/h).
        vd_L: Volume of distribution (L).
        kge_fasted_per_h: Gastric emptying rate in fasted state (1/h).
        kge_fed_per_h: Gastric emptying rate in fed state (1/h).
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        GastricEmptyingResult dataclass.
    """
    kge = kge_fasted_per_h if food_state == "fasted" else kge_fed_per_h
    _validate_inputs(dose_mg, food_state, f_abs, ka_per_h, cl_L_per_h, vd_L, kge)

    ke = cl_L_per_h / vd_L

    # Initial conditions
    a_stomach = dose_mg
    a_intestine = 0.0
    c_plasma = 0.0

    times_h = []
    a_stomach_list = []
    a_intestine_list = []
    c_plasma_list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        times_h.append(t)
        a_stomach_list.append(a_stomach)
        a_intestine_list.append(a_intestine)
        c_plasma_list.append(c_plasma)

        # Forward Euler
        da_stomach = -kge * a_stomach
        da_intestine = kge * a_stomach - ka_per_h * a_intestine
        dc_plasma = (ka_per_h * f_abs * a_intestine / vd_L) - ke * c_plasma

        a_stomach = max(0.0, a_stomach + da_stomach * dt_h)
        a_intestine = max(0.0, a_intestine + da_intestine * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)

        t += dt_h

    # PK metrics
    cmax = max(c_plasma_list)
    tmax = times_h[c_plasma_list.index(cmax)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])

    t_half = math.log(2) / ke

    # Lag time: first time A_intestine > 5% of its maximum
    max_intestine = max(a_intestine_list) if max(a_intestine_list) > 0 else 1.0
    threshold = 0.05 * max_intestine
    lag_time_h = 0.0
    for i, ai in enumerate(a_intestine_list):
        if ai > threshold:
            lag_time_h = times_h[i]
            break

    tmax_without_lag = max(0.0, tmax - lag_time_h)

    notes = (
        f"Food state: {food_state}. kge={kge:.2f}/h. "
        f"Lag time ~{lag_time_h:.2f} h before significant intestinal absorption. "
        f"t½={t_half:.2f} h."
    )

    return GastricEmptyingResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        food_state=food_state,
        kge_per_h=kge,
        times_h=times_h,
        a_stomach_mg=a_stomach_list,
        a_intestine_mg=a_intestine_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_h=t_half,
        lag_time_h=lag_time_h,
        tmax_without_lag_h=tmax_without_lag,
        food_delay_ratio=0.0,
        notes=notes,
    )


def compare_fed_fasted(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float = 1.5,
    f_abs: float = 0.9,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
) -> dict:
    """Compare fed vs fasted gastric emptying PK.

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose in mg.
        ka_per_h: Intestinal absorption rate constant (1/h).
        f_abs: Fraction absorbed from intestine.
        cl_L_per_h: Systemic clearance (L/h).
        vd_L: Volume of distribution (L).

    Returns:
        dict with keys: fasted, fed, tmax_delay_h, auc_ratio, notes.
    """
    fasted_result = simulate_gastric_emptying_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        food_state="fasted",
        ka_per_h=ka_per_h,
        f_abs=f_abs,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
    )
    fed_result = simulate_gastric_emptying_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        food_state="fed",
        ka_per_h=ka_per_h,
        f_abs=f_abs,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
    )

    tmax_delay_h = fed_result.tmax_h - fasted_result.tmax_h
    auc_ratio = (
        fed_result.auc_mg_h_per_L / fasted_result.auc_mg_h_per_L
        if fasted_result.auc_mg_h_per_L > 0
        else 1.0
    )

    notes = (
        f"Fed state delays Tmax by {tmax_delay_h:.2f} h. "
        f"AUC ratio (fed/fasted) = {auc_ratio:.3f}. "
        f"Fasted kge={fasted_result.kge_per_h}/h, fed kge={fed_result.kge_per_h}/h."
    )

    return {
        "fasted": fasted_result,
        "fed": fed_result,
        "tmax_delay_h": tmax_delay_h,
        "auc_ratio": auc_ratio,
        "notes": notes,
    }
