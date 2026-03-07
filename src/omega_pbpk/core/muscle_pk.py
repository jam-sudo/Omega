"""Skeletal muscle pharmacokinetics — 2-compartment plasma + muscle model.

Models drug distribution into skeletal muscle using perfusion-limited kinetics.
Muscle blood flow determines the rate of drug transfer between central plasma
and the muscle compartment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass
class MusclePKResult:
    """Result of skeletal muscle PK simulation."""

    drug_name: str
    dose_mg: float
    kp_muscle: float
    muscle_blood_flow_L_per_h: float
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_muscle_mg_L: list[float] = field(default_factory=list)
    cmax_plasma: float = 0.0
    auc_plasma: float = 0.0
    muscle_plasma_ss_ratio: float = 0.0
    notes: str = ""


def simulate_muscle_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    muscle_blood_flow_L_per_h: float,
    kp_muscle: float,
    t_end_h: float,
    dt_h: float = 0.01,
    route: str = "iv",
    ka_per_h: float = 1.0,
) -> MusclePKResult:
    """Simulate drug distribution into skeletal muscle using 2-compartment model.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose in mg.
    cl_L_per_h : float
        Total clearance (L/h); must be > 0.
    vd_L : float
        Total volume of distribution (L); must be > 0.
    muscle_blood_flow_L_per_h : float
        Skeletal muscle blood flow (L/h); must be > 0.
    kp_muscle : float
        Muscle-to-plasma partition coefficient; must be > 0.
    t_end_h : float
        Simulation end time (h); must be > 0.
    dt_h : float
        Integration time step (h); must be > 0.
    route : str
        Administration route: "iv" (bolus) or "oral".
    ka_per_h : float
        First-order oral absorption rate (1/h); used only when route="oral".

    Returns
    -------
    MusclePKResult
    """
    # --- Input validation ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if muscle_blood_flow_L_per_h <= 0:
        raise ValueError("muscle_blood_flow_L_per_h must be > 0")
    if kp_muscle <= 0:
        raise ValueError("kp_muscle must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")

    # --- Volume partitioning ---
    # Muscle is ~40% of body volume
    vd_muscle = vd_L * 0.4 * kp_muscle
    vd_central = max(1.0, vd_L - vd_muscle)

    # --- Elimination rate constant (from central compartment) ---
    ke = cl_L_per_h / vd_central

    # --- Time array ---
    n_steps = max(2, int(math.ceil(t_end_h / dt_h)) + 1)
    times = [i * dt_h for i in range(n_steps)]
    # Clamp last time to t_end_h
    if times[-1] > t_end_h:
        times[-1] = t_end_h

    # --- Initial conditions ---
    c_central = 0.0
    c_muscle = 0.0
    gut_mass = 0.0  # only used for oral route

    if route == "iv":
        c_central = dose_mg / vd_central
    else:
        # oral: drug starts in gut
        gut_mass = dose_mg

    c_plasma_list: list[float] = []
    c_muscle_list: list[float] = []

    # --- Forward Euler integration ---
    for i, _t in enumerate(times):
        c_plasma_list.append(c_central)
        c_muscle_list.append(c_muscle)

        if i == len(times) - 1:
            break

        dt_step = times[i + 1] - times[i]

        # Perfusion-limited transfer (plasma→muscle driving force)
        transfer = muscle_blood_flow_L_per_h * (c_central - c_muscle / kp_muscle)

        # Oral absorption term
        abs_rate = 0.0
        if route == "oral":
            abs_rate = ka_per_h * gut_mass
            d_gut = -abs_rate * dt_step
            gut_mass = max(0.0, gut_mass + d_gut)

        # dC_central/dt
        dc_central = (
            -ke * c_central
            - transfer / vd_central
            + abs_rate / vd_central
        )
        # dC_muscle/dt
        dc_muscle = transfer / vd_muscle

        c_central = max(0.0, c_central + dc_central * dt_step)
        c_muscle = max(0.0, c_muscle + dc_muscle * dt_step)

    # --- PK metrics ---
    t_arr = np.array(times, dtype=np.float64)
    cp_arr = np.array(c_plasma_list, dtype=np.float64)

    cmax_plasma = float(np.max(cp_arr))
    auc_plasma = float(np_trapz(cp_arr, t_arr))

    # Steady-state muscle-to-plasma ratio: C_muscle_ss / C_plasma_ss = kp_muscle
    # (at true steady state, transfer rate = 0 → C_muscle = kp_muscle * C_plasma)
    # Estimate from terminal phase if profile has sufficient length
    muscle_plasma_ss_ratio = kp_muscle  # theoretical steady-state value

    # Refine from last 10% of simulation if concentrations are non-zero
    tail_start = max(0, int(len(times) * 0.9))
    if tail_start < len(times):
        cp_tail = cp_arr[tail_start:]
        cm_tail = np.array(c_muscle_list[tail_start:], dtype=np.float64)
        nz_mask = cp_tail > 1e-12
        if np.any(nz_mask):
            ratios = cm_tail[nz_mask] / cp_tail[nz_mask]
            muscle_plasma_ss_ratio = float(np.mean(ratios))

    notes = (
        f"2-compartment plasma+muscle model; route={route}; "
        f"Vd_central={vd_central:.2f} L; Vd_muscle={vd_muscle:.2f} L; "
        f"ke={ke:.4f} /h; theoretical SS ratio=kp_muscle={kp_muscle}"
    )

    return MusclePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        kp_muscle=kp_muscle,
        muscle_blood_flow_L_per_h=muscle_blood_flow_L_per_h,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_muscle_mg_L=c_muscle_list,
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        muscle_plasma_ss_ratio=muscle_plasma_ss_ratio,
        notes=notes,
    )


def exercise_effect(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    kp_muscle: float,
    muscle_flow_normal: float,
    muscle_flow_exercise: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> dict:
    """Compare drug muscle PK under normal vs. exercise conditions.

    During exercise, muscle blood flow increases 3-5x, enhancing drug
    distribution into skeletal muscle.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        IV bolus dose (mg).
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Total volume of distribution (L).
    kp_muscle : float
        Muscle-to-plasma partition coefficient.
    muscle_flow_normal : float
        Resting skeletal muscle blood flow (L/h); must be > 0.
    muscle_flow_exercise : float
        Exercise skeletal muscle blood flow (L/h); must be > muscle_flow_normal.
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    dict
        Keys: auc_plasma_normal, auc_plasma_exercise, cmax_plasma_normal,
        cmax_plasma_exercise, cmax_muscle_normal, cmax_muscle_exercise,
        auc_ratio_exercise_vs_normal, flow_ratio, notes.
    """
    # --- Input validation ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if kp_muscle <= 0:
        raise ValueError("kp_muscle must be > 0")
    if muscle_flow_normal <= 0:
        raise ValueError("muscle_flow_normal must be > 0")
    if muscle_flow_exercise <= 0:
        raise ValueError("muscle_flow_exercise must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    normal_result = simulate_muscle_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        muscle_blood_flow_L_per_h=muscle_flow_normal,
        kp_muscle=kp_muscle,
        t_end_h=t_end_h,
        dt_h=dt_h,
        route="iv",
    )

    exercise_result = simulate_muscle_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        muscle_blood_flow_L_per_h=muscle_flow_exercise,
        kp_muscle=kp_muscle,
        t_end_h=t_end_h,
        dt_h=dt_h,
        route="iv",
    )

    cmax_muscle_normal = (
        float(max(normal_result.c_muscle_mg_L)) if normal_result.c_muscle_mg_L else 0.0
    )
    cmax_muscle_exercise = (
        float(max(exercise_result.c_muscle_mg_L)) if exercise_result.c_muscle_mg_L else 0.0
    )

    flow_ratio = muscle_flow_exercise / muscle_flow_normal
    auc_ratio = (
        exercise_result.auc_plasma / normal_result.auc_plasma
        if normal_result.auc_plasma > 0
        else float("nan")
    )

    return {
        "auc_plasma_normal": normal_result.auc_plasma,
        "auc_plasma_exercise": exercise_result.auc_plasma,
        "cmax_plasma_normal": normal_result.cmax_plasma,
        "cmax_plasma_exercise": exercise_result.cmax_plasma,
        "cmax_muscle_normal": cmax_muscle_normal,
        "cmax_muscle_exercise": cmax_muscle_exercise,
        "auc_ratio_exercise_vs_normal": auc_ratio,
        "flow_ratio": flow_ratio,
        "notes": (
            f"Exercise increases muscle blood flow {flow_ratio:.1f}x "
            f"({muscle_flow_normal:.1f} → {muscle_flow_exercise:.1f} L/h). "
            f"Muscle Cmax {cmax_muscle_normal:.3f} → {cmax_muscle_exercise:.3f} mg/L."
        ),
    }


__all__ = ["MusclePKResult", "simulate_muscle_pk", "exercise_effect"]
