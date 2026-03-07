"""
Phase 981 — Muscle Drug Distribution (core/muscle_pk_p981.py)

Models drug distribution into skeletal muscle tissue using a 2-compartment
perfusion-limited model:
  - Plasma compartment (central)
  - Muscle tissue compartment

Relevant for: antibiotics (aminoglycosides), local anesthetics, muscle relaxants.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MusclePKResult:
    """Result of a muscle PK simulation (Phase 981, 2-compartment model)."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_muscle_mg_L: list = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    cmax_muscle_mg_L: float = 0.0
    tmax_muscle_h: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_muscle_mg_h_per_L: float = 0.0
    muscle_to_plasma_ratio: float = 0.0
    kp_muscle: float = 0.0
    notes: str = ""


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        total += 0.5 * (values[i] + values[i - 1]) * dt
    return total


def simulate_muscle_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 1.0,
    fup: float = 0.5,
    route: str = "iv",
    v_plasma_L: float = 5.0,
    v_muscle_L: float = 28.0,
    q_muscle_L_per_h: float = 0.75,
    cl_plasma_L_per_h: float = 5.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> MusclePKResult:
    """Simulate drug distribution into skeletal muscle tissue.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    logp:
        Octanol-water partition coefficient (log scale).
    fup:
        Fraction unbound in plasma (0-1).
    route:
        Administration route: "iv" or "oral".
    v_plasma_L:
        Plasma volume (L).
    v_muscle_L:
        Skeletal muscle volume (L); ~28 L for 70 kg human.
    q_muscle_L_per_h:
        Muscle blood flow / perfusion rate (L/h).
    cl_plasma_L_per_h:
        Systemic plasma clearance (L/h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    MusclePKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if v_plasma_L <= 0:
        raise ValueError("v_plasma_L must be > 0")
    if v_muscle_L <= 0:
        raise ValueError("v_muscle_L must be > 0")
    if cl_plasma_L_per_h <= 0:
        raise ValueError("cl_plasma_L_per_h must be > 0")
    if route not in {"iv", "oral"}:
        raise ValueError("route must be 'iv' or 'oral'")

    # --- Kp calculation (logP-driven, fup-adjusted) ---
    kp_muscle = max(0.1, (10.0 ** (0.3 * logp)) * fup)

    # --- Oral absorption parameters ---
    ka = 0.5  # absorption rate constant (h^-1)
    f_oral = 0.8  # oral bioavailability

    # --- Initial conditions ---
    c_plasma = dose_mg / v_plasma_L if route == "iv" else 0.0
    c_muscle = 0.0
    a_gut = dose_mg if route == "oral" else 0.0

    times: list = []
    c_plasma_list: list = []
    c_muscle_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _i in range(n_steps + 1):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_muscle_list.append(c_muscle)

        # --- Compute input rate from gut (oral only) ---
        input_rate = 0.0
        if route == "oral" and a_gut > 0.0:
            input_rate = ka * f_oral * a_gut  # mg/h absorbed into plasma
            da_gut = -ka * a_gut * dt_h
        else:
            da_gut = 0.0

        # --- ODE: perfusion-limited 2-compartment ---
        # dC_plasma/dt = input_rate/v_plasma
        #               - (q_muscle/v_plasma) * C_plasma
        #               + (q_muscle/v_muscle) * C_muscle/kp_muscle
        #               - (cl_plasma/v_plasma) * C_plasma
        dc_plasma = (
            input_rate / v_plasma_L
            - (q_muscle_L_per_h / v_plasma_L) * c_plasma
            + (q_muscle_L_per_h / v_muscle_L) * c_muscle / kp_muscle
            - (cl_plasma_L_per_h / v_plasma_L) * c_plasma
        )

        # dC_muscle/dt = (q_muscle/v_muscle) * C_plasma
        #               - (q_muscle/v_muscle) * C_muscle/kp_muscle
        dc_muscle = (q_muscle_L_per_h / v_muscle_L) * c_plasma - (
            q_muscle_L_per_h / v_muscle_L
        ) * c_muscle / kp_muscle

        # --- Forward Euler update ---
        c_plasma += dc_plasma * dt_h
        c_muscle += dc_muscle * dt_h
        a_gut += da_gut

        # Clamp to zero (numerical safety)
        if c_plasma < 0.0:
            c_plasma = 0.0
        if c_muscle < 0.0:
            c_muscle = 0.0
        if a_gut < 0.0:
            a_gut = 0.0

        t += dt_h

    # --- PK metrics ---
    cmax_plasma_mg_L = max(c_plasma_list)
    cmax_muscle_mg_L = max(c_muscle_list)

    # tmax_muscle_h: time at which muscle concentration is maximal
    idx_tmax_muscle = c_muscle_list.index(cmax_muscle_mg_L)
    tmax_muscle_h = times[idx_tmax_muscle]

    auc_plasma = _trapz(times, c_plasma_list)
    auc_muscle = _trapz(times, c_muscle_list)

    muscle_to_plasma_ratio = auc_muscle / auc_plasma if auc_plasma > 0.0 else 0.0

    notes = (
        f"Route={route}, logP={logp:.2f}, fup={fup:.2f}, "
        f"kp_muscle={kp_muscle:.3f}, muscle:plasma AUC ratio={muscle_to_plasma_ratio:.3f}"
    )

    return MusclePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_muscle_mg_L=c_muscle_list,
        cmax_plasma_mg_L=cmax_plasma_mg_L,
        cmax_muscle_mg_L=cmax_muscle_mg_L,
        tmax_muscle_h=tmax_muscle_h,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_muscle_mg_h_per_L=auc_muscle,
        muscle_to_plasma_ratio=muscle_to_plasma_ratio,
        kp_muscle=kp_muscle,
        notes=notes,
    )
