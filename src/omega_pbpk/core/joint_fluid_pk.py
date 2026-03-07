"""Phase 995 — Synovial joint fluid PK model.

Models drug distribution into synovial joint fluid via a 2-compartment
Forward Euler ODE (plasma + synovial fluid). Relevant for NSAIDs,
corticosteroids, and DMARDs in rheumatoid arthritis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class JointFluidPKResult:
    """Result of a joint fluid PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_joint_mg_L: list = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    cmax_joint_mg_L: float = 0.0
    tmax_joint_h: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_joint_mg_h_per_L: float = 0.0
    joint_to_plasma_ratio: float = 0.0
    kp_joint: float = 0.0
    t_lag_to_joint_h: float = 0.0
    notes: str = ""


def simulate_joint_fluid_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 1.0,
    fup: float = 0.5,
    route: str = "oral",
    v_plasma_L: float = 5.0,
    v_joint_mL: float = 3.0,
    cl_plasma_L_per_h: float = 5.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> JointFluidPKResult:
    """Simulate drug distribution into synovial joint fluid.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    logp:
        Octanol-water partition coefficient (log scale).
    fup:
        Fraction unbound in plasma (0-1).
    route:
        Administration route: "oral" or "iv".
    v_plasma_L:
        Plasma volume in litres.
    v_joint_mL:
        Synovial fluid volume in millilitres.
    cl_plasma_L_per_h:
        Plasma clearance in L/h.
    t_end_h:
        Simulation duration in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    JointFluidPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if v_plasma_L <= 0:
        raise ValueError(f"v_plasma_L must be > 0, got {v_plasma_L}")
    if v_joint_mL <= 0:
        raise ValueError(f"v_joint_mL must be > 0, got {v_joint_mL}")
    if cl_plasma_L_per_h <= 0:
        raise ValueError(f"cl_plasma_L_per_h must be > 0, got {cl_plasma_L_per_h}")
    if route not in {"oral", "iv"}:
        raise ValueError(f"route must be 'oral' or 'iv', got {route!r}")

    # --- Derived parameters ---
    v_joint_L = v_joint_mL / 1000.0

    # Kp for synovial joint fluid (less lipophilicity-dependent than tissues)
    kp_joint = max(0.2, 0.5 + 0.15 * logp * fup)

    # Synovial membrane permeation rate constant (hydrophilic drugs cross faster
    # via aqueous channels; lipophilic drugs are slower through this route)
    k_syn = max(0.05, 0.2 - 0.03 * logp)  # /h

    # Rate constants
    k_elim = cl_plasma_L_per_h / v_plasma_L  # /h
    # Transfer rate joint→plasma (back-flux) to reach equilibrium at kp_joint
    k_back = k_syn / kp_joint  # /h

    # Absorption parameters for oral
    ka = 0.8  # /h
    f_abs = 0.9  # fraction absorbed

    # --- Initialise state ---
    c_plasma = 0.0  # mg/L
    c_joint = 0.0  # mg/L
    a_gut = 0.0  # mg

    if route == "iv":
        c_plasma = dose_mg / v_plasma_L
    else:
        a_gut = dose_mg * f_abs

    # --- Forward Euler integration ---
    n_steps = int(math.ceil(t_end_h / dt_h))

    times: list[float] = []
    c_plasma_arr: list[float] = []
    c_joint_arr: list[float] = []

    for step in range(n_steps + 1):
        t = step * dt_h
        times.append(t)
        c_plasma_arr.append(c_plasma)
        c_joint_arr.append(c_joint)

        if step == n_steps:
            break

        # Absorption input (mg/h)
        if route == "oral":
            dA_gut = -ka * a_gut
            absorption_rate = ka * a_gut / v_plasma_L  # mg/L/h
        else:
            dA_gut = 0.0
            absorption_rate = 0.0

        # ODEs
        # dC_plasma/dt = input - k_elim*C_plasma - k_syn*C_plasma + k_back*C_joint
        d_plasma = absorption_rate - k_elim * c_plasma - k_syn * c_plasma + k_back * c_joint
        # dC_joint/dt = k_syn * C_plasma * (V_plasma / V_joint) - k_back * C_joint
        d_joint = k_syn * c_plasma * (v_plasma_L / v_joint_L) - k_back * c_joint

        # Euler update
        a_gut += dA_gut * dt_h
        if a_gut < 0.0:
            a_gut = 0.0

        c_plasma += d_plasma * dt_h
        c_joint += d_joint * dt_h

        if c_plasma < 0.0:
            c_plasma = 0.0
        if c_joint < 0.0:
            c_joint = 0.0

    # --- Summary statistics ---
    cmax_plasma = max(c_plasma_arr)
    cmax_joint = max(c_joint_arr)

    # tmax_joint: time at which joint concentration is maximum
    idx_cmax_joint = c_joint_arr.index(cmax_joint)
    tmax_joint = times[idx_cmax_joint]

    # t_lag: first time when C_joint > 10% of Cmax_joint
    threshold = 0.1 * cmax_joint
    t_lag = 0.0
    for i, cj in enumerate(c_joint_arr):
        if cj > threshold:
            t_lag = times[i]
            break

    # Trapezoidal AUC
    auc_plasma = 0.0
    auc_joint = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        auc_plasma += 0.5 * (c_plasma_arr[i] + c_plasma_arr[i + 1]) * dt
        auc_joint += 0.5 * (c_joint_arr[i] + c_joint_arr[i + 1]) * dt

    joint_to_plasma_ratio = auc_joint / auc_plasma if auc_plasma > 0.0 else 0.0

    notes = (
        f"k_syn={k_syn:.4f}/h, kp_joint={kp_joint:.3f}, "
        f"k_elim={k_elim:.4f}/h, V_joint={v_joint_L * 1000:.1f} mL"
    )

    return JointFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma_arr,
        c_joint_mg_L=c_joint_arr,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_joint_mg_L=cmax_joint,
        tmax_joint_h=tmax_joint,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_joint_mg_h_per_L=auc_joint,
        joint_to_plasma_ratio=joint_to_plasma_ratio,
        kp_joint=kp_joint,
        t_lag_to_joint_h=t_lag,
        notes=notes,
    )
