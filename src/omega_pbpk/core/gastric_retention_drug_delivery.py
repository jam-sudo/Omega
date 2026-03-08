"""Phase 505 — Gastric Retention Drug Delivery PK model.

PK model for floating/swelling gastric-retentive dosage forms that prolong
gastric residence time and modify absorption kinetics.

Two-phase model:
- Gastric phase (0 to retention_time_h): drug releases at k_release_per_h
  (zero-order from device) into stomach, absorbed at ka_gastric.
- Intestinal phase (after retention_time_h): remaining device drug dumps to GI,
  absorbed at ka_intestine.

Systemic PK: 1-compartment, k_el = CL / Vd.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "GastricRetentionResult",
    "simulate_gastric_retention",
    "compare_retention_times",
]

_VALID_FORMULATION_TYPES = {"floating", "swelling"}


@dataclass
class GastricRetentionResult:
    """Result from gastric retention drug delivery simulation."""

    drug_name: str
    dose_mg: float
    formulation_type: str
    times_h: list
    c_systemic_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    retention_time_h: float
    f_absorbed: float
    absorption_phase: str
    t_half_h: float
    notes: str


def _validate_inputs(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    retention_time_h: float,
    formulation_type: str,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if retention_time_h <= 0:
        raise ValueError(f"retention_time_h must be > 0, got {retention_time_h}")
    if formulation_type not in _VALID_FORMULATION_TYPES:
        raise ValueError(
            f"formulation_type must be one of {sorted(_VALID_FORMULATION_TYPES)}, "
            f"got '{formulation_type}'"
        )


def simulate_gastric_retention(
    drug_name: str,
    dose_mg: float,
    formulation_type: str,
    cl_L_per_h: float,
    vd_L: float,
    retention_time_h: float = 4.0,
    ka_gastric: float = 0.3,
    ka_intestine: float = 1.0,
    k_release_per_h: float = 0.2,
    f_abs: float = 0.9,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> GastricRetentionResult:
    """Simulate gastric-retentive drug delivery PK using Forward Euler ODE.

    Compartments
    ------------
    A_device (mg)  : drug remaining in the retentive device
    A_gi (mg)      : drug in GI lumen available for absorption
    C_sys (mg/L)   : systemic plasma concentration

    Phase transitions
    -----------------
    Gastric phase (t < retention_time_h):
        dA_device/dt = -k_release_per_h * A_device
        dA_gi/dt    = k_release_per_h * A_device * f_abs - ka_gastric * A_gi
        dC_sys/dt   = ka_gastric * A_gi / Vd - k_el * C_sys

    Intestinal phase (t >= retention_time_h):
        At transition: A_gi += A_device; A_device = 0
        dA_gi/dt    = -ka_intestine * A_gi
        dC_sys/dt   = ka_intestine * A_gi / Vd - k_el * C_sys

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    dose_mg : float
        Total dose in the retentive device (mg).
    formulation_type : str
        "floating" or "swelling".
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    retention_time_h : float
        Duration of gastric retention (h). Default 4.0.
    ka_gastric : float
        Gastric absorption rate constant (1/h). Default 0.3.
    ka_intestine : float
        Intestinal absorption rate constant (1/h). Default 1.0.
    k_release_per_h : float
        First-order release rate from device (1/h). Default 0.2.
    f_abs : float
        Fraction of released drug absorbed (bioavailability modifier). Default 0.9.
    t_end_h : float
        Simulation end time (h). Default 24.0.
    dt_h : float
        Forward Euler step size (h). Default 0.1.

    Returns
    -------
    GastricRetentionResult
    """
    _validate_inputs(dose_mg, cl_L_per_h, vd_L, retention_time_h, formulation_type)

    k_el = cl_L_per_h / vd_L

    # Initial conditions
    a_device = dose_mg
    a_gi = 0.0
    c_sys = 0.0
    dumped = False

    times_h: list[float] = []
    c_sys_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    dose_absorbed = 0.0  # track cumulative absorbed amount

    for _step in range(n_steps + 1):
        times_h.append(t)
        c_sys_list.append(c_sys)

        # Transition from gastric to intestinal phase
        if t >= retention_time_h and not dumped:
            a_gi += a_device
            a_device = 0.0
            dumped = True

        # Select absorption rate constant
        if not dumped:
            # Gastric phase
            rel = k_release_per_h * a_device * dt_h
            rel = min(rel, a_device)  # cannot release more than available

            d_a_device = -k_release_per_h * a_device
            d_a_gi = k_release_per_h * a_device * f_abs - ka_gastric * a_gi
            absorbed_rate = ka_gastric * a_gi
        else:
            # Intestinal phase
            d_a_device = 0.0
            d_a_gi = -ka_intestine * a_gi
            absorbed_rate = ka_intestine * a_gi

        d_c_sys = absorbed_rate / vd_L - k_el * c_sys

        dose_absorbed += absorbed_rate * dt_h

        # Forward Euler update
        a_device = max(0.0, a_device + d_a_device * dt_h)
        a_gi = max(0.0, a_gi + d_a_gi * dt_h)
        c_sys = max(0.0, c_sys + d_c_sys * dt_h)

        t += dt_h

    # PK metrics
    cmax_mg_L = max(c_sys_list) if c_sys_list else 0.0
    tmax_h = times_h[c_sys_list.index(cmax_mg_L)] if c_sys_list else 0.0

    # Manual trapezoidal AUC
    auc = sum(
        0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Fraction absorbed (dose absorbed / dose_mg, capped at 1)
    f_absorbed = min(1.0, dose_absorbed / dose_mg) if dose_mg > 0 else 0.0

    # Half-life
    t_half_h = math.log(2.0) / k_el if k_el > 0 else float("inf")

    # Determine dominant absorption phase
    absorption_phase = "gastric" if retention_time_h >= t_end_h * 0.5 else "mixed"

    notes = (
        f"{formulation_type.capitalize()} gastric-retentive formulation; "
        f"retention={retention_time_h:.1f}h; "
        f"ka_gastric={ka_gastric:.2f}/h; ka_intestine={ka_intestine:.2f}/h; "
        f"k_release={k_release_per_h:.3f}/h; f_abs={f_abs:.2f}"
    )

    return GastricRetentionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        times_h=times_h,
        c_systemic_mg_L=c_sys_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        retention_time_h=retention_time_h,
        f_absorbed=f_absorbed,
        absorption_phase=absorption_phase,
        t_half_h=t_half_h,
        notes=notes,
    )


def compare_retention_times(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    retention_times_h: list,
) -> list:
    """Compare multiple gastric retention times and return results sorted by AUC descending.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    dose_mg : float
        Total dose in mg.
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    retention_times_h : list[float]
        List of retention times to compare.

    Returns
    -------
    list[GastricRetentionResult] sorted by auc_mg_h_per_L descending.
    """
    results = [
        simulate_gastric_retention(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation_type="floating",
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            retention_time_h=rt,
        )
        for rt in retention_times_h
    ]
    return sorted(results, key=lambda r: r.auc_mg_h_per_L, reverse=True)
