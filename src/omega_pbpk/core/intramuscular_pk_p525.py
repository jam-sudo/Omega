"""Intramuscular (IM) injection PK model with depot absorption (Phase 525).

Models drug release from an IM depot into systemic circulation using
Forward Euler ODE:

    dA_depot/dt = -ka_im * A_depot
    dC_sys/dt   = ka_im * A_depot / Vd_sys - k_el * C_sys

Three formulation types are supported, each with a distinct ka_im:
    - aqueous:     ka_im = 0.5 /h  (fast absorption)
    - suspension:  ka_im = 0.1 /h  (slow, prolonged release)
    - oil_depot:   ka_im = 0.02 /h (very slow, sustained release)

No first-pass effect; F = 1 (bioavailability equivalent to IV).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "IntramuscularPKResult",
    "simulate_intramuscular_pk",
    "compare_formulations_im",
]

# Formulation-specific absorption rate constants (/h)
_KA_MAP: dict[str, float] = {
    "aqueous": 0.5,
    "suspension": 0.1,
    "oil_depot": 0.02,
}

_VALID_FORMULATIONS = set(_KA_MAP.keys())


@dataclass
class IntramuscularPKResult:
    """Results from an IM pharmacokinetic simulation.

    Attributes
    ----------
    drug_name : Drug identifier.
    dose_mg : Administered dose (mg).
    formulation_type : One of 'aqueous', 'suspension', 'oil_depot'.
    ka_im_per_h : Absorption rate constant used (1/h).
    t_half_absorption_h : Absorption half-life = ln(2) / ka_im (h).
    times_h : Simulation time points (h).
    c_systemic_mg_L : Systemic concentration over time (mg/L).
    cmax_mg_L : Peak systemic concentration (mg/L).
    tmax_h : Time of Cmax (h).
    auc_mg_h_per_L : AUC 0-to-t_end (mg*h/L), trapezoidal.
    duration_above_half_cmax_h : Duration with C_sys > 0.5 * Cmax (h).
    t_half_systemic_h : Systemic elimination half-life = ln(2) / k_el (h).
    vs_iv_tmax_factor : Conceptual ratio tmax_im / t_half_systemic.
    notes : Summary text.
    """

    drug_name: str
    dose_mg: float
    formulation_type: str
    ka_im_per_h: float
    t_half_absorption_h: float
    times_h: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    duration_above_half_cmax_h: float = 0.0
    t_half_systemic_h: float = 0.0
    vs_iv_tmax_factor: float = 0.0
    notes: str = ""


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC integration."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_intramuscular_pk(
    drug_name: str,
    dose_mg: float,
    formulation_type: str,
    cl_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> IntramuscularPKResult:
    """Simulate intramuscular PK using forward Euler integration.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Administered dose (mg). Must be > 0.
    formulation_type : 'aqueous', 'suspension', or 'oil_depot'.
    cl_L_per_h : Systemic clearance (L/h). Must be > 0.
    vd_sys_L : Systemic volume of distribution (L). Must be > 0.
    t_end_h : Simulation end time (h). Default 48 h.
    dt_h : Forward Euler step size (h). Default 0.1 h.

    Returns
    -------
    IntramuscularPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if formulation_type not in _VALID_FORMULATIONS:
        raise ValueError(f"formulation_type must be one of {sorted(_VALID_FORMULATIONS)}")

    ka_im = _KA_MAP[formulation_type]
    k_el = cl_L_per_h / vd_sys_L
    t_half_absorption = math.log(2.0) / ka_im
    t_half_systemic = math.log(2.0) / k_el

    n_steps = max(int(round(t_end_h / dt_h)), 1)

    times: list[float] = []
    c_sys_list: list[float] = []

    # Initial state
    a_depot = dose_mg  # mg remaining in depot
    c_sys = 0.0  # mg/L

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        c_sys_list.append(c_sys)

        if i < n_steps:
            # Forward Euler update
            da_depot = -ka_im * a_depot
            dc_sys = (ka_im * a_depot) / vd_sys_L - k_el * c_sys

            a_depot = max(a_depot + da_depot * dt_h, 0.0)
            c_sys = max(c_sys + dc_sys * dt_h, 0.0)

    # Derived PK parameters
    cmax = max(c_sys_list)
    tmax = times[c_sys_list.index(cmax)]
    auc = _trapezoidal_auc(times, c_sys_list)

    # Duration above half-Cmax
    half_cmax = 0.5 * cmax
    duration_above = 0.0
    for i in range(1, len(times)):
        c0 = c_sys_list[i - 1]
        c1 = c_sys_list[i]
        dt = times[i] - times[i - 1]
        if c0 > half_cmax and c1 > half_cmax:
            duration_above += dt
        elif c0 > half_cmax or c1 > half_cmax:
            # Partial segment: linear interpolation to find crossing
            duration_above += dt * 0.5

    vs_iv_tmax_factor = tmax / t_half_systemic if t_half_systemic > 0 else 0.0

    notes = (
        f"IM {formulation_type} PK for {drug_name}. "
        f"ka_im={ka_im}/h, t_half_abs={t_half_absorption:.2f}h, "
        f"k_el={k_el:.4f}/h, t_half_sys={t_half_systemic:.2f}h. "
        f"Cmax={cmax:.4f}mg/L at Tmax={tmax:.2f}h, AUC={auc:.3f}mg*h/L."
    )

    return IntramuscularPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        ka_im_per_h=ka_im,
        t_half_absorption_h=t_half_absorption,
        times_h=times,
        c_systemic_mg_L=c_sys_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        duration_above_half_cmax_h=duration_above,
        t_half_systemic_h=t_half_systemic,
        vs_iv_tmax_factor=vs_iv_tmax_factor,
        notes=notes,
    )


def compare_formulations_im(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_sys_L: float,
) -> list:
    """Compare all three IM formulations, sorted by tmax ascending (fastest first).

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Administered dose (mg).
    cl_L_per_h : Systemic clearance (L/h).
    vd_sys_L : Systemic volume of distribution (L).

    Returns
    -------
    list of IntramuscularPKResult sorted by tmax_h ascending.
    """
    results = [
        simulate_intramuscular_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation_type=ftype,
            cl_L_per_h=cl_L_per_h,
            vd_sys_L=vd_sys_L,
            t_end_h=120.0,
            dt_h=0.1,
        )
        for ftype in _VALID_FORMULATIONS
    ]
    return sorted(results, key=lambda r: r.tmax_h)
