"""
Phase 387 — Depot Injection PK Model

Simulate long-acting depot injection pharmacokinetics (IM or SC) with
dissolution/release from depot, absorption into systemic circulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Formulation release-rate modifiers
# ---------------------------------------------------------------------------

_FORMULATION_MODIFIERS: dict[str, float] = {
    "suspension": 1.0,
    "microsphere": 0.3,
    "implant": 0.05,
    "oil_solution": 0.5,
}


# ---------------------------------------------------------------------------
# Result dataclass (plain — has list fields)
# ---------------------------------------------------------------------------


@dataclass
class DepotInjectionResult:
    """Results from depot injection PK simulation."""

    drug_name: str
    formulation_type: str
    dose_mg: float
    times_h: list
    c_depot_mg: list  # drug remaining at injection site (mg)
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_apparent_h: float  # apparent half-life of plasma decline
    duration_above_mic_h: float
    depot_t90_h: float  # time for 90 % of depot absorbed
    flip_flop_kinetics: bool
    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_depot_injection(
    drug_name: str,
    dose_mg: float,
    formulation_type: str,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    k_release_per_h: float = 0.02,
    t_end_h: float = 720.0,
    dt_h: float = 0.5,
    mic_mg_L: float = 0.0,
) -> DepotInjectionResult:
    """Simulate depot injection PK using a 2-compartment forward-Euler model.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Dose loaded into the depot (mg).
    formulation_type:
        One of "suspension", "microsphere", "implant", "oil_solution".
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    k_release_per_h:
        Base release rate constant from depot (1/h) before formulation modifier.
    t_end_h:
        Simulation duration (h).
    dt_h:
        Forward-Euler step size (h).
    mic_mg_L:
        Minimum inhibitory (or effective) concentration for duration calculation.

    Returns
    -------
    DepotInjectionResult
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if k_release_per_h <= 0:
        raise ValueError(f"k_release_per_h must be > 0, got {k_release_per_h}")
    if formulation_type not in _FORMULATION_MODIFIERS:
        raise ValueError(
            f"formulation_type must be one of {list(_FORMULATION_MODIFIERS)}, "
            f"got '{formulation_type}'"
        )

    # ------------------------------------------------------------------
    # Effective release rate
    # ------------------------------------------------------------------
    eff_k_rel = k_release_per_h * _FORMULATION_MODIFIERS[formulation_type]
    k_elim = cl_sys_L_per_h / vd_sys_L  # elimination rate constant (1/h)

    # ------------------------------------------------------------------
    # Forward-Euler simulation
    # ------------------------------------------------------------------
    n_steps = int(t_end_h / dt_h) + 1
    times_h: list[float] = []
    c_depot_mg: list[float] = []
    c_plasma_mg_L: list[float] = []

    m_depot = dose_mg
    c_plasma = 0.0

    for i in range(n_steps):
        t = i * dt_h
        times_h.append(t)
        c_depot_mg.append(m_depot)
        c_plasma_mg_L.append(c_plasma)

        # ODEs
        dm_depot_dt = -eff_k_rel * m_depot
        dc_plasma_dt = eff_k_rel * m_depot / vd_sys_L - k_elim * c_plasma

        # Euler update
        m_depot = max(0.0, m_depot + dm_depot_dt * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma_dt * dt_h)

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    # Cmax and Tmax
    cmax_mg_L = max(c_plasma_mg_L)
    tmax_h = times_h[c_plasma_mg_L.index(cmax_mg_L)]

    # AUC (trapezoidal)
    auc_mg_h_per_L = sum(
        0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Duration above MIC
    duration_above_mic_h = 0.0
    if mic_mg_L > 0.0:
        for i in range(1, len(times_h)):
            if c_plasma_mg_L[i] > mic_mg_L:
                duration_above_mic_h += dt_h

    # Depot T90: analytical (time for 90 % absorbed = 10 % remaining)
    # M_depot(t) = dose * exp(-eff_k_rel * t)  => t = ln(10) / eff_k_rel
    depot_t90_h = math.log(10.0) / eff_k_rel

    # Apparent half-life: time for plasma to drop to 50 % of Cmax after Tmax
    t_half_apparent_h = 0.0
    tmax_idx = c_plasma_mg_L.index(cmax_mg_L)
    half_cmax = cmax_mg_L * 0.5
    t_half_found = False
    for i in range(tmax_idx + 1, len(c_plasma_mg_L)):
        if c_plasma_mg_L[i] <= half_cmax:
            # Linear interpolation
            c_prev = c_plasma_mg_L[i - 1]
            c_curr = c_plasma_mg_L[i]
            t_prev = times_h[i - 1]
            t_curr = times_h[i]
            if c_prev != c_curr:
                frac = (c_prev - half_cmax) / (c_prev - c_curr)
                t_half_point = t_prev + frac * (t_curr - t_prev)
            else:
                t_half_point = t_curr
            t_half_apparent_h = t_half_point - tmax_h
            t_half_found = True
            break
    if not t_half_found:
        # Fallback: analytical apparent half-life dominated by slower rate
        dominant_rate = min(eff_k_rel, k_elim)
        t_half_apparent_h = math.log(2.0) / dominant_rate if dominant_rate > 0 else 0.0

    # Flip-flop kinetics: absorption rate << elimination rate
    flip_flop_kinetics = eff_k_rel < k_elim * 0.5

    # Notes
    notes = (
        f"Formulation: {formulation_type}; "
        f"eff_k_release={eff_k_rel:.4f}/h; "
        f"k_elim={k_elim:.4f}/h; "
        f"flip_flop={'yes' if flip_flop_kinetics else 'no'}"
    )

    return DepotInjectionResult(
        drug_name=drug_name,
        formulation_type=formulation_type,
        dose_mg=dose_mg,
        times_h=times_h,
        c_depot_mg=c_depot_mg,
        c_plasma_mg_L=c_plasma_mg_L,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc_mg_h_per_L,
        t_half_apparent_h=t_half_apparent_h,
        duration_above_mic_h=duration_above_mic_h,
        depot_t90_h=depot_t90_h,
        flip_flop_kinetics=flip_flop_kinetics,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Formulation comparison
# ---------------------------------------------------------------------------


def compare_formulations(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    k_release_per_h: float = 0.02,
) -> list:
    """Compare all 4 formulations for the same drug/dose.

    Returns a list of DepotInjectionResult sorted by auc_mg_h_per_L descending.
    """
    results = []
    for formulation_type in _FORMULATION_MODIFIERS:
        result = simulate_depot_injection(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation_type=formulation_type,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
            k_release_per_h=k_release_per_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
