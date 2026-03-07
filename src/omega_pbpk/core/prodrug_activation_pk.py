"""
Phase 889 — Prodrug Activation Kinetics

Models pharmacokinetics of prodrug-to-active metabolite conversion.
Pure Python / stdlib only (math, dataclasses). No numpy, no scipy.
Forward Euler ODE integration. Manual trapezoidal AUC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_ACTIVATION_SITES = {"plasma", "liver", "intestine", "gut_wall"}


@dataclass
class ProdrugPKResult:
    """Result of a prodrug activation PK simulation."""

    drug_name: str
    active_metabolite_name: str
    dose_mg: float
    activation_site: str
    times_h: list
    c_prodrug_mg_L: list
    c_active_mg_L: list
    cmax_prodrug: float
    tmax_prodrug_h: float
    auc_prodrug: float
    cmax_active: float
    tmax_active_h: float
    auc_active: float
    f_activation: float
    active_to_prodrug_auc_ratio: float
    notes: str


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Compute AUC via the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt
    return auc


def _find_cmax_tmax(times: list, concs: list) -> tuple:
    """Return (cmax, tmax_h) for the given concentration profile."""
    cmax = 0.0
    tmax_h = 0.0
    for t, c in zip(times, concs):
        if c > cmax:
            cmax = c
            tmax_h = t
    return cmax, tmax_h


def simulate_prodrug_pk(
    drug_name: str,
    active_metabolite_name: str,
    dose_mg: float,
    activation_site: str = "liver",
    k_abs_per_h: float = 1.0,
    k_act_per_h: float = 0.5,
    f_activation: float = 0.7,
    cl_prodrug_L_per_h: float = 8.0,
    cl_active_L_per_h: float = 5.0,
    vd_L: float = 30.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> ProdrugPKResult:
    """
    Simulate 3-compartment prodrug->active metabolite PK using Forward Euler.

    Compartments:
        a_gut       -- amount in GI tract (mg)
        A_prodrug   -- amount of prodrug in plasma (mg)
        A_active    -- amount of active metabolite in plasma (mg)
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0.0 <= f_activation <= 1.0):
        raise ValueError("f_activation must be in [0, 1]")
    if cl_prodrug_L_per_h <= 0:
        raise ValueError("cl_prodrug_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if activation_site not in _VALID_ACTIVATION_SITES:
        raise ValueError(
            f"activation_site must be one of {_VALID_ACTIVATION_SITES}, got '{activation_site}'"
        )

    # --- Derived rate constants ---
    ke_prodrug = cl_prodrug_L_per_h / vd_L
    ke_active = cl_active_L_per_h / vd_L

    # --- Initial conditions ---
    a_gut = dose_mg
    A_prodrug = 0.0
    A_active = 0.0

    times = []
    c_prodrug_list = []
    c_active_list = []

    n_steps = int(math.ceil(t_end_h / dt_h))

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        c_prodrug_list.append(A_prodrug / vd_L)
        c_active_list.append(A_active / vd_L)

        if i < n_steps:
            d_gut = -k_abs_per_h * a_gut
            d_prod = k_abs_per_h * a_gut - (k_act_per_h + ke_prodrug) * A_prodrug
            d_active = k_act_per_h * f_activation * A_prodrug - ke_active * A_active

            a_gut += d_gut * dt_h
            A_prodrug += d_prod * dt_h
            A_active += d_active * dt_h

            if a_gut < 0.0:
                a_gut = 0.0
            if A_prodrug < 0.0:
                A_prodrug = 0.0
            if A_active < 0.0:
                A_active = 0.0

    # --- PK metrics ---
    cmax_prodrug, tmax_prodrug_h = _find_cmax_tmax(times, c_prodrug_list)
    cmax_active, tmax_active_h = _find_cmax_tmax(times, c_active_list)
    auc_prodrug = _trapezoidal_auc(times, c_prodrug_list)
    auc_active = _trapezoidal_auc(times, c_active_list)

    ratio = auc_active / auc_prodrug if auc_prodrug > 0 else 0.0

    if ratio > 3:
        notes = "Active metabolite dominant — prodrug highly efficient"
    elif ratio > 1:
        notes = "Balanced prodrug/active exposure"
    else:
        notes = "Prodrug-dominant PK — consider activation efficiency"

    return ProdrugPKResult(
        drug_name=drug_name,
        active_metabolite_name=active_metabolite_name,
        dose_mg=dose_mg,
        activation_site=activation_site,
        times_h=times,
        c_prodrug_mg_L=c_prodrug_list,
        c_active_mg_L=c_active_list,
        cmax_prodrug=cmax_prodrug,
        tmax_prodrug_h=tmax_prodrug_h,
        auc_prodrug=auc_prodrug,
        cmax_active=cmax_active,
        tmax_active_h=tmax_active_h,
        auc_active=auc_active,
        f_activation=f_activation,
        active_to_prodrug_auc_ratio=ratio,
        notes=notes,
    )


def compare_activation_sites(
    drug_name: str,
    active_metabolite_name: str,
    dose_mg: float,
    **kwargs,
) -> list:
    """
    Simulate prodrug PK for all four activation sites.

    Returns results sorted by auc_active descending.
    Extra keyword arguments are forwarded to simulate_prodrug_pk.
    """
    results = []
    for site in _VALID_ACTIVATION_SITES:
        result = simulate_prodrug_pk(
            drug_name=drug_name,
            active_metabolite_name=active_metabolite_name,
            dose_mg=dose_mg,
            activation_site=site,
            **kwargs,
        )
        results.append(result)
    results.sort(key=lambda r: r.auc_active, reverse=True)
    return results
