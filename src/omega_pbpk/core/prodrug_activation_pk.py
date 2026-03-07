"""Prodrug → active drug conversion and pharmacokinetic simulation.

Models two-compartment prodrug/active-drug PK using forward-Euler ODEs.
Supports oral and IV routes with configurable conversion and elimination rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ProdrugPKResult",
    "simulate_prodrug_pk",
    "compare_prodrug_strategies",
]


@dataclass
class ProdrugPKResult:
    """Result of prodrug activation PK simulation."""

    prodrug_name: str
    active_drug_name: str
    dose_mg: float
    route: str  # "oral" or "iv"
    f_conversion: float  # fraction converted to active drug
    ka_prodrug_per_h: float  # absorption rate of prodrug
    kconv_per_h: float  # conversion rate prodrug->active
    ke_prodrug_per_h: float  # elimination rate of prodrug
    ke_active_per_h: float  # elimination rate of active drug
    # Time-course arrays
    times_h: list
    c_prodrug_mg_L: list
    c_active_mg_L: list
    # Summary metrics (active drug)
    cmax_active_mg_L: float
    tmax_active_h: float
    auc_active_mg_h_per_L: float
    t_half_active_h: float
    # Summary metrics (prodrug)
    cmax_prodrug_mg_L: float
    tmax_prodrug_h: float
    auc_prodrug_mg_h_per_L: float
    active_to_prodrug_auc_ratio: float
    notes: str


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


def simulate_prodrug_pk(
    prodrug_name: str,
    active_drug_name: str,
    dose_mg: float,
    vd_prodrug_L: float = 30.0,
    vd_active_L: float = 50.0,
    ka_prodrug_per_h: float = 1.0,
    kconv_per_h: float = 0.5,
    ke_prodrug_per_h: float = 0.3,
    ke_active_per_h: float = 0.15,
    f_oral_prodrug: float = 0.8,
    f_conversion: float = 0.8,
    route: str = "oral",
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> ProdrugPKResult:
    """Simulate prodrug → active drug PK using forward-Euler ODEs.

    Compartments
    ------------
    - gut_depot (mg): oral only; initial dose * f_oral_prodrug
    - prodrug_plasma (mg/L)
    - active_plasma (mg/L)

    ODEs
    ----
    d(gut)/dt   = -ka * gut
    d(prod)/dt  = ka * gut / vd_prodrug  [oral]
                  OR dose/vd_prodrug at t=0 [iv]
                  - (kconv + ke_prodrug) * prodrug
    d(active)/dt = kconv * prodrug * vd_prodrug * f_conversion / vd_active
                  - ke_active * active
    """
    # --- Validation ---
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vd_prodrug_L <= 0.0:
        raise ValueError(f"vd_prodrug_L must be > 0, got {vd_prodrug_L}")
    if vd_active_L <= 0.0:
        raise ValueError(f"vd_active_L must be > 0, got {vd_active_L}")
    if kconv_per_h < 0.0:
        raise ValueError(f"kconv_per_h must be >= 0, got {kconv_per_h}")
    if ke_prodrug_per_h < 0.0:
        raise ValueError(f"ke_prodrug_per_h must be >= 0, got {ke_prodrug_per_h}")
    if ke_active_per_h < 0.0:
        raise ValueError(f"ke_active_per_h must be >= 0, got {ke_active_per_h}")
    if route not in ("oral", "iv"):
        raise ValueError(f"route must be 'oral' or 'iv', got {route!r}")

    # --- Initial conditions ---
    if route == "oral":
        gut = dose_mg * f_oral_prodrug
        prodrug = 0.0
    else:
        # IV: bolus directly into plasma at t=0
        gut = 0.0
        prodrug = dose_mg / vd_prodrug_L

    active = 0.0

    times: list[float] = []
    c_prodrug: list[float] = []
    c_active: list[float] = []

    n_steps = int(round(t_end_h / dt_h))
    t = 0.0

    for step in range(n_steps + 1):
        times.append(t)
        c_prodrug.append(prodrug)
        c_active.append(active)

        if step < n_steps:
            # Derivatives
            d_gut = -ka_prodrug_per_h * gut

            if route == "oral":
                absorption_rate_mg_h_per_L = ka_prodrug_per_h * gut / vd_prodrug_L
            else:
                absorption_rate_mg_h_per_L = 0.0

            d_prodrug = absorption_rate_mg_h_per_L - (kconv_per_h + ke_prodrug_per_h) * prodrug

            # Conversion from prodrug compartment (mg/h) -> active (mg/L/h)
            rate_conv_mg_h = kconv_per_h * prodrug * vd_prodrug_L
            d_active = rate_conv_mg_h * f_conversion / vd_active_L - ke_active_per_h * active

            # Forward Euler update
            gut = max(0.0, gut + dt_h * d_gut)
            prodrug = max(0.0, prodrug + dt_h * d_prodrug)
            active = max(0.0, active + dt_h * d_active)
            t += dt_h

    # --- Summary metrics: active drug ---
    cmax_active = max(c_active)
    tmax_active_idx = c_active.index(cmax_active)
    tmax_active = times[tmax_active_idx]
    auc_active = _trapz(times, c_active)
    t_half_active = math.log(2.0) / ke_active_per_h if ke_active_per_h > 0 else float("inf")

    # --- Summary metrics: prodrug ---
    cmax_prodrug = max(c_prodrug)
    tmax_prodrug_idx = c_prodrug.index(cmax_prodrug)
    tmax_prodrug = times[tmax_prodrug_idx]
    auc_prodrug = _trapz(times, c_prodrug)

    # Ratio active/prodrug AUC
    if auc_prodrug > 0.0:
        ratio = auc_active / auc_prodrug
    else:
        ratio = float("inf")

    notes_parts = [
        f"Route: {route}",
        f"f_conversion={f_conversion:.2f}",
        f"kconv={kconv_per_h:.3f}/h",
        f"ke_active={ke_active_per_h:.3f}/h",
        f"t1/2_active={t_half_active:.2f}h",
    ]

    return ProdrugPKResult(
        prodrug_name=prodrug_name,
        active_drug_name=active_drug_name,
        dose_mg=dose_mg,
        route=route,
        f_conversion=f_conversion,
        ka_prodrug_per_h=ka_prodrug_per_h,
        kconv_per_h=kconv_per_h,
        ke_prodrug_per_h=ke_prodrug_per_h,
        ke_active_per_h=ke_active_per_h,
        times_h=times,
        c_prodrug_mg_L=c_prodrug,
        c_active_mg_L=c_active,
        cmax_active_mg_L=cmax_active,
        tmax_active_h=tmax_active,
        auc_active_mg_h_per_L=auc_active,
        t_half_active_h=t_half_active,
        cmax_prodrug_mg_L=cmax_prodrug,
        tmax_prodrug_h=tmax_prodrug,
        auc_prodrug_mg_h_per_L=auc_prodrug,
        active_to_prodrug_auc_ratio=ratio,
        notes="; ".join(notes_parts),
    )


def compare_prodrug_strategies(
    prodrug_name: str,
    active_drug_name: str,
    dose_mg: float,
    strategies: list[dict],
) -> list[ProdrugPKResult]:
    """Compare multiple prodrug conversion strategies.

    Parameters
    ----------
    strategies:
        List of dicts, each with keys: "kconv_per_h", "f_conversion", "route".

    Returns
    -------
    List of ProdrugPKResult sorted by auc_active_mg_h_per_L descending.
    """
    results: list[ProdrugPKResult] = []
    for strat in strategies:
        res = simulate_prodrug_pk(
            prodrug_name=prodrug_name,
            active_drug_name=active_drug_name,
            dose_mg=dose_mg,
            kconv_per_h=strat["kconv_per_h"],
            f_conversion=strat["f_conversion"],
            route=strat["route"],
        )
        results.append(res)

    results.sort(key=lambda r: r.auc_active_mg_h_per_L, reverse=True)
    return results
