"""Polymer-based depot drug delivery (PLGA microspheres, implants)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "PolymerDepotResult",
    "simulate_polymer_depot",
    "compare_depot_formulations",
]

POLYMER_PARAMS: dict[str, dict[str, float]] = {
    "PLGA_50_50": {"t_half_deg_days": 7.0,  "burst_pct": 15.0, "lag_h": 0.0},
    "PLGA_75_25": {"t_half_deg_days": 28.0, "burst_pct": 10.0, "lag_h": 24.0},
    "PLGA_85_15": {"t_half_deg_days": 60.0, "burst_pct": 5.0,  "lag_h": 48.0},
    "PLA":        {"t_half_deg_days": 180.0, "burst_pct": 2.0,  "lag_h": 72.0},
}


@dataclass(frozen=True)
class PolymerDepotResult:
    drug_name: str
    polymer_type: str       # "PLGA_50_50", "PLGA_75_25", "PLGA_85_15", "PLA"
    dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    release_rate_mg_per_h: list[float]
    cumulative_released_pct: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_mg_L_h: float
    duration_days: float    # effective duration (to 80% release)
    burst_release_pct: float  # initial burst in first 24h
    notes: str


def simulate_polymer_depot(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    polymer_type: str = "PLGA_50_50",
    drug_loading_pct: float = 10.0,
    particle_size_um: float = 50.0,
    t_end_h: float = 2160.0,
    dt_h: float = 1.0,
) -> PolymerDepotResult:
    """Simulate drug release and plasma PK from a polymer depot formulation."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 < drug_loading_pct < 100):
        raise ValueError("drug_loading_pct must be in (0, 100)")
    if polymer_type not in POLYMER_PARAMS:
        raise ValueError(
            f"polymer_type must be one of {list(POLYMER_PARAMS.keys())}, got '{polymer_type}'"
        )

    params = POLYMER_PARAMS[polymer_type]
    k_deg = math.log(2) / (params["t_half_deg_days"] * 24.0)  # h^-1
    burst_drug = dose_mg * params["burst_pct"] / 100.0
    core_drug = dose_mg - burst_drug
    lag_h = params["lag_h"]

    ke = cl_L_per_h / vd_L

    times = np.arange(0.0, t_end_h + dt_h / 2.0, dt_h)
    n = len(times)

    c_plasma_arr = np.zeros(n)
    release_rate_arr = np.zeros(n)
    cumulative_pct_arr = np.zeros(n)

    a_depot = core_drug
    a_burst = burst_drug
    released_cumulative = 0.0
    c_plasma = 0.0

    burst_released_24h = 0.0

    for i in range(1, n):
        t = times[i]

        # Burst: fast release in first 24h
        if t <= 24.0:
            k_burst = 0.1  # fast first-order burst h^-1
            burst_flux = k_burst * a_burst * dt_h
        else:
            burst_flux = 0.0

        # Core: first-order degradation release
        if t >= lag_h:
            core_flux = k_deg * a_depot * dt_h
        else:
            core_flux = 0.0

        total_flux = burst_flux + core_flux
        a_depot = max(0.0, a_depot - core_flux)
        a_burst = max(0.0, a_burst - burst_flux)
        released_cumulative += total_flux

        # Track burst in first 24h
        if t <= 24.0:
            burst_released_24h += burst_flux

        release_rate_arr[i] = total_flux / dt_h  # mg/h

        # PK: released drug → plasma
        dc_plasma = (total_flux / vd_L - ke * c_plasma) * dt_h
        c_plasma = max(0.0, c_plasma + dc_plasma)
        c_plasma_arr[i] = c_plasma
        cumulative_pct_arr[i] = released_cumulative / dose_mg * 100.0

    # Cmax and Tmax
    cmax_idx = int(np.argmax(c_plasma_arr))
    cmax_mg_L = float(c_plasma_arr[cmax_idx])
    tmax_h = float(times[cmax_idx])

    # AUC
    auc_mg_L_h = float(np_trapz(c_plasma_arr, times))

    # Duration: time when 80% released
    duration_h = t_end_h  # default: full simulation
    for i in range(n):
        if cumulative_pct_arr[i] >= 80.0:
            duration_h = float(times[i])
            break
    duration_days = duration_h / 24.0

    # Burst release %: fraction released in first 24h relative to dose
    burst_release_pct = burst_released_24h / dose_mg * 100.0

    notes = (
        f"Polymer: {polymer_type}, t½ degradation: {params['t_half_deg_days']:.0f} days, "
        f"burst: {params['burst_pct']:.0f}%, lag: {lag_h:.0f} h. "
        f"Duration to 80% release: {duration_days:.1f} days."
    )

    return PolymerDepotResult(
        drug_name=drug_name,
        polymer_type=polymer_type,
        dose_mg=dose_mg,
        times_h=times.tolist(),
        c_plasma_mg_L=c_plasma_arr.tolist(),
        release_rate_mg_per_h=release_rate_arr.tolist(),
        cumulative_released_pct=cumulative_pct_arr.tolist(),
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_L_h=auc_mg_L_h,
        duration_days=duration_days,
        burst_release_pct=burst_release_pct,
        notes=notes,
    )


def compare_depot_formulations(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    **kwargs: object,
) -> list[PolymerDepotResult]:
    """Compare all 4 polymer depot formulations for a given drug."""
    results = []
    for polymer_type in POLYMER_PARAMS:
        result = simulate_polymer_depot(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            polymer_type=polymer_type,
            **kwargs,  # type: ignore[arg-type]
        )
        results.append(result)
    return results
