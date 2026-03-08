"""Phase 254: PK model for drug penetration into tumor spheroids.

Simulates drug transport through an avascular tumor spheroid using a
3-compartment model: culture medium -> spheroid surface -> spheroid core.

Reference: Minchinton & Tannock, Nature Reviews Cancer, 2006;
Tredan et al., J Natl Cancer Inst, 2007.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TumorSpheroidPKResult:
    """Result of tumor spheroid PK simulation."""

    drug_name: str
    dose_mg_per_L: float
    times_h: list
    c_medium_mg_L: list
    c_surface_mg_L: list
    c_core_mg_L: list
    cmax_core_mg_L: float
    tmax_core_h: float
    auc_medium_mg_h_per_L: float
    auc_core_mg_h_per_L: float
    penetration_efficiency_pct: float
    k_penetration_per_h: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(times: list, values: list) -> float:
    """Trapezoidal AUC integration."""
    if len(times) < 2:
        return 0.0
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


# Default ODE parameters
_V_MEDIUM = 1000.0  # mL, culture medium volume
_V_SURFACE = 0.003  # mL, spheroid surface shell volume
_V_CORE = 0.001  # mL, spheroid core volume
_K_IN = 0.5  # /h, medium->surface transfer rate
_K_OUT = 0.3  # /h, surface->medium back-transfer rate
_K_PEN = 0.1  # /h, surface->core penetration rate
_K_ELIM = 0.05  # /h, core elimination rate


def _simulate_odes(
    dose_mg_per_L: float,
    k_in: float,
    k_out: float,
    k_pen: float,
    k_elim: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list, list, list, list]:
    """Forward Euler ODE integration for 3-cpt tumor spheroid model.

    ODEs:
        dC_medium/dt  = -k_in * C_medium + k_out * C_surface * V_surface / V_medium
        dC_surface/dt =  k_in * C_medium * V_medium / V_surface
                         - k_out * C_surface - k_pen * C_surface
        dC_core/dt    =  k_pen * C_surface * V_surface / V_core
                         - k_elim * C_core

    Returns (times, c_medium, c_surface, c_core).
    """
    v_medium = _V_MEDIUM
    v_surface = _V_SURFACE
    v_core = _V_CORE

    c_medium = dose_mg_per_L
    c_surface = 0.0
    c_core = 0.0

    n_steps = max(1, int(round(t_end_h / dt_h)))

    times: list = [0.0]
    c_med_list: list = [c_medium]
    c_surf_list: list = [c_surface]
    c_core_list: list = [c_core]

    for _ in range(n_steps):
        dcm = -k_in * c_medium + k_out * c_surface * (v_surface / v_medium)
        dcs = k_in * c_medium * (v_medium / v_surface) - k_out * c_surface - k_pen * c_surface
        dcc = k_pen * c_surface * (v_surface / v_core) - k_elim * c_core

        c_medium = max(0.0, c_medium + dt_h * dcm)
        c_surface = max(0.0, c_surface + dt_h * dcs)
        c_core = max(0.0, c_core + dt_h * dcc)

        times.append(times[-1] + dt_h)
        c_med_list.append(c_medium)
        c_surf_list.append(c_surface)
        c_core_list.append(c_core)

    return times, c_med_list, c_surf_list, c_core_list


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_spheroid_pk(
    drug_name: str,
    dose_mg_per_L: float,
    logp: float = 2.0,
    mw: float = 400.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> TumorSpheroidPKResult:
    """Simulate drug PK in a tumor spheroid.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg_per_L:
        Initial drug concentration in the culture medium (mg/L).
    logp:
        Partition coefficient (log scale). More lipophilic -> faster uptake.
    mw:
        Molecular weight (Da). Larger MW -> slower core penetration.
    t_end_h:
        Simulation end time (hours). Default 48.
    dt_h:
        Time step for forward Euler (hours). Default 0.1.

    Returns
    -------
    TumorSpheroidPKResult
    """
    if dose_mg_per_L <= 0:
        raise ValueError("dose_mg_per_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # logP effect on k_in: more lipophilic -> faster medium->surface uptake
    k_in = _K_IN * (1.0 + 0.1 * max(0.0, logp - 1.0))

    # MW effect on k_pen: larger MW -> slower surface->core penetration
    k_pen = _K_PEN * max(0.3, math.exp(-0.002 * max(0.0, mw - 300.0)))

    k_out = _K_OUT
    k_elim = _K_ELIM

    times, c_med, c_surf, c_core = _simulate_odes(
        dose_mg_per_L=dose_mg_per_L,
        k_in=k_in,
        k_out=k_out,
        k_pen=k_pen,
        k_elim=k_elim,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # PK metrics for core compartment
    cmax_core = max(c_core)
    tmax_core = times[c_core.index(cmax_core)]

    auc_medium = _trapz(times, c_med)
    auc_core = _trapz(times, c_core)

    if auc_medium > 0:
        # Penetration efficiency: fraction of drug exposure reaching core.
        # Capped at 100% as core concentration amplification due to small
        # volume does not reflect true mass-based efficiency > 100%.
        raw_efficiency = (auc_core / auc_medium) * 100.0
        penetration_efficiency_pct = min(100.0, raw_efficiency)
    else:
        penetration_efficiency_pct = 0.0

    # Effective penetration rate (ratio of k_pen after MW adjustment)
    k_penetration_per_h = k_pen

    notes_parts: list[str] = []
    if logp < 0:
        notes_parts.append("Low logP; poor membrane permeability expected")
    if mw > 500:
        notes_parts.append("MW > 500 Da; reduced spheroid penetration expected")
    if penetration_efficiency_pct < 5.0:
        notes_parts.append("Poor core penetration (<5%); may limit anti-tumor efficacy")
    elif penetration_efficiency_pct > 30.0:
        notes_parts.append("Good core penetration (>30%)")

    notes = "; ".join(notes_parts) if notes_parts else "Simulation completed successfully"

    return TumorSpheroidPKResult(
        drug_name=drug_name,
        dose_mg_per_L=dose_mg_per_L,
        times_h=times,
        c_medium_mg_L=c_med,
        c_surface_mg_L=c_surf,
        c_core_mg_L=c_core,
        cmax_core_mg_L=cmax_core,
        tmax_core_h=tmax_core,
        auc_medium_mg_h_per_L=auc_medium,
        auc_core_mg_h_per_L=auc_core,
        penetration_efficiency_pct=penetration_efficiency_pct,
        k_penetration_per_h=k_penetration_per_h,
        notes=notes,
    )


def compare_drug_properties(
    drug_name: str,
    dose_mg_per_L: float,
    logp_values: list | None = None,
) -> list[TumorSpheroidPKResult]:
    """Compare spheroid penetration across a range of logP values.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg_per_L:
        Initial drug concentration in the culture medium (mg/L).
    logp_values:
        List of logP values to simulate. Defaults to [0.0, 1.0, 2.0, 3.0, 4.0].

    Returns
    -------
    list[TumorSpheroidPKResult]
        Sorted by penetration_efficiency_pct descending.
    """
    if logp_values is None:
        logp_values = [0.0, 1.0, 2.0, 3.0, 4.0]

    results: list[TumorSpheroidPKResult] = []
    for lp in logp_values:
        result = simulate_spheroid_pk(
            drug_name=drug_name,
            dose_mg_per_L=dose_mg_per_L,
            logp=lp,
        )
        results.append(result)

    results.sort(key=lambda r: r.penetration_efficiency_pct, reverse=True)
    return results


__all__ = [
    "TumorSpheroidPKResult",
    "simulate_spheroid_pk",
    "compare_drug_properties",
]
