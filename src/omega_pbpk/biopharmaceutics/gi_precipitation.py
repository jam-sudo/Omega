"""Drug precipitation in GI tract — Phase 342.

Spring-and-parachute model with extended two-compartment (stomach + intestine)
precipitation kinetics for dissolved and precipitated drug pools.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GIPrecipitationResult",
    "simulate_gi_precipitation",
    "effect_of_supersaturation_inhibitors",
]


@dataclass(frozen=True)
class GIPrecipitationResult:
    """Results of GI tract precipitation simulation."""

    drug_name: str
    dose_mg: float
    solubility_stomach_mg_mL: float
    solubility_intestine_mg_mL: float

    times_h: list[float]
    c_dissolved_stomach_mg_mL: list[float]
    c_dissolved_intestine_mg_mL: list[float]
    c_precipitated_stomach_mg: list[float]
    c_precipitated_intestine_mg: list[float]

    # Summary
    f_precipitated_stomach: float  # fraction that precipitated in stomach
    f_precipitated_intestine: float  # fraction that precipitated in intestine
    f_absorbed: float  # fraction absorbed
    f_lost: float  # fraction never absorbed (precipitated + excreted)

    peak_supersaturation_ratio: float  # max(C_dissolved / solubility)
    time_to_precipitation_h: float  # time until first precipitation event
    bioavailability_reduction_pct: float  # % loss vs ideal (no precipitation)
    notes: str


def simulate_gi_precipitation(
    drug_name: str,
    dose_mg: float,
    solubility_stomach_mg_mL: float = 0.01,
    solubility_intestine_mg_mL: float = 0.1,
    gastric_volume_mL: float = 250.0,
    intestinal_volume_mL: float = 500.0,
    gastric_emptying_h: float = 0.5,
    k_precip_per_h: float = 2.0,
    k_redissolve_per_h: float = 0.3,
    ka_intestinal_per_h: float = 1.5,
    t_end_h: float = 6.0,
    dt_h: float = 0.05,
) -> GIPrecipitationResult:
    """Simulate drug precipitation in stomach and intestine using forward Euler.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose (mg, > 0).
    solubility_stomach_mg_mL : float
        Drug solubility in stomach fluid (mg/mL, > 0).
    solubility_intestine_mg_mL : float
        Drug solubility in intestinal fluid (mg/mL, > 0).
    gastric_volume_mL : float
        Volume of gastric fluid (mL, > 0).
    intestinal_volume_mL : float
        Volume of intestinal fluid (mL, > 0).
    gastric_emptying_h : float
        Gastric emptying half-life in hours (> 0), used as first-order rate 1/h.
    k_precip_per_h : float
        First-order precipitation rate constant (1/h).
    k_redissolve_per_h : float
        First-order redissolution rate constant (1/h).
    ka_intestinal_per_h : float
        Intestinal absorption rate constant (1/h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration time step (h).

    Returns
    -------
    GIPrecipitationResult
    """
    # Input validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_stomach_mg_mL <= 0:
        raise ValueError("solubility_stomach_mg_mL must be > 0")
    if solubility_intestine_mg_mL <= 0:
        raise ValueError("solubility_intestine_mg_mL must be > 0")
    if gastric_volume_mL <= 0:
        raise ValueError("gastric_volume_mL must be > 0")
    if intestinal_volume_mL <= 0:
        raise ValueError("intestinal_volume_mL must be > 0")
    if gastric_emptying_h <= 0:
        raise ValueError("gastric_emptying_h must be > 0")

    k_empty = 1.0 / gastric_emptying_h  # first-order gastric emptying rate (1/h)
    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    # Initial state
    max_dissolved_stomach = solubility_stomach_mg_mL * gastric_volume_mL  # mg
    if dose_mg <= max_dissolved_stomach:
        c_stomach = dose_mg / gastric_volume_mL  # mg/mL
        m_precip_stomach = 0.0
    else:
        c_stomach = solubility_stomach_mg_mL
        m_precip_stomach = dose_mg - solubility_stomach_mg_mL * gastric_volume_mL

    c_intestine = 0.0
    m_precip_intestine = 0.0
    m_absorbed = 0.0

    # Storage arrays
    c_stom_arr = np.zeros(n_steps + 1)
    c_int_arr = np.zeros(n_steps + 1)
    mp_stom_arr = np.zeros(n_steps + 1)
    mp_int_arr = np.zeros(n_steps + 1)

    c_stom_arr[0] = c_stomach
    c_int_arr[0] = c_intestine
    mp_stom_arr[0] = m_precip_stomach
    mp_int_arr[0] = m_precip_intestine

    time_to_precipitation_h = t_end_h  # default: no precipitation

    for i in range(n_steps):
        # ── Gastric emptying (dissolved + precipitated) ──────────────────
        flow_dissolved_mg = k_empty * c_stomach * gastric_volume_mL * dt_h
        flow_dissolved_mg = min(flow_dissolved_mg, c_stomach * gastric_volume_mL)
        flow_precip_mg = k_empty * m_precip_stomach * dt_h
        flow_precip_mg = min(flow_precip_mg, m_precip_stomach)

        # ── Stomach precipitation / redissolution ────────────────────────
        precip_stom = k_precip_per_h * max(0.0, c_stomach - solubility_stomach_mg_mL) * gastric_volume_mL
        rediss_stom = k_redissolve_per_h * m_precip_stomach

        # ── Intestinal processes ─────────────────────────────────────────
        absorption_mg_h = ka_intestinal_per_h * c_intestine * intestinal_volume_mL
        precip_int = (
            k_precip_per_h
            * max(0.0, c_intestine - solubility_intestine_mg_mL)
            * intestinal_volume_mL
        )
        rediss_int = k_redissolve_per_h * m_precip_intestine

        # ── Update stomach ───────────────────────────────────────────────
        dm_stom_dissolved = (
            -flow_dissolved_mg
            - precip_stom * dt_h
            + rediss_stom * dt_h
        )
        dm_precip_stom = (
            precip_stom * dt_h
            - rediss_stom * dt_h
            - flow_precip_mg
        )

        m_stom_dissolved = max(0.0, c_stomach * gastric_volume_mL + dm_stom_dissolved)
        m_precip_stomach = max(0.0, m_precip_stomach + dm_precip_stom)
        c_stomach = m_stom_dissolved / gastric_volume_mL

        # ── Update intestine ─────────────────────────────────────────────
        # Incoming from stomach (dissolved transfers directly, precipitate adds to intestinal precipitate)
        incoming_dissolved_mg = flow_dissolved_mg
        incoming_precip_mg = flow_precip_mg

        dm_int_dissolved = (
            incoming_dissolved_mg
            - absorption_mg_h * dt_h
            - precip_int * dt_h
            + rediss_int * dt_h
        )
        dm_precip_int = (
            precip_int * dt_h
            - rediss_int * dt_h
            + incoming_precip_mg
        )
        dm_absorbed = absorption_mg_h * dt_h

        m_int_dissolved = max(0.0, c_intestine * intestinal_volume_mL + dm_int_dissolved)
        m_precip_intestine = max(0.0, m_precip_intestine + dm_precip_int)
        c_intestine = m_int_dissolved / intestinal_volume_mL
        m_absorbed += dm_absorbed

        c_stom_arr[i + 1] = c_stomach
        c_int_arr[i + 1] = c_intestine
        mp_stom_arr[i + 1] = m_precip_stomach
        mp_int_arr[i + 1] = m_precip_intestine

        # Track time to first precipitation
        total_precip = m_precip_stomach + m_precip_intestine
        if total_precip > 0.01 and time_to_precipitation_h == t_end_h:
            time_to_precipitation_h = times[i + 1]

    # Summary statistics
    f_absorbed = min(1.0, m_absorbed / dose_mg)
    f_precipitated_stomach = mp_stom_arr[-1] / dose_mg
    f_precipitated_intestine = mp_int_arr[-1] / dose_mg
    f_lost = 1.0 - f_absorbed

    # Peak supersaturation in intestine
    with np.errstate(divide="ignore", invalid="ignore"):
        ss_int = c_int_arr / solubility_intestine_mg_mL
    peak_supersaturation_ratio = float(np.max(ss_int))
    if peak_supersaturation_ratio < 0:
        peak_supersaturation_ratio = 0.0

    bioavailability_reduction_pct = float((1.0 - f_absorbed) * 100.0)
    bioavailability_reduction_pct = max(0.0, min(100.0, bioavailability_reduction_pct))

    notes = (
        f"Forward Euler GI precipitation simulation (dt={dt_h} h). "
        f"k_precip={k_precip_per_h}/h, k_redissolve={k_redissolve_per_h}/h. "
        f"Gastric emptying 1/h={k_empty:.3f}. "
        f"f_absorbed={f_absorbed:.3f}, RID={bioavailability_reduction_pct:.1f}%."
    )

    return GIPrecipitationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        solubility_stomach_mg_mL=solubility_stomach_mg_mL,
        solubility_intestine_mg_mL=solubility_intestine_mg_mL,
        times_h=times.tolist(),
        c_dissolved_stomach_mg_mL=c_stom_arr.tolist(),
        c_dissolved_intestine_mg_mL=c_int_arr.tolist(),
        c_precipitated_stomach_mg=mp_stom_arr.tolist(),
        c_precipitated_intestine_mg=mp_int_arr.tolist(),
        f_precipitated_stomach=float(f_precipitated_stomach),
        f_precipitated_intestine=float(f_precipitated_intestine),
        f_absorbed=float(f_absorbed),
        f_lost=float(f_lost),
        peak_supersaturation_ratio=peak_supersaturation_ratio,
        time_to_precipitation_h=float(time_to_precipitation_h),
        bioavailability_reduction_pct=bioavailability_reduction_pct,
        notes=notes,
    )


def effect_of_supersaturation_inhibitors(
    drug_name: str,
    dose_mg: float,
    solubility_intestine_mg_mL: float,
    k_precip_values: list[float] | None = None,
) -> list[GIPrecipitationResult]:
    """Evaluate impact of polymer precipitation inhibitors on GI absorption.

    Lower k_precip values represent more effective inhibitors (e.g., HPMC, PVP).

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Oral dose (mg).
    solubility_intestine_mg_mL : float
        Intestinal solubility (mg/mL).
    k_precip_values : list[float] or None
        Precipitation rate constants to evaluate. Defaults to [5.0, 2.0, 0.5, 0.1].

    Returns
    -------
    list[GIPrecipitationResult]
        Results sorted by f_absorbed descending (best absorption first).
    """
    if k_precip_values is None:
        k_precip_values = [5.0, 2.0, 0.5, 0.1]

    results = []
    for k_p in k_precip_values:
        result = simulate_gi_precipitation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            solubility_intestine_mg_mL=solubility_intestine_mg_mL,
            k_precip_per_h=k_p,
        )
        results.append(result)

    return sorted(results, key=lambda r: r.f_absorbed, reverse=True)
