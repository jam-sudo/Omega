"""Osmotic pump (OROS) drug release and PK simulation.

Models zero-order release from an osmotic controlled-release oral delivery
system.  Drug is released at a constant rate until the tablet is exhausted,
then absorbed from the GI lumen (first-order) into a one-compartment plasma
model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class OsmoticPumpResult:
    """Result of an OROS simulation."""

    drug_name: str
    dose_mg: float
    release_duration_h: float
    times_h: list[float]
    m_tablet_mg: list[float]  # remaining drug mass in tablet (mg)
    c_gi_mg_mL: list[float]  # drug concentration in GI lumen (mg/mL, proxy)
    c_plasma_mg_L: list[float]  # plasma concentration (mg/L)
    cmax_mg_L: float  # peak plasma concentration
    cmin_mg_L: float  # minimum plasma concentration after peak
    tmax_h: float  # time of peak plasma concentration
    auc_mg_h_per_L: float  # area under the plasma concentration-time curve
    peak_trough_ratio: float  # Cmax / Cmin (Cmin measured after peak)
    fluctuation_pct: float  # (Cmax - Cmin) / Cmin * 100
    release_complete_h: float  # time at which tablet is exhausted
    notes: str


def simulate_osmotic_pump(
    drug_name: str,
    dose_mg: float,
    release_duration_h: float,
    ka_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    k_transit_per_h: float = 0.02,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> OsmoticPumpResult:
    """Simulate OROS tablet PK using forward Euler integration.

    The tablet releases drug at a constant zero-order rate until exhausted.
    Drug in the GI lumen is absorbed by first-order kinetics into plasma, which
    is described by a one-compartment model with first-order elimination.

    Parameters
    ----------
    drug_name:
        Name of the drug (for labelling).
    dose_mg:
        Total dose loaded in the osmotic tablet (mg).
    release_duration_h:
        Duration over which the tablet releases drug at constant rate (h).
    ka_per_h:
        First-order absorption rate constant from GI lumen to plasma (1/h).
    cl_L_per_h:
        Total plasma clearance (L/h).
    vd_L:
        Volume of distribution (L).
    k_transit_per_h:
        First-order GI transit/loss rate constant (1/h).  Default 0.02 /h.
    t_end_h:
        Simulation end time (h).  Default 24 h.
    dt_h:
        Integration step size (h).  Default 0.05 h.

    Returns
    -------
    OsmoticPumpResult
    """
    # --- input validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if release_duration_h <= 0:
        raise ValueError(f"release_duration_h must be positive, got {release_duration_h}")
    if ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be positive, got {ka_per_h}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")

    # --- derived constants ---
    ke_per_h = cl_L_per_h / vd_L  # elimination rate constant
    q_release_mg_per_h = dose_mg / release_duration_h  # zero-order release rate

    # --- state initialisation ---
    # m_tab: drug remaining in tablet (mg)
    # m_gi : drug mass in GI lumen (mg)  — proxy; divide by a nominal volume
    #        for concentration display but PK uses mass balance
    # a_p  : drug amount in plasma (mg)
    m_tab = dose_mg
    m_gi = 0.0
    a_p = 0.0  # amount in plasma (mg); C_plasma = a_p / vd_L

    n_steps = int(t_end_h / dt_h) + 1
    times = np.linspace(0.0, t_end_h, n_steps)

    out_m_tab = np.empty(n_steps)
    out_m_gi = np.empty(n_steps)
    out_c_plasma = np.empty(n_steps)

    release_complete_h = release_duration_h  # default (may be clamped)

    for i, _t in enumerate(times):
        out_m_tab[i] = m_tab
        out_m_gi[i] = m_gi
        out_c_plasma[i] = a_p / vd_L  # mg/L

        # zero-order release from tablet
        if m_tab > 0.0:
            rate_in = min(q_release_mg_per_h, m_tab / dt_h)
            dm_tab = -rate_in * dt_h
            dm_gi_from_tablet = rate_in * dt_h
        else:
            dm_tab = 0.0
            dm_gi_from_tablet = 0.0

        # absorption + transit from GI to plasma
        absorbed = ka_per_h * m_gi * dt_h
        transit_loss = k_transit_per_h * m_gi * dt_h

        dm_gi = dm_gi_from_tablet - absorbed - transit_loss
        da_p = absorbed - ke_per_h * a_p * dt_h

        m_tab = max(0.0, m_tab + dm_tab)
        m_gi = max(0.0, m_gi + dm_gi)
        a_p = max(0.0, a_p + da_p)

    c_plasma = out_c_plasma

    # --- PK metrics ---
    idx_peak = int(np.argmax(c_plasma))
    cmax = float(c_plasma[idx_peak])
    tmax = float(times[idx_peak])

    # Cmin: minimum plasma concentration *after* the peak
    c_after_peak = c_plasma[idx_peak:]
    if len(c_after_peak) > 0:
        cmin = float(np.min(c_after_peak))
    else:
        cmin = float(c_plasma[-1])

    # guard against zero cmin (e.g., drug fully eliminated before end)
    cmin = max(cmin, 1e-9)

    auc = float(np_trapz(c_plasma, times))
    ptr = cmax / cmin
    fluct_pct = (cmax - cmin) / cmin * 100.0

    # release_complete_h: first time tablet is essentially empty
    tol = dose_mg * 1e-4
    idx_exhaust = np.where(out_m_tab <= tol)[0]
    if len(idx_exhaust) > 0:
        release_complete_h = float(times[idx_exhaust[0]])
    else:
        release_complete_h = t_end_h

    # nominal GI concentration (divide by 250 mL stomach volume)
    gi_vol_mL = 250.0
    c_gi_mg_mL = list(out_m_gi / gi_vol_mL)

    notes = (
        f"OROS simulation: {release_duration_h:.1f} h zero-order release, "
        f"ke={ke_per_h:.3f} /h, ka={ka_per_h:.3f} /h"
    )

    return OsmoticPumpResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        release_duration_h=release_duration_h,
        times_h=list(times),
        m_tablet_mg=list(out_m_tab),
        c_gi_mg_mL=c_gi_mg_mL,
        c_plasma_mg_L=list(c_plasma),
        cmax_mg_L=cmax,
        cmin_mg_L=cmin,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        peak_trough_ratio=ptr,
        fluctuation_pct=fluct_pct,
        release_complete_h=release_complete_h,
        notes=notes,
    )


def compare_release_durations(
    drug_name: str,
    dose_mg: float,
    durations_h: list[float],
    ka_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    **kwargs,
) -> list[OsmoticPumpResult]:
    """Simulate the same drug/dose across multiple release durations.

    Returns results sorted by *peak_trough_ratio* ascending (lower ratio means
    more controlled, smoother release).

    Parameters
    ----------
    drug_name, dose_mg, ka_per_h, cl_L_per_h, vd_L:
        As in :func:`simulate_osmotic_pump`.
    durations_h:
        List of release durations to compare.
    **kwargs:
        Forwarded to :func:`simulate_osmotic_pump` (e.g. ``t_end_h``,
        ``dt_h``, ``k_transit_per_h``).
    """
    if not durations_h:
        raise ValueError("durations_h must not be empty")

    results = [
        simulate_osmotic_pump(
            drug_name=drug_name,
            dose_mg=dose_mg,
            release_duration_h=d,
            ka_per_h=ka_per_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            **kwargs,
        )
        for d in durations_h
    ]

    return sorted(results, key=lambda r: r.peak_trough_ratio)
