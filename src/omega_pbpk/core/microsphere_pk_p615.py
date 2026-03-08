"""Microsphere Drug Release PK model (Phase 615).

Models drug release from biodegradable PLGA microspheres and subsequent
systemic pharmacokinetics. Uses a biphasic release model:
  - Phase 1 (burst): first-order burst release in first 24h (15% of dose)
  - Phase 2 (erosion): PLGA erosion-controlled release of remaining 85%

References
----------
- Siepmann J & Gopferich A, Adv Drug Deliv Rev. 2001
- Fredenberg S et al., Int J Pharm. 2011
- Anderson JM & Shive MS, Adv Drug Deliv Rev. 1997
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MicrospherePKResult:
    """Results from microsphere drug release PK simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    dose_mg : Total dose (mg).
    microsphere_diameter_um : Microsphere diameter (micrometers).
    times_h : Time points (h).
    fraction_released : Cumulative fraction released from microspheres (0-1).
    c_systemic_mg_L : Systemic plasma concentration (mg/L).
    cmax_mg_L : Peak plasma concentration (mg/L).
    tmax_h : Time of peak plasma concentration (h).
    auc_mg_h_per_L : AUC 0 to t_end (mg*h/L).
    t_half_release_h : Time for 50% cumulative release (h).
    mean_residence_time_h : MRT = AUMC/AUC (h).
    sustained_release_duration_h : Time for 90% release (h).
    notes : Summary text.
    """

    drug_name: str
    dose_mg: float
    microsphere_diameter_um: float
    times_h: list
    fraction_released: list
    c_systemic_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_release_h: float
    mean_residence_time_h: float
    sustained_release_duration_h: float
    notes: str


def simulate_microsphere_pk(
    drug_name: str,
    dose_mg: float,
    microsphere_diameter_um: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    drug_loading_pct: float = 10.0,
    plga_mw_kDa: float = 50.0,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> MicrospherePKResult:
    """Simulate microsphere drug release and systemic PK.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Total dose in mg.
    microsphere_diameter_um : Microsphere diameter in micrometers.
    cl_sys_L_per_h : Systemic clearance (L/h).
    vd_sys_L : Volume of distribution (L).
    drug_loading_pct : Drug loading percentage (default 10%).
    plga_mw_kDa : PLGA molecular weight in kDa (default 50 kDa).
    t_end_h : Simulation end time in hours (default 720h = 30 days).
    dt_h : Time step in hours (default 1h).

    Returns
    -------
    MicrospherePKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if microsphere_diameter_um <= 0:
        raise ValueError("microsphere_diameter_um must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if not (0 < drug_loading_pct <= 100):
        raise ValueError("drug_loading_pct must be in (0, 100]")

    # --- Release rate constants ---
    # PLGA degradation half-life: larger polymer = slower degradation
    t_half_deg = plga_mw_kDa * 2.0  # hours

    # Burst phase: 15% of dose released in first 24h first-order
    f_burst = 0.15
    k1 = -math.log(1.0 - f_burst) / 24.0  # /h

    # Erosion phase: remaining 85%, rate from PLGA half-life
    k2 = math.log(2.0) / t_half_deg  # /h

    # Microsphere size effect: smaller = faster release
    # Reference diameter = 50 µm
    size_factor = 50.0 / microsphere_diameter_um
    # Clamp to [0.1, 10.0]
    size_factor = max(0.1, min(10.0, size_factor))

    k1 *= size_factor
    k2 *= size_factor

    # --- Initial conditions ---
    B = dose_mg * 0.15  # burst depot (mg)
    E = dose_mg * 0.85  # erosion depot (mg)
    C = 0.0  # systemic concentration (mg/L)

    # --- Simulate ---
    n_steps = int(t_end_h / dt_h)
    times_h: list[float] = []
    fraction_released: list[float] = []
    c_systemic: list[float] = []

    t = 0.0
    for i in range(n_steps + 1):
        fr = 1.0 - (B + E) / dose_mg
        fr = max(0.0, min(1.0, fr))
        times_h.append(t)
        fraction_released.append(fr)
        c_systemic.append(max(0.0, C))

        if i < n_steps:
            # Release rates
            release_rate = k1 * B + k2 * E  # mg/h

            # ODEs
            dB = -k1 * B
            dE = -k2 * E
            dC = release_rate / vd_sys_L - (cl_sys_L_per_h / vd_sys_L) * C

            # Forward Euler
            B = max(0.0, B + dB * dt_h)
            E = max(0.0, E + dE * dt_h)
            C = max(0.0, C + dC * dt_h)
            t += dt_h

    # --- Derived metrics ---
    cmax_mg_L = max(c_systemic)
    tmax_h = times_h[c_systemic.index(cmax_mg_L)]

    # AUC via trapezoidal rule
    auc = sum(
        0.5 * (c_systemic[i] + c_systemic[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # AUMC via trapezoidal rule
    aumc = sum(
        0.5
        * (times_h[i] * c_systemic[i] + times_h[i - 1] * c_systemic[i - 1])
        * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    mrt = aumc / auc if auc > 0.0 else t_end_h

    # t_half_release: first time fraction_released >= 0.5
    t_half_release = t_end_h
    for i, fr in enumerate(fraction_released):
        if fr >= 0.5:
            t_half_release = times_h[i]
            break

    # sustained_release_duration: first time fraction_released >= 0.9
    sustained_duration = t_end_h
    for i, fr in enumerate(fraction_released):
        if fr >= 0.9:
            sustained_duration = times_h[i]
            break

    notes = (
        f"PLGA microsphere PK: diameter={microsphere_diameter_um} um, "
        f"MW={plga_mw_kDa} kDa, t_half_deg={t_half_deg:.1f} h, "
        f"size_factor={size_factor:.3f}, k1={k1:.5f}/h, k2={k2:.5f}/h. "
        f"Cmax={cmax_mg_L:.4f} mg/L at {tmax_h:.1f}h, "
        f"50% release at {t_half_release:.1f}h, 90% release at {sustained_duration:.1f}h."
    )

    return MicrospherePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        microsphere_diameter_um=microsphere_diameter_um,
        times_h=times_h,
        fraction_released=fraction_released,
        c_systemic_mg_L=c_systemic,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        t_half_release_h=t_half_release,
        mean_residence_time_h=mrt,
        sustained_release_duration_h=sustained_duration,
        notes=notes,
    )


def compare_microsphere_sizes(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    diameters: list = None,
) -> list:
    """Compare microsphere release PK across different diameters.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Total dose (mg).
    cl_sys_L_per_h : Systemic clearance (L/h).
    vd_sys_L : Volume of distribution (L).
    diameters : List of microsphere diameters in um (default [10, 50, 100]).

    Returns
    -------
    list of MicrospherePKResult sorted by sustained_release_duration_h descending.
    """
    if diameters is None:
        diameters = [10.0, 50.0, 100.0]

    results = []
    for d in diameters:
        r = simulate_microsphere_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            microsphere_diameter_um=d,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(r)

    # Sort by sustained_release_duration_h descending (larger = more sustained)
    results.sort(key=lambda r: r.sustained_release_duration_h, reverse=True)
    return results
