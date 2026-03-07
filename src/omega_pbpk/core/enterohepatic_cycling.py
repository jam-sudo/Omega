"""Advanced enterohepatic recirculation (EHR) model with gallbladder emptying and bile secretion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class EnterohepatiCyclingResult:
    drug_name: str
    dose_mg: float
    f_biliary: float
    t_half_h: float
    t_half_no_ehr_h: float

    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_bile_mg_L: list[float]
    m_gallbladder_mg: list[float]

    secondary_peaks: int
    auc_total_mg_h_L: float
    auc_no_ehr_mg_h_L: float
    ehr_auc_ratio: float
    notes: str


def _validate_inputs(
    cl_L_per_h: float,
    vd_L: float,
    dose_mg: float,
    f_biliary: float,
    ka_per_h: float,
    f_oral: float,
    gallbladder_empty_interval_h: float,
    gallbladder_empty_fraction: float,
) -> None:
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if f_biliary < 0 or f_biliary >= 1.0:
        raise ValueError("f_biliary must be in [0, 1)")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if f_oral <= 0 or f_oral > 1.0:
        raise ValueError("f_oral must be in (0, 1]")
    if gallbladder_empty_interval_h <= 0:
        raise ValueError("gallbladder_empty_interval_h must be > 0")
    if gallbladder_empty_fraction <= 0 or gallbladder_empty_fraction > 1.0:
        raise ValueError("gallbladder_empty_fraction must be in (0, 1]")


def _run_simulation(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_biliary: float,
    ka_per_h: float,
    f_oral: float,
    gallbladder_empty_interval_h: float,
    gallbladder_empty_fraction: float,
    k_gb_fill_per_h: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Run the ODE forward Euler simulation. Returns (times, c_plasma, c_bile, m_gb)."""
    n_steps = int(round(t_end_h / dt_h)) + 1
    times = [i * dt_h for i in range(n_steps)]

    # Rate constants
    ke = (cl_L_per_h * (1.0 - f_biliary)) / vd_L  # non-biliary hepatic elimination
    k_biliary = (f_biliary * cl_L_per_h) / vd_L  # biliary secretion rate constant

    # State variables
    a_gut = 0.0  # mg in GI tract
    c_plasma = 0.0  # mg/L plasma concentration
    m_gb = 0.0  # mg in gallbladder

    c_plasma_list: list[float] = []
    c_bile_list: list[float] = []
    m_gb_list: list[float] = []

    # Initial dose into gut
    a_gut = f_oral * dose_mg

    for i, t in enumerate(times):
        # Record state
        c_plasma_list.append(max(0.0, c_plasma))
        c_bile_list.append(max(0.0, k_biliary * c_plasma))
        m_gb_list.append(max(0.0, m_gb))

        # Gallbladder emptying: impulsive at each interval (check if t is a multiple)
        gb_dump = 0.0
        if i > 0 and gallbladder_empty_interval_h > 0:
            t_mod = t % gallbladder_empty_interval_h
            # Within dt_h/2 tolerance of a multiple
            tol = dt_h * 0.5
            if t_mod < tol or (gallbladder_empty_interval_h - t_mod) < tol:
                gb_dump = gallbladder_empty_fraction * m_gb
                m_gb -= gb_dump
                m_gb = max(0.0, m_gb)
                a_gut += gb_dump

        # ODEs (forward Euler)
        da_gut = -ka_per_h * a_gut
        dc_plasma = (ka_per_h * a_gut) / vd_L - ke * c_plasma - k_biliary * c_plasma
        dm_gb = k_biliary * c_plasma * vd_L  # filling from plasma

        a_gut = max(0.0, a_gut + da_gut * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        m_gb = max(0.0, m_gb + dm_gb * dt_h)

    return times, c_plasma_list, c_bile_list, m_gb_list


def _count_secondary_peaks(times: list[float], c_plasma: list[float]) -> int:
    """Count sign changes from negative to positive in d(c_plasma)/dt after t > 3h."""
    n = len(c_plasma)
    count = 0
    for i in range(1, n - 1):
        if times[i] <= 3.0:
            continue
        deriv_prev = c_plasma[i] - c_plasma[i - 1]
        deriv_next = c_plasma[i + 1] - c_plasma[i]
        if deriv_prev < 0 and deriv_next > 0:
            count += 1
    return count


def _estimate_apparent_half_life(times: list[float], c_plasma: list[float]) -> float:
    """Estimate apparent half-life from terminal slope of log-plasma concentration."""
    n = len(c_plasma)
    # Use last 30% of time points
    start_idx = int(n * 0.7)
    # Filter positive concentrations
    t_term = []
    lnc_term = []
    for i in range(start_idx, n):
        if c_plasma[i] > 1e-12:
            t_term.append(times[i])
            lnc_term.append(math.log(c_plasma[i]))

    if len(t_term) < 2:
        return float("nan")

    # Linear regression ln(C) vs t
    t_arr = np.array(t_term)
    lnc_arr = np.array(lnc_term)
    slope, _ = np.polyfit(t_arr, lnc_arr, 1)

    if slope >= 0:
        return float("nan")

    return math.log(2.0) / (-slope)


def simulate_enterohepatic_cycling(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_biliary: float = 0.3,
    ka_per_h: float = 1.2,
    f_oral: float = 0.8,
    gallbladder_empty_interval_h: float = 4.0,
    gallbladder_empty_fraction: float = 0.8,
    k_gb_fill_per_h: float = 0.5,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> EnterohepatiCyclingResult:
    """Simulate advanced enterohepatic cycling with gallbladder emptying.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    cl_L_per_h:
        Total hepatic clearance in L/h.
    vd_L:
        Volume of distribution in L.
    f_biliary:
        Fraction of hepatic clearance via bile (0 to <1).
    ka_per_h:
        First-order absorption rate constant (1/h).
    f_oral:
        Oral bioavailability fraction.
    gallbladder_empty_interval_h:
        Interval between gallbladder emptying events (hours).
    gallbladder_empty_fraction:
        Fraction of gallbladder contents emptied per event.
    k_gb_fill_per_h:
        Gallbladder fill rate constant (not used directly, included for API consistency).
    t_end_h:
        Simulation end time (hours).
    dt_h:
        ODE step size (hours).

    Returns
    -------
    EnterohepatiCyclingResult
    """
    _validate_inputs(
        cl_L_per_h,
        vd_L,
        dose_mg,
        f_biliary,
        ka_per_h,
        f_oral,
        gallbladder_empty_interval_h,
        gallbladder_empty_fraction,
    )

    # Full EHR simulation
    times, c_plasma, c_bile, m_gb = _run_simulation(
        dose_mg,
        cl_L_per_h,
        vd_L,
        f_biliary,
        ka_per_h,
        f_oral,
        gallbladder_empty_interval_h,
        gallbladder_empty_fraction,
        k_gb_fill_per_h,
        t_end_h,
        dt_h,
    )

    # No-EHR simulation (f_biliary = 0)
    _, c_plasma_no_ehr, _, _ = _run_simulation(
        dose_mg,
        cl_L_per_h,
        vd_L,
        0.0,
        ka_per_h,
        f_oral,
        gallbladder_empty_interval_h,
        gallbladder_empty_fraction,
        k_gb_fill_per_h,
        t_end_h,
        dt_h,
    )

    times_arr = np.array(times)
    c_arr = np.array(c_plasma)
    c_no_ehr_arr = np.array(c_plasma_no_ehr)

    auc_total = float(np_trapz(c_arr, times_arr))
    auc_no_ehr = float(np_trapz(c_no_ehr_arr, times_arr))

    if auc_no_ehr > 0:
        ehr_auc_ratio = auc_total / auc_no_ehr
    else:
        ehr_auc_ratio = 1.0

    secondary_peaks = _count_secondary_peaks(times, c_plasma)
    t_half = _estimate_apparent_half_life(times, c_plasma)
    t_half_no_ehr = _estimate_apparent_half_life(times, c_plasma_no_ehr)

    # Fallback half-life from PK theory if regression fails
    ke_total = cl_L_per_h / vd_L
    t_half_theory = math.log(2.0) / ke_total if ke_total > 0 else float("nan")
    if math.isnan(t_half):
        t_half = t_half_theory
    if math.isnan(t_half_no_ehr):
        t_half_no_ehr = t_half_theory

    notes = (
        f"{drug_name}: f_biliary={f_biliary:.2f}, GB empties every "
        f"{gallbladder_empty_interval_h:.1f}h ({gallbladder_empty_fraction * 100:.0f}% emptied). "
        f"EHR AUC ratio={ehr_auc_ratio:.2f}."
    )

    return EnterohepatiCyclingResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_biliary=f_biliary,
        t_half_h=t_half,
        t_half_no_ehr_h=t_half_no_ehr,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_bile_mg_L=c_bile,
        m_gallbladder_mg=m_gb,
        secondary_peaks=secondary_peaks,
        auc_total_mg_h_L=auc_total,
        auc_no_ehr_mg_h_L=auc_no_ehr,
        ehr_auc_ratio=ehr_auc_ratio,
        notes=notes,
    )


def compare_biliary_fractions(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    f_biliary_values: list[float] | None = None,
) -> list[EnterohepatiCyclingResult]:
    """Simulate EHR for multiple biliary fractions and return sorted by ehr_auc_ratio.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    cl_L_per_h:
        Total hepatic clearance in L/h.
    vd_L:
        Volume of distribution in L.
    f_biliary_values:
        List of biliary fraction values to simulate. Defaults to [0.0, 0.1, 0.3, 0.5, 0.7].

    Returns
    -------
    list[EnterohepatiCyclingResult] sorted by ehr_auc_ratio descending.
    """
    if f_biliary_values is None:
        f_biliary_values = [0.0, 0.1, 0.3, 0.5, 0.7]

    results = []
    for fb in f_biliary_values:
        result = simulate_enterohepatic_cycling(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            f_biliary=fb,
        )
        results.append(result)

    results.sort(key=lambda r: r.ehr_auc_ratio, reverse=True)
    return results
