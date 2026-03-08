"""
Phase 408 — Drug Absorption Lag Time Model

Model oral drug absorption with lag time (t_lag) due to gastric emptying,
tablet disintegration, or intestinal transit to absorption site.
"""

import math
from dataclasses import dataclass


@dataclass
class AbsorptionLagResult:
    """Result of absorption lag time simulation."""

    drug_name: str
    dose_mg: float
    t_lag_h: float
    ka_per_h: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float  # actual tmax observed
    auc_mg_h_per_L: float
    t_half_apparent_h: float
    lag_time_effect_pct: float  # % reduction in Cmax vs no-lag scenario
    absorption_complete_h: float  # time for 95% absorption
    notes: str


def _validate_inputs(dose_mg, t_lag_h, ka_per_h, cl_L_per_h, vd_L, f_oral, t_end_h, dt_h):
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if t_lag_h < 0:
        raise ValueError("t_lag_h must be >= 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")


def _simulate_ode(dose_mg, t_lag_h, ka_per_h, cl_L_per_h, vd_L, f_oral, t_end_h, dt_h):
    """
    Forward Euler simulation of 1-cpt PK with lag time.

    Before t_lag: drug stays in gut depot (no absorption), plasma
                  elimination continues.
    After t_lag: first-order absorption starts from gut depot.
    """
    ke = cl_L_per_h / vd_L

    times = []
    c_plasma = []
    m_gut_history = []

    m_gut = dose_mg  # gut depot (mg)
    c = 0.0  # plasma concentration (mg/L)
    t = 0.0

    while t <= t_end_h + 1e-9:
        times.append(t)
        c_plasma.append(max(c, 0.0))
        m_gut_history.append(m_gut)

        # Forward Euler step
        if t < t_lag_h:
            # Lag phase: no absorption; plasma only eliminated
            dc = -ke * c
            dm_gut = 0.0
        else:
            # Absorption phase
            dm_gut = -ka_per_h * m_gut
            # Absorbed amount per unit time contributes to plasma
            dc = ka_per_h * f_oral * m_gut / vd_L - ke * c

        c = c + dc * dt_h
        c = max(c, 0.0)
        m_gut = m_gut + dm_gut * dt_h
        m_gut = max(m_gut, 0.0)
        t += dt_h

    return times, c_plasma, m_gut_history


def _trapezoidal_auc(x, y):
    """Manual trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_absorption_lag(
    drug_name: str,
    dose_mg: float,
    t_lag_h: float,
    ka_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> AbsorptionLagResult:
    """
    Simulate oral PK with lag time using Forward Euler.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    t_lag_h : float  lag time (h); 0 = no lag
    ka_per_h : float  first-order absorption rate constant
    cl_L_per_h : float  clearance
    vd_L : float  volume of distribution
    f_oral : float  oral bioavailability
    t_end_h : float  simulation end time
    dt_h : float  integration step size

    Returns
    -------
    AbsorptionLagResult
    """
    _validate_inputs(dose_mg, t_lag_h, ka_per_h, cl_L_per_h, vd_L, f_oral, t_end_h, dt_h)

    times, c_plasma, m_gut_history = _simulate_ode(
        dose_mg, t_lag_h, ka_per_h, cl_L_per_h, vd_L, f_oral, t_end_h, dt_h
    )

    # Cmax and tmax
    cmax = max(c_plasma)
    tmax_idx = c_plasma.index(cmax)
    tmax_h = times[tmax_idx]

    # AUC
    auc = _trapezoidal_auc(times, c_plasma)

    # Apparent half-life
    ke = cl_L_per_h / vd_L
    t_half = math.log(2.0) / ke if ke > 0 else float("inf")

    # Absorption complete: time when m_gut <= 0.05 * dose_mg
    absorption_complete_h = t_end_h  # fallback
    threshold = 0.05 * dose_mg
    for _i, (t, mg) in enumerate(zip(times, m_gut_history)):
        if t >= t_lag_h and mg <= threshold:
            absorption_complete_h = t
            break

    # Lag time effect: compare Cmax with and without lag
    if t_lag_h > 0:
        times_no_lag, c_no_lag, _ = _simulate_ode(
            dose_mg, 0.0, ka_per_h, cl_L_per_h, vd_L, f_oral, t_end_h, dt_h
        )
        cmax_no_lag = max(c_no_lag)
        if cmax_no_lag > 0:
            lag_effect_pct = (cmax_no_lag - cmax) / cmax_no_lag * 100.0
        else:
            lag_effect_pct = 0.0
    else:
        lag_effect_pct = 0.0

    lag_effect_pct = max(lag_effect_pct, 0.0)

    notes = (
        f"ke={ke:.4f}/h, t_half={t_half:.2f}h, "
        f"AUC_theoretical={f_oral * dose_mg / cl_L_per_h:.2f} mg*h/L"
    )

    return AbsorptionLagResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        t_lag_h=t_lag_h,
        ka_per_h=ka_per_h,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        cmax_mg_L=cmax,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        t_half_apparent_h=t_half,
        lag_time_effect_pct=lag_effect_pct,
        absorption_complete_h=absorption_complete_h,
        notes=notes,
    )


def compare_lag_times(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float,
    cl_L_per_h: float,
    vd_L: float,
    f_oral: float = 1.0,
    lag_times_h: list[float] | None = None,
) -> list[AbsorptionLagResult]:
    """
    Compare absorption profiles across multiple lag times.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    ka_per_h : float
    cl_L_per_h : float
    vd_L : float
    f_oral : float
    lag_times_h : list of float or None (default = [0.0, 0.5, 1.0, 2.0, 4.0])

    Returns
    -------
    list of AbsorptionLagResult sorted by tmax_h ascending
    """
    if lag_times_h is None:
        lag_times_h = [0.0, 0.5, 1.0, 2.0, 4.0]

    results = []
    for lag in lag_times_h:
        result = simulate_absorption_lag(
            drug_name=drug_name,
            dose_mg=dose_mg,
            t_lag_h=lag,
            ka_per_h=ka_per_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            f_oral=f_oral,
        )
        results.append(result)

    results.sort(key=lambda r: r.tmax_h)
    return results
