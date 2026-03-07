"""Pharmacokinetics model for drugs in bioreactor systems.

Models drug distribution and metabolism in cell culture, hollow fiber, and
fermentation bioreactors — relevant for bioprocessing of mAbs and cell
therapy products.

Compartments
------------
1. Medium (culture) — extracellular drug concentration
2. Intracellular — drug taken up by cells
3. Product/metabolite — accumulation of intracellular metabolic product

ODEs (Forward Euler)
--------------------
dC_med/dt  = -k_uptake * C_med + k_efflux * C_cell * volume_ratio
dC_cell/dt =  k_uptake * C_med / volume_ratio - k_efflux * C_cell - k_met * C_cell
dC_prod/dt =  k_met * C_cell - k_degrad * C_prod
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BioreactorPKResult:
    """Results from a bioreactor PK simulation.

    Attributes
    ----------
    drug_name : Compound identifier.
    initial_conc_mg_L : Starting drug concentration in medium (mg/L).
    times_h : Simulation time points (h).
    c_medium_mg_L : Drug concentration in medium over time (mg/L).
    c_intracell_mg_L : Intracellular drug concentration over time (mg/L).
    c_product_mg_L : Product/metabolite concentration over time (mg/L).
    cmax_medium : Peak medium drug concentration (mg/L).
    tmax_medium_h : Time of peak medium concentration (h).
    auc_medium : AUC of medium concentration (mg/L·h), trapezoidal.
    auc_product : AUC of product concentration (mg/L·h), trapezoidal.
    uptake_rate_avg : Average cellular uptake rate (mg/L/h).
    product_yield : Final product concentration at end of simulation (mg/L).
    half_life_medium_h : Apparent half-life of drug in medium (h).
    notes : Summary notes.
    """

    drug_name: str
    initial_conc_mg_L: float
    times_h: list
    c_medium_mg_L: list
    c_intracell_mg_L: list
    c_product_mg_L: list
    cmax_medium: float
    tmax_medium_h: float
    auc_medium: float
    auc_product: float
    uptake_rate_avg: float
    product_yield: float
    half_life_medium_h: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC."""
    total = 0.0
    for i in range(len(times) - 1):
        total += (concs[i] + concs[i + 1]) / 2.0 * (times[i + 1] - times[i])
    return total


def _apparent_half_life(times: list, concs: list, initial: float) -> float:
    """Estimate apparent half-life from time to reach 50% of initial."""
    half = initial / 2.0
    for i in range(1, len(concs)):
        if concs[i] <= half:
            # Linear interpolation between steps i-1 and i
            c0, c1 = concs[i - 1], concs[i]
            t0, t1 = times[i - 1], times[i]
            if c0 > c1:
                frac = (c0 - half) / (c0 - c1)
                return t0 + frac * (t1 - t0)
    # Never crossed — estimate from ln2 / apparent elimination rate
    # Use first and last non-zero points
    if concs[-1] > 0 and concs[0] > 0 and concs[-1] < concs[0]:
        k_app = math.log(concs[0] / concs[-1]) / (times[-1] - times[0])
        if k_app > 0:
            return math.log(2.0) / k_app
    return times[-1]  # fallback


def _run_ode(
    c_med0: float,
    k_uptake: float,
    k_efflux: float,
    k_metabolism: float,
    k_degradation: float,
    volume_ratio: float,
    t_end_h: float,
    dt_h: float,
    feed_schedule: list[tuple[float, float]] | None = None,
) -> tuple[list, list, list, list]:
    """Run forward Euler ODE. Returns (times, c_med, c_cell, c_prod)."""
    n_steps = int(round(t_end_h / dt_h)) + 1
    times = [0.0]
    c_med = [c_med0]
    c_cell = [0.0]
    c_prod = [0.0]

    # Build feed lookup: map step index → amount to add
    feed_events: dict[int, float] = {}
    if feed_schedule:
        for feed_t, feed_amt in feed_schedule:
            step_idx = int(round(feed_t / dt_h))
            feed_events[step_idx] = feed_events.get(step_idx, 0.0) + feed_amt

    cm = c_med0
    cc = 0.0
    cp = 0.0

    for i in range(1, n_steps):
        t_new = i * dt_h

        # Apply feeding event at step i (before ODE update for this step)
        if i in feed_events:
            cm = cm + feed_events[i]

        # ODE derivatives
        dcm = -k_uptake * cm + k_efflux * cc * volume_ratio
        dcc = k_uptake * cm / volume_ratio - k_efflux * cc - k_metabolism * cc
        dcp = k_metabolism * cc - k_degradation * cp

        cm = max(cm + dcm * dt_h, 0.0)
        cc = max(cc + dcc * dt_h, 0.0)
        cp = max(cp + dcp * dt_h, 0.0)

        times.append(t_new)
        c_med.append(cm)
        c_cell.append(cc)
        c_prod.append(cp)

    return times, c_med, c_cell, c_prod


def _compute_results(
    drug_name: str,
    initial_conc_mg_L: float,
    times: list,
    c_med: list,
    c_cell: list,
    c_prod: list,
    k_uptake: float,
    notes: str,
) -> BioreactorPKResult:
    """Compute summary metrics and build result."""
    cmax_medium = max(c_med)
    tmax_medium_h = times[c_med.index(cmax_medium)]

    auc_medium = _trapz_auc(times, c_med)
    auc_product = _trapz_auc(times, c_prod)

    # Average uptake rate: k_uptake * C_med (time-averaged)
    uptake_rates = [k_uptake * cm for cm in c_med]
    auc_uptake = _trapz_auc(times, uptake_rates)
    t_total = times[-1] if times[-1] > 0 else 1.0
    uptake_rate_avg = auc_uptake / t_total

    product_yield = c_prod[-1]
    half_life_medium_h = _apparent_half_life(times, c_med, initial_conc_mg_L)

    return BioreactorPKResult(
        drug_name=drug_name,
        initial_conc_mg_L=initial_conc_mg_L,
        times_h=times,
        c_medium_mg_L=c_med,
        c_intracell_mg_L=c_cell,
        c_product_mg_L=c_prod,
        cmax_medium=cmax_medium,
        tmax_medium_h=tmax_medium_h,
        auc_medium=auc_medium,
        auc_product=auc_product,
        uptake_rate_avg=uptake_rate_avg,
        product_yield=product_yield,
        half_life_medium_h=half_life_medium_h,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_bioreactor_pk(
    drug_name: str,
    initial_conc_mg_L: float,
    k_uptake_per_h: float = 0.5,
    k_efflux_per_h: float = 0.1,
    k_metabolism_per_h: float = 0.3,
    k_degradation_per_h: float = 0.05,
    volume_ratio: float = 0.1,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> BioreactorPKResult:
    """Simulate drug PK in a 3-compartment bioreactor model.

    Parameters
    ----------
    drug_name : Compound identifier.
    initial_conc_mg_L : Starting drug concentration in medium (mg/L). Must be > 0.
    k_uptake_per_h : Cellular uptake rate constant (1/h). Must be > 0.
    k_efflux_per_h : Drug efflux from cells to medium (1/h).
    k_metabolism_per_h : Intracellular metabolism to product (1/h).
    k_degradation_per_h : Product degradation rate (1/h).
    volume_ratio : Cell volume / total bioreactor volume (0 < ratio < 1).
    t_end_h : Simulation end time (h). Must be > 0.
    dt_h : Integration time step (h).

    Returns
    -------
    BioreactorPKResult

    Raises
    ------
    ValueError
        If any validation constraint is violated.
    """
    if initial_conc_mg_L <= 0:
        raise ValueError(f"initial_conc_mg_L must be > 0, got {initial_conc_mg_L}")
    if k_uptake_per_h <= 0:
        raise ValueError(f"k_uptake_per_h must be > 0, got {k_uptake_per_h}")
    if not (0 < volume_ratio < 1):
        raise ValueError(f"volume_ratio must be in (0, 1), got {volume_ratio}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")

    times, c_med, c_cell, c_prod = _run_ode(
        c_med0=initial_conc_mg_L,
        k_uptake=k_uptake_per_h,
        k_efflux=k_efflux_per_h,
        k_metabolism=k_metabolism_per_h,
        k_degradation=k_degradation_per_h,
        volume_ratio=volume_ratio,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    notes = (
        f"Bioreactor PK for {drug_name}: batch mode, {t_end_h:.0f}h simulation. "
        f"k_uptake={k_uptake_per_h}/h, k_met={k_metabolism_per_h}/h, "
        f"volume_ratio={volume_ratio}."
    )

    return _compute_results(
        drug_name, initial_conc_mg_L, times, c_med, c_cell, c_prod, k_uptake_per_h, notes
    )


def optimize_feeding_strategy(
    drug_name: str,
    initial_conc_mg_L: float,
    feed_intervals_h: list[float],
    feed_amount_mg_L: list[float],
    k_uptake_per_h: float = 0.5,
    k_efflux_per_h: float = 0.1,
    k_metabolism_per_h: float = 0.3,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> BioreactorPKResult:
    """Simulate bioreactor PK with fed-batch additions at specified intervals.

    Parameters
    ----------
    drug_name : Compound identifier.
    initial_conc_mg_L : Starting drug concentration in medium (mg/L). Must be > 0.
    feed_intervals_h : List of times (h) at which drug is added to the medium.
    feed_amount_mg_L : Concentration increments (mg/L) added at each interval.
    k_uptake_per_h : Cellular uptake rate constant (1/h). Must be > 0.
    k_efflux_per_h : Drug efflux from cells to medium (1/h).
    k_metabolism_per_h : Intracellular metabolism to product (1/h).
    t_end_h : Simulation end time (h).
    dt_h : Integration time step (h).

    Returns
    -------
    BioreactorPKResult

    Raises
    ------
    ValueError
        If feed_intervals_h and feed_amount_mg_L have mismatched lengths, or
        initial_conc_mg_L <= 0, or k_uptake_per_h <= 0.
    """
    if initial_conc_mg_L <= 0:
        raise ValueError(f"initial_conc_mg_L must be > 0, got {initial_conc_mg_L}")
    if k_uptake_per_h <= 0:
        raise ValueError(f"k_uptake_per_h must be > 0, got {k_uptake_per_h}")
    if len(feed_intervals_h) != len(feed_amount_mg_L):
        raise ValueError("feed_intervals_h and feed_amount_mg_L must have the same length.")

    feed_schedule = list(zip(feed_intervals_h, feed_amount_mg_L))

    times, c_med, c_cell, c_prod = _run_ode(
        c_med0=initial_conc_mg_L,
        k_uptake=k_uptake_per_h,
        k_efflux=k_efflux_per_h,
        k_metabolism=k_metabolism_per_h,
        k_degradation=0.05,
        volume_ratio=0.1,
        t_end_h=t_end_h,
        dt_h=dt_h,
        feed_schedule=feed_schedule,
    )

    n_feeds = len(feed_intervals_h)
    notes = (
        f"Fed-batch bioreactor PK for {drug_name}: {n_feeds} feed event(s) over "
        f"{t_end_h:.0f}h. k_uptake={k_uptake_per_h}/h, k_met={k_metabolism_per_h}/h."
    )

    return _compute_results(
        drug_name, initial_conc_mg_L, times, c_med, c_cell, c_prod, k_uptake_per_h, notes
    )
