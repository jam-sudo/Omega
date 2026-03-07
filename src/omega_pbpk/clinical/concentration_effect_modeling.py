"""PK/PD concentration-effect modeling: Emax, sigmoid Emax, linear, threshold, and biophase."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ConcentrationEffectResult:
    model: str
    ec50: float
    emax: float
    e0: float
    hill_n: float
    r_squared: float
    predicted_effects: tuple
    effect_at_ec50: float
    effect_at_2x_ec50: float
    therapeutic_threshold_conc: float
    notes: str


def emax_model(
    concentrations: list,
    emax: float,
    ec50: float,
    e0: float = 0.0,
) -> list:
    """E = E0 + Emax * C / (EC50 + C)."""
    if not concentrations:
        raise ValueError("concentrations must be non-empty")
    if emax <= 0:
        raise ValueError(f"emax must be > 0, got {emax}")
    if ec50 <= 0:
        raise ValueError(f"ec50 must be > 0, got {ec50}")

    effects = []
    for c in concentrations:
        if c < 0:
            raise ValueError(f"concentrations must be non-negative, got {c}")
        e = e0 + emax * c / (ec50 + c)
        effects.append(e)
    return effects


def sigmoid_emax_model(
    concentrations: list,
    emax: float,
    ec50: float,
    hill_n: float = 1.0,
    e0: float = 0.0,
) -> list:
    """E = E0 + Emax * C^n / (EC50^n + C^n)."""
    if not concentrations:
        raise ValueError("concentrations must be non-empty")
    if emax <= 0:
        raise ValueError(f"emax must be > 0, got {emax}")
    if ec50 <= 0:
        raise ValueError(f"ec50 must be > 0, got {ec50}")
    if hill_n <= 0:
        raise ValueError(f"hill_n must be > 0, got {hill_n}")

    effects = []
    for c in concentrations:
        if c < 0:
            raise ValueError(f"concentrations must be non-negative, got {c}")
        if c == 0:
            effects.append(e0)
        else:
            cn = c**hill_n
            ec50n = ec50**hill_n
            e = e0 + emax * cn / (ec50n + cn)
            effects.append(e)
    return effects


def linear_model(concentrations: list, slope: float, intercept: float = 0.0) -> list:
    """E = intercept + slope * C."""
    if not concentrations:
        raise ValueError("concentrations must be non-empty")
    return [intercept + slope * c for c in concentrations]


def fit_emax_model(concentrations: list, observed_effects: list) -> ConcentrationEffectResult:
    """Grid-search fit of sigmoid Emax model. Returns best-fit parameters + R²."""
    if not concentrations:
        raise ValueError("concentrations must be non-empty")
    if len(concentrations) != len(observed_effects):
        raise ValueError("concentrations and observed_effects must have the same length")
    if any(c <= 0 for c in concentrations):
        raise ValueError("all concentrations must be positive for fitting")

    n_obs = len(observed_effects)
    obs_mean = sum(observed_effects) / n_obs
    ss_tot = sum((y - obs_mean) ** 2 for y in observed_effects)

    max_obs = max(observed_effects)
    min_conc = min(concentrations)
    max_conc = max(concentrations)

    emax_candidates = [
        max_obs * 0.9,
        max_obs * 1.0,
        max_obs * 1.1,
        max_obs * 1.3,
        max_obs * 1.5,
    ]
    # 10 log-spaced EC50 candidates
    if min_conc <= 0 or max_conc <= 0:
        ec50_candidates = [max_conc * 0.5]
    else:
        log_min = math.log10(min_conc)
        log_max = math.log10(max_conc)
        ec50_candidates = [10 ** (log_min + i * (log_max - log_min) / 9) for i in range(10)]
    hill_candidates = [0.5, 1.0, 1.5, 2.0, 3.0]

    best_sse = float("inf")
    best_emax = emax_candidates[1]
    best_ec50 = ec50_candidates[len(ec50_candidates) // 2]
    best_hill = 1.0
    e0_fixed = 0.0

    for em in emax_candidates:
        if em <= 0:
            continue
        for ec50 in ec50_candidates:
            if ec50 <= 0:
                continue
            for hn in hill_candidates:
                preds = sigmoid_emax_model(concentrations, em, ec50, hn, e0_fixed)
                sse = sum((p - o) ** 2 for p, o in zip(preds, observed_effects))
                if sse < best_sse:
                    best_sse = sse
                    best_emax = em
                    best_ec50 = ec50
                    best_hill = hn

    r_squared = 1.0 - best_sse / ss_tot if ss_tot > 0 else 1.0
    r_squared = max(-1.0, min(1.0, r_squared))

    best_preds = sigmoid_emax_model(concentrations, best_emax, best_ec50, best_hill, e0_fixed)

    effect_at_ec50 = e0_fixed + best_emax / 2.0
    two_n = 2.0**best_hill
    effect_at_2x_ec50 = e0_fixed + best_emax * two_n / (1.0 + two_n)

    return ConcentrationEffectResult(
        model="sigmoid_emax",
        ec50=best_ec50,
        emax=best_emax,
        e0=e0_fixed,
        hill_n=best_hill,
        r_squared=r_squared,
        predicted_effects=tuple(best_preds),
        effect_at_ec50=effect_at_ec50,
        effect_at_2x_ec50=effect_at_2x_ec50,
        therapeutic_threshold_conc=best_ec50,
        notes=f"Grid-search fit: Emax={best_emax:.3f}, EC50={best_ec50:.3f}, hill={best_hill:.2f}",
    )


def effect_site_model(
    times_h: list,
    c_plasma: list,
    ke0_per_h: float,
    emax: float,
    ec50: float,
    hill_n: float = 1.0,
    e0: float = 0.0,
    dt_h: float = 0.1,
) -> tuple:
    """Simulate effect-site (biophase) concentration and effect over time.

    dCe/dt = ke0 * (Cp - Ce)

    Returns (c_effect_site, effect_values) as lists.
    """
    if not times_h:
        raise ValueError("times_h must be non-empty")
    if ke0_per_h <= 0:
        raise ValueError(f"ke0_per_h must be > 0, got {ke0_per_h}")
    if emax <= 0:
        raise ValueError(f"emax must be > 0, got {emax}")
    if ec50 <= 0:
        raise ValueError(f"ec50 must be > 0, got {ec50}")
    if hill_n <= 0:
        raise ValueError(f"hill_n must be > 0, got {hill_n}")

    # Interpolate plasma concentration at requested times using linear interp
    def interp_plasma(t: float) -> float:
        if len(times_h) == 1:
            return c_plasma[0]
        if t <= times_h[0]:
            return c_plasma[0]
        if t >= times_h[-1]:
            return c_plasma[-1]
        for i in range(len(times_h) - 1):
            if times_h[i] <= t <= times_h[i + 1]:
                frac = (t - times_h[i]) / (times_h[i + 1] - times_h[i])
                return c_plasma[i] + frac * (c_plasma[i + 1] - c_plasma[i])
        return c_plasma[-1]

    t_end = times_h[-1]
    t_start = times_h[0]

    # Forward Euler simulation on a fine grid
    fine_times = []
    t = t_start
    while t <= t_end + dt_h * 0.5:
        fine_times.append(t)
        t += dt_h

    ce = 0.0
    ce_fine: list[float] = []
    t_fine_list: list[float] = []

    for ft in fine_times:
        cp = interp_plasma(ft)
        ce_fine.append(ce)
        t_fine_list.append(ft)
        dce = ke0_per_h * (cp - ce) * dt_h
        ce = max(0.0, ce + dce)

    # Interpolate effect-site concentration at requested output times
    def interp_ce(t: float) -> float:
        if len(t_fine_list) == 0:
            return 0.0
        if t <= t_fine_list[0]:
            return ce_fine[0]
        if t >= t_fine_list[-1]:
            return ce_fine[-1]
        for i in range(len(t_fine_list) - 1):
            if t_fine_list[i] <= t <= t_fine_list[i + 1]:
                frac = (t - t_fine_list[i]) / (t_fine_list[i + 1] - t_fine_list[i])
                return ce_fine[i] + frac * (ce_fine[i + 1] - ce_fine[i])
        return ce_fine[-1]

    c_effect_site = [interp_ce(t) for t in times_h]

    # Compute effect using sigmoid Emax at each effect-site concentration
    effect_values = []
    for ce_val in c_effect_site:
        if ce_val <= 0:
            effect_values.append(e0)
        else:
            cen = ce_val**hill_n
            ec50n = ec50**hill_n
            eff = e0 + emax * cen / (ec50n + cen)
            effect_values.append(eff)

    return (c_effect_site, effect_values)
