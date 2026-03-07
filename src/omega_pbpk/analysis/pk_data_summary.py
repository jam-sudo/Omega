"""Phase 560 — PK data summarization and descriptive statistics.

Pure Python, no numpy/scipy.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    """Population std dev (ddof=1 for sample)."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _median(values: list[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def _trapz(times: list[float], concs: list[float]) -> float:
    """Trapezoidal AUC from paired time/concentration lists."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt
    return auc


def _linreg(x: list[float], y: list[float]):
    """Simple linear regression y = a + b*x; returns (a, b)."""
    n = len(x)
    mx = _mean(x)
    my = _mean(y)
    ssxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    ssxx = sum((x[i] - mx) ** 2 for i in range(n))
    if ssxx == 0.0:
        return my, 0.0
    b = ssxy / ssxx
    a = my - b * mx
    return a, b


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def nca_summary(
    times_h: list[float],
    concentrations: list[float],
    dose_mg: float,
    route: str = "iv",
) -> dict:
    """Standard non-compartmental analysis (NCA).

    Returns: cmax, tmax_h, auc_0_last, auc_0_inf, ke_terminal,
             t_half_h, cl_L_per_h, vz_L, pct_extrap.
    """
    if len(times_h) != len(concentrations):
        raise ValueError("times_h and concentrations must have the same length.")
    if len(times_h) < 3:
        raise ValueError("Need at least 3 time points for NCA.")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0.")

    # Cmax and tmax
    cmax = max(concentrations)
    tmax_h = times_h[concentrations.index(cmax)]

    # AUC0-last (trapezoidal)
    auc_0_last = _trapz(times_h, concentrations)

    # ke_terminal: log-linear regression on last 3 points or last half
    n = len(times_h)
    n_terminal = max(3, n // 2)
    t_term = times_h[n - n_terminal :]
    c_term = concentrations[n - n_terminal :]

    # Filter out zero/negative concentrations for log transform
    pairs = [(t, c) for t, c in zip(t_term, c_term) if c > 0]
    if len(pairs) < 2:
        raise ValueError("Insufficient positive concentration values for terminal slope.")

    t_vals = [p[0] for p in pairs]
    log_c_vals = [math.log(p[1]) for p in pairs]
    _, slope = _linreg(t_vals, log_c_vals)
    ke_terminal = -slope  # ke > 0 for declining curve

    if ke_terminal <= 0:
        ke_terminal = 1e-6  # guard against non-declining terminal phase

    t_half_h = math.log(2.0) / ke_terminal

    # AUC0-inf = AUC0-last + Clast/ke_terminal
    c_last = concentrations[-1]
    auc_0_inf = auc_0_last + c_last / ke_terminal

    pct_extrap = (c_last / ke_terminal / auc_0_inf) * 100.0 if auc_0_inf > 0 else 0.0

    # CL and Vz
    cl = dose_mg / auc_0_inf  # L/h (assuming auc in mg/L*h)
    vz = cl / ke_terminal

    return {
        "cmax": cmax,
        "tmax_h": tmax_h,
        "auc_0_last": auc_0_last,
        "auc_0_inf": auc_0_inf,
        "ke_terminal": ke_terminal,
        "t_half_h": t_half_h,
        "cl_L_per_h": cl,
        "vz_L": vz,
        "pct_extrap": pct_extrap,
    }


def coefficient_of_variation(values: list[float]) -> float:
    """CV% = std / mean * 100.

    Raises ValueError if < 2 values or mean == 0.
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 values to compute CV.")
    m = _mean(values)
    if m == 0.0:
        raise ValueError("Mean is zero; CV is undefined.")
    return _std(values) / abs(m) * 100.0


def geometric_mean(values: list[float]) -> float:
    """Geometric mean = exp(mean of log(v)).

    Raises ValueError if any value <= 0.
    """
    if not values:
        raise ValueError("values list is empty.")
    for v in values:
        if v <= 0:
            raise ValueError(f"All values must be > 0 for geometric mean; got {v}.")
    log_mean = _mean([math.log(v) for v in values])
    return math.exp(log_mean)


def summarize_population_pk(
    individual_results: list[dict],
    pk_params: list[str] | None = None,
) -> dict:
    """Descriptive statistics across individuals for key PK parameters.

    Default pk_params: ['cmax', 'auc_0_inf', 't_half_h', 'cl_L_per_h'].

    Returns nested dict: {param: {mean, median, std, cv_pct, geo_mean, min, max}}.
    Skips params not present in all individuals.
    """
    if pk_params is None:
        pk_params = ["cmax", "auc_0_inf", "t_half_h", "cl_L_per_h"]

    if not individual_results:
        return {}

    summary = {}
    for param in pk_params:
        # Only include param if present in ALL individuals
        if not all(param in ind for ind in individual_results):
            continue

        vals = [ind[param] for ind in individual_results]

        try:
            cv = coefficient_of_variation(vals) if len(vals) >= 2 else 0.0
        except ValueError:
            cv = float("nan")

        try:
            gm = geometric_mean(vals)
        except ValueError:
            gm = float("nan")

        std = _std(vals) if len(vals) >= 2 else 0.0

        summary[param] = {
            "mean": _mean(vals),
            "median": _median(vals),
            "std": std,
            "cv_pct": cv,
            "geo_mean": gm,
            "min": min(vals),
            "max": max(vals),
        }

    return summary


def accumulation_ratio_ss(t_half_h: float, tau_h: float) -> float:
    """Accumulation ratio at steady state.

    Rac = 1 / (1 - exp(-ke * tau))  where ke = ln2 / t_half_h.

    Both t_half_h and tau_h must be > 0.
    """
    if t_half_h <= 0:
        raise ValueError("t_half_h must be > 0.")
    if tau_h <= 0:
        raise ValueError("tau_h must be > 0.")

    ke = math.log(2.0) / t_half_h
    return 1.0 / (1.0 - math.exp(-ke * tau_h))


def time_above_threshold(
    times_h: list[float],
    concentrations: list[float],
    threshold: float,
) -> dict:
    """Calculate the time (hours) and fraction where concentration > threshold.

    Uses linear interpolation to estimate crossing times.

    Returns: time_above_h, fraction_above, n_crossings.
    """
    if len(times_h) != len(concentrations):
        raise ValueError("times_h and concentrations must have the same length.")
    if len(times_h) < 2:
        raise ValueError("Need at least 2 time points.")

    total_duration = times_h[-1] - times_h[0]
    if total_duration <= 0:
        raise ValueError("Total duration must be > 0.")

    time_above = 0.0
    n_crossings = 0

    for i in range(1, len(times_h)):
        t0, t1 = times_h[i - 1], times_h[i]
        c0, c1 = concentrations[i - 1], concentrations[i]
        dt = t1 - t0

        both_above = c0 > threshold and c1 > threshold
        both_below = c0 <= threshold and c1 <= threshold

        if both_above:
            time_above += dt
        elif both_below:
            pass  # no time above
        else:
            # Crossing: linear interpolation for crossing time
            # c(t) = c0 + (c1 - c0) * (t - t0) / dt = threshold
            t_cross = t0 + (threshold - c0) / (c1 - c0) * dt
            n_crossings += 1

            if c0 > threshold:
                # starts above, goes below
                time_above += t_cross - t0
            else:
                # starts below, goes above
                time_above += t1 - t_cross

    fraction_above = time_above / total_duration

    return {
        "time_above_h": time_above,
        "fraction_above": fraction_above,
        "n_crossings": n_crossings,
    }
