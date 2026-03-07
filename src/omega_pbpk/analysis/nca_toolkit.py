"""Extended non-compartmental analysis toolkit — Phase 484."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "NCAResult",
    "compute_nca_full",
    "compute_aumc",
    "compute_terminal_slope",
    "compute_accumulation_ratio",
    "superposition_steady_state",
]

_VALID_ROUTES = {"iv", "oral"}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class NCAResult:
    """Result from full NCA analysis."""

    route: str
    dose_mg: float
    cmax_mg_L: float
    tmax_h: float
    auc_last_mg_h_L: float
    auc_inf_mg_h_L: float
    aumc_mg_h2_L: float
    mrt_h: float
    t_half_h: float
    lambda_z_per_h: float
    cl_obs_L_per_h: float
    vd_obs_L: float
    pct_auc_extrapolated: float
    r_squared_terminal: float
    n_terminal_points: int
    notes: str = field(default="")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(xs: list[float], ys: list[float]) -> float:
    """Linear trapezoidal integration of y over x."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("xs and ys must have the same length >= 2")
    total = 0.0
    for i in range(len(xs) - 1):
        total += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    return total


def _log_trapz(xs: list[float], ys: list[float]) -> float:
    """Log-linear trapezoidal integration (for terminal phase AUC)."""
    total = 0.0
    for i in range(len(xs) - 1):
        dt = xs[i + 1] - xs[i]
        c1, c2 = ys[i], ys[i + 1]
        if c1 <= 0 or c2 <= 0 or c1 == c2:
            # Fall back to linear trap for flat/zero segments
            total += 0.5 * (c1 + c2) * dt
        else:
            total += (c1 - c2) * dt / math.log(c1 / c2)
    return total


def _ols_slope_intercept(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Ordinary least squares: y = slope*x + intercept.

    Returns (slope, intercept, r_squared).
    """
    n = len(xs)
    if n < 2:
        raise ValueError("Need at least 2 points for OLS")
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=True))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-15:
        raise ValueError("OLS denominator is zero (collinear data)")

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R-squared
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    if ss_tot == 0.0:
        r_sq = 1.0
    else:
        y_pred = [slope * x + intercept for x in xs]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(ys, y_pred, strict=True))
        r_sq = 1.0 - ss_res / ss_tot

    return slope, intercept, r_sq


def _validate_profile(times_h: list[float], concs_mg_L: list[float]) -> None:
    if len(times_h) < 3:
        raise ValueError("At least 3 time points required")
    if len(times_h) != len(concs_mg_L):
        raise ValueError("times_h and concs_mg_L must have the same length")
    if any(t < 0 for t in times_h):
        raise ValueError("All time values must be >= 0")
    if any(c < 0 for c in concs_mg_L):
        raise ValueError("All concentration values must be >= 0")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_terminal_slope(
    times_h: list[float],
    concs_mg_L: list[float],
    n_terminal: int = 3,
) -> tuple[float, float]:
    """Estimate terminal elimination rate constant (lambda_z) via log-linear OLS.

    Parameters
    ----------
    times_h:
        Time points in hours.
    concs_mg_L:
        Corresponding plasma concentrations in mg/L.
    n_terminal:
        Number of terminal data points to use (>= 2).

    Returns
    -------
    (lambda_z_per_h, r_squared)
        lambda_z is positive; r_squared measures goodness of fit.
    """
    if n_terminal < 2:
        raise ValueError("n_terminal must be >= 2")
    if len(times_h) < n_terminal:
        raise ValueError(f"Not enough data points for n_terminal={n_terminal}")
    if len(times_h) != len(concs_mg_L):
        raise ValueError("times_h and concs_mg_L must have the same length")

    # Use last n_terminal points with positive concentrations
    terminal_t = times_h[-n_terminal:]
    terminal_c = concs_mg_L[-n_terminal:]

    # Filter out zeros/negatives for log transform
    pairs = [(t, c) for t, c in zip(terminal_t, terminal_c, strict=True) if c > 0]
    if len(pairs) < 2:
        raise ValueError("Fewer than 2 positive concentration points in terminal phase")

    t_vals = [p[0] for p in pairs]
    ln_c_vals = [math.log(p[1]) for p in pairs]

    slope, _intercept, r_sq = _ols_slope_intercept(t_vals, ln_c_vals)

    # slope of ln(C) vs t is -lambda_z
    lambda_z = -slope
    if lambda_z <= 0:
        raise ValueError(f"Terminal slope is non-negative ({slope:.4f}); profile is not declining")

    return lambda_z, r_sq


def compute_aumc(times_h: list[float], concs_mg_L: list[float]) -> float:
    """Compute AUMC (area under first moment curve) using linear trapezoidal rule.

    AUMC = integral(t * C(t)) dt

    Parameters
    ----------
    times_h:
        Time points in hours.
    concs_mg_L:
        Concentrations in mg/L.

    Returns
    -------
    float
        AUMC in mg·h²/L.
    """
    if len(times_h) != len(concs_mg_L):
        raise ValueError("times_h and concs_mg_L must have the same length")
    if len(times_h) < 2:
        raise ValueError("At least 2 time points required")

    tc = [t * c for t, c in zip(times_h, concs_mg_L, strict=True)]
    return _trapz(times_h, tc)


def compute_accumulation_ratio(auc_ss: float, auc_single: float) -> float:
    """Compute accumulation ratio = AUC_ss / AUC_single_dose.

    Parameters
    ----------
    auc_ss:
        AUC over one dosing interval at steady state.
    auc_single:
        AUC after single dose (AUC_inf or AUC over same interval).

    Returns
    -------
    float
        Accumulation ratio (>= 1).
    """
    if auc_single <= 0:
        raise ValueError("auc_single must be > 0")
    if auc_ss < 0:
        raise ValueError("auc_ss must be >= 0")
    return auc_ss / auc_single


def compute_nca_full(
    times_h: list[float],
    concs_mg_L: list[float],
    dose_mg: float,
    route: str = "iv",
    weight_kg: float = 70.0,
) -> NCAResult:
    """Compute a full suite of NCA parameters from a plasma concentration–time profile.

    Parameters
    ----------
    times_h:
        Observed time points in hours (must be sorted ascending, length >= 3).
    concs_mg_L:
        Observed plasma concentrations in mg/L (must match times_h in length).
    dose_mg:
        Administered dose in milligrams (> 0).
    route:
        Either ``'iv'`` or ``'oral'``.
    weight_kg:
        Body weight in kg (used for weight-normalised parameters; default 70).

    Returns
    -------
    NCAResult
    """
    # --- Validation ---
    _validate_profile(times_h, concs_mg_L)
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")

    notes_parts: list[str] = []

    # --- Basic metrics ---
    cmax_mg_L = max(concs_mg_L)
    tmax_h = times_h[concs_mg_L.index(cmax_mg_L)]

    # AUC_last via linear trapezoidal
    auc_last = _trapz(times_h, concs_mg_L)

    # --- Terminal slope: try n=3, fall back to n=2 ---
    n_terminal = 3
    try:
        lambda_z, r_sq = compute_terminal_slope(times_h, concs_mg_L, n_terminal=n_terminal)
    except ValueError:
        n_terminal = 2
        try:
            lambda_z, r_sq = compute_terminal_slope(times_h, concs_mg_L, n_terminal=n_terminal)
            notes_parts.append("Only 2 terminal points used for lambda_z estimation.")
        except ValueError as exc:
            raise ValueError(f"Could not estimate terminal slope: {exc}") from exc

    t_half_h = math.log(2.0) / lambda_z

    # AUC extrapolation: C_last / lambda_z
    t_last = times_h[-1]
    # Use positive last concentration
    pos_concs = [(c, t) for c, t in zip(concs_mg_L, times_h, strict=True) if c > 0]
    if pos_concs:
        c_last_pos, _t_last_pos = pos_concs[-1]
    else:
        c_last_pos = 1e-12

    auc_extrap = c_last_pos / lambda_z
    auc_inf = auc_last + auc_extrap
    pct_extrap = 100.0 * auc_extrap / auc_inf if auc_inf > 0 else 0.0

    if pct_extrap > 20.0:
        notes_parts.append(
            f"AUC extrapolation is {pct_extrap:.1f}% — profile may not reach terminal phase."
        )

    # --- AUMC_last then extrapolate ---
    aumc_last = compute_aumc(times_h, concs_mg_L)
    # AUMC extrapolation from last observed point: t_last*C_last/lambda_z + C_last/lambda_z^2
    aumc_extrap = t_last * c_last_pos / lambda_z + c_last_pos / (lambda_z**2)
    aumc_inf = aumc_last + aumc_extrap

    # MRT = AUMC_inf / AUC_inf  (for IV: MRT = Vss/CL = 1/ke for mono-exp)
    mrt_h = aumc_inf / auc_inf if auc_inf > 0 else 0.0

    # CL and Vd
    # For IV: F=1; for oral: F assumed 1 (non-compartmental, apparent CL)
    cl_obs = dose_mg / auc_inf if auc_inf > 0 else float("inf")
    vd_obs = cl_obs / lambda_z

    return NCAResult(
        route=route,
        dose_mg=dose_mg,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_last_mg_h_L=auc_last,
        auc_inf_mg_h_L=auc_inf,
        aumc_mg_h2_L=aumc_inf,
        mrt_h=mrt_h,
        t_half_h=t_half_h,
        lambda_z_per_h=lambda_z,
        cl_obs_L_per_h=cl_obs,
        vd_obs_L=vd_obs,
        pct_auc_extrapolated=pct_extrap,
        r_squared_terminal=r_sq,
        n_terminal_points=n_terminal,
        notes=" ".join(notes_parts),
    )


def superposition_steady_state(
    times_h: list[float],
    concs_mg_L: list[float],
    dose_interval_h: float,
) -> dict:
    """Estimate steady-state PK by linear superposition of single-dose profiles.

    Parameters
    ----------
    times_h:
        Single-dose time points in hours.
    concs_mg_L:
        Single-dose plasma concentrations in mg/L.
    dose_interval_h:
        Dosing interval (tau) in hours (> 0).

    Returns
    -------
    dict with keys:
        - ``'cmax_ss'``: predicted Cmax at steady state (mg/L)
        - ``'cmin_ss'``: predicted trough concentration (mg/L)
        - ``'auc_ss'``: AUC over one dosing interval at SS (mg·h/L)
        - ``'accumulation_ratio'``: AUC_ss / AUC_single
    """
    if dose_interval_h <= 0:
        raise ValueError("dose_interval_h must be > 0")
    if len(times_h) < 2:
        raise ValueError("At least 2 time points required")
    if len(times_h) != len(concs_mg_L):
        raise ValueError("times_h and concs_mg_L must have the same length")

    # Estimate lambda_z for extrapolation
    try:
        lambda_z, _r_sq = compute_terminal_slope(times_h, concs_mg_L, n_terminal=3)
    except ValueError:
        lambda_z, _r_sq = compute_terminal_slope(times_h, concs_mg_L, n_terminal=2)

    # Superposition: accumulate infinite doses at -tau, -2tau, -3tau ...
    # C_ss(t) = sum_{k=0}^{inf} C_single(t + k*tau)
    # We truncate when C_single contribution < 0.1% of current total

    # Build a fine interpolation grid over [0, tau]
    n_grid = max(200, len(times_h) * 5)
    dt = dose_interval_h / n_grid
    grid_t = [i * dt for i in range(n_grid + 1)]

    # Interpolate single-dose profile onto any time point
    t_max_obs = times_h[-1]

    def interp_single(t: float) -> float:
        """Linear interpolation + exponential extrapolation."""
        if t <= 0:
            return 0.0
        if t > t_max_obs:
            # Extrapolate using terminal slope
            c_last = concs_mg_L[-1]
            return c_last * math.exp(-lambda_z * (t - t_max_obs))
        # Binary search for bracket
        lo, hi = 0, len(times_h) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if times_h[mid] <= t:
                lo = mid
            else:
                hi = mid
        t0, t1 = times_h[lo], times_h[hi]
        c0, c1 = concs_mg_L[lo], concs_mg_L[hi]
        if t1 == t0:
            return c0
        frac = (t - t0) / (t1 - t0)
        return c0 + frac * (c1 - c0)

    # Compute C_ss on grid by superposition
    max_doses = 200
    c_ss = [0.0] * (n_grid + 1)
    for k in range(max_doses):
        offset = k * dose_interval_h
        max_contrib = 0.0
        for i, t in enumerate(grid_t):
            contrib = interp_single(t + offset)
            c_ss[i] += contrib
            if contrib > max_contrib:
                max_contrib = contrib
        # Convergence: if contribution is tiny relative to current SS
        if k > 0 and max_contrib < 1e-6 * max(c_ss):
            break

    cmax_ss = max(c_ss)
    cmin_ss = min(c_ss)
    auc_ss = _trapz(grid_t, c_ss)

    # AUC_single
    auc_single = _trapz(times_h, concs_mg_L)
    # Extrapolate AUC_single to infinity for accumulation ratio
    try:
        c_last_pos = next(c for c in reversed(concs_mg_L) if c > 0)
    except StopIteration:
        c_last_pos = 1e-12
    auc_single_inf = auc_single + c_last_pos / lambda_z

    acc_ratio = compute_accumulation_ratio(auc_ss, auc_single_inf)

    return {
        "cmax_ss": cmax_ss,
        "cmin_ss": cmin_ss,
        "auc_ss": auc_ss,
        "accumulation_ratio": acc_ratio,
    }
