"""Drug exposure metrics calculator for PK time-course data (Phase 317)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ExposureMetrics",
    "NCAResult",
    "compute_exposure_metrics",
    "compute_nca_parameters",
    "compare_exposures",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExposureMetrics:
    """Standard NCA exposure metrics from a PK time-course."""

    times_h: tuple[float, ...]
    concentrations: tuple[float, ...]
    dose_mg: float
    route: str
    auc0t: float  # AUC0-t (trapezoidal), mg*h/L
    auc0inf: float  # AUC0-inf, mg*h/L
    aumc: float  # Area under the first moment curve, mg*h^2/L
    mrt_h: float  # Mean residence time, h
    cmax: float  # Maximum observed concentration, mg/L
    tmax_h: float  # Time of maximum concentration, h
    t_half_h: float  # Terminal half-life, h
    lambda_z_per_h: float  # Terminal elimination rate constant, 1/h
    cl_f_L_per_h: float  # Apparent clearance (CL/F), L/h
    vd_f_L: float  # Apparent volume of distribution (Vd/F), L
    auc_pct_extrap: float  # % of AUC0-inf extrapolated beyond last sample
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NCAResult:
    """Full NCA result including multiple-dose metrics."""

    # Core exposure metrics
    times_h: tuple[float, ...]
    concentrations: tuple[float, ...]
    dose_mg: float
    route: str
    auc0t: float
    auc0inf: float
    aumc: float
    mrt_h: float
    cmax: float
    tmax_h: float
    t_half_h: float
    lambda_z_per_h: float
    cl_f_L_per_h: float
    vd_f_L: float
    auc_pct_extrap: float
    # Multiple-dose metrics
    tau_h: float | None
    cavg: float | None  # Average concentration over tau
    swing_pct: float | None  # (Cmax-Cmin)/Cavg * 100
    accumulation_ratio: float | None  # R = 1/(1-exp(-lambda_z*tau))
    time_above_mec_h: float | None
    notes: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    times_h: list[float],
    concentrations: list[float],
    dose_mg: float,
    min_points: int = 3,
) -> None:
    if len(times_h) < min_points:
        raise ValueError(f"At least {min_points} data points are required.")
    if len(times_h) != len(concentrations):
        raise ValueError("times_h and concentrations must have the same length.")
    for i in range(1, len(times_h)):
        if times_h[i] <= times_h[i - 1]:
            raise ValueError("times_h must be strictly monotonically increasing.")
    if any(c < 0 for c in concentrations):
        raise ValueError("concentrations must be >= 0.")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive.")


# ---------------------------------------------------------------------------
# Core numerical helpers
# ---------------------------------------------------------------------------


def _auc_linear_trapezoidal(times: list[float], conc: list[float]) -> float:
    """Linear trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += (conc[i] + conc[i - 1]) * (times[i] - times[i - 1]) / 2.0
    return auc


def _auc_lin_log_trapezoidal(times: list[float], conc: list[float]) -> float:
    """Lin-log trapezoidal AUC (log interpolation in terminal phase)."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        c0, c1 = conc[i - 1], conc[i]
        if c0 <= 0 or c1 <= 0:
            # Fallback to linear for zero/negative concentrations
            auc += (c0 + c1) * dt / 2.0
        elif c1 < c0:
            # Log trapezoidal (elimination phase)
            auc += dt * (c0 - c1) / math.log(c0 / c1)
        else:
            # Linear trapezoidal (absorption phase or flat)
            auc += (c0 + c1) * dt / 2.0
    return auc


def _aumc_trapezoidal(times: list[float], conc: list[float]) -> float:
    """Trapezoidal AUMC = integral of t*C dt."""
    tc = [t * c for t, c in zip(times, conc, strict=True)]
    return _auc_linear_trapezoidal(times, tc)


def _estimate_lambda_z(
    times: list[float],
    conc: list[float],
    lloq: float | None = None,
) -> tuple[float, float, int]:
    """Estimate terminal elimination rate constant via log-linear regression.

    Returns (lambda_z, r_squared, n_points_used).
    Uses the last N >= 3 points (excluding LLOQ-censored values).
    Selects the best window by maximising adjusted R^2.
    """
    # Filter out LLOQ and zero concentrations from the end
    valid_pairs = [
        (t, c) for t, c in zip(times, conc, strict=True) if c > 0 and (lloq is None or c >= lloq)
    ]
    if len(valid_pairs) < 3:
        raise ValueError(
            "Fewer than 3 valid (above LLOQ, positive) points for lambda_z estimation."
        )

    t_arr = np.array([p[0] for p in valid_pairs])
    c_arr = np.array([p[1] for p in valid_pairs])
    log_c = np.log(c_arr)

    best_lz = None
    best_adj_r2 = -np.inf
    best_n = 0

    n_total = len(t_arr)
    for n in range(3, n_total + 1):
        t_seg = t_arr[-n:]
        lc_seg = log_c[-n:]
        # Linear regression: log_c = a - lambda_z * t
        coeffs = np.polyfit(t_seg, lc_seg, 1)
        lz = -coeffs[0]
        if lz <= 0:
            continue
        predicted = np.polyval(coeffs, t_seg)
        ss_res = np.sum((lc_seg - predicted) ** 2)
        ss_tot = np.sum((lc_seg - np.mean(lc_seg)) ** 2)
        if ss_tot < 1e-15:
            r2 = 1.0
        else:
            r2 = 1.0 - ss_res / ss_tot
        # Adjusted R^2
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - 2, 1)
        if adj_r2 > best_adj_r2:
            best_adj_r2 = adj_r2
            best_lz = lz
            best_n = n

    if best_lz is None or best_lz <= 0:
        raise ValueError("Could not estimate a positive lambda_z from the data.")

    return best_lz, best_adj_r2, best_n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_exposure_metrics(
    times_h: list[float],
    concentrations: list[float],
    dose_mg: float,
    route: str = "oral",
    lloq: float | None = None,
) -> ExposureMetrics:
    """Compute comprehensive drug exposure metrics from PK time-course data.

    Parameters
    ----------
    times_h:
        Sampling times in hours (monotonically increasing, starting from 0 or pre-dose).
    concentrations:
        Measured drug concentrations in mg/L (or same units as dose/volume).
    dose_mg:
        Administered dose in mg.
    route:
        Dosing route: "oral" or "iv".
    lloq:
        Lower limit of quantification; concentrations below are excluded from
        lambda_z regression.

    Returns
    -------
    ExposureMetrics
        Frozen dataclass with all NCA metrics.
    """
    _validate_inputs(times_h, concentrations, dose_mg)

    t = list(times_h)
    c = list(concentrations)
    notes: list[str] = []

    # LLOQ handling: replace BLQ with 0 for AUC but exclude from lambda_z
    c_for_auc = [ci if (lloq is None or ci >= lloq) else 0.0 for ci in c]

    # Cmax and Tmax
    cmax = max(c_for_auc)
    tmax_idx = c_for_auc.index(cmax)
    tmax_h = t[tmax_idx]

    # AUC0-t using lin-log trapezoidal
    auc0t = _auc_lin_log_trapezoidal(t, c_for_auc)

    # AUMC using linear trapezoidal on t*C
    aumc = _aumc_trapezoidal(t, c_for_auc)

    # Terminal elimination rate constant
    try:
        lambda_z, r2, n_lz = _estimate_lambda_z(t, c, lloq=lloq)
    except ValueError as exc:
        notes.append(f"lambda_z estimation failed: {exc}")
        lambda_z = float("nan")
        r2 = float("nan")
        n_lz = 0

    t_half_h = 0.693147 / lambda_z if not math.isnan(lambda_z) else float("nan")

    # Last observed concentration
    # Find last non-zero (or above LLOQ) concentration
    c_last = 0.0
    for ci in reversed(c_for_auc):
        if ci > 0:
            c_last = ci
            break

    # AUC0-inf = AUC0-t + C_last/lambda_z
    if not math.isnan(lambda_z) and lambda_z > 0:
        auc_extrap = c_last / lambda_z
        auc0inf = auc0t + auc_extrap
        auc_pct_extrap = (auc_extrap / auc0inf * 100.0) if auc0inf > 0 else 0.0
        if auc_pct_extrap > 20.0:
            notes.append(
                f"AUC extrapolated fraction {auc_pct_extrap:.1f}% > 20%; "
                "AUC0-inf estimate may be unreliable."
            )
    else:
        auc0inf = float("nan")
        auc_pct_extrap = float("nan")

    # MRT = AUMC / AUC0-inf (or AUC0-t if AUC0-inf unavailable)
    auc_for_mrt = auc0inf if not math.isnan(auc0inf) else auc0t
    mrt_h = aumc / auc_for_mrt if auc_for_mrt > 0 else float("nan")

    # Apparent clearance: CL/F = dose / AUC0-inf
    if not math.isnan(auc0inf) and auc0inf > 0:
        cl_f = dose_mg / auc0inf
    else:
        cl_f = float("nan")

    # Apparent volume: Vd/F = CL/F / lambda_z
    if not math.isnan(cl_f) and not math.isnan(lambda_z) and lambda_z > 0:
        vd_f = cl_f / lambda_z
    else:
        vd_f = float("nan")

    if n_lz > 0:
        notes.append(f"lambda_z estimated from last {n_lz} points (adj-R²={r2:.4f}).")

    return ExposureMetrics(
        times_h=tuple(t),
        concentrations=tuple(c),
        dose_mg=dose_mg,
        route=route,
        auc0t=auc0t,
        auc0inf=auc0inf,
        aumc=aumc,
        mrt_h=mrt_h,
        cmax=cmax,
        tmax_h=tmax_h,
        t_half_h=t_half_h,
        lambda_z_per_h=lambda_z,
        cl_f_L_per_h=cl_f,
        vd_f_L=vd_f,
        auc_pct_extrap=auc_pct_extrap,
        notes=tuple(notes),
    )


def compute_nca_parameters(
    times_h: list[float],
    concentrations: list[float],
    dose_mg: float,
    route: str = "oral",
    lloq: float | None = None,
    tau_h: float | None = None,
    mec_mg_L: float | None = None,
) -> NCAResult:
    """Full NCA including multiple-dose accumulation metrics.

    Parameters
    ----------
    tau_h:
        Dosing interval (h). Required for Cavg, swing, and accumulation ratio.
    mec_mg_L:
        Minimum effective concentration (mg/L). Used to compute time above MEC.
    """
    base = compute_exposure_metrics(times_h, concentrations, dose_mg, route, lloq)
    notes = list(base.notes)

    # Multiple-dose metrics
    cavg: float | None = None
    swing_pct: float | None = None
    accumulation_ratio: float | None = None
    time_above_mec_h: float | None = None

    if tau_h is not None:
        if tau_h <= 0:
            raise ValueError("tau_h must be positive.")
        # Cavg = AUC0-tau / tau; approximate as AUC0-t / last_time if tau > last time
        t_last = times_h[-1]
        if tau_h <= t_last:
            # Integrate only up to tau
            idx = [i for i, ti in enumerate(times_h) if ti <= tau_h]
            t_tau = [times_h[i] for i in idx]
            c_tau = [concentrations[i] for i in idx]
            # Add linearly interpolated point at tau if needed
            if t_tau[-1] < tau_h and len(t_tau) < len(times_h):
                next_i = idx[-1] + 1
                t0, t1 = times_h[next_i - 1], times_h[next_i]
                c0, c1 = concentrations[next_i - 1], concentrations[next_i]
                frac = (tau_h - t0) / (t1 - t0)
                c_interp = c0 + frac * (c1 - c0)
                t_tau.append(tau_h)
                c_tau.append(c_interp)
            auc_tau = _auc_lin_log_trapezoidal(t_tau, c_tau)
        else:
            # tau > observation window; extrapolate using AUC0-inf
            if not math.isnan(base.auc0inf):
                auc_tau = min(base.auc0inf, base.auc0t + (base.auc0inf - base.auc0t))
                notes.append("tau_h exceeds observation window; Cavg approximated from AUC0-inf.")
            else:
                auc_tau = base.auc0t
        cavg = auc_tau / tau_h if tau_h > 0 else None

        # Swing = (Cmax - Cmin) / Cavg * 100
        # Approximate Cmin as last concentration at time tau (or last observed)
        if tau_h <= t_last:
            c_vals = list(concentrations)
            t_vals = list(times_h)
            # Interpolate at tau
            cmin = float(np.interp(tau_h, t_vals, c_vals))
        else:
            cmin = concentrations[-1]

        if cavg and cavg > 0:
            swing_pct = (base.cmax - cmin) / cavg * 100.0
        else:
            swing_pct = None

        # Accumulation ratio R = 1 / (1 - exp(-lambda_z * tau))
        lz = base.lambda_z_per_h
        if not math.isnan(lz) and lz > 0:
            exp_term = math.exp(-lz * tau_h)
            if abs(1.0 - exp_term) > 1e-12:
                accumulation_ratio = 1.0 / (1.0 - exp_term)
            else:
                accumulation_ratio = float("inf")
        else:
            accumulation_ratio = None

    # Time above MEC
    if mec_mg_L is not None:
        if mec_mg_L < 0:
            raise ValueError("mec_mg_L must be non-negative.")
        time_above_mec_h = _time_above_threshold_value(times_h, concentrations, mec_mg_L)

    return NCAResult(
        times_h=base.times_h,
        concentrations=base.concentrations,
        dose_mg=base.dose_mg,
        route=base.route,
        auc0t=base.auc0t,
        auc0inf=base.auc0inf,
        aumc=base.aumc,
        mrt_h=base.mrt_h,
        cmax=base.cmax,
        tmax_h=base.tmax_h,
        t_half_h=base.t_half_h,
        lambda_z_per_h=base.lambda_z_per_h,
        cl_f_L_per_h=base.cl_f_L_per_h,
        vd_f_L=base.vd_f_L,
        auc_pct_extrap=base.auc_pct_extrap,
        tau_h=tau_h,
        cavg=cavg,
        swing_pct=swing_pct,
        accumulation_ratio=accumulation_ratio,
        time_above_mec_h=time_above_mec_h,
        notes=tuple(notes),
    )


def compare_exposures(
    pk_data_list: list[dict],
    dose_mg: float,
) -> list[ExposureMetrics]:
    """Process multiple PK profiles and return sorted by AUC0-inf descending.

    Each entry in pk_data_list should contain:
        - "times_h": list[float]
        - "concentrations": list[float]
        - "route": str (optional, default "oral")
        - "lloq": float (optional)
    """
    results: list[ExposureMetrics] = []
    for entry in pk_data_list:
        times_h = entry["times_h"]
        concentrations = entry["concentrations"]
        route = entry.get("route", "oral")
        lloq = entry.get("lloq", None)
        em = compute_exposure_metrics(times_h, concentrations, dose_mg, route, lloq)
        results.append(em)

    # Sort by AUC0-inf descending; NaN values go to the end
    def sort_key(em: ExposureMetrics) -> float:
        return em.auc0inf if not math.isnan(em.auc0inf) else -1.0

    return sorted(results, key=sort_key, reverse=True)


# ---------------------------------------------------------------------------
# Internal helper (shared with threshold module logic)
# ---------------------------------------------------------------------------


def _time_above_threshold_value(
    times_h: list[float],
    concentrations: list[float],
    threshold: float,
) -> float:
    """Compute total time above a given concentration threshold (h)."""
    total = 0.0
    t = times_h
    c = concentrations
    n = len(t)
    for i in range(1, n):
        c0, c1 = c[i - 1], c[i]
        t0, t1 = t[i - 1], t[i]
        dt = t1 - t0
        above0 = c0 > threshold
        above1 = c1 > threshold
        if above0 and above1:
            total += dt
        elif above0 and not above1:
            # Crosses downward: interpolate crossing time
            frac = (c0 - threshold) / (c0 - c1)
            total += frac * dt
        elif not above0 and above1:
            frac = (threshold - c0) / (c1 - c0)
            total += (1.0 - frac) * dt
        # else: both below
    return total
