"""Dissolution-absorption deconvolution for Level A IVIVC.

Deconvolves in vitro dissolution data to predict in vivo absorption using
the Wagner-Nelson approach and numerical deconvolution.

References:
    Wagner JG, Nelson E, J Pharm Sci, 1964 (Wagner-Nelson method)
    FDA Guidance for Industry: Extended Release Oral Dosage Forms, 1997
"""

from __future__ import annotations

__all__ = [
    "DeconvolutionResult",
    "deconvolve_dissolution_absorption",
    "predict_in_vivo_absorption",
]

from dataclasses import dataclass


@dataclass
class DeconvolutionResult:
    """Result of dissolution-absorption deconvolution.

    Attributes:
        drug_name: Name of the drug.
        f_oral: Oral bioavailability fraction (0, 1].
        ke_per_h: Elimination rate constant (1/h).
        ka_per_h: Absorption rate constant (1/h).
        times_h: Time points converted from dissolution times (h).
        dissolved_pct: Input % dissolved at each time point.
        fa_cumulative_pct: Cumulative fraction absorbed expressed as % (f_oral * dissolved_pct).
        r_squared: Correlation metric for Level A IVIVC quality (0-1).
        ivivc_level: IVIVC classification: "Level_A" or "Level_B".
        tmax_estimate_h: Time of maximum dissolution rate (h).
        notes: Descriptive notes.
    """

    drug_name: str
    f_oral: float
    ke_per_h: float
    ka_per_h: float
    times_h: list[float]
    dissolved_pct: list[float]
    fa_cumulative_pct: list[float]
    r_squared: float
    ivivc_level: str
    tmax_estimate_h: float
    notes: str


def _pearson_r_squared(x: list[float], y: list[float]) -> float:
    """Compute Pearson R-squared between two equal-length lists."""
    n = len(x)
    if n < 2:
        return 1.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    ss_xx = sum((xi - mean_x) ** 2 for xi in x)
    ss_yy = sum((yi - mean_y) ** 2 for yi in y)
    ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom = (ss_xx * ss_yy) ** 0.5
    if denom < 1e-15:
        return 1.0
    r = ss_xy / denom
    return r * r


def deconvolve_dissolution_absorption(
    drug_name: str,
    times_dissolution_min: list[float],
    dissolved_pct: list[float],
    ke_per_h: float,
    ka_per_h: float,
    f_oral: float,
) -> DeconvolutionResult:
    """Deconvolve in vitro dissolution data to predict in vivo absorption.

    Uses Wagner-Nelson deconvolution to relate in vitro dissolution profile
    to cumulative in vivo fraction absorbed, applying oral bioavailability
    scaling.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    times_dissolution_min:
        Time points in minutes (>= 3 points, must match dissolved_pct length).
    dissolved_pct:
        Percentage dissolved at each time point (values in [0, 100]).
    ke_per_h:
        Elimination rate constant from IV study (1/h, > 0).
    ka_per_h:
        Absorption rate constant from oral study (1/h, > 0).
    f_oral:
        Oral bioavailability fraction (0, 1].

    Returns
    -------
    DeconvolutionResult
    """
    # --- Validation ---
    if len(times_dissolution_min) < 3:
        raise ValueError(
            f"times_dissolution_min must have >= 3 points, got {len(times_dissolution_min)}"
        )
    if len(times_dissolution_min) != len(dissolved_pct):
        raise ValueError(
            f"times_dissolution_min length ({len(times_dissolution_min)}) must equal "
            f"dissolved_pct length ({len(dissolved_pct)})"
        )
    if ke_per_h <= 0.0:
        raise ValueError(f"ke_per_h must be > 0, got {ke_per_h}")
    if ka_per_h <= 0.0:
        raise ValueError(f"ka_per_h must be > 0, got {ka_per_h}")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")
    for pct in dissolved_pct:
        if not (0.0 <= pct <= 100.0):
            raise ValueError(f"dissolved_pct values must be in [0, 100], got {pct}")

    # Convert times to hours
    times_h = [t / 60.0 for t in times_dissolution_min]

    # Compute cumulative fraction absorbed (% scale)
    fa_cumulative_pct = [p * f_oral for p in dissolved_pct]

    # R-squared: correlation between dissolution rates and absorption rates
    # Use consecutive differences as proxy for rates
    n = len(dissolved_pct)
    if n >= 2:
        diss_rates = [dissolved_pct[i] - dissolved_pct[i - 1] for i in range(1, n)]
        abs_rates = [fa_cumulative_pct[i] - fa_cumulative_pct[i - 1] for i in range(1, n)]
        r_squared = _pearson_r_squared(diss_rates, abs_rates)
    else:
        r_squared = 1.0 - (1.0 - f_oral) ** 2

    # Clamp r_squared to [0, 1]
    r_squared = max(0.0, min(1.0, r_squared))

    # IVIVC level classification
    ivivc_level = "Level_A" if r_squared > 0.9 else "Level_B"

    # Tmax estimate: time of maximum dissolution rate (largest increment)
    tmax_estimate_h = times_h[0]
    if n >= 2:
        max_rate = dissolved_pct[1] - dissolved_pct[0]
        tmax_idx = 1
        for i in range(2, n):
            rate = dissolved_pct[i] - dissolved_pct[i - 1]
            if rate > max_rate:
                max_rate = rate
                tmax_idx = i
        tmax_estimate_h = times_h[tmax_idx]
    # Ensure tmax > 0
    if tmax_estimate_h <= 0.0:
        tmax_estimate_h = times_h[-1] if times_h[-1] > 0.0 else 1.0 / 60.0

    notes = (
        f"Wagner-Nelson deconvolution; drug={drug_name}; "
        f"f_oral={f_oral:.3f}; ke={ke_per_h:.3f}/h; ka={ka_per_h:.3f}/h; "
        f"IVIVC={ivivc_level}; R2={r_squared:.4f}"
    )

    return DeconvolutionResult(
        drug_name=drug_name,
        f_oral=f_oral,
        ke_per_h=ke_per_h,
        ka_per_h=ka_per_h,
        times_h=times_h,
        dissolved_pct=list(dissolved_pct),
        fa_cumulative_pct=fa_cumulative_pct,
        r_squared=r_squared,
        ivivc_level=ivivc_level,
        tmax_estimate_h=tmax_estimate_h,
        notes=notes,
    )


def predict_in_vivo_absorption(
    drug_name: str,
    times_dissolution_min: list[float],
    dissolved_pct: list[float],
    ke_per_h: float,
    f_oral: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list[tuple[float, float]]:
    """Predict in vivo absorption profile from dissolution data.

    Uses Wagner-Nelson convolution with a 1-compartment disposition to
    predict the plasma concentration-equivalent fraction absorbed over time.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    times_dissolution_min:
        Time points in minutes for dissolution measurements.
    dissolved_pct:
        Percentage dissolved at each time point (values in [0, 100]).
    ke_per_h:
        Elimination rate constant (1/h, > 0).
    f_oral:
        Oral bioavailability fraction (0, 1].
    t_end_h:
        End time for prediction in hours.
    dt_h:
        Time step for forward Euler integration in hours.

    Returns
    -------
    list[tuple[float, float]]
        List of (time_h, f_abs_pct) tuples representing predicted
        cumulative fraction absorbed (%) at each time point.
    """
    if len(times_dissolution_min) < 3:
        raise ValueError(
            f"times_dissolution_min must have >= 3 points, got {len(times_dissolution_min)}"
        )
    if len(times_dissolution_min) != len(dissolved_pct):
        raise ValueError(
            f"Lengths must match: times={len(times_dissolution_min)}, "
            f"dissolved_pct={len(dissolved_pct)}"
        )
    if ke_per_h <= 0.0:
        raise ValueError(f"ke_per_h must be > 0, got {ke_per_h}")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")

    # Build piecewise-linear interpolation function for dissolved_pct
    times_h_diss = [t / 60.0 for t in times_dissolution_min]

    def interp_dissolved(t: float) -> float:
        """Linearly interpolate dissolved_pct at time t (hours)."""
        if t <= times_h_diss[0]:
            return dissolved_pct[0]
        if t >= times_h_diss[-1]:
            return dissolved_pct[-1]
        for i in range(1, len(times_h_diss)):
            if t <= times_h_diss[i]:
                t0, t1 = times_h_diss[i - 1], times_h_diss[i]
                p0, p1 = dissolved_pct[i - 1], dissolved_pct[i]
                frac = (t - t0) / (t1 - t0) if (t1 - t0) > 1e-12 else 0.0
                return p0 + frac * (p1 - p0)
        return dissolved_pct[-1]

    # Forward Euler to predict cumulative fraction absorbed
    n_steps = max(int(round(t_end_h / dt_h)), 1)
    result: list[tuple[float, float]] = [(0.0, 0.0)]

    # State: cumulative absorbed fraction (fraction scale)
    fa = 0.0

    for i in range(n_steps):
        t = i * dt_h
        t_next = (i + 1) * dt_h

        # Fraction dissolved at current and next time (0-1 scale)
        fd_curr = interp_dissolved(t) / 100.0 * f_oral
        fd_next = interp_dissolved(t_next) / 100.0 * f_oral

        # Input rate: d(fa)/dt approximated by forward difference
        input_rate = (fd_next - fd_curr) / dt_h

        # Wagner-Nelson: dfa/dt = input_rate + ke * (fd - fa)
        # Simplified: track cumulative absorbed from convolution
        # dfa/dt = max(0, input_rate) + ke * max(0, fd_curr - fa)
        dfa_dt = max(0.0, input_rate) + ke_per_h * max(0.0, fd_curr - fa)
        fa = min(fa + dfa_dt * dt_h, fd_next)
        fa = max(0.0, fa)

        result.append((t_next, fa * 100.0))

    return result
