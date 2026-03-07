"""Drug exposure metrics from concentration-time data (Phase 642)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ExposureMetrics",
    "compute_exposure_metrics",
    "compare_exposures",
]


@dataclass(frozen=True)
class ExposureMetrics:
    """Standard exposure metrics computed from a PK concentration-time profile."""

    auc_total: float
    auc_0_to_tau: float
    cmax: float
    cmin: float
    tmax_h: float
    peak_trough_ratio: float
    time_above_threshold_h: float
    time_weighted_avg_mg_L: float
    auc_first_half: float
    auc_second_half: float
    accumulation_ratio: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC over the full range."""
    total = 0.0
    for i in range(1, len(times)):
        total += (concs[i] + concs[i - 1]) / 2.0 * (times[i] - times[i - 1])
    return total


def _trapezoidal_auc_until_tau(times: list[float], concs: list[float], tau: float) -> float:
    """AUC from t[0] to tau using trapezoidal rule (sum steps where t[i] < tau)."""
    total = 0.0
    for i in range(1, len(times)):
        t0, t1 = times[i - 1], times[i]
        c0, c1 = concs[i - 1], concs[i]
        if t0 >= tau:
            break
        if t1 <= tau:
            total += (c0 + c1) / 2.0 * (t1 - t0)
        else:
            # Linear interpolation at tau
            frac = (tau - t0) / (t1 - t0)
            c_tau = c0 + frac * (c1 - c0)
            total += (c0 + c_tau) / 2.0 * (tau - t0)
            break
    return total


def _time_above_threshold(times: list[float], concs: list[float], threshold: float) -> float:
    """Total time where concentration exceeds threshold (linear interpolation at crossings)."""
    total = 0.0
    n = len(times)
    for i in range(1, n):
        t0, t1 = times[i - 1], times[i]
        c0, c1 = concs[i - 1], concs[i]
        dt = t1 - t0
        above0 = c0 > threshold
        above1 = c1 > threshold
        if above0 and above1:
            total += dt
        elif above0 and not above1:
            # Crossing downward
            frac = (c0 - threshold) / (c0 - c1)
            total += frac * dt
        elif not above0 and above1:
            # Crossing upward
            frac = (threshold - c0) / (c1 - c0)
            total += (1.0 - frac) * dt
        # else: both below
    return total


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(
    times_h: list[float],
    concs_mg_L: list[float],
    tau_h: float | None,
) -> None:
    if len(times_h) < 2:
        raise ValueError(f"times_h must have at least 2 points; got {len(times_h)}.")
    if len(times_h) != len(concs_mg_L):
        raise ValueError(
            f"times_h and concs_mg_L must have the same length; "
            f"got {len(times_h)} vs {len(concs_mg_L)}."
        )
    for i in range(1, len(times_h)):
        if times_h[i] <= times_h[i - 1]:
            raise ValueError(
                f"times_h must be strictly monotonically increasing; "
                f"times_h[{i - 1}]={times_h[i - 1]}, times_h[{i}]={times_h[i]}."
            )
    for c in concs_mg_L:
        if c < 0.0:
            raise ValueError(f"All concentrations must be >= 0; got {c}.")
    if tau_h is not None and tau_h <= 0.0:
        raise ValueError(f"tau_h must be positive; got {tau_h}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_exposure_metrics(
    times_h: list[float],
    concs_mg_L: list[float],
    tau_h: float | None = None,
    threshold_mg_L: float = 0.0,
    accumulation_ratio: float = 1.0,
) -> ExposureMetrics:
    """Compute drug exposure metrics from a concentration-time profile.

    Parameters
    ----------
    times_h:
        Sampling times in hours (monotonically increasing).
    concs_mg_L:
        Drug concentrations in mg/L (all >= 0).
    tau_h:
        Dosing interval (h) for AUC_0_tau computation. If None, set to total duration.
    threshold_mg_L:
        Concentration threshold for time-above-threshold computation.
    accumulation_ratio:
        Known accumulation ratio (pass 1.0 for single dose).

    Returns
    -------
    ExposureMetrics
    """
    _validate(times_h, concs_mg_L, tau_h)

    t = list(times_h)
    c = list(concs_mg_L)
    n = len(t)
    duration = t[-1] - t[0]

    # AUC total
    auc_total = _trapezoidal_auc(t, c)

    # AUC 0 to tau
    if tau_h is None:
        auc_0_to_tau = auc_total
    elif tau_h >= t[-1]:
        auc_0_to_tau = auc_total
    else:
        auc_0_to_tau = _trapezoidal_auc_until_tau(t, c, tau_h)

    # Cmax and tmax
    cmax = max(c)
    tmax_idx = c.index(cmax)
    tmax_h = t[tmax_idx]

    # Cmin: minimum concentration excluding t=0 value
    if n > 1:
        cmin = min(c[1:])
    else:
        cmin = c[0]

    # Peak-trough ratio
    if cmin > 0.0:
        peak_trough_ratio = cmax / cmin
    else:
        peak_trough_ratio = 999.0

    # Time above threshold
    time_above_threshold_h = _time_above_threshold(t, c, threshold_mg_L)

    # Time-weighted average
    if duration > 0.0:
        time_weighted_avg_mg_L = auc_total / duration
    else:
        time_weighted_avg_mg_L = c[0]

    # AUC first half and second half
    midpoint = t[0] + duration / 2.0
    auc_first_half = _trapezoidal_auc_until_tau(t, c, midpoint)
    # Second half: total AUC minus first half
    auc_second_half = auc_total - auc_first_half

    notes = (
        f"Cmax={cmax:.3f} mg/L at t={tmax_h:.2f}h, "
        f"Cmin={cmin:.3f} mg/L, AUC_total={auc_total:.3f} mg*h/L, "
        f"duration={duration:.2f}h, "
        f"time_above_{threshold_mg_L:.2f}={time_above_threshold_h:.2f}h."
    )

    return ExposureMetrics(
        auc_total=auc_total,
        auc_0_to_tau=auc_0_to_tau,
        cmax=cmax,
        cmin=cmin,
        tmax_h=tmax_h,
        peak_trough_ratio=peak_trough_ratio,
        time_above_threshold_h=time_above_threshold_h,
        time_weighted_avg_mg_L=time_weighted_avg_mg_L,
        auc_first_half=auc_first_half,
        auc_second_half=auc_second_half,
        accumulation_ratio=accumulation_ratio,
        notes=notes,
    )


def compare_exposures(
    scenarios: list[dict],
    threshold_mg_L: float = 0.0,
) -> list[dict]:
    """Compute exposure metrics for multiple scenarios, sorted by auc_total descending.

    Each dict in scenarios must contain:
        - "name": str
        - "times_h": list[float]
        - "concs_mg_L": list[float]
        - "tau_h": float (optional)

    Returns
    -------
    list[dict]
        Each element has "name" key plus all ExposureMetrics fields, sorted by
        auc_total descending.
    """
    results: list[dict] = []
    for scenario in scenarios:
        name = scenario["name"]
        times_h = scenario["times_h"]
        concs_mg_L = scenario["concs_mg_L"]
        tau_h = scenario.get("tau_h", None)
        em = compute_exposure_metrics(
            times_h=times_h,
            concs_mg_L=concs_mg_L,
            tau_h=tau_h,
            threshold_mg_L=threshold_mg_L,
        )
        row = {"name": name}
        row["auc_total"] = em.auc_total
        row["auc_0_to_tau"] = em.auc_0_to_tau
        row["cmax"] = em.cmax
        row["cmin"] = em.cmin
        row["tmax_h"] = em.tmax_h
        row["peak_trough_ratio"] = em.peak_trough_ratio
        row["time_above_threshold_h"] = em.time_above_threshold_h
        row["time_weighted_avg_mg_L"] = em.time_weighted_avg_mg_L
        row["auc_first_half"] = em.auc_first_half
        row["auc_second_half"] = em.auc_second_half
        row["accumulation_ratio"] = em.accumulation_ratio
        row["notes"] = em.notes
        results.append(row)

    results.sort(key=lambda d: d["auc_total"], reverse=True)
    return results
