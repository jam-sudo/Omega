"""Advanced concentration-time profile analysis: peak detection, AUC partitioning, profile comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ProfileComparisonResult",
    "find_local_maxima",
    "compute_partial_auc",
    "compare_profiles",
    "compute_fluctuation_index",
    "interpolate_concentration",
]


@dataclass(frozen=True)
class ProfileComparisonResult:
    """Result of comparing two concentration-time profiles."""

    auc_ratio: float
    cmax_ratio: float
    tmax_difference_h: float
    rmsd: float
    normalized_rmsd: float
    profile_similarity: str
    notes: str


def _validate_profile(times_h: list[float], concs_mg_L: list[float]) -> None:
    """Validate that times and concs are compatible and strictly increasing."""
    if len(times_h) != len(concs_mg_L):
        raise ValueError(
            f"times_h and concs_mg_L must have same length, got {len(times_h)} and {len(concs_mg_L)}"
        )
    if len(times_h) < 2:
        raise ValueError("Profile must have at least 2 points")
    for i in range(1, len(times_h)):
        if times_h[i] <= times_h[i - 1]:
            raise ValueError(
                f"times_h must be strictly increasing, got {times_h[i - 1]} >= {times_h[i]} at index {i}"
            )


def find_local_maxima(
    times_h: list[float],
    concs_mg_L: list[float],
    min_prominence: float = 0.05,
) -> list[dict]:
    """Find local maxima (peaks) in a concentration-time profile.

    A point is a local maximum if it is greater than both its neighbors.
    Only peaks with value > min_prominence * global_cmax are returned.

    Returns:
        List of dicts with keys: time_h, conc_mg_L, index
    """
    _validate_profile(times_h, concs_mg_L)

    if len(concs_mg_L) == 0:
        return []

    cmax_global = max(concs_mg_L)
    threshold = min_prominence * cmax_global

    peaks = []
    n = len(concs_mg_L)
    for i in range(1, n - 1):
        if concs_mg_L[i] > concs_mg_L[i - 1] and concs_mg_L[i] > concs_mg_L[i + 1]:
            if concs_mg_L[i] > threshold:
                peaks.append(
                    {
                        "time_h": times_h[i],
                        "conc_mg_L": concs_mg_L[i],
                        "index": i,
                    }
                )

    return peaks


def interpolate_concentration(
    times_h: list[float], concs_mg_L: list[float], t_query: float
) -> float:
    """Linear interpolation at time t_query.

    Returns 0.0 if t_query is outside the range of times_h.
    """
    _validate_profile(times_h, concs_mg_L)

    if t_query < times_h[0] or t_query > times_h[-1]:
        return 0.0

    # Find the interval containing t_query
    for i in range(len(times_h) - 1):
        if times_h[i] <= t_query <= times_h[i + 1]:
            t0, t1 = times_h[i], times_h[i + 1]
            c0, c1 = concs_mg_L[i], concs_mg_L[i + 1]
            if math.isclose(t1, t0):
                return c0
            frac = (t_query - t0) / (t1 - t0)
            return c0 + frac * (c1 - c0)

    # Exact match with last point
    return concs_mg_L[-1]


def compute_partial_auc(
    times_h: list[float],
    concs_mg_L: list[float],
    t_start: float,
    t_end: float,
) -> float:
    """Compute trapezoidal AUC from t_start to t_end.

    Uses linear interpolation at boundaries to handle partial intervals.
    """
    _validate_profile(times_h, concs_mg_L)

    if t_start >= t_end:
        return 0.0

    # Clamp to profile range
    t_start = max(t_start, times_h[0])
    t_end = min(t_end, times_h[-1])

    if t_start >= t_end:
        return 0.0

    # Build list of (time, conc) for integration window
    pts: list[tuple[float, float]] = []

    # Add interpolated start point
    c_start = interpolate_concentration(times_h, concs_mg_L, t_start)
    pts.append((t_start, c_start))

    # Add all interior profile points
    for i, t in enumerate(times_h):
        if t_start < t < t_end:
            pts.append((t, concs_mg_L[i]))

    # Add interpolated end point
    c_end = interpolate_concentration(times_h, concs_mg_L, t_end)
    pts.append((t_end, c_end))

    # Sort by time (should already be sorted, but be safe)
    pts.sort(key=lambda x: x[0])

    # Trapezoidal integration
    auc = 0.0
    for i in range(len(pts) - 1):
        dt = pts[i + 1][0] - pts[i][0]
        auc += 0.5 * (pts[i][1] + pts[i + 1][1]) * dt

    return auc


def compare_profiles(
    times_a: list[float],
    concs_a: list[float],
    times_b: list[float],
    concs_b: list[float],
) -> ProfileComparisonResult:
    """Compare two concentration-time profiles.

    Interpolates both profiles to a common time grid (union of all time points),
    then computes AUC ratio, Cmax ratio, tmax difference, RMSD, and nRMSD.
    """
    _validate_profile(times_a, concs_a)
    _validate_profile(times_b, concs_b)

    # Build common time grid as union of both time vectors, restricted to overlap
    t_start = max(times_a[0], times_b[0])
    t_end = min(times_a[-1], times_b[-1])

    common_times_set: set[float] = set()
    for t in times_a:
        if t_start <= t <= t_end:
            common_times_set.add(t)
    for t in times_b:
        if t_start <= t <= t_end:
            common_times_set.add(t)

    common_times = sorted(common_times_set)

    if len(common_times) < 2:
        # Minimal overlap — use endpoints
        common_times = [t_start, t_end]

    # Interpolate both profiles at common times
    ca_interp = [interpolate_concentration(times_a, concs_a, t) for t in common_times]
    cb_interp = [interpolate_concentration(times_b, concs_b, t) for t in common_times]

    # AUC via trapezoidal
    def _trapz(ts: list[float], cs: list[float]) -> float:
        auc = 0.0
        for i in range(len(ts) - 1):
            auc += 0.5 * (cs[i] + cs[i + 1]) * (ts[i + 1] - ts[i])
        return auc

    auc_a = _trapz(common_times, ca_interp)
    auc_b = _trapz(common_times, cb_interp)

    auc_ratio = auc_a / auc_b if auc_b != 0.0 else float("inf")

    # Cmax
    cmax_a = max(concs_a)
    cmax_b = max(concs_b)
    cmax_ratio = cmax_a / cmax_b if cmax_b != 0.0 else float("inf")

    # tmax (first occurrence of Cmax)
    tmax_a = times_a[concs_a.index(cmax_a)]
    tmax_b = times_b[concs_b.index(cmax_b)]
    tmax_diff = tmax_a - tmax_b

    # RMSD
    n = len(common_times)
    sq_sum = sum((ca_interp[i] - cb_interp[i]) ** 2 for i in range(n))
    rmsd = math.sqrt(sq_sum / n)

    # Normalized RMSD
    mean_b = sum(cb_interp) / n if n > 0 else 1.0
    normalized_rmsd = rmsd / mean_b if mean_b != 0.0 else float("inf")

    # Profile similarity classification
    if normalized_rmsd < 0.2:
        profile_similarity = "similar"
    elif normalized_rmsd <= 0.5:
        profile_similarity = "moderate"
    else:
        profile_similarity = "dissimilar"

    notes = (
        f"AUC ratio={auc_ratio:.3f}, Cmax ratio={cmax_ratio:.3f}, "
        f"nRMSD={normalized_rmsd:.3f}, similarity={profile_similarity}"
    )

    return ProfileComparisonResult(
        auc_ratio=auc_ratio,
        cmax_ratio=cmax_ratio,
        tmax_difference_h=tmax_diff,
        rmsd=rmsd,
        normalized_rmsd=normalized_rmsd,
        profile_similarity=profile_similarity,
        notes=notes,
    )


def compute_fluctuation_index(times_h: list[float], concs_mg_L: list[float]) -> dict:
    """Compute steady-state fluctuation metrics.

    Returns:
        dict with keys: cmax, cmin, cavg, ptf, swing
        ptf = (Cmax - Cmin) / Cavg (peak-trough fluctuation)
        swing = (Cmax - Cmin) / Cmin
    """
    _validate_profile(times_h, concs_mg_L)

    cmax = max(concs_mg_L)
    cmin = min(concs_mg_L)

    # Average concentration via trapezoidal AUC / time span
    t_span = times_h[-1] - times_h[0]
    auc = 0.0
    for i in range(len(times_h) - 1):
        dt = times_h[i + 1] - times_h[i]
        auc += 0.5 * (concs_mg_L[i] + concs_mg_L[i + 1]) * dt

    cavg = auc / t_span if t_span > 0.0 else (cmax + cmin) / 2.0

    ptf = (cmax - cmin) / cavg if cavg != 0.0 else 0.0
    swing = (cmax - cmin) / cmin if cmin != 0.0 else float("inf")

    return {
        "cmax": cmax,
        "cmin": cmin,
        "cavg": cavg,
        "ptf": ptf,
        "swing": swing,
    }
