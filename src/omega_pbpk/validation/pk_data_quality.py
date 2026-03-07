"""PK data quality assessment for model building.

Evaluates pharmacokinetic datasets for adequacy of sampling design,
presence of statistical outliers, and handling of concentrations
below the lower limit of quantification (BLQ/LLOQ).

Quality criteria follow best-practice guidance (FDA, EMA, NONMEM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SamplingQualityResult",
    "OutlierDetectionResult",
    "BLQResult",
    "DataQualityReport",
    "assess_sampling_schedule",
    "detect_outliers_pk",
    "assess_blq_handling",
    "pk_data_quality_report",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_TOTAL_POINTS = 6
_MIN_PHASE_POINTS = 3
_IQR_MULTIPLIER = 1.5
_ZSCORE_THRESHOLD = 2.5
_TERMINAL_FRACTION = 0.30  # last 30% of timespan


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingQualityResult:
    """Result of sampling schedule adequacy assessment."""

    n_points: int
    """Total number of sampling time-points."""

    has_absorption_points: bool
    """True if at least one point is in the absorption phase."""

    has_distribution_points: bool
    """True if Cmax region is covered by >=3 points within tmax±50%."""

    has_terminal_points: bool
    """True if >=3 points in the terminal elimination phase."""

    quality_score: int
    """Integer score 0–5 (sum of individual criteria)."""

    recommendations: list[str]
    """List of actionable recommendations to improve the schedule."""


@dataclass(frozen=True)
class OutlierDetectionResult:
    """Result of PK concentration outlier detection."""

    n_outliers: int
    """Number of outliers detected."""

    outlier_indices: list[int]
    """Indices (into original arrays) of outlier observations."""

    outlier_times: list[float]
    """Time values of outlier observations."""

    outlier_concs: list[float]
    """Concentration values of outlier observations."""

    cleaned_times: list[float]
    """Time array with outliers removed."""

    cleaned_concs: list[float]
    """Concentration array with outliers removed."""

    method: str
    """Detection method used: 'iqr' or 'zscore'."""

    notes: str
    """Summary of findings."""


@dataclass(frozen=True)
class BLQResult:
    """Result of BLQ (below LLOQ) data handling assessment."""

    n_blq: int
    """Number of BLQ observations."""

    pct_blq: float
    """Percentage of observations that are BLQ."""

    blq_location: str
    """'none', 'terminal', or 'mid_profile' (problematic if mid_profile)."""

    recommendation: str
    """Recommended BLQ handling strategy."""

    notes: str
    """Additional context."""


@dataclass(frozen=True)
class DataQualityReport:
    """Comprehensive PK data quality report."""

    sampling: SamplingQualityResult
    outliers: OutlierDetectionResult
    blq: BLQResult
    overall_score: float
    """Weighted quality score 0–100."""

    overall_grade: str
    """Letter grade: 'A' (>=90), 'B' (>=75), 'C' (>=60), 'D' (>=40), 'F' (<40)."""

    notes: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_pk_data(
    times_h: list[float], concentrations: list[float]
) -> tuple[list[float], list[float]]:
    """Validate basic PK data requirements. Returns (times, concs) unchanged."""
    if len(times_h) != len(concentrations):
        raise ValueError(
            f"times_h and concentrations must have the same length, "
            f"got {len(times_h)} and {len(concentrations)}"
        )
    if len(times_h) < 3:
        raise ValueError(
            f"At least 3 time-points required, got {len(times_h)}"
        )
    for i in range(len(times_h) - 1):
        if times_h[i] >= times_h[i + 1]:
            raise ValueError(
                f"times_h must be strictly increasing; "
                f"times_h[{i}]={times_h[i]} >= times_h[{i+1}]={times_h[i+1]}"
            )
    for c in concentrations:
        if c < 0:
            raise ValueError(
                f"Concentrations must be >= 0, got {c}"
            )
    return times_h, concentrations


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def assess_sampling_schedule(
    times_h: list[float],
    route: str = "oral",
) -> SamplingQualityResult:
    """Assess PK sampling schedule adequacy.

    Checks five criteria, each worth 1 point:
      1. At least 6 total time-points.
      2. At least 1 point in the absorption phase (before Cmax).
      3. At least 3 points in the distribution/Cmax region (tmax ± 50%).
      4. At least 3 points in the terminal phase (last 30% of timespan).
      5. Both absorption and elimination phases are covered.

    Parameters
    ----------
    times_h:
        Sampling times in hours (must be monotonically increasing, ≥ 3 points).
    route:
        Dosing route ('oral' or 'iv'). IV profiles have no absorption phase.

    Returns
    -------
    SamplingQualityResult
    """
    if len(times_h) < 1:
        raise ValueError("times_h must not be empty")
    for i in range(len(times_h) - 1):
        if times_h[i] >= times_h[i + 1]:
            raise ValueError("times_h must be strictly increasing")

    t_arr = list(times_h)
    n = len(t_arr)
    t_min = t_arr[0]
    t_max = t_arr[-1]
    t_span = t_max - t_min if t_max > t_min else 1.0

    route_lower = route.lower()
    is_oral = route_lower != "iv"

    # ---- Criterion 1: minimum number of points ----
    c1_sufficient_points = n >= _MIN_TOTAL_POINTS

    # ---- Estimate tmax heuristic ----
    # For oral: tmax is roughly the first quarter of the profile
    # For IV: no tmax (starts at peak)
    if is_oral:
        tmax_est = t_min + 0.25 * t_span
    else:
        tmax_est = t_min  # IV peaks at t=0

    # ---- Criterion 2: absorption phase points (oral only) ----
    if is_oral:
        absorption_pts = [t for t in t_arr if t < tmax_est]
        c2_absorption = len(absorption_pts) >= 1
    else:
        # IV: absorption phase not applicable; criterion auto-passed
        absorption_pts = []
        c2_absorption = True

    # ---- Criterion 3: distribution/Cmax region points ----
    tmax_lower = tmax_est * 0.5 if tmax_est > 0 else 0.0
    tmax_upper = tmax_est * 1.5 + 0.5 * t_span * 0.1  # generous window
    dist_pts = [t for t in t_arr if tmax_lower <= t <= tmax_upper]
    # For IV, Cmax region is first 20% of timespan
    if not is_oral:
        cmax_cutoff = t_min + 0.20 * t_span
        dist_pts = [t for t in t_arr if t <= cmax_cutoff]
    c3_distribution = len(dist_pts) >= _MIN_PHASE_POINTS

    # ---- Criterion 4: terminal phase points ----
    terminal_start = t_max - _TERMINAL_FRACTION * t_span
    terminal_pts = [t for t in t_arr if t >= terminal_start]
    c4_terminal = len(terminal_pts) >= _MIN_PHASE_POINTS

    # ---- Criterion 5: coverage of both phases ----
    if is_oral:
        has_early = any(t <= t_min + 0.33 * t_span for t in t_arr)
        has_late = any(t >= t_min + 0.67 * t_span for t in t_arr)
        c5_coverage = has_early and has_late
    else:
        # IV: just need coverage across early and late
        has_early = any(t <= t_min + 0.40 * t_span for t in t_arr)
        has_late = any(t >= t_min + 0.60 * t_span for t in t_arr)
        c5_coverage = has_early and has_late

    score = sum([c1_sufficient_points, c2_absorption, c3_distribution, c4_terminal, c5_coverage])

    recommendations: list[str] = []
    if not c1_sufficient_points:
        recommendations.append(
            f"Add more samples: minimum {_MIN_TOTAL_POINTS} points required (have {n})."
        )
    if is_oral and not c2_absorption:
        recommendations.append(
            "Add early time-points to capture the absorption phase (before estimated tmax)."
        )
    if not c3_distribution:
        recommendations.append(
            "Add samples around Cmax to characterise the peak concentration."
        )
    if not c4_terminal:
        recommendations.append(
            f"Add at least {_MIN_PHASE_POINTS} samples in the terminal elimination phase "
            f"(last {int(_TERMINAL_FRACTION*100)}% of sampling window)."
        )
    if not c5_coverage:
        recommendations.append(
            "Improve coverage: ensure samples span both early and late portions of the profile."
        )

    return SamplingQualityResult(
        n_points=n,
        has_absorption_points=c2_absorption,
        has_distribution_points=c3_distribution,
        has_terminal_points=c4_terminal,
        quality_score=score,
        recommendations=recommendations,
    )


def detect_outliers_pk(
    times_h: list[float],
    concentrations: list[float],
    method: str = "iqr",
) -> OutlierDetectionResult:
    """Detect outlier concentration values in a PK profile.

    Methods
    -------
    'iqr'    : Tukey fences — outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    'zscore' : |z-score| > 2.5 (on log-transformed concentrations for PK).

    Parameters
    ----------
    times_h:
        Sampling times in hours.
    concentrations:
        Measured plasma concentrations (same units, ≥ 0).
    method:
        'iqr' or 'zscore'.

    Returns
    -------
    OutlierDetectionResult
    """
    _validate_pk_data(times_h, concentrations)
    method_lower = method.lower()
    if method_lower not in ("iqr", "zscore"):
        raise ValueError(f"method must be 'iqr' or 'zscore', got '{method}'")

    concs = np.array(concentrations, dtype=float)

    # Work on positive concentrations for statistical detection
    positive_mask = concs > 0
    positive_concs = concs[positive_mask]

    outlier_mask = np.zeros(len(concs), dtype=bool)

    if len(positive_concs) >= 3:
        if method_lower == "iqr":
            q1 = float(np.percentile(positive_concs, 25))
            q3 = float(np.percentile(positive_concs, 75))
            iqr = q3 - q1
            lo = q1 - _IQR_MULTIPLIER * iqr
            hi = q3 + _IQR_MULTIPLIER * iqr
            for i, c in enumerate(concs):
                if positive_mask[i] and (c < lo or c > hi):
                    outlier_mask[i] = True
        else:  # zscore — modified z-score using median/MAD in log-space (outlier-robust)
            log_concs = np.log(positive_concs)
            median_log = float(np.median(log_concs))
            # Median Absolute Deviation; scale factor 0.6745 for normal consistency
            mad = float(np.median(np.abs(log_concs - median_log)))
            mad_scaled = mad / 0.6745 if mad > 0 else 0.0
            if mad_scaled > 0:
                for i, c in enumerate(concs):
                    if positive_mask[i]:
                        z = abs(math.log(c) - median_log) / mad_scaled
                        if z > _ZSCORE_THRESHOLD:
                            outlier_mask[i] = True
            elif len(log_concs) > 1:
                # Fallback: standard z-score when MAD = 0 (all same value)
                std_log = float(np.std(log_concs, ddof=1))
                if std_log > 0:
                    for i, c in enumerate(concs):
                        if positive_mask[i]:
                            z = abs(math.log(c) - median_log) / std_log
                            if z > _ZSCORE_THRESHOLD:
                                outlier_mask[i] = True

    outlier_indices = [int(i) for i in np.where(outlier_mask)[0]]
    outlier_times = [times_h[i] for i in outlier_indices]
    outlier_concs_list = [concentrations[i] for i in outlier_indices]
    cleaned_times = [times_h[i] for i in range(len(times_h)) if not outlier_mask[i]]
    cleaned_concs = [concentrations[i] for i in range(len(concentrations)) if not outlier_mask[i]]

    n_out = len(outlier_indices)
    notes = (
        f"Method: {method_lower}. {n_out} outlier(s) detected out of "
        f"{len(concentrations)} observations."
    )

    return OutlierDetectionResult(
        n_outliers=n_out,
        outlier_indices=outlier_indices,
        outlier_times=outlier_times,
        outlier_concs=outlier_concs_list,
        cleaned_times=cleaned_times,
        cleaned_concs=cleaned_concs,
        method=method_lower,
        notes=notes,
    )


def assess_blq_handling(
    times_h: list[float],
    concentrations: list[float],
    lloq: float,
) -> BLQResult:
    """Assess BLQ (below LLOQ) observations and recommend handling.

    BLQ values in the terminal phase are acceptable; BLQ values in the
    middle of a profile are problematic and indicate assay or sampling issues.

    Parameters
    ----------
    times_h:
        Sampling times in hours.
    concentrations:
        Measured concentrations (≥ 0; 0 may represent BLQ).
    lloq:
        Lower limit of quantification.

    Returns
    -------
    BLQResult
    """
    _validate_pk_data(times_h, concentrations)
    if lloq < 0:
        raise ValueError(f"lloq must be >= 0, got {lloq}")

    n = len(concentrations)
    blq_mask = [c < lloq for c in concentrations]
    n_blq = sum(blq_mask)
    pct_blq = 100.0 * n_blq / n if n > 0 else 0.0

    if n_blq == 0:
        location = "none"
        recommendation = "No BLQ values; no special handling required."
        notes = "All observations are above LLOQ."
    else:
        t_max = times_h[-1]
        t_min = times_h[0]
        t_span = t_max - t_min if t_max > t_min else 1.0
        terminal_start = t_max - _TERMINAL_FRACTION * t_span

        blq_times = [times_h[i] for i, b in enumerate(blq_mask) if b]
        mid_blq = [t for t in blq_times if t < terminal_start]

        if mid_blq:
            location = "mid_profile"
            recommendation = (
                "Mid-profile BLQ values detected. Consider M3 method (NONMEM) "
                "for likelihood-based BLQ handling. Investigate assay sensitivity."
            )
            notes = (
                f"{len(mid_blq)} BLQ observation(s) in mid-profile at "
                f"times: {mid_blq}. This may indicate an assay or sampling issue."
            )
        else:
            location = "terminal"
            recommendation = (
                "Terminal BLQ values are acceptable. Replace with LLOQ/2 for "
                "NCA, or use M3 method for population PK."
            )
            notes = (
                f"{n_blq} BLQ observation(s) all in terminal phase — "
                "consistent with normal drug elimination."
            )

    return BLQResult(
        n_blq=n_blq,
        pct_blq=pct_blq,
        blq_location=location,
        recommendation=recommendation,
        notes=notes,
    )


def pk_data_quality_report(
    times_h: list[float],
    concentrations: list[float],
    route: str = "oral",
    lloq: float | None = None,
) -> DataQualityReport:
    """Generate a comprehensive PK data quality report.

    Combines sampling schedule assessment, outlier detection, and BLQ
    handling guidance into a single weighted quality score (0–100).

    Scoring weights:
      - Sampling quality : 50% (score 0–5 → 0–50)
      - Outlier burden   : 30% (0 outliers = 30, penalty per outlier%)
      - BLQ burden       : 20% (none=20, terminal=15, mid=5)

    Parameters
    ----------
    times_h:
        Sampling times in hours.
    concentrations:
        Measured plasma concentrations (≥ 0).
    route:
        Dosing route ('oral' or 'iv').
    lloq:
        Lower limit of quantification. If None, BLQ assessment is skipped.

    Returns
    -------
    DataQualityReport
    """
    _validate_pk_data(times_h, concentrations)

    sampling = assess_sampling_schedule(times_h, route=route)
    outliers = detect_outliers_pk(times_h, concentrations, method="iqr")

    if lloq is not None:
        blq = assess_blq_handling(times_h, concentrations, lloq=lloq)
    else:
        blq = BLQResult(
            n_blq=0,
            pct_blq=0.0,
            blq_location="none",
            recommendation="No LLOQ provided; BLQ assessment skipped.",
            notes="Set lloq parameter to enable BLQ assessment.",
        )

    # ---- Scoring ----
    sampling_score = (sampling.quality_score / 5.0) * 50.0  # 0–50

    n_total = len(times_h)
    outlier_pct = 100.0 * outliers.n_outliers / n_total if n_total > 0 else 0.0
    # Penalise: 30 - 3 per 10% outliers (floor at 0)
    outlier_score = max(0.0, 30.0 - outlier_pct * 0.3)

    if blq.blq_location == "none":
        blq_score = 20.0
    elif blq.blq_location == "terminal":
        blq_score = 15.0
    else:
        blq_score = 5.0

    overall = sampling_score + outlier_score + blq_score

    if overall >= 90:
        grade = "A"
    elif overall >= 75:
        grade = "B"
    elif overall >= 60:
        grade = "C"
    elif overall >= 40:
        grade = "D"
    else:
        grade = "F"

    notes = (
        f"Sampling score {sampling.quality_score}/5, "
        f"outlier burden {outlier_pct:.1f}%, "
        f"BLQ location '{blq.blq_location}'. "
        f"Overall: {overall:.1f}/100 ({grade})."
    )

    return DataQualityReport(
        sampling=sampling,
        outliers=outliers,
        blq=blq,
        overall_score=overall,
        overall_grade=grade,
        notes=notes,
    )
