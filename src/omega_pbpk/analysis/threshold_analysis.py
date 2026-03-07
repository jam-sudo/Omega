"""Concentration threshold crossing analysis for PK time-course data (Phase 318)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .exposure_metrics import _auc_lin_log_trapezoidal

__all__ = [
    "ThresholdResult",
    "CoverageResult",
    "AntibioticPKPDResult",
    "time_above_threshold",
    "time_below_threshold",
    "therapeutic_coverage",
    "pk_pd_breakpoint_analysis",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdResult:
    """Result of a threshold crossing analysis."""

    threshold: float
    total_time_above_h: float
    fraction_above: float
    n_intervals: int
    intervals_above: tuple[tuple[float, float], ...]  # (start_h, end_h) pairs
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CoverageResult:
    """Time in therapeutic window analysis."""

    lower_threshold: float
    upper_threshold: float
    time_therapeutic_h: float
    time_toxic_h: float
    time_subtherapeutic_h: float
    pct_therapeutic: float
    pct_toxic: float
    pct_subtherapeutic: float
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AntibioticPKPDResult:
    """PK/PD breakpoint analysis for antibiotic dosing."""

    mic_mg_L: float
    pkpd_index: str  # "auc_mic", "t_above_mic", "cmax_mic"
    auc_total: float
    auc_mic_ratio: float
    fT_above_mic_pct: float
    cmax_mic_ratio: float
    target_attained: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_threshold_inputs(
    times_h: list[float],
    concentrations: list[float],
    threshold: float,
    min_points: int = 2,
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
    if threshold <= 0:
        raise ValueError("threshold must be positive.")


def _validate_mic(mic_mg_L: float) -> None:
    if mic_mg_L <= 0:
        raise ValueError("mic_mg_L must be positive.")


# ---------------------------------------------------------------------------
# Crossing-time interpolation
# ---------------------------------------------------------------------------


def _find_crossing_intervals_above(
    times_h: list[float],
    concentrations: list[float],
    threshold: float,
) -> list[tuple[float, float]]:
    """Return list of (t_enter, t_exit) intervals where concentration > threshold."""
    intervals: list[tuple[float, float]] = []
    n = len(times_h)
    in_interval = False
    t_enter = 0.0

    for i in range(n):
        t_i = times_h[i]
        c_i = concentrations[i]
        above = c_i > threshold

        if i == 0:
            if above:
                in_interval = True
                t_enter = t_i
            continue

        t_prev = times_h[i - 1]
        c_prev = concentrations[i - 1]
        above_prev = c_prev > threshold

        if not above_prev and above:
            # Crossing upward — interpolate entry time
            frac = (threshold - c_prev) / (c_i - c_prev)
            t_enter = t_prev + frac * (t_i - t_prev)
            in_interval = True

        elif above_prev and not above:
            # Crossing downward — interpolate exit time
            frac = (c_prev - threshold) / (c_prev - c_i)
            t_exit = t_prev + frac * (t_i - t_prev)
            if in_interval:
                intervals.append((t_enter, t_exit))
            in_interval = False

    # Close any open interval at the last time point
    if in_interval:
        intervals.append((t_enter, times_h[-1]))

    return intervals


def _total_interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def time_above_threshold(
    times_h: list[float],
    concentrations: list[float],
    threshold: float,
) -> ThresholdResult:
    """Compute total time concentration exceeds a given threshold.

    Parameters
    ----------
    times_h:
        Sampling times (h), monotonically increasing.
    concentrations:
        Drug concentrations (mg/L), non-negative.
    threshold:
        Concentration threshold (mg/L), must be > 0.

    Returns
    -------
    ThresholdResult
        Frozen dataclass with timing metrics and crossing intervals.
    """
    _validate_threshold_inputs(times_h, concentrations, threshold)

    intervals = _find_crossing_intervals_above(times_h, concentrations, threshold)
    total_time = _total_interval_duration(intervals)
    obs_duration = times_h[-1] - times_h[0]
    fraction_above = total_time / obs_duration if obs_duration > 0 else 0.0

    notes: list[str] = []
    if fraction_above > 0.99:
        notes.append("Concentration is above threshold for nearly the entire observation.")

    return ThresholdResult(
        threshold=threshold,
        total_time_above_h=total_time,
        fraction_above=fraction_above,
        n_intervals=len(intervals),
        intervals_above=tuple(tuple(iv) for iv in intervals),  # type: ignore[arg-type]
        notes=tuple(notes),
    )


def time_below_threshold(
    times_h: list[float],
    concentrations: list[float],
    threshold: float,
) -> ThresholdResult:
    """Compute total time concentration is below a given threshold (e.g., MEC).

    Parameters
    ----------
    times_h:
        Sampling times (h), monotonically increasing.
    concentrations:
        Drug concentrations (mg/L), non-negative.
    threshold:
        Concentration threshold (mg/L), must be > 0.

    Returns
    -------
    ThresholdResult
        A ThresholdResult where ``total_time_above_h`` and ``fraction_above``
        represent the time **below** the threshold, and ``intervals_above``
        contains intervals where concentration is below the threshold.
    """
    _validate_threshold_inputs(times_h, concentrations, threshold)

    obs_duration = times_h[-1] - times_h[0]
    above_result = _find_crossing_intervals_above(times_h, concentrations, threshold)
    time_above = _total_interval_duration(above_result)
    time_below = obs_duration - time_above

    # Compute intervals below as complement of intervals above
    intervals_above = above_result
    intervals_below: list[tuple[float, float]] = []
    prev_end = times_h[0]
    for start, end in intervals_above:
        if start > prev_end:
            intervals_below.append((prev_end, start))
        prev_end = end
    if prev_end < times_h[-1]:
        intervals_below.append((prev_end, times_h[-1]))

    fraction_below = time_below / obs_duration if obs_duration > 0 else 0.0

    notes: list[str] = []
    if fraction_below > 0.5:
        notes.append(
            f"Concentration is below threshold for {fraction_below * 100:.1f}% "
            "of the observation period."
        )

    return ThresholdResult(
        threshold=threshold,
        total_time_above_h=time_below,
        fraction_above=fraction_below,
        n_intervals=len(intervals_below),
        intervals_above=tuple(tuple(iv) for iv in intervals_below),  # type: ignore[arg-type]
        notes=tuple(notes),
    )


def therapeutic_coverage(
    times_h: list[float],
    concentrations: list[float],
    lower_threshold: float,
    upper_threshold: float,
) -> CoverageResult:
    """Analyse time in therapeutic window, toxic range, and sub-therapeutic range.

    Parameters
    ----------
    times_h:
        Sampling times (h), monotonically increasing.
    concentrations:
        Drug concentrations (mg/L), non-negative.
    lower_threshold:
        Minimum effective concentration (mg/L); must be > 0.
    upper_threshold:
        Maximum safe concentration (mg/L); must be > lower_threshold.

    Returns
    -------
    CoverageResult
        Times and percentages in each pharmacological zone.
    """
    if upper_threshold <= lower_threshold:
        raise ValueError("upper_threshold must be greater than lower_threshold.")
    _validate_threshold_inputs(times_h, concentrations, lower_threshold)

    obs_duration = times_h[-1] - times_h[0]
    if obs_duration <= 0:
        raise ValueError("Observation window must be > 0.")

    # Time above upper (toxic)
    toxic_intervals = _find_crossing_intervals_above(times_h, concentrations, upper_threshold)
    time_toxic = _total_interval_duration(toxic_intervals)

    # Time above lower
    above_lower_intervals = _find_crossing_intervals_above(
        times_h, concentrations, lower_threshold
    )
    time_above_lower = _total_interval_duration(above_lower_intervals)

    # Time in therapeutic = time above lower - time above upper
    time_therapeutic = max(0.0, time_above_lower - time_toxic)
    time_subtherapeutic = max(0.0, obs_duration - time_above_lower)

    pct_therapeutic = time_therapeutic / obs_duration * 100.0
    pct_toxic = time_toxic / obs_duration * 100.0
    pct_subtherapeutic = time_subtherapeutic / obs_duration * 100.0

    notes: list[str] = []
    if pct_therapeutic < 50.0:
        notes.append(
            f"Only {pct_therapeutic:.1f}% of dosing interval is within therapeutic range."
        )
    if pct_toxic > 20.0:
        notes.append(f"Concentration exceeds upper threshold for {pct_toxic:.1f}% of the time.")

    return CoverageResult(
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        time_therapeutic_h=time_therapeutic,
        time_toxic_h=time_toxic,
        time_subtherapeutic_h=time_subtherapeutic,
        pct_therapeutic=pct_therapeutic,
        pct_toxic=pct_toxic,
        pct_subtherapeutic=pct_subtherapeutic,
        notes=tuple(notes),
    )


def pk_pd_breakpoint_analysis(
    times_h: list[float],
    concentrations: list[float],
    mic_mg_L: float,
    pkpd_index: str = "auc_mic",
) -> AntibioticPKPDResult:
    """PK/PD breakpoint analysis for antibiotic dosing.

    Parameters
    ----------
    times_h:
        Sampling times (h), monotonically increasing.
    concentrations:
        Drug concentrations (mg/L), non-negative.
    mic_mg_L:
        Minimum inhibitory concentration (mg/L), must be > 0.
    pkpd_index:
        PK/PD driver: "auc_mic" (AUC/MIC >= 125), "t_above_mic" (fT>MIC >= 40%),
        or "cmax_mic" (Cmax/MIC >= 8).

    Returns
    -------
    AntibioticPKPDResult
        All three PK/PD indices with target attainment assessment.
    """
    valid_indices = {"auc_mic", "t_above_mic", "cmax_mic"}
    if pkpd_index not in valid_indices:
        raise ValueError(f"pkpd_index must be one of {valid_indices}, got '{pkpd_index}'.")
    _validate_mic(mic_mg_L)
    if len(times_h) < 2:
        raise ValueError("At least 2 data points are required.")
    if len(times_h) != len(concentrations):
        raise ValueError("times_h and concentrations must have the same length.")
    for i in range(1, len(times_h)):
        if times_h[i] <= times_h[i - 1]:
            raise ValueError("times_h must be strictly monotonically increasing.")
    if any(c < 0 for c in concentrations):
        raise ValueError("concentrations must be >= 0.")

    notes: list[str] = []

    # AUC/MIC
    auc_total = _auc_lin_log_trapezoidal(times_h, concentrations)
    auc_mic_ratio = auc_total / mic_mg_L

    # fT>MIC
    obs_duration = times_h[-1] - times_h[0]
    intervals_above = _find_crossing_intervals_above(times_h, concentrations, mic_mg_L)
    time_above_mic = _total_interval_duration(intervals_above)
    ft_above_mic_pct = (time_above_mic / obs_duration * 100.0) if obs_duration > 0 else 0.0

    # Cmax/MIC
    cmax = max(concentrations)
    cmax_mic_ratio = cmax / mic_mg_L

    # Target attainment thresholds (standard PK/PD breakpoints)
    TARGET_AUC_MIC = 125.0  # AUC/MIC >= 125 (fluoroquinolones, aminoglycosides)
    TARGET_FT_MIC = 40.0  # fT>MIC >= 40% (beta-lactams)
    TARGET_CMAX_MIC_LOW = 8.0  # Cmax/MIC >= 8 (aminoglycosides)
    TARGET_CMAX_MIC_HIGH = 12.0  # Cmax/MIC >= 12 (optimal)

    if pkpd_index == "auc_mic":
        target_attained = auc_mic_ratio >= TARGET_AUC_MIC
        notes.append(
            f"AUC/MIC target: >= {TARGET_AUC_MIC}; achieved {auc_mic_ratio:.1f}."
        )
    elif pkpd_index == "t_above_mic":
        target_attained = ft_above_mic_pct >= TARGET_FT_MIC
        notes.append(
            f"fT>MIC target: >= {TARGET_FT_MIC}%; achieved {ft_above_mic_pct:.1f}%."
        )
    else:  # cmax_mic
        target_attained = cmax_mic_ratio >= TARGET_CMAX_MIC_LOW
        if cmax_mic_ratio >= TARGET_CMAX_MIC_HIGH:
            notes.append(
                f"Cmax/MIC {cmax_mic_ratio:.1f} >= {TARGET_CMAX_MIC_HIGH} (optimal)."
            )
        else:
            notes.append(
                f"Cmax/MIC target: >= {TARGET_CMAX_MIC_LOW}; achieved {cmax_mic_ratio:.1f}."
            )

    return AntibioticPKPDResult(
        mic_mg_L=mic_mg_L,
        pkpd_index=pkpd_index,
        auc_total=auc_total,
        auc_mic_ratio=auc_mic_ratio,
        fT_above_mic_pct=ft_above_mic_pct,
        cmax_mic_ratio=cmax_mic_ratio,
        target_attained=target_attained,
        notes=tuple(notes),
    )
