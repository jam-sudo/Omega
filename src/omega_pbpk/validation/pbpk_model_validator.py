"""PBPK model validation against observed PK data — Phase 512."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PKValidationResult:
    """Quantitative validation metrics comparing PBPK predictions to observed PK data."""

    n_points: int
    aafe: float
    afe: float
    mfe: float
    within_2fold_pct: float
    within_3fold_pct: float
    rmse: float
    r_squared: float
    mean_pred_obs_ratio: float
    model_bias: str
    validation_passed: bool
    notes: str


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def _validate_paired(observed: list[float], predicted: list[float]) -> None:
    """Validate that observed and predicted are non-empty, same length, and positive."""
    if len(observed) != len(predicted):
        raise ValueError(
            f"observed and predicted must have the same length, "
            f"got {len(observed)} vs {len(predicted)}"
        )
    if len(observed) == 0:
        raise ValueError("observed and predicted must not be empty")
    for i, (o, p) in enumerate(zip(observed, predicted)):
        if o <= 0:
            raise ValueError(f"observed[{i}] must be positive, got {o}")
        if p <= 0:
            raise ValueError(f"predicted[{i}] must be positive, got {p}")


def calculate_aafe(observed: list[float], predicted: list[float]) -> float:
    """Compute Average Absolute Fold Error.

    AAFE = 10^( (1/n) * sum(|log10(P/O)|) )

    AAFE = 1.0 is perfect; AAFE < 2.0 is acceptable for PBPK.
    """
    _validate_paired(observed, predicted)
    n = len(observed)
    mean_abs_log = sum(abs(math.log10(p / o)) for o, p in zip(observed, predicted)) / n
    return 10.0**mean_abs_log


def calculate_afe(observed: list[float], predicted: list[float]) -> float:
    """Compute Average Fold Error (signed bias).

    AFE = 10^( (1/n) * sum(log10(P/O)) )

    AFE > 1 → over-prediction; AFE < 1 → under-prediction.
    """
    _validate_paired(observed, predicted)
    n = len(observed)
    mean_log = sum(math.log10(p / o) for o, p in zip(observed, predicted)) / n
    return 10.0**mean_log


def calculate_mfe(observed: list[float], predicted: list[float]) -> float:
    """Compute Mean Fold Error (arithmetic mean of P/O ratios).

    MFE = (1/n) * sum(P/O)
    """
    _validate_paired(observed, predicted)
    n = len(observed)
    return sum(p / o for o, p in zip(observed, predicted)) / n


def within_fold(
    observed: list[float],
    predicted: list[float],
    fold: float = 2.0,
) -> float:
    """Compute fraction of predictions within n-fold of observed.

    A prediction is within n-fold if |log10(P/O)| <= log10(fold).

    Returns fraction in [0, 1].
    """
    _validate_paired(observed, predicted)
    if fold <= 1.0:
        raise ValueError(f"fold must be > 1.0, got {fold}")
    threshold = math.log10(fold)
    n = len(observed)
    count = sum(1 for o, p in zip(observed, predicted) if abs(math.log10(p / o)) <= threshold)
    return count / n


# ---------------------------------------------------------------------------
# Linear interpolation helper
# ---------------------------------------------------------------------------


def _interpolate(x_known: list[float], y_known: list[float], x_query: float) -> float | None:
    """Linear interpolation/extrapolation for a single query point.

    Returns None if x_known is empty or query is outside range (no extrapolation).
    """
    if len(x_known) < 2:
        return None
    if x_query < x_known[0] or x_query > x_known[-1]:
        return None
    # Find bracket
    for i in range(len(x_known) - 1):
        x0, x1 = x_known[i], x_known[i + 1]
        if x0 <= x_query <= x1:
            if x1 == x0:
                return y_known[i]
            t = (x_query - x0) / (x1 - x0)
            return y_known[i] + t * (y_known[i + 1] - y_known[i])
    return None


# ---------------------------------------------------------------------------
# Profile-level validation
# ---------------------------------------------------------------------------


def validate_pk_profile(
    observed_times: list[float],
    observed_concs: list[float],
    predicted_times: list[float],
    predicted_concs: list[float],
) -> PKValidationResult:
    """Validate a full PK concentration-time profile.

    Interpolates predicted concentrations to observed timepoints, then computes
    all metrics. Observed and predicted must have corresponding positive values.

    Parameters
    ----------
    observed_times:
        Timepoints of observed data (must be sorted ascending).
    observed_concs:
        Observed plasma concentrations at observed_times. Must be positive.
    predicted_times:
        Timepoints for predicted profile (must be sorted ascending).
    predicted_concs:
        Predicted concentrations at predicted_times. Must be positive.

    Returns
    -------
    PKValidationResult with full set of validation metrics.
    """
    if len(observed_times) != len(observed_concs):
        raise ValueError("observed_times and observed_concs must have the same length")
    if len(predicted_times) != len(predicted_concs):
        raise ValueError("predicted_times and predicted_concs must have the same length")
    if len(observed_times) == 0:
        raise ValueError("observed profile must not be empty")
    if len(predicted_times) == 0:
        raise ValueError("predicted profile must not be empty")

    # Validate positive values
    for i, c in enumerate(observed_concs):
        if c <= 0:
            raise ValueError(f"observed_concs[{i}] must be positive, got {c}")
    for i, c in enumerate(predicted_concs):
        if c <= 0:
            raise ValueError(f"predicted_concs[{i}] must be positive, got {c}")

    # Interpolate predicted to observed timepoints
    obs_list: list[float] = []
    pred_list: list[float] = []

    for t, c_obs in zip(observed_times, observed_concs):
        c_pred = _interpolate(predicted_times, predicted_concs, t)
        if c_pred is not None and c_pred > 0:
            obs_list.append(c_obs)
            pred_list.append(c_pred)

    if len(obs_list) == 0:
        raise ValueError("No overlapping timepoints between observed and predicted profiles")

    n = len(obs_list)
    aafe = calculate_aafe(obs_list, pred_list)
    afe = calculate_afe(obs_list, pred_list)
    mfe = calculate_mfe(obs_list, pred_list)
    w2 = within_fold(obs_list, pred_list, 2.0) * 100.0
    w3 = within_fold(obs_list, pred_list, 3.0) * 100.0

    # RMSE
    rmse = math.sqrt(sum((p - o) ** 2 for o, p in zip(obs_list, pred_list)) / n)

    # R² on log10 scale (Pearson r²)
    log_obs = [math.log10(o) for o in obs_list]
    log_pred = [math.log10(p) for p in pred_list]
    mean_lo = sum(log_obs) / n
    mean_lp = sum(log_pred) / n
    ss_xy = sum((lo - mean_lo) * (lp - mean_lp) for lo, lp in zip(log_obs, log_pred))
    ss_xx = sum((lo - mean_lo) ** 2 for lo in log_obs)
    ss_yy = sum((lp - mean_lp) ** 2 for lp in log_pred)
    denom = math.sqrt(ss_xx * ss_yy)
    r_squared = (ss_xy / denom) ** 2 if denom > 0 else 0.0

    # Mean P/O ratio
    mean_ratio = sum(p / o for o, p in zip(obs_list, pred_list)) / n

    # Model bias
    if afe > 1.05:
        model_bias = "over-prediction"
    elif afe < 0.952:  # 1/1.05
        model_bias = "under-prediction"
    else:
        model_bias = "balanced"

    # Acceptance criteria (FDA/EMA PBPK guidance)
    validation_passed = aafe < 2.0 and w2 > 50.0

    notes_parts = []
    if aafe < 2.0:
        notes_parts.append(f"AAFE={aafe:.2f} (acceptable, <2.0)")
    else:
        notes_parts.append(f"AAFE={aafe:.2f} (exceeds 2.0 threshold)")
    notes_parts.append(f"{w2:.1f}% within 2-fold")
    notes_parts.append(f"Bias: {model_bias}")
    notes = "; ".join(notes_parts)

    return PKValidationResult(
        n_points=n,
        aafe=aafe,
        afe=afe,
        mfe=mfe,
        within_2fold_pct=w2,
        within_3fold_pct=w3,
        rmse=rmse,
        r_squared=r_squared,
        mean_pred_obs_ratio=mean_ratio,
        model_bias=model_bias,
        validation_passed=validation_passed,
        notes=notes,
    )


def acceptance_criteria_check(
    aafe: float,
    afe: float,
    within_2fold_pct: float,
) -> dict:
    """Check whether PBPK model meets FDA/EMA acceptance criteria.

    Parameters
    ----------
    aafe:
        Average Absolute Fold Error (should be < 2.0).
    afe:
        Average Fold Error (signed; ideally near 1.0).
    within_2fold_pct:
        Percentage of predictions within 2-fold of observed (should be > 50%).

    Returns
    -------
    dict with keys: 'passed', 'aafe_passed', 'within_2fold_passed',
    'bias_acceptable', 'details'.
    """
    aafe_passed = aafe < 2.0
    within_2fold_passed = within_2fold_pct > 50.0
    bias_acceptable = 0.5 <= afe <= 2.0
    passed = aafe_passed and within_2fold_passed

    details = {
        "aafe": aafe,
        "aafe_threshold": 2.0,
        "aafe_passed": aafe_passed,
        "afe": afe,
        "bias_acceptable": bias_acceptable,
        "within_2fold_pct": within_2fold_pct,
        "within_2fold_threshold": 50.0,
        "within_2fold_passed": within_2fold_passed,
        "passed": passed,
    }
    return details
