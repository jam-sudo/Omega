"""Exposure-response (ER) modeling linking drug exposure metrics to efficacy or toxicity probability."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ExposureResponseResult",
    "fit_logistic_er",
    "predict_response_probability",
    "er_quartile_analysis",
]


@dataclass(frozen=True)
class ExposureResponseResult:
    """Result of exposure-response model fitting."""

    compound_name: str
    model_type: str
    alpha: float
    beta: float
    ec50: float
    auc_roc: float
    n_subjects: int
    notes: str


def predict_response_probability(exposure: float, alpha: float, beta: float) -> float:
    """Return P = 1 / (1 + exp(-(alpha + beta * exposure)))."""
    arg = alpha + beta * exposure
    # Clamp to avoid overflow
    if arg > 500:
        return 1.0
    if arg < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-arg))


def _neg_log_likelihood(
    exposures: list[float], responses: list[int], alpha: float, beta: float
) -> float:
    """Compute negative log-likelihood for logistic regression."""
    nll = 0.0
    eps = 1e-15
    for x, y in zip(exposures, responses):
        p = predict_response_probability(x, alpha, beta)
        p = max(eps, min(1.0 - eps, p))
        nll += -y * math.log(p) - (1 - y) * math.log(1 - p)
    return nll


def _compute_auc_roc(
    exposures: list[float], responses: list[int], alpha: float, beta: float
) -> float:
    """Compute AUC-ROC using trapezoidal rule over threshold grid."""
    # Generate predicted probabilities
    probs = [predict_response_probability(x, alpha, beta) for x in exposures]

    # Collect thresholds from 0 to 1 in 101 steps
    thresholds = [i / 100.0 for i in range(101)]

    roc_points = []
    for thresh in thresholds:
        tp = sum(1 for p, y in zip(probs, responses) if p >= thresh and y == 1)
        fp = sum(1 for p, y in zip(probs, responses) if p >= thresh and y == 0)
        fn = sum(1 for p, y in zip(probs, responses) if p < thresh and y == 1)
        tn = sum(1 for p, y in zip(probs, responses) if p < thresh and y == 0)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        roc_points.append((fpr, tpr))

    # Sort by fpr ascending
    roc_points.sort(key=lambda pt: pt[0])

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(roc_points)):
        x0, y0 = roc_points[i - 1]
        x1, y1 = roc_points[i]
        auc += 0.5 * (y0 + y1) * (x1 - x0)

    return max(0.0, min(1.0, auc))


def fit_logistic_er(
    exposures: list[float],
    responses: list[int],
    n_grid: int = 200,
) -> dict:
    """Fit logistic exposure-response model via grid search.

    Parameters
    ----------
    exposures:
        List of exposure values (AUC, Cmax, etc.). Must be >= 0.
    responses:
        Binary response indicators (0 or 1).
    n_grid:
        Number of grid steps for alpha and beta each.

    Returns
    -------
    dict with keys: alpha, beta, ec50, hill, auc_roc, n_subjects, notes
    """
    if len(exposures) != len(responses):
        raise ValueError(
            f"exposures and responses must have equal length, got {len(exposures)} vs {len(responses)}"
        )
    if len(exposures) < 4:
        raise ValueError(f"Need at least 4 subjects for ER fitting, got {len(exposures)}")
    if any(r not in (0, 1) for r in responses):
        raise ValueError("All responses must be 0 or 1")
    if any(x < 0 for x in exposures):
        raise ValueError("All exposures must be >= 0")

    best_alpha = 0.0
    best_beta = 0.0
    best_nll = float("inf")

    alpha_min, alpha_max = -10.0, 10.0
    beta_min, beta_max = -5.0, 5.0

    alpha_step = (alpha_max - alpha_min) / (n_grid - 1) if n_grid > 1 else 0.0
    beta_step = (beta_max - beta_min) / (n_grid - 1) if n_grid > 1 else 0.0

    for i in range(n_grid):
        alpha = alpha_min + i * alpha_step
        for j in range(n_grid):
            beta = beta_min + j * beta_step
            nll = _neg_log_likelihood(exposures, responses, alpha, beta)
            if nll < best_nll:
                best_nll = nll
                best_alpha = alpha
                best_beta = beta

    # Compute EC50: exposure at which P=0.5, i.e. alpha + beta*ec50 = 0
    if abs(best_beta) > 1e-12:
        ec50 = -best_alpha / best_beta
    else:
        ec50 = float("nan")

    auc_roc = _compute_auc_roc(exposures, responses, best_alpha, best_beta)

    notes = (
        f"Logistic ER model fitted via grid search ({n_grid}x{n_grid}). "
        f"NLL={best_nll:.3f}. AUC-ROC={auc_roc:.3f}."
    )

    return {
        "alpha": best_alpha,
        "beta": best_beta,
        "ec50": ec50,
        "hill": 1,
        "auc_roc": auc_roc,
        "n_subjects": len(exposures),
        "notes": notes,
    }


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) from sorted list using linear interpolation."""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_vals[0]
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def er_quartile_analysis(exposures: list[float], responses: list[int]) -> dict:
    """Split exposures into quartiles and compute response rate per quartile.

    Parameters
    ----------
    exposures:
        List of exposure values.
    responses:
        Binary response indicators (0 or 1).

    Returns
    -------
    dict with keys: quartile_bounds (5 boundaries), response_rates (4 rates), trend, notes
    """
    if len(exposures) != len(responses):
        raise ValueError(
            f"exposures and responses must have equal length, got {len(exposures)} vs {len(responses)}"
        )
    if len(exposures) < 4:
        raise ValueError(f"Need at least 4 subjects for quartile analysis, got {len(exposures)}")

    sorted_exposures = sorted(exposures)
    q0 = sorted_exposures[0]
    q1 = _percentile(sorted_exposures, 25.0)
    q2 = _percentile(sorted_exposures, 50.0)
    q3 = _percentile(sorted_exposures, 75.0)
    q4 = sorted_exposures[-1]

    quartile_bounds = [q0, q1, q2, q3, q4]

    # Compute response rates per quartile
    response_rates: list[float] = []
    quartile_pairs = [
        (q0, q1),
        (q1, q2),
        (q2, q3),
        (q3, q4),
    ]

    paired = list(zip(exposures, responses))
    for idx, (lo, hi) in enumerate(quartile_pairs):
        if idx < 3:
            # Include lo, exclude hi (except last quartile)
            members = [(x, y) for x, y in paired if lo <= x < hi]
        else:
            # Last quartile: include both endpoints
            members = [(x, y) for x, y in paired if lo <= x <= hi]

        if len(members) == 0:
            response_rates.append(float("nan"))
        else:
            rate = sum(y for _, y in members) / len(members)
            response_rates.append(rate)

    # Determine trend from valid rates
    valid_rates = [r for r in response_rates if not math.isnan(r)]
    if len(valid_rates) >= 2:
        first = valid_rates[0]
        last = valid_rates[-1]
        diff = last - first
        if diff > 0.1:
            trend = "increasing"
        elif diff < -0.1:
            trend = "decreasing"
        else:
            trend = "flat"
    else:
        trend = "flat"

    notes = (
        f"Quartile analysis of {len(exposures)} subjects. "
        f"Bounds: [{q0:.3g}, {q1:.3g}, {q2:.3g}, {q3:.3g}, {q4:.3g}]. "
        f"Trend: {trend}."
    )

    return {
        "quartile_bounds": quartile_bounds,
        "response_rates": response_rates,
        "trend": trend,
        "notes": notes,
    }
