"""Exposure-response modeling for clinical PK/PD analysis.

Models the relationship between drug exposure metrics (AUC, Cmax) and
clinical response using Emax, linear, and logistic models.

References
----------
- Sheiner LB, Steimer JL. Clin Pharmacol Ther 2000;68:564-575.
- Bhattaram VA et al. AAPS J 2005;7:E503-E512.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "ExposureResponseResult",
    "fit_exposure_response",
    "quartile_analysis",
]

_VALID_MODEL_TYPES = ("emax", "linear", "logistic")


@dataclass
class ExposureResponseResult:
    """Results from exposure-response model fitting.

    Attributes
    ----------
    model_type : Model used ("emax", "linear", "logistic").
    n_subjects : Number of subjects/observations.
    fitted_params : Dictionary of fitted parameters.
    r_squared : Coefficient of determination (R² for emax/linear) or
        pseudo-R² (McFadden) for logistic.
    ed50_or_ec50 : EC50 for emax model, None for others.
    predicted_responses : Model-predicted responses at each exposure.
    notes : Interpretation notes.
    """

    model_type: str
    n_subjects: int
    fitted_params: dict
    r_squared: float
    ed50_or_ec50: float | None
    predicted_responses: list[float] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def _ss_tot(responses: list[float]) -> float:
    mu = _mean(responses)
    return sum((r - mu) ** 2 for r in responses)


def _r_squared(responses: list[float], predicted: list[float]) -> float:
    ss_res = sum((r - p) ** 2 for r, p in zip(responses, predicted, strict=True))
    ss_tot = _ss_tot(responses)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _emax_predict(x: float, emax: float, ec50: float, n: float) -> float:
    """Hill Emax equation."""
    if x <= 0.0:
        return 0.0
    xn = x**n
    ec50n = ec50**n
    return emax * xn / (ec50n + xn)


def _fit_emax(exposures: list[float], responses: list[float]) -> dict:
    """Grid search over Emax, EC50, n for Hill Emax model."""
    best_sse = math.inf
    best_params = {"emax": 1.0, "ec50": 1.0, "n": 1.0}

    max_resp = max(responses)
    max_exp = max(exposures)
    min_exp_pos = min(e for e in exposures if e > 0) if any(e > 0 for e in exposures) else 0.01

    # Build grids
    emax_grid = [max_resp * f for f in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0)]
    ec50_grid = [min_exp_pos * f for f in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)]
    ec50_grid = [e for e in ec50_grid if e <= max_exp * 2]
    n_grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    for emax in emax_grid:
        for ec50 in ec50_grid:
            for n in n_grid:
                sse = sum(
                    (_emax_predict(x, emax, ec50, n) - r) ** 2
                    for x, r in zip(exposures, responses, strict=True)
                )
                if sse < best_sse:
                    best_sse = sse
                    best_params = {"emax": emax, "ec50": ec50, "n": n}

    # Local refinement via gradient-free coordinate descent
    emax = best_params["emax"]
    ec50 = best_params["ec50"]
    n = best_params["n"]

    for _ in range(200):
        improved = False
        for param_name in ("emax", "ec50", "n"):
            current = {"emax": emax, "ec50": ec50, "n": n}[param_name]
            for factor in (0.95, 1.05, 0.9, 1.1, 0.8, 1.2):
                candidate = current * factor
                if candidate <= 0:
                    continue
                trial = {"emax": emax, "ec50": ec50, "n": n}
                trial[param_name] = candidate
                sse = sum(
                    (_emax_predict(x, trial["emax"], trial["ec50"], trial["n"]) - r) ** 2
                    for x, r in zip(exposures, responses, strict=True)
                )
                if sse < best_sse:
                    best_sse = sse
                    if param_name == "emax":
                        emax = candidate
                    elif param_name == "ec50":
                        ec50 = candidate
                    else:
                        n = candidate
                    improved = True
                    break
        if not improved:
            break

    return {"emax": emax, "ec50": ec50, "n": n}


def _fit_linear(exposures: list[float], responses: list[float]) -> dict:
    """OLS linear regression: E = a + b * X."""
    n = len(exposures)
    sx = sum(exposures)
    sy = sum(responses)
    sxy = sum(x * y for x, y in zip(exposures, responses, strict=True))
    sxx = sum(x * x for x in exposures)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-15:
        b = 0.0
        a = sy / n
    else:
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
    return {"a": a, "b": b}


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _fit_logistic(exposures: list[float], responses: list[float]) -> dict:
    """Gradient descent logistic regression: P = 1/(1+exp(-(a+b*X)))."""
    # Validate binary responses
    vals = set(responses)
    if not vals.issubset({0.0, 1.0}):
        # Binarize at median
        med = sorted(responses)[len(responses) // 2]
        responses = [1.0 if r >= med else 0.0 for r in responses]

    a = 0.0
    b = 0.0
    lr = 0.01
    n = len(exposures)

    for _ in range(1000):
        grad_a = 0.0
        grad_b = 0.0
        for x, y in zip(exposures, responses, strict=True):
            p = _sigmoid(a + b * x)
            err = p - y
            grad_a += err
            grad_b += err * x
        a -= lr * grad_a / n
        b -= lr * grad_b / n

    return {"a": a, "b": b}


def _log_likelihood_null(responses: list[float]) -> float:
    """Log-likelihood of null logistic model (intercept only)."""
    n = len(responses)
    p_bar = sum(responses) / n
    p_bar = max(1e-12, min(1 - 1e-12, p_bar))
    return sum(
        y * math.log(p_bar) + (1 - y) * math.log(1 - p_bar) for y in responses
    )


def _log_likelihood_logistic(
    exposures: list[float], responses: list[float], a: float, b: float
) -> float:
    ll = 0.0
    for x, y in zip(exposures, responses, strict=True):
        p = _sigmoid(a + b * x)
        p = max(1e-12, min(1 - 1e-12, p))
        ll += y * math.log(p) + (1 - y) * math.log(1 - p)
    return ll


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_exposure_response(
    exposures: list[float],
    responses: list[float],
    model_type: str,
) -> ExposureResponseResult:
    """Fit an exposure-response model to paired exposure/response data.

    Parameters
    ----------
    exposures:
        Drug exposure values (e.g., AUC in mg·h/L or Cmax in mg/L).
        Must have at least 3 observations.
    responses:
        Observed responses paired with each exposure value.
        For "logistic", values should be binary (0 or 1).
    model_type:
        One of "emax" (Hill Emax), "linear" (OLS), or "logistic"
        (binary logistic regression via gradient descent).

    Returns
    -------
    ExposureResponseResult
        Fitted parameters, goodness-of-fit, and predicted values.

    Raises
    ------
    ValueError
        If inputs are invalid, lengths differ, or model_type is unknown.
    """
    if not exposures:
        raise ValueError("exposures must not be empty")
    if not responses:
        raise ValueError("responses must not be empty")
    if len(exposures) != len(responses):
        raise ValueError(
            f"exposures and responses must have equal length, got "
            f"{len(exposures)} vs {len(responses)}"
        )
    if len(exposures) < 3:
        raise ValueError("At least 3 observations required for fitting")
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"model_type must be one of {_VALID_MODEL_TYPES}, got '{model_type}'"
        )
    if any(not math.isfinite(v) for v in exposures):
        raise ValueError("exposures contains non-finite values")
    if any(not math.isfinite(v) for v in responses):
        raise ValueError("responses contains non-finite values")

    exposures = list(exposures)
    responses = list(responses)
    n = len(exposures)

    if model_type == "emax":
        params = _fit_emax(exposures, responses)
        predicted = [
            _emax_predict(x, params["emax"], params["ec50"], params["n"])
            for x in exposures
        ]
        r2 = _r_squared(responses, predicted)
        ec50 = params["ec50"]
        notes = (
            f"Hill Emax model: Emax={params['emax']:.3g}, "
            f"EC50={ec50:.3g}, n={params['n']:.3g}. R²={r2:.3f}"
        )
        return ExposureResponseResult(
            model_type=model_type,
            n_subjects=n,
            fitted_params=params,
            r_squared=r2,
            ed50_or_ec50=ec50,
            predicted_responses=predicted,
            notes=notes,
        )

    elif model_type == "linear":
        params = _fit_linear(exposures, responses)
        predicted = [params["a"] + params["b"] * x for x in exposures]
        r2 = _r_squared(responses, predicted)
        notes = (
            f"Linear model: E = {params['a']:.3g} + {params['b']:.3g}·X. "
            f"R²={r2:.3f}"
        )
        return ExposureResponseResult(
            model_type=model_type,
            n_subjects=n,
            fitted_params=params,
            r_squared=r2,
            ed50_or_ec50=None,
            predicted_responses=predicted,
            notes=notes,
        )

    else:  # logistic
        params = _fit_logistic(exposures, responses)
        a, b = params["a"], params["b"]
        predicted = [_sigmoid(a + b * x) for x in exposures]

        # McFadden pseudo-R²
        ll_model = _log_likelihood_logistic(exposures, responses, a, b)
        ll_null = _log_likelihood_null(responses)
        if abs(ll_null) < 1e-15:
            pseudo_r2 = 0.0
        else:
            pseudo_r2 = max(0.0, 1.0 - ll_model / ll_null)

        notes = (
            f"Logistic model: P = sigmoid({a:.3g} + {b:.3g}·X). "
            f"McFadden R²={pseudo_r2:.3f}"
        )
        return ExposureResponseResult(
            model_type=model_type,
            n_subjects=n,
            fitted_params=params,
            r_squared=pseudo_r2,
            ed50_or_ec50=None,
            predicted_responses=predicted,
            notes=notes,
        )


def quartile_analysis(
    exposures: list[float],
    responses: list[float],
) -> dict:
    """Divide subjects into exposure quartiles and summarize response.

    Parameters
    ----------
    exposures:
        Drug exposure values (any units).
    responses:
        Observed response for each subject.

    Returns
    -------
    dict
        Keys:
        - "quartile_boundaries": [Q25, Q50, Q75] exposure thresholds
        - "quartile_means": dict Q1..Q4 → mean response
        - "quartile_sizes": dict Q1..Q4 → number of subjects
        - "fold_change_q4_q1": Q4 mean / Q1 mean (or None if Q1 mean ≈ 0)
        - "notes": interpretation string

    Raises
    ------
    ValueError
        If inputs have different lengths or fewer than 4 observations.
    """
    if not exposures:
        raise ValueError("exposures must not be empty")
    if len(exposures) != len(responses):
        raise ValueError(
            f"exposures and responses must have equal length, got "
            f"{len(exposures)} vs {len(responses)}"
        )
    if len(exposures) < 4:
        raise ValueError("At least 4 observations required for quartile analysis")

    pairs = sorted(zip(exposures, responses, strict=True), key=lambda p: p[0])

    # Quartile boundaries (25th, 50th, 75th percentile of exposures)
    sorted_exp = [p[0] for p in pairs]
    q25 = _percentile(sorted_exp, 25)
    q50 = _percentile(sorted_exp, 50)
    q75 = _percentile(sorted_exp, 75)

    def _quartile_idx(x: float) -> str:
        if x <= q25:
            return "Q1"
        elif x <= q50:
            return "Q2"
        elif x <= q75:
            return "Q3"
        else:
            return "Q4"

    groups: dict[str, list[float]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for exp, resp in pairs:
        groups[_quartile_idx(exp)].append(resp)

    quartile_means: dict[str, float] = {}
    quartile_sizes: dict[str, int] = {}
    for q in ("Q1", "Q2", "Q3", "Q4"):
        vals = groups[q]
        quartile_sizes[q] = len(vals)
        quartile_means[q] = _mean(vals) if vals else 0.0

    q1_mean = quartile_means["Q1"]
    q4_mean = quartile_means["Q4"]
    if abs(q1_mean) < 1e-15:
        fold_change: float | None = None
    else:
        fold_change = q4_mean / q1_mean

    notes = (
        f"Quartile boundaries: Q25={q25:.3g}, Q50={q50:.3g}, Q75={q75:.3g}. "
        f"Q1 mean={q1_mean:.3g}, Q4 mean={q4_mean:.3g}. "
        f"Fold-change Q4/Q1: {fold_change:.2f}" if fold_change is not None
        else f"Quartile boundaries: Q25={q25:.3g}, Q50={q50:.3g}, Q75={q75:.3g}. "
             "Q1 mean ≈ 0; fold-change undefined."
    )

    return {
        "quartile_boundaries": [q25, q50, q75],
        "quartile_means": quartile_means,
        "quartile_sizes": quartile_sizes,
        "fold_change_q4_q1": fold_change,
        "notes": notes,
    }


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear interpolation percentile of a sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = pct / 100.0 * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])
