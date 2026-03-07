"""Exposure-response modeling for population-level efficacy and safety analysis.

Models population-level exposure-response relationships using linear, Emax,
and logistic models. Supports quartile analysis and therapeutic window identification.

References
----------
- Bhattaram VA et al. AAPS J 2005;7:E503-E512.
- FDA Exposure-Response Relationships: Study Design, Data Analysis, and Regulatory
  Applications (2003).
"""

from __future__ import annotations

import math

__all__ = [
    "fit_linear_er",
    "fit_emax_er",
    "quartile_analysis",
    "probability_of_response",
    "er_safety_window",
]


def _validate_matched_lists(exposures: list, responses: list, name: str = "responses") -> None:
    """Validate that exposures and responses are non-empty and same length."""
    if not exposures:
        raise ValueError("exposures must not be empty")
    if len(exposures) != len(responses):
        raise ValueError(f"exposures and {name} must have the same length")


def fit_linear_er(exposures: list[float], responses: list[float]) -> dict:
    """Fit a linear exposure-response model using OLS.

    Parameters
    ----------
    exposures:
        List of exposure values (e.g. AUC).
    responses:
        List of response values.

    Returns
    -------
    dict with keys: slope, intercept, r_squared, n
    """
    _validate_matched_lists(exposures, responses)
    n = len(exposures)
    if n < 2:
        raise ValueError("At least 2 data points required for linear fit")

    sumx = sum(exposures)
    sumy = sum(responses)
    sumxy = sum(x * y for x, y in zip(exposures, responses))
    sumx2 = sum(x * x for x in exposures)

    denom = n * sumx2 - sumx * sumx
    if abs(denom) < 1e-12:
        # All x values identical — undefined slope
        slope = 0.0
        intercept = sumy / n
    else:
        slope = (n * sumxy - sumx * sumy) / denom
        intercept = (sumy - slope * sumx) / n

    # R-squared
    y_mean = sumy / n
    ss_tot = sum((y - y_mean) ** 2 for y in responses)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(exposures, responses))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "n": n,
    }


def fit_emax_er(
    exposures: list[float],
    responses: list[float],
    emax_init: float | None = None,
) -> dict:
    """Fit an Emax exposure-response model via grid search.

    Model: E = Emax * AUC / (AUC50 + AUC)

    Parameters
    ----------
    exposures:
        List of exposure values.
    responses:
        List of response values.
    emax_init:
        Unused; kept for API compatibility.

    Returns
    -------
    dict with keys: emax, auc50, r_squared, predicted (list)
    """
    _validate_matched_lists(exposures, responses)
    n = len(exposures)
    if n < 2:
        raise ValueError("At least 2 data points required for Emax fit")

    max_resp = max(responses)
    min_exp = min(exposures)
    max_exp = max(exposures)

    # Grid: Emax in [max_response*0.8, max_response*1.5] (5 pts)
    #       AUC50 in [min_exp, max_exp] (10 pts)
    emax_lo = max_resp * 0.8
    emax_hi = max_resp * 1.5
    n_emax = 5
    n_auc50 = 10

    emax_candidates = [
        emax_lo + i * (emax_hi - emax_lo) / max(n_emax - 1, 1) for i in range(n_emax)
    ]
    if max_exp > min_exp:
        auc50_candidates = [
            min_exp + i * (max_exp - min_exp) / max(n_auc50 - 1, 1) for i in range(n_auc50)
        ]
    else:
        auc50_candidates = [min_exp * 0.5 + i * min_exp * 0.1 for i in range(n_auc50)]

    best_sse = float("inf")
    best_emax = emax_hi
    best_auc50 = (min_exp + max_exp) / 2.0

    for emax in emax_candidates:
        for auc50 in auc50_candidates:
            if auc50 <= 0:
                auc50 = 1e-9
            sse = 0.0
            for x, y in zip(exposures, responses):
                pred = emax * x / (auc50 + x)
                sse += (y - pred) ** 2
            if sse < best_sse:
                best_sse = sse
                best_emax = emax
                best_auc50 = auc50

    predicted = [best_emax * x / (best_auc50 + x) for x in exposures]

    y_mean = sum(responses) / n
    ss_tot = sum((y - y_mean) ** 2 for y in responses)
    ss_res = sum((y - p) ** 2 for y, p in zip(responses, predicted))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0

    return {
        "emax": best_emax,
        "auc50": best_auc50,
        "r_squared": r_squared,
        "predicted": predicted,
    }


def quartile_analysis(
    exposures: list[float],
    responses: list[float],
    n_quartiles: int = 4,
) -> list[dict]:
    """Divide exposures into n_quartiles equal-size groups and summarise each.

    Parameters
    ----------
    exposures:
        List of exposure values.
    responses:
        List of response values.
    n_quartiles:
        Number of equally-sized groups (default 4).

    Returns
    -------
    List of dicts: quartile, exp_range, mean_exp, mean_response, n, sem
    """
    _validate_matched_lists(exposures, responses)
    if n_quartiles < 1:
        raise ValueError("n_quartiles must be >= 1")

    n = len(exposures)
    # Sort by exposure
    paired = sorted(zip(exposures, responses), key=lambda t: t[0])

    # Split into n_quartiles roughly equal groups
    result = []
    for q in range(n_quartiles):
        start_idx = q * n // n_quartiles
        end_idx = (q + 1) * n // n_quartiles if q < n_quartiles - 1 else n
        group = paired[start_idx:end_idx]
        if not group:
            continue
        g_exp = [t[0] for t in group]
        g_resp = [t[1] for t in group]
        ng = len(group)
        mean_exp = sum(g_exp) / ng
        mean_resp = sum(g_resp) / ng
        if ng > 1:
            variance = sum((r - mean_resp) ** 2 for r in g_resp) / (ng - 1)
            sem = math.sqrt(variance) / math.sqrt(ng)
        else:
            sem = 0.0

        result.append(
            {
                "quartile": q + 1,
                "exp_range": (min(g_exp), max(g_exp)),
                "mean_exp": mean_exp,
                "mean_response": mean_resp,
                "n": ng,
                "sem": sem,
            }
        )

    return result


def probability_of_response(
    exposures: list[float],
    responses: list[float],
    response_threshold: float,
) -> dict:
    """Fit a logistic E-R model and return response probabilities.

    Binary response is 1 if response >= threshold, else 0.

    Grid search: a in [-5, 5] (10 pts), b in [0, 5] (10 pts).

    Parameters
    ----------
    exposures:
        List of exposure values.
    responses:
        List of observed responses.
    response_threshold:
        Threshold to binarise responses.

    Returns
    -------
    dict: coefficients (a, b), prob_at_exposure, ec50_eff
    """
    _validate_matched_lists(exposures, responses)

    binary = [1 if r >= response_threshold else 0 for r in responses]

    # Grid search over logistic parameters
    n_a = 10
    n_b = 10
    a_vals = [-5.0 + i * 10.0 / max(n_a - 1, 1) for i in range(n_a)]
    b_vals = [0.0 + i * 5.0 / max(n_b - 1, 1) for i in range(n_b)]

    best_nll = float("inf")
    best_a = 0.0
    best_b = 1.0

    for a in a_vals:
        for b in b_vals:
            nll = 0.0
            for x, y in zip(exposures, binary):
                log_odds = a + b * x
                # Clip for numerical stability
                log_odds = max(-500.0, min(500.0, log_odds))
                p = 1.0 / (1.0 + math.exp(-log_odds))
                p = max(1e-12, min(1 - 1e-12, p))
                nll -= y * math.log(p) + (1 - y) * math.log(1 - p)
            if nll < best_nll:
                best_nll = nll
                best_a = a
                best_b = b

    # Probabilities at each unique exposure
    unique_exps = sorted(set(exposures))
    prob_at_exposure = {}
    for x in unique_exps:
        log_odds = best_a + best_b * x
        log_odds = max(-500.0, min(500.0, log_odds))
        prob_at_exposure[x] = 1.0 / (1.0 + math.exp(-log_odds))

    # EC50_eff: exposure where p=0.5 → a + b*x = 0 → x = -a/b
    if abs(best_b) > 1e-12:
        ec50_eff = -best_a / best_b
    else:
        ec50_eff = float("nan")

    return {
        "coefficients": {"a": best_a, "b": best_b},
        "prob_at_exposure": prob_at_exposure,
        "ec50_eff": ec50_eff,
    }


def er_safety_window(
    exposures: list[float],
    efficacy_responses: list[float],
    safety_responses: list[float],
    efficacy_threshold: float,
    safety_threshold: float,
) -> dict:
    """Identify therapeutic window using Emax models for efficacy and safety.

    Parameters
    ----------
    exposures:
        List of exposure values.
    efficacy_responses:
        Efficacy response values at each exposure.
    safety_responses:
        Safety (toxicity) response values at each exposure.
    efficacy_threshold:
        Minimum efficacy response required.
    safety_threshold:
        Maximum acceptable safety response.

    Returns
    -------
    dict: auc_min_efficacy, auc_max_safety, therapeutic_window_exists, safety_margin
    """
    _validate_matched_lists(exposures, efficacy_responses, "efficacy_responses")
    _validate_matched_lists(exposures, safety_responses, "safety_responses")

    eff_fit = fit_emax_er(exposures, efficacy_responses)
    saf_fit = fit_emax_er(exposures, safety_responses)

    eff_emax = eff_fit["emax"]
    eff_auc50 = eff_fit["auc50"]
    saf_emax = saf_fit["emax"]
    saf_auc50 = saf_fit["auc50"]

    # Find AUC_min_efficacy: Emax_eff * AUC / (AUC50_eff + AUC) = efficacy_threshold
    # => AUC = efficacy_threshold * AUC50_eff / (Emax_eff - efficacy_threshold)
    if eff_emax > efficacy_threshold:
        auc_min_efficacy = efficacy_threshold * eff_auc50 / (eff_emax - efficacy_threshold)
    else:
        # Efficacy threshold can never be reached
        auc_min_efficacy = float("inf")

    # Find AUC_max_safety: Emax_saf * AUC / (AUC50_saf + AUC) = safety_threshold
    # => AUC = safety_threshold * AUC50_saf / (Emax_saf - safety_threshold)
    if saf_emax > safety_threshold:
        auc_max_safety = safety_threshold * saf_auc50 / (saf_emax - safety_threshold)
    else:
        # Safety threshold never reached — no upper limit from safety model
        auc_max_safety = float("inf")

    therapeutic_window_exists = auc_min_efficacy < auc_max_safety and auc_min_efficacy != float(
        "inf"
    )

    if auc_min_efficacy > 0 and auc_max_safety != float("inf"):
        safety_margin = auc_max_safety / auc_min_efficacy
    elif auc_max_safety == float("inf") and auc_min_efficacy > 0:
        safety_margin = float("inf")
    else:
        safety_margin = float("nan")

    return {
        "auc_min_efficacy": auc_min_efficacy,
        "auc_max_safety": auc_max_safety,
        "therapeutic_window_exists": therapeutic_window_exists,
        "safety_margin": safety_margin,
    }
