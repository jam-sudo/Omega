"""Dose-response analysis tools — Phase 457."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DoseResponseResult",
    "four_param_logistic",
    "fit_dose_response",
    "calculate_ec50",
    "calculate_selectivity_ratio",
    "hill_equation",
    "inhibition_constant",
]


@dataclass
class DoseResponseResult:
    """Result from dose-response model fitting."""

    model: str
    ec50: float
    hill_slope: float
    bottom: float
    top: float
    emax: float
    doses: list
    fitted_responses: list
    r_squared: float
    notes: str


def four_param_logistic(
    dose: float,
    bottom: float,
    top: float,
    ec50: float,
    hill_slope: float,
) -> float:
    """4-parameter logistic (4PL) dose-response model.

    Parameters
    ----------
    dose:
        Dose or concentration value (must be >= 0).
    bottom:
        Lower asymptote (minimum response).
    top:
        Upper asymptote (maximum response).
    ec50:
        Dose producing 50% effect (midpoint of the curve).
    hill_slope:
        Hill slope (steepness of the sigmoid).

    Returns
    -------
    float
        Predicted response at the given dose.
    """
    if ec50 <= 0:
        raise ValueError("ec50 must be positive")
    if dose < 0:
        raise ValueError("dose must be non-negative")
    if dose == 0:
        return bottom
    # 4PL: bottom + (top - bottom) / (1 + (ec50/dose)^hill_slope)
    ratio = (ec50 / dose) ** hill_slope
    return bottom + (top - bottom) / (1.0 + ratio)


def _r_squared(observed: list, fitted: list) -> float:
    """Compute coefficient of determination R²."""
    n = len(observed)
    if n < 2:
        return 1.0
    mean_obs = sum(observed) / n
    ss_tot = sum((y - mean_obs) ** 2 for y in observed)
    ss_res = sum((observed[i] - fitted[i]) ** 2 for i in range(n))
    if ss_tot == 0.0:
        return 1.0
    r2 = 1.0 - ss_res / ss_tot
    # Clamp to [0, 1] for poorly constrained fits
    return max(0.0, min(1.0, r2))


def _golden_section_search(
    f,
    lo: float,
    hi: float,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """Minimize f on [lo, hi] using golden-section search."""
    phi = (math.sqrt(5) - 1) / 2  # golden ratio conjugate ≈ 0.618
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - phi * (b - a)
        d = a + phi * (b - a)
    return (a + b) / 2.0


def fit_dose_response(
    doses: list,
    responses: list,
    model: str = "4pl",
) -> DoseResponseResult:
    """Fit a dose-response model to data.

    Parameters
    ----------
    doses:
        Sequence of dose/concentration values (must be positive).
    responses:
        Corresponding response measurements.
    model:
        Model type; currently only '4pl' is supported.

    Returns
    -------
    DoseResponseResult
        Fitted model parameters and goodness-of-fit metrics.
    """
    if not doses or not responses:
        raise ValueError("doses and responses must not be empty")
    if len(doses) != len(responses):
        raise ValueError("doses and responses must have the same length")
    if any(d <= 0 for d in doses):
        raise ValueError("all dose values must be positive")
    if model not in ("4pl",):
        raise ValueError(f"unsupported model '{model}'; choose '4pl'")

    doses_f = [float(d) for d in doses]
    responses_f = [float(r) for r in responses]

    bottom = min(responses_f)
    top = max(responses_f)

    # Find EC50 via golden-section on log-dose space
    log_doses = [math.log10(d) for d in doses_f]
    lo_log = min(log_doses) - 1.0
    hi_log = max(log_doses) + 1.0

    def _ec50_objective(log_ec50: float) -> float:
        ec50_candidate = 10.0 ** log_ec50
        # Use Hill slope = 1 for EC50 search (refine afterwards)
        fitted = [
            four_param_logistic(d, bottom, top, ec50_candidate, 1.0)
            for d in doses_f
        ]
        return sum((responses_f[i] - fitted[i]) ** 2 for i in range(len(responses_f)))

    best_log_ec50 = _golden_section_search(_ec50_objective, lo_log, hi_log)
    ec50 = 10.0 ** best_log_ec50

    # Estimate Hill slope from shape: use two points straddling EC50
    # slope = log(81) / log(d90/d10) where d90, d10 are doses at 90% and 10%
    # Approximate using observed data or fall back to 1.0
    hill_slope = 1.0
    if len(doses_f) >= 3:
        # Numerically search Hill slope that minimises residuals
        def _hs_objective(hs: float) -> float:
            if hs <= 0:
                return 1e12
            fitted = [
                four_param_logistic(d, bottom, top, ec50, hs)
                for d in doses_f
            ]
            return sum((responses_f[i] - fitted[i]) ** 2 for i in range(len(responses_f)))

        hill_slope = _golden_section_search(_hs_objective, 0.1, 10.0)

    fitted_responses = [
        four_param_logistic(d, bottom, top, ec50, hill_slope) for d in doses_f
    ]
    r2 = _r_squared(responses_f, fitted_responses)

    emax = top - bottom

    return DoseResponseResult(
        model=model,
        ec50=ec50,
        hill_slope=hill_slope,
        bottom=bottom,
        top=top,
        emax=emax,
        doses=doses_f,
        fitted_responses=fitted_responses,
        r_squared=r2,
        notes="Fit via golden-section search; bottom/top fixed to min/max of data.",
    )


def calculate_ec50(doses: list, responses: list) -> float:
    """Estimate EC50 from dose-response data using a 4PL fit.

    Parameters
    ----------
    doses:
        Dose values (must be positive).
    responses:
        Response values corresponding to each dose.

    Returns
    -------
    float
        Estimated EC50.
    """
    result = fit_dose_response(doses, responses, model="4pl")
    return result.ec50


def calculate_selectivity_ratio(
    ec50_target: float,
    ec50_off_target: float,
) -> float:
    """Compute selectivity ratio (EC50_target / EC50_off_target).

    A ratio < 1 indicates the compound is more potent at the target
    (lower EC50) than at the off-target, which is the desired case.

    Parameters
    ----------
    ec50_target:
        EC50 at the primary target of interest.
    ec50_off_target:
        EC50 at an off-target or selectivity counterpart.

    Returns
    -------
    float
        Selectivity ratio (lower is more selective).
    """
    if ec50_target <= 0:
        raise ValueError("ec50_target must be positive")
    if ec50_off_target <= 0:
        raise ValueError("ec50_off_target must be positive")
    return ec50_target / ec50_off_target


def hill_equation(
    concentration: float,
    emax: float,
    ec50: float,
    n: float = 1.0,
) -> float:
    """Hill / Emax sigmoid equation.

    E(C) = Emax * C^n / (EC50^n + C^n)

    At C = EC50, E = Emax / 2 regardless of n.

    Parameters
    ----------
    concentration:
        Drug concentration (must be >= 0).
    emax:
        Maximum achievable effect.
    ec50:
        Concentration producing 50% of Emax.
    n:
        Hill exponent (steepness).

    Returns
    -------
    float
        Effect at the given concentration.
    """
    if ec50 <= 0:
        raise ValueError("ec50 must be positive")
    if concentration < 0:
        raise ValueError("concentration must be non-negative")
    if n <= 0:
        raise ValueError("Hill exponent n must be positive")
    if concentration == 0.0:
        return 0.0
    cn = concentration ** n
    ec50n = ec50 ** n
    return emax * cn / (ec50n + cn)


def inhibition_constant(
    ic50: float,
    hill_slope: float = 1.0,
    substrate_conc: float | None = None,
    km: float | None = None,
) -> float:
    """Calculate inhibition constant Ki via the Cheng-Prusoff equation.

    When substrate_conc and km are provided the competitive correction
    is applied:  Ki = IC50 / (1 + [S]/Km).

    Without substrate data the simplified form is used:
        Ki = IC50 / (2 ^ (1 / hill_slope))
    which reduces to Ki = IC50 / 2 when hill_slope = 1 and [S] = Km.

    Parameters
    ----------
    ic50:
        Observed half-maximal inhibitory concentration.
    hill_slope:
        Hill slope of the inhibition curve (default 1.0).
    substrate_conc:
        Substrate concentration used in the assay (optional).
    km:
        Michaelis constant for the substrate (optional).

    Returns
    -------
    float
        Estimated Ki.
    """
    if ic50 <= 0:
        raise ValueError("ic50 must be positive")
    if hill_slope <= 0:
        raise ValueError("hill_slope must be positive")

    if substrate_conc is not None and km is not None:
        if km <= 0:
            raise ValueError("km must be positive")
        if substrate_conc < 0:
            raise ValueError("substrate_conc must be non-negative")
        # Cheng-Prusoff: Ki = IC50 / (1 + [S]/Km)
        return ic50 / (1.0 + substrate_conc / km)

    # Simplified: Ki = IC50 / 2^(1/hill_slope)
    return ic50 / (2.0 ** (1.0 / hill_slope))
