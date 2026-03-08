"""Hill equation fitting for dose-response pharmacodynamic relationships — Phase 192.

Fits the Hill equation to dose-response data via least-squares grid search
and characterizes the pharmacodynamic relationship (cooperativity, TI, etc.).

Hill equation: E(D) = Emax * D^n / (EC50^n + D^n)

References
----------
- Hill AV, J Physiol. 1910;40:iv-vii (original Hill equation)
- Gesztelyi R et al., Arch Hist Exact Sci. 2012;66(4):427-38 (history/review)
- FDA Guidance for Industry: Exposure-Response Relationships (2003)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HillFitResult:
    """Result of Hill equation fitting to dose-response data.

    Attributes
    ----------
    drug_name : Name of the drug.
    emax : Fitted maximum effect.
    ec50 : Fitted half-maximal effective concentration.
    hill_n : Fitted Hill coefficient (shape/cooperativity parameter).
    r_squared : Coefficient of determination for the fit (0-1).
    cooperativity_class : 'sub-Hill' (n<1), 'Michaelis-Menten' (n=1), or 'cooperative' (n>1).
    fitted_responses : Fitted effect values at input doses (tuple).
    residuals : Residuals (observed - fitted) at input doses (tuple).
    notes : Summary notes.
    """

    drug_name: str
    emax: float
    ec50: float
    hill_n: float
    r_squared: float
    cooperativity_class: str
    fitted_responses: tuple
    residuals: tuple
    notes: str


def _hill(dose: float, emax: float, ec50: float, n: float) -> float:
    """Evaluate Hill equation at a single dose."""
    if dose <= 0.0:
        return 0.0
    dn = dose**n
    ec50n = ec50**n
    return emax * dn / (ec50n + dn)


def _r_squared(observed: list, fitted: list) -> float:
    """Calculate coefficient of determination R²."""
    n = len(observed)
    if n == 0:
        return 0.0
    mean_obs = sum(observed) / n
    ss_tot = sum((y - mean_obs) ** 2 for y in observed)
    ss_res = sum((o - f) ** 2 for o, f in zip(observed, fitted))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _classify_cooperativity(hill_n: float) -> str:
    """Classify cooperativity from Hill coefficient."""
    if hill_n < 0.95:
        return "sub-Hill"
    elif hill_n <= 1.05:
        return "Michaelis-Menten"
    else:
        return "cooperative"


def fit_hill_equation(
    drug_name: str,
    doses: list,
    responses: list,
    emax_steps: int = 10,
    ec50_steps: int = 20,
    n_steps_count: int = 15,
) -> HillFitResult:
    """Fit Hill equation to dose-response data via grid search.

    Grid search over:
    - Emax: max(responses)*0.5 to max(responses)*2.0 in emax_steps steps
    - EC50: geometric range from min(doses) to max(doses) in ec50_steps steps
    - n: 0.5 to 4.0 in steps of 0.25 (n_steps_count steps)

    Parameters
    ----------
    drug_name : str
        Name of the drug/compound.
    doses : list[float]
        Dose levels (must be > 0, same length as responses).
    responses : list[float]
        Observed effect values (must be > 0, same length as doses).
    emax_steps : int
        Number of grid points for Emax search.
    ec50_steps : int
        Number of grid points for EC50 search (geometric spacing).
    n_steps_count : int
        Number of grid points for Hill coefficient search.

    Returns
    -------
    HillFitResult

    Raises
    ------
    ValueError
        If doses and responses have different lengths, or responses contain
        non-positive values, or fewer than 2 data points provided.
    """
    # --- Validation ---
    if len(doses) != len(responses):
        raise ValueError(
            f"doses and responses must have the same length, got {len(doses)} vs {len(responses)}"
        )
    if len(doses) < 2:
        raise ValueError("At least 2 data points required for fitting")
    if any(r <= 0 for r in responses):
        raise ValueError("All response values must be > 0")
    if any(d <= 0 for d in doses):
        raise ValueError("All dose values must be > 0")

    max_resp = max(responses)
    min_dose = min(doses)
    max_dose = max(doses)

    # Build grid
    emax_grid = [
        max_resp * 0.5 + (max_resp * 1.5) * i / max(emax_steps - 1, 1) for i in range(emax_steps)
    ]
    # Geometric spacing for EC50
    log_min = math.log(min_dose)
    log_max = math.log(max_dose)
    ec50_grid = [
        math.exp(log_min + (log_max - log_min) * i / max(ec50_steps - 1, 1))
        for i in range(ec50_steps)
    ]
    # Hill n from 0.5 to 4.0
    n_grid = [0.5 + 0.25 * i for i in range(n_steps_count)]

    # Grid search: minimize sum of squared residuals
    best_sse = float("inf")
    best_emax = emax_grid[len(emax_grid) // 2]
    best_ec50 = ec50_grid[len(ec50_grid) // 2]
    best_n = 1.0

    for emax in emax_grid:
        for ec50 in ec50_grid:
            for n in n_grid:
                sse = 0.0
                for d, obs in zip(doses, responses):
                    pred = _hill(d, emax, ec50, n)
                    sse += (obs - pred) ** 2
                if sse < best_sse:
                    best_sse = sse
                    best_emax = emax
                    best_ec50 = ec50
                    best_n = n

    # Compute fitted values and residuals
    fitted = [_hill(d, best_emax, best_ec50, best_n) for d in doses]
    residuals = tuple(obs - fit for obs, fit in zip(responses, fitted))
    r2 = _r_squared(list(responses), fitted)
    coop_class = _classify_cooperativity(best_n)

    # Build notes
    notes_parts = []
    if r2 < 0.9:
        notes_parts.append(f"Poor fit: R²={r2:.3f} < 0.9; consider alternative models")
    if best_n > 3.0:
        notes_parts.append(f"Very high cooperativity: n={best_n:.2f}")
    if r2 >= 0.99:
        notes_parts.append(f"Excellent fit: R²={r2:.3f}")
    notes = "; ".join(notes_parts) if notes_parts else f"Hill fit R²={r2:.3f}"

    return HillFitResult(
        drug_name=drug_name,
        emax=best_emax,
        ec50=best_ec50,
        hill_n=best_n,
        r_squared=r2,
        cooperativity_class=coop_class,
        fitted_responses=tuple(fitted),
        residuals=residuals,
        notes=notes,
    )


def predict_effect(
    drug_name: str,
    dose: float,
    emax: float,
    ec50: float,
    n: float,
) -> float:
    """Predict effect at a given dose using the Hill equation.

    Parameters
    ----------
    drug_name : str
        Name of the drug (used for context/logging).
    dose : float
        Dose at which to predict effect.
    emax : float
        Maximum effect.
    ec50 : float
        Half-maximal effective concentration.
    n : float
        Hill coefficient.

    Returns
    -------
    float
        Predicted effect E(dose).
    """
    return _hill(dose, emax, ec50, n)


def calculate_therapeutic_index(
    drug_name: str,
    ec50_efficacy: float,
    ec50_toxicity: float,
    n_eff: float = 1.0,
    n_tox: float = 1.0,
) -> dict:
    """Calculate therapeutic index from efficacy and toxicity EC50 values.

    TI = EC50_toxicity / EC50_efficacy

    Safety margin classification:
    - TI >= 10  : Wide margin (safe)
    - TI >= 3   : Adequate margin
    - TI >= 1   : Narrow margin
    - TI < 1    : Toxic before effective (dangerous)

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    ec50_efficacy : float
        EC50 for the desired therapeutic effect.
    ec50_toxicity : float
        EC50 for the toxic/adverse effect.
    n_eff : float
        Hill coefficient for efficacy curve.
    n_tox : float
        Hill coefficient for toxicity curve.

    Returns
    -------
    dict
        Keys: drug_name, TI, ec50_efficacy, ec50_toxicity, n_eff, n_tox,
              safety_margin, selectivity_window.
    """
    if ec50_efficacy <= 0:
        raise ValueError("ec50_efficacy must be > 0")
    if ec50_toxicity <= 0:
        raise ValueError("ec50_toxicity must be > 0")

    ti = ec50_toxicity / ec50_efficacy

    if ti >= 10.0:
        safety_margin = "wide"
    elif ti >= 3.0:
        safety_margin = "adequate"
    elif ti >= 1.0:
        safety_margin = "narrow"
    else:
        safety_margin = "dangerous"

    # Selectivity window: dose range where efficacy >> toxicity
    # Approximated as EC50_tox / EC50_eff
    selectivity_window = ti

    return {
        "drug_name": drug_name,
        "TI": ti,
        "ec50_efficacy": ec50_efficacy,
        "ec50_toxicity": ec50_toxicity,
        "n_eff": n_eff,
        "n_tox": n_tox,
        "safety_margin": safety_margin,
        "selectivity_window": selectivity_window,
    }
