"""Cosolvent Solubility Enhancement Predictor — Phase 818.

Implements the Yalkowsky log-linear cosolvency model:
    log10(S_mix) = log10(S_water) + sigma * f_cosolvent

where sigma is the cosolvency power of the cosolvent for the given drug logP.

Provides:
- CosolventSolubilityResult dataclass
- predict_cosolvent_solubility()
- optimize_cosolvent_fraction() via grid search
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Cosolvent database
# ---------------------------------------------------------------------------

# name → (sigma_slope, sigma_max, max_safe_fraction)
# sigma = slope * logP, clamped to [0, sigma_max]
_COSOLVENT_DB: dict[str, tuple[float, float, float]] = {
    "PEG400": (0.9, 4.0, 0.5),
    "Ethanol": (1.0, 5.0, 0.1),
    "Propylene Glycol": (0.7, 3.5, 0.5),
    "Glycerol": (0.4, 2.0, 0.6),
    "DMSO": (1.2, 5.0, 0.1),
}

_UNKNOWN_SLOPE = 0.5
_UNKNOWN_SIGMA_MAX = 3.0
_UNKNOWN_MAX_SAFE_FRACTION = 0.3

# Grid search step size for optimize_cosolvent_fraction
_GRID_STEP = 0.05


def _get_cosolvent_params(cosolvent_name: str) -> tuple[float, float, float]:
    """Return (slope, sigma_max, max_safe_fraction) for the named cosolvent."""
    if cosolvent_name in _COSOLVENT_DB:
        return _COSOLVENT_DB[cosolvent_name]
    return (_UNKNOWN_SLOPE, _UNKNOWN_SIGMA_MAX, _UNKNOWN_MAX_SAFE_FRACTION)


def _compute_sigma(slope: float, sigma_max: float, logp: float) -> float:
    """Compute cosolvency power: slope * logP clamped to [0, sigma_max]."""
    raw = slope * logp
    return max(0.0, min(sigma_max, raw))


def _log_linear_solubility(
    s_water_mg_ml: float,
    sigma: float,
    fraction: float,
) -> float:
    """Return S_mix (mg/mL) using log-linear cosolvency model."""
    log_s_mix = math.log10(s_water_mg_ml) + sigma * fraction
    return 10.0**log_s_mix


@dataclass(frozen=True)
class CosolventSolubilityResult:
    """Result of a cosolvent solubility enhancement prediction."""

    drug_name: str
    cosolvent_name: str
    cosolvent_fraction: float  # volume fraction 0-1
    aqueous_solubility_mg_mL: float
    enhanced_solubility_mg_mL: float
    enhancement_fold: float
    sigma: float  # cosolvency power
    max_enhancement_fold: float  # at max safe fraction
    max_safe_fraction: float
    formulation_feasible: bool  # enhanced_solubility >= target_mg_mL
    notes: str


def predict_cosolvent_solubility(
    drug_name: str,
    logp: float,
    aqueous_solubility_mg_mL: float,
    cosolvent_name: str,
    cosolvent_fraction: float,
    target_mg_mL: float = 1.0,
) -> CosolventSolubilityResult:
    """Predict solubility enhancement using the Yalkowsky log-linear cosolvency model.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logp:
        Octanol-water partition coefficient.
    aqueous_solubility_mg_mL:
        Intrinsic aqueous solubility in mg/mL (must be > 0).
    cosolvent_name:
        Name of cosolvent (see _COSOLVENT_DB for recognised entries).
    cosolvent_fraction:
        Volume fraction of cosolvent in the final mixture (must be in [0, 1]).
    target_mg_mL:
        Desired solubility target in mg/mL for feasibility assessment.

    Returns
    -------
    CosolventSolubilityResult

    Raises
    ------
    ValueError
        If cosolvent_fraction is outside [0, 1] or aqueous_solubility_mg_mL <= 0.
    """
    if not (0.0 <= cosolvent_fraction <= 1.0):
        raise ValueError(f"cosolvent_fraction must be in [0, 1]; got {cosolvent_fraction}")
    if aqueous_solubility_mg_mL <= 0.0:
        raise ValueError(f"aqueous_solubility_mg_mL must be > 0; got {aqueous_solubility_mg_mL}")

    slope, sigma_max, max_safe_frac = _get_cosolvent_params(cosolvent_name)
    sigma = _compute_sigma(slope, sigma_max, logp)

    enhanced = _log_linear_solubility(aqueous_solubility_mg_mL, sigma, cosolvent_fraction)
    fold = enhanced / aqueous_solubility_mg_mL

    max_enhanced = _log_linear_solubility(aqueous_solubility_mg_mL, sigma, max_safe_frac)
    max_fold = max_enhanced / aqueous_solubility_mg_mL

    feasible = enhanced >= target_mg_mL

    known = cosolvent_name in _COSOLVENT_DB
    note_parts = [
        f"sigma={sigma:.3f} (logP={logp:.2f})",
        f"f={cosolvent_fraction:.2f}",
        f"S_water={aqueous_solubility_mg_mL:.4f} mg/mL",
        f"S_mix={enhanced:.4f} mg/mL",
    ]
    if not known:
        note_parts.append(f"cosolvent '{cosolvent_name}' not in database; default parameters used")

    return CosolventSolubilityResult(
        drug_name=drug_name,
        cosolvent_name=cosolvent_name,
        cosolvent_fraction=cosolvent_fraction,
        aqueous_solubility_mg_mL=aqueous_solubility_mg_mL,
        enhanced_solubility_mg_mL=enhanced,
        enhancement_fold=fold,
        sigma=sigma,
        max_enhancement_fold=max_fold,
        max_safe_fraction=max_safe_frac,
        formulation_feasible=feasible,
        notes="; ".join(note_parts),
    )


def optimize_cosolvent_fraction(
    drug_name: str,
    logp: float,
    aqueous_solubility_mg_mL: float,
    cosolvent_name: str,
    target_mg_mL: float = 1.0,
) -> CosolventSolubilityResult:
    """Find the minimum cosolvent fraction that achieves target_mg_mL via grid search.

    Grid searches fractions [0, 0.05, 0.10, ..., max_safe_fraction].
    Returns the first fraction at which the target is achieved, or the result at
    max_safe_fraction if the target is never met.

    Parameters
    ----------
    drug_name, logp, aqueous_solubility_mg_mL, cosolvent_name, target_mg_mL:
        Same as predict_cosolvent_solubility.

    Returns
    -------
    CosolventSolubilityResult at the optimal (or maximum safe) fraction.
    """
    if aqueous_solubility_mg_mL <= 0.0:
        raise ValueError(f"aqueous_solubility_mg_mL must be > 0; got {aqueous_solubility_mg_mL}")

    _, _, max_safe_frac = _get_cosolvent_params(cosolvent_name)

    # Build grid: 0.00, 0.05, 0.10, ..., max_safe_fraction
    n_steps = int(round(max_safe_frac / _GRID_STEP))
    fractions: list[float] = []
    for i in range(n_steps + 1):
        f = round(i * _GRID_STEP, 10)
        if f <= max_safe_frac + 1e-9:
            fractions.append(min(f, max_safe_frac))

    # Ensure max_safe_frac is included
    if not fractions or abs(fractions[-1] - max_safe_frac) > 1e-9:
        fractions.append(max_safe_frac)

    best_fraction = max_safe_frac
    for f in fractions:
        result = predict_cosolvent_solubility(
            drug_name, logp, aqueous_solubility_mg_mL, cosolvent_name, f, target_mg_mL
        )
        if result.formulation_feasible:
            best_fraction = f
            break

    return predict_cosolvent_solubility(
        drug_name, logp, aqueous_solubility_mg_mL, cosolvent_name, best_fraction, target_mg_mL
    )
