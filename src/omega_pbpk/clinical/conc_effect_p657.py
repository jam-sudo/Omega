"""Concentration-effect relationship modeling.

Models concentration-effect (C-E) relationships using linear, Emax,
sigmoidal Emax (Hill), and indirect response pharmacodynamic models.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConcEffectResult:
    """Result of concentration-effect relationship modeling."""

    drug_name: str
    model_type: str
    ec50_mg_L: float
    emax: float
    hill_coefficient: float
    baseline_effect: float
    effect_at_ec50: float
    ec90_mg_L: float
    ec10_mg_L: float
    therapeutic_index_approx: float
    model_equation: str
    notes: str


_VALID_MODELS = {"linear", "emax", "sigmoid", "indirect"}


def _evaluate_single(
    model_type: str,
    ec50: float,
    emax: float,
    hill_n: float,
    baseline: float,
    conc: float,
) -> float:
    """Evaluate a single concentration through the specified model."""
    if model_type == "linear":
        return baseline + emax * conc / ec50
    if model_type == "emax":
        return baseline + emax * conc / (ec50 + conc)
    if model_type == "sigmoid":
        return baseline + emax * conc**hill_n / (ec50**hill_n + conc**hill_n)
    # indirect (inhibitory)
    return baseline + emax * (1.0 - conc / (ec50 + conc))


def model_concentration_effect(
    drug_name: str,
    model_type: str,
    ec50_mg_L: float,
    emax: float = 1.0,
    hill_n: float = 1.0,
    baseline_effect: float = 0.0,
    toxic_effect_threshold: float | None = None,
) -> ConcEffectResult:
    """Model a concentration-effect relationship.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    model_type : str
        One of "linear", "emax", "sigmoid", "indirect".
    ec50_mg_L : float
        Concentration producing 50% of Emax (mg/L). Must be > 0.
    emax : float
        Maximum effect magnitude. Must be > 0.
    hill_n : float
        Hill coefficient (steepness). Must be > 0.
    baseline_effect : float
        Baseline (no-drug) effect level.
    toxic_effect_threshold : float or None
        If provided, used to estimate therapeutic index.

    Returns
    -------
    ConcEffectResult
    """
    if model_type not in _VALID_MODELS:
        raise ValueError(f"model_type must be one of {sorted(_VALID_MODELS)}, got '{model_type}'")
    if ec50_mg_L <= 0:
        raise ValueError(f"ec50_mg_L must be > 0, got {ec50_mg_L}")
    if emax <= 0:
        raise ValueError(f"emax must be > 0, got {emax}")
    if hill_n <= 0:
        raise ValueError(f"hill_n must be > 0, got {hill_n}")

    effect_at_ec50 = _evaluate_single(
        model_type, ec50_mg_L, emax, hill_n, baseline_effect, ec50_mg_L
    )

    # EC90 and EC10 calculations
    if model_type == "sigmoid":
        ec90 = ec50_mg_L * 9.0 ** (1.0 / hill_n)
        ec10 = ec50_mg_L * (1.0 / 9.0) ** (1.0 / hill_n)
    elif model_type in ("emax", "indirect"):
        ec90 = ec50_mg_L * 9.0
        ec10 = ec50_mg_L / 9.0
    else:  # linear
        ec90 = ec50_mg_L * 9.0
        ec10 = ec50_mg_L / 9.0

    # Therapeutic index
    if toxic_effect_threshold is not None:
        ti = toxic_effect_threshold / ec50_mg_L
    else:
        ti = -1.0

    # Model equation string
    equations = {
        "linear": f"E(C) = {baseline_effect} + {emax} * C / {ec50_mg_L}",
        "emax": f"E(C) = {baseline_effect} + {emax} * C / ({ec50_mg_L} + C)",
        "sigmoid": (
            f"E(C) = {baseline_effect} + {emax} * C^{hill_n} / ({ec50_mg_L}^{hill_n} + C^{hill_n})"
        ),
        "indirect": (f"E(C) = {baseline_effect} + {emax} * (1 - C / ({ec50_mg_L} + C))"),
    }

    notes_parts = [f"{model_type} model"]
    if model_type == "sigmoid" and hill_n != 1.0:
        notes_parts.append(f"Hill coefficient = {hill_n}")
    if toxic_effect_threshold is not None:
        notes_parts.append(f"TI approx = {ti:.2f}")

    return ConcEffectResult(
        drug_name=drug_name,
        model_type=model_type,
        ec50_mg_L=ec50_mg_L,
        emax=emax,
        hill_coefficient=hill_n,
        baseline_effect=baseline_effect,
        effect_at_ec50=effect_at_ec50,
        ec90_mg_L=ec90,
        ec10_mg_L=ec10,
        therapeutic_index_approx=ti,
        model_equation=equations[model_type],
        notes="; ".join(notes_parts),
    )


def evaluate_effect(
    drug_name: str,
    model_type: str,
    ec50_mg_L: float,
    concentrations: list[float],
    emax: float = 1.0,
    hill_n: float = 1.0,
    baseline_effect: float = 0.0,
) -> list[tuple[float, float]]:
    """Evaluate effect at multiple concentrations.

    Returns list of (concentration, effect) tuples.
    """
    if model_type not in _VALID_MODELS:
        raise ValueError(f"model_type must be one of {sorted(_VALID_MODELS)}, got '{model_type}'")
    if ec50_mg_L <= 0:
        raise ValueError(f"ec50_mg_L must be > 0, got {ec50_mg_L}")

    return [
        (c, _evaluate_single(model_type, ec50_mg_L, emax, hill_n, baseline_effect, c))
        for c in concentrations
    ]
