"""
Phase 659 — PK Sensitivity Ranking

One-at-a-time (OAT) sensitivity analysis for ranking PK parameters by their
influence on drug exposure outcomes (Cmax, AUC, t½).

Science:
    OAT sensitivity index: S_i = (ΔY/Y_base) / (ΔX_i/X_i_base)
    Rank parameters by |S_i| descending.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityIndex:
    """Standardized sensitivity index for a single parameter."""

    parameter_name: str
    nominal_value: float
    perturbed_low: float
    perturbed_high: float
    outcome_at_low: float
    outcome_at_high: float
    outcome_at_nominal: float
    s_index: float  # standardized sensitivity index
    abs_s_index: float  # |s_index|
    rank: int  # rank by |s_index| (1 = most influential)
    direction: str  # "positive" or "negative" (effect direction)
    notes: str


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_sensitivity_index(
    param_name: str,
    nominal_value: float,
    low_value: float,
    high_value: float,
    outcome_low: float,
    outcome_high: float,
    outcome_nominal: float,
) -> SensitivityIndex:
    """Compute the standardized sensitivity index for a single parameter.

    Parameters
    ----------
    param_name:
        Name of the parameter.
    nominal_value:
        Baseline (central) value of the parameter.
    low_value:
        Perturbed low value (must be < nominal_value).
    high_value:
        Perturbed high value (must be > nominal_value).
    outcome_low:
        Outcome evaluated at low_value.
    outcome_high:
        Outcome evaluated at high_value.
    outcome_nominal:
        Outcome evaluated at nominal_value.

    Returns
    -------
    SensitivityIndex
        Dataclass with s_index, abs_s_index, direction, rank placeholder 0.
    """
    if low_value >= nominal_value:
        raise ValueError(
            f"low_value ({low_value}) must be strictly less than nominal_value ({nominal_value})"
        )
    if nominal_value >= high_value:
        raise ValueError(
            f"nominal_value ({nominal_value}) must be strictly less than high_value ({high_value})"
        )

    delta_y = outcome_high - outcome_low
    # Normalized perturbation: ΔX/X = (high - low) / nominal
    # This gives the standard elasticity S = (ΔY/Y) / (ΔX/X)
    delta_x_normalized = (high_value - low_value) / nominal_value

    if outcome_nominal != 0.0 and delta_x_normalized != 0.0:
        s = (delta_y / outcome_nominal) / delta_x_normalized
    else:
        s = 0.0

    abs_s = abs(s)
    direction = "positive" if s >= 0.0 else "negative"

    notes = (
        f"OAT sensitivity for '{param_name}': nominal={nominal_value:.4g}, "
        f"range=[{low_value:.4g}, {high_value:.4g}], "
        f"outcome range=[{outcome_low:.4g}, {outcome_high:.4g}], "
        f"S={s:.4f}"
    )

    return SensitivityIndex(
        parameter_name=param_name,
        nominal_value=nominal_value,
        perturbed_low=low_value,
        perturbed_high=high_value,
        outcome_at_low=outcome_low,
        outcome_at_high=outcome_high,
        outcome_at_nominal=outcome_nominal,
        s_index=s,
        abs_s_index=abs_s,
        rank=0,  # placeholder; assigned by rank_pk_sensitivity
        direction=direction,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


def rank_pk_sensitivity(
    param_ranges: dict[str, tuple[float, float, float]],  # {name: (low, nominal, high)}
    outcome_function: Callable[[dict[str, float]], float],
    outcome_name: str = "outcome",
) -> list[SensitivityIndex]:
    """Rank parameters by their OAT sensitivity on a given outcome.

    Each parameter is perturbed to its low and high values while all other
    parameters are held at their nominal values.

    Parameters
    ----------
    param_ranges:
        Mapping of parameter name to (low, nominal, high) tuple.
    outcome_function:
        Callable ``f(param_dict) -> float`` that evaluates the outcome.
    outcome_name:
        Descriptive label for the outcome (used in notes).

    Returns
    -------
    list[SensitivityIndex]
        Sorted by abs_s_index descending (most influential first), with ranks
        assigned 1..N.
    """
    if not param_ranges:
        raise ValueError("param_ranges must contain at least one parameter")

    # Build nominal dict
    nominal_dict: dict[str, float] = {name: trio[1] for name, trio in param_ranges.items()}

    # Evaluate outcome at full nominal
    outcome_nominal = outcome_function(nominal_dict)

    raw_results: list[SensitivityIndex] = []

    for param_name, (low_val, nominal_val, high_val) in param_ranges.items():
        # Evaluate at low
        low_dict = dict(nominal_dict)
        low_dict[param_name] = low_val
        outcome_low = outcome_function(low_dict)

        # Evaluate at high
        high_dict = dict(nominal_dict)
        high_dict[param_name] = high_val
        outcome_high = outcome_function(high_dict)

        si = compute_sensitivity_index(
            param_name=param_name,
            nominal_value=nominal_val,
            low_value=low_val,
            high_value=high_val,
            outcome_low=outcome_low,
            outcome_high=outcome_high,
            outcome_nominal=outcome_nominal,
        )
        raw_results.append(si)

    # Sort by abs_s_index descending, then by param name for stability
    raw_results.sort(key=lambda x: (-x.abs_s_index, x.parameter_name))

    # Assign ranks (1-based, unique)
    ranked: list[SensitivityIndex] = []
    for rank_idx, si in enumerate(raw_results, start=1):
        ranked.append(
            SensitivityIndex(
                parameter_name=si.parameter_name,
                nominal_value=si.nominal_value,
                perturbed_low=si.perturbed_low,
                perturbed_high=si.perturbed_high,
                outcome_at_low=si.outcome_at_low,
                outcome_at_high=si.outcome_at_high,
                outcome_at_nominal=si.outcome_at_nominal,
                s_index=si.s_index,
                abs_s_index=si.abs_s_index,
                rank=rank_idx,
                direction=si.direction,
                notes=si.notes + f" | outcome='{outcome_name}'",
            )
        )

    return ranked


def sensitivity_heatmap_data(
    param_ranges: dict[str, tuple[float, float, float]],
    outcome_functions: dict[str, Callable[[dict[str, float]], float]],
) -> dict[str, list[SensitivityIndex]]:
    """Run OAT sensitivity analysis for multiple outcomes.

    Parameters
    ----------
    param_ranges:
        Mapping of parameter name to (low, nominal, high).
    outcome_functions:
        Dict of {outcome_name: callable}.

    Returns
    -------
    dict[str, list[SensitivityIndex]]
        Keys are outcome names; values are ranked SensitivityIndex lists.
    """
    results: dict[str, list[SensitivityIndex]] = {}
    for outcome_name, func in outcome_functions.items():
        results[outcome_name] = rank_pk_sensitivity(
            param_ranges=param_ranges,
            outcome_function=func,
            outcome_name=outcome_name,
        )
    return results
