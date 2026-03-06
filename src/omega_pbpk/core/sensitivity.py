"""Local sensitivity analysis for PK parameters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SensitivityResult",
    "local_sensitivity",
    "normalized_sensitivity_matrix",
    "tornado_plot_data",
]


@dataclass
class SensitivityResult:
    parameter_name: str
    nominal_value: float
    perturbed_up_value: float
    perturbed_down_value: float
    output_metric: str
    nominal_output: float
    output_up: float
    output_down: float
    sensitivity_index: float
    relative_sensitivity: float
    rank: int = field(default=0)


def _perturb_simulate(
    param_dict: dict,
    param_name: str,
    direction: str,
    h: float,
    simulate_fn: Callable[[dict], float],
) -> float:
    """Perturb a single parameter and call simulate_fn, returning the result."""
    perturbed = dict(param_dict)
    nominal = perturbed[param_name]
    if direction == "up":
        perturbed[param_name] = nominal * (1.0 + h)
    else:
        perturbed[param_name] = nominal * (1.0 - h)
    return float(simulate_fn(perturbed))


def local_sensitivity(
    simulate_fn: Callable[[dict], float],
    param_dict: dict,
    output_metric: str = "auc",
    h: float = 0.1,
) -> list[SensitivityResult]:
    """Compute local (one-at-a-time) sensitivity for each parameter.

    The normalized sensitivity index is:
        S = (output_up - output_down) / (2 * nominal_output) / (2 * h)

    Parameters
    ----------
    simulate_fn:
        Callable that accepts a parameter dict and returns a scalar float.
    param_dict:
        Dict mapping parameter names to their nominal values.
    output_metric:
        Label for the output quantity (e.g. 'auc', 'cmax').
    h:
        Fractional perturbation size (0 < h < 1).

    Returns
    -------
    List of SensitivityResult sorted by abs(sensitivity_index) descending,
    with rank assigned starting at 1.
    """
    if h <= 0 or h >= 1:
        raise ValueError(f"h must be in (0, 1), got {h}")

    nominal_output = float(simulate_fn(param_dict))
    results: list[SensitivityResult] = []

    for name, nominal_value in param_dict.items():
        output_up = _perturb_simulate(param_dict, name, "up", h, simulate_fn)
        output_down = _perturb_simulate(param_dict, name, "down", h, simulate_fn)

        if nominal_output != 0.0:
            sensitivity_index = (output_up - output_down) / (2.0 * nominal_output) / h
        else:
            sensitivity_index = float(np.nan)

        results.append(
            SensitivityResult(
                parameter_name=name,
                nominal_value=float(nominal_value),
                perturbed_up_value=float(nominal_value) * (1.0 + h),
                perturbed_down_value=float(nominal_value) * (1.0 - h),
                output_metric=output_metric,
                nominal_output=nominal_output,
                output_up=output_up,
                output_down=output_down,
                sensitivity_index=sensitivity_index,
                relative_sensitivity=sensitivity_index,
            )
        )

    results.sort(
        key=lambda r: abs(r.sensitivity_index) if not np.isnan(r.sensitivity_index) else -1.0,
        reverse=True,
    )
    for rank, result in enumerate(results, start=1):
        result.rank = rank

    return results


def normalized_sensitivity_matrix(
    simulate_fn: Callable[[dict], float],
    param_dict: dict,
    output_metrics: list[str],
    h: float = 0.1,
) -> dict[str, list[SensitivityResult]]:
    """Compute sensitivity for each output metric.

    Parameters
    ----------
    simulate_fn:
        Callable that accepts a parameter dict and returns a float.
        The caller is responsible for providing a wrapper per metric.
    param_dict:
        Dict of parameter names to nominal values.
    output_metrics:
        List of output metric labels.
    h:
        Fractional perturbation size.

    Returns
    -------
    Dict keyed by metric name, each value being a list of SensitivityResult.
    """
    if h <= 0 or h >= 1:
        raise ValueError(f"h must be in (0, 1), got {h}")

    return {
        metric: local_sensitivity(simulate_fn, param_dict, output_metric=metric, h=h)
        for metric in output_metrics
    }


def tornado_plot_data(sensitivity_results: list[SensitivityResult]) -> list[dict]:
    """Produce tornado plot data from sensitivity results.

    Returns a list of dicts with keys:
        param_name, low_output, high_output, delta

    Sorted by abs(delta) descending.
    """
    rows = [
        {
            "param_name": r.parameter_name,
            "low_output": r.output_down,
            "high_output": r.output_up,
            "delta": r.output_up - r.output_down,
        }
        for r in sensitivity_results
    ]
    rows.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return rows
