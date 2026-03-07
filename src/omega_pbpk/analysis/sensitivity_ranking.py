"""Parameter sensitivity ranking via one-at-a-time (OAT) analysis."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "SensitivityRankResult",
    "rank_parameters",
    "tornado_chart_data",
]


@dataclass
class SensitivityRankResult:
    """Result of OAT sensitivity ranking."""

    param_names: list[str] = field(default_factory=list)
    sensitivity_indices: list[float] = field(default_factory=list)  # same order as param_names
    ranked_params: list[str] = field(default_factory=list)  # sorted descending
    ranked_indices: list[float] = field(default_factory=list)
    output_key: str = ""
    base_output: float = 0.0
    notes: str = ""


def rank_parameters(
    base_params: dict,
    param_ranges: dict,
    pk_function: Callable,
    output_key: str,
    n_samples: int = 20,
) -> SensitivityRankResult:
    """Rank model parameters by one-at-a-time (OAT) sensitivity analysis.

    Each parameter is varied ±20% from its base value while all others remain
    fixed. The normalized sensitivity index is:

        SI = |output_high - output_low| / (0.4 * base_value) / |output_base|

    where 0.4 * base_value is the total parameter range (±20%).

    Parameters
    ----------
    base_params : dict
        Mapping of parameter name → nominal value. All values must be numeric.
    param_ranges : dict
        Mapping of parameter name → (low, high) tuple specifying the absolute
        range to sweep. If None or not provided for a parameter, ±20% of base
        is used. Keys must be a subset of base_params keys.
    pk_function : callable
        Function that accepts a dict of parameters and returns a dict (or
        object with attribute access) whose ``output_key`` field is the scalar
        metric of interest.
    output_key : str
        Key to extract from the return value of ``pk_function``.
    n_samples : int
        Number of uniformly-spaced samples within each parameter range.
        Must be ≥ 2. Default is 20.

    Returns
    -------
    SensitivityRankResult
        Parameters ranked by sensitivity index (descending).
    """
    if not base_params:
        raise ValueError("base_params must be a non-empty dict.")
    if not isinstance(param_ranges, dict):
        raise ValueError("param_ranges must be a dict.")
    if not callable(pk_function):
        raise ValueError("pk_function must be callable.")
    if not output_key or not isinstance(output_key, str):
        raise ValueError("output_key must be a non-empty string.")
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}.")

    # Validate param_ranges keys are in base_params
    for key in param_ranges:
        if key not in base_params:
            raise ValueError(f"param_ranges key '{key}' not found in base_params.")

    # Evaluate base output
    base_result = pk_function(dict(base_params))
    if isinstance(base_result, dict):
        output_base = float(base_result[output_key])
    else:
        output_base = float(getattr(base_result, output_key))

    if output_base == 0.0:
        raise ValueError(
            f"Base output '{output_key}' is zero — cannot compute normalized sensitivity."
        )

    param_names: list[str] = []
    sensitivity_indices: list[float] = []

    for param, base_val in base_params.items():
        base_val = float(base_val)

        # Determine low/high sweep range
        if param in param_ranges:
            lo, hi = float(param_ranges[param][0]), float(param_ranges[param][1])
        else:
            # Default ±20%
            lo = base_val * 0.80
            hi = base_val * 1.20

        if lo >= hi:
            raise ValueError(f"For param '{param}', low ({lo}) must be < high ({hi}).")

        # Sample n_samples points across [lo, hi]
        step = (hi - lo) / (n_samples - 1) if n_samples > 1 else 0.0
        outputs: list[float] = []
        for i in range(n_samples):
            test_val = lo + i * step
            test_params = dict(base_params)
            test_params[param] = test_val
            result = pk_function(test_params)
            if isinstance(result, dict):
                outputs.append(float(result[output_key]))
            else:
                outputs.append(float(getattr(result, output_key)))

        output_low = outputs[0]
        output_high = outputs[-1]

        # Normalized sensitivity index: |ΔOutput / ΔParam| / |output_base|
        delta_output = abs(output_high - output_low)
        delta_param = hi - lo
        # Avoid division by zero for constant parameters
        if delta_param == 0.0 or abs(output_base) == 0.0:
            si = 0.0
        else:
            si = (delta_output / delta_param) * (abs(base_val) / abs(output_base))

        param_names.append(param)
        sensitivity_indices.append(si)

    # Sort descending by sensitivity index
    paired = sorted(
        zip(param_names, sensitivity_indices, strict=True), key=lambda x: x[1], reverse=True
    )
    ranked_params = [p for p, _ in paired]
    ranked_indices = [s for _, s in paired]

    notes = (
        f"OAT sensitivity with ±20% perturbation (or custom range); "
        f"output: '{output_key}'; base_output={output_base:.4g}; "
        f"n_params={len(param_names)}; n_samples={n_samples}."
    )

    return SensitivityRankResult(
        param_names=param_names,
        sensitivity_indices=sensitivity_indices,
        ranked_params=ranked_params,
        ranked_indices=ranked_indices,
        output_key=output_key,
        base_output=output_base,
        notes=notes,
    )


def tornado_chart_data(result: SensitivityRankResult) -> list[dict]:
    """Generate data for a tornado (butterfly) chart from a sensitivity result.

    Returns a list of dicts, one per parameter (in ranked order, most sensitive
    first), each containing:

        - ``param``       : parameter name
        - ``sensitivity`` : sensitivity index
        - ``rank``        : 1-based rank (1 = most sensitive)

    Parameters
    ----------
    result : SensitivityRankResult
        Output from :func:`rank_parameters`.

    Returns
    -------
    list[dict]
        Tornado chart data records sorted from most to least sensitive.
    """
    if not isinstance(result, SensitivityRankResult):
        raise ValueError("result must be a SensitivityRankResult instance.")
    if not result.ranked_params:
        raise ValueError("SensitivityRankResult has no ranked parameters.")

    chart_data: list[dict] = []
    for rank_idx, (param, si) in enumerate(
        zip(result.ranked_params, result.ranked_indices, strict=True), start=1
    ):
        chart_data.append(
            {
                "param": param,
                "sensitivity": si,
                "rank": rank_idx,
            }
        )
    return chart_data
