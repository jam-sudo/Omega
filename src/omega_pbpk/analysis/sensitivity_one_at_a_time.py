"""
One-at-a-time (OAT) sensitivity analysis.

Phase 583: Vary each parameter independently while holding all others at base
values. Compute normalized sensitivity (elasticity), ranking, and tornado
chart data.
"""

from __future__ import annotations


def oat_sensitivity(
    base_params: dict,
    param_ranges: dict,
    model_fn: callable,
    output_key: str,
    n_levels: int = 5,
) -> list[dict]:
    """
    Perform one-at-a-time sensitivity analysis.

    For each parameter in param_ranges, vary it from low to high (n_levels
    evenly spaced), keeping all other parameters at their base values.

    Parameters
    ----------
    base_params:
        Default parameter values used when a parameter is NOT being varied.
    param_ranges:
        Mapping of param_name -> (low, high) defining the variation range.
    model_fn:
        Callable(**params) -> dict.  Must return a dict containing output_key.
    output_key:
        Key in the result dict to use as the scalar output.
    n_levels:
        Number of evenly-spaced levels between low and high (inclusive).

    Returns
    -------
    List of dicts with keys:
        param_name, value, output, normalized_sensitivity
    """
    # Compute base output once
    base_result = model_fn(**base_params)
    base_output = base_result[output_key]

    results: list[dict] = []

    for param_name, (low, high) in param_ranges.items():
        base_val = base_params[param_name]

        # Generate n_levels evenly spaced values
        if n_levels <= 1:
            levels = [low]
        else:
            step = (high - low) / (n_levels - 1)
            levels = [low + i * step for i in range(n_levels)]

        for value in levels:
            # Build parameter dict with this param varied
            params = dict(base_params)
            params[param_name] = value

            result = model_fn(**params)
            output = result[output_key]

            # Normalized sensitivity (elasticity)
            norm_sens = local_elasticity(value, base_val, output, base_output)

            results.append(
                {
                    "param_name": param_name,
                    "value": value,
                    "output": output,
                    "normalized_sensitivity": norm_sens,
                    "base_output": base_output,
                }
            )

    return results


def sensitivity_ranking(oat_results: list[dict]) -> list[dict]:
    """
    Compute max absolute normalized_sensitivity for each parameter and rank.

    Returns a list of dicts (param_name, max_sensitivity, rank) sorted by
    sensitivity_magnitude descending (rank=1 is most sensitive).
    """
    # Aggregate max |sensitivity| per parameter
    param_max: dict[str, float] = {}
    for entry in oat_results:
        name = entry["param_name"]
        abs_sens = abs(entry["normalized_sensitivity"])
        if name not in param_max or abs_sens > param_max[name]:
            param_max[name] = abs_sens

    # Sort descending by sensitivity
    sorted_params = sorted(param_max.items(), key=lambda kv: kv[1], reverse=True)

    ranking = []
    for rank, (param_name, max_sensitivity) in enumerate(sorted_params, start=1):
        ranking.append(
            {
                "param_name": param_name,
                "max_sensitivity": max_sensitivity,
                "rank": rank,
            }
        )

    return ranking


def tornado_chart_data(oat_results: list[dict], base_output: float) -> list[dict]:
    """
    Compute tornado chart data.

    For each parameter, find output at its minimum and maximum sampled value.

    Returns list of dicts (param_name, output_low, output_high, swing),
    sorted by swing descending.
    """
    # Collect all entries per parameter
    param_entries: dict[str, list[dict]] = {}
    for entry in oat_results:
        name = entry["param_name"]
        param_entries.setdefault(name, []).append(entry)

    chart_data = []
    for param_name, entries in param_entries.items():
        # Sort by sampled value
        sorted_entries = sorted(entries, key=lambda e: e["value"])
        output_low = sorted_entries[0]["output"]
        output_high = sorted_entries[-1]["output"]
        swing = output_high - output_low

        chart_data.append(
            {
                "param_name": param_name,
                "output_low": output_low,
                "output_high": output_high,
                "swing": swing,
            }
        )

    # Sort by |swing| descending
    chart_data.sort(key=lambda d: abs(d["swing"]), reverse=True)
    return chart_data


def local_elasticity(
    param_value: float,
    base_param: float,
    output_value: float,
    base_output: float,
) -> float:
    """
    Compute local elasticity (normalised sensitivity).

    E = (dOutput/Output) / (dParam/Param)
      = (output - base_output) / base_output / ((param_value - base_param) / base_param)

    Returns 0.0 if base_param == 0 or base_output == 0 or param_value == base_param.
    """
    if base_param == 0.0 or base_output == 0.0:
        return 0.0
    d_param = param_value - base_param
    if d_param == 0.0:
        return 0.0
    d_output_frac = (output_value - base_output) / base_output
    d_param_frac = d_param / base_param
    return d_output_frac / d_param_frac
