"""Side-by-side comparison of multiple PK simulations (Phase 420)."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "SimulationComparisonResult",
    "compare_pk_simulations",
    "fold_change_matrix",
]

_VALID_METRICS = {"cmax", "auc", "tmax", "t_half", "auc_normalized"}

# Metrics where a higher value is "better" (ranked descending)
_DESCENDING_METRICS = {"cmax", "auc", "auc_normalized"}


@dataclass
class SimulationComparisonResult:
    """Result of a side-by-side PK simulation comparison."""

    metric: str
    n_simulations: int
    simulation_names: list[str]
    metric_values: list[float]
    ranking: list[int]  # indices into simulation_names, ordered best → worst
    best_simulation_name: str
    worst_simulation_name: str
    notes: str


# ---------------------------------------------------------------------------
# Internal NCA helpers (manual implementation — no scipy)
# ---------------------------------------------------------------------------


def _auc_trapezoidal(times: list[float], conc: list[float]) -> float:
    """Linear trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += (conc[i] + conc[i - 1]) * (times[i] - times[i - 1]) / 2.0
    return auc


def _estimate_lambda_z(times: list[float], conc: list[float]) -> float:
    """Estimate terminal elimination rate constant via log-linear regression.

    Uses the last max(3, 40% of data) points.  Returns positive lambda_z or
    raises ``ValueError`` if a positive value cannot be determined.
    """
    positive = [(t, c) for t, c in zip(times, conc, strict=True) if c > 0]
    if len(positive) < 3:
        raise ValueError("Fewer than 3 positive concentrations for lambda_z estimation.")

    n_term = max(3, int(math.ceil(0.4 * len(positive))))
    seg = positive[-n_term:]
    t_seg = [p[0] for p in seg]
    lc_seg = [math.log(p[1]) for p in seg]

    # OLS: lc = a + b*t  =>  lambda_z = -b
    n = len(t_seg)
    t_mean = sum(t_seg) / n
    lc_mean = sum(lc_seg) / n
    num = sum((t_seg[i] - t_mean) * (lc_seg[i] - lc_mean) for i in range(n))
    den = sum((t_seg[i] - t_mean) ** 2 for i in range(n))
    if abs(den) < 1e-15:
        raise ValueError("Degenerate time values; cannot estimate lambda_z.")
    slope = num / den
    if slope >= 0:
        raise ValueError(f"Non-negative slope {slope:.4f}; cannot estimate lambda_z.")
    return -slope


def _compute_metric(
    times: list[float],
    conc: list[float],
    dose_mg: float,
    metric: str,
) -> float:
    """Compute a single NCA metric for a simulation profile."""
    if metric == "cmax":
        return max(conc)

    if metric == "tmax":
        cmax_val = max(conc)
        return float(times[conc.index(cmax_val)])

    if metric in {"auc", "auc_normalized"}:
        auc = _auc_trapezoidal(times, conc)
        # Extrapolate using C_last / lambda_z when possible
        try:
            lz = _estimate_lambda_z(times, conc)
            c_last = next((c for c in reversed(conc) if c > 0), 0.0)
            auc += c_last / lz
        except ValueError:
            pass  # Use observed AUC only
        if metric == "auc_normalized":
            if dose_mg <= 0:
                raise ValueError(f"dose_mg must be > 0 for auc_normalized, got {dose_mg}")
            return auc / dose_mg
        return auc

    if metric == "t_half":
        try:
            lz = _estimate_lambda_z(times, conc)
            return math.log(2.0) / lz
        except ValueError:
            return float("nan")

    raise ValueError(f"Unknown metric '{metric}'")  # pragma: no cover


def _validate_simulation(sim: dict, index: int) -> None:
    name = sim.get("name", f"simulation_{index}")
    times = sim.get("times")
    conc = sim.get("concentrations")
    dose = sim.get("dose_mg")

    if times is None or conc is None:
        raise ValueError(f"Simulation '{name}' must have 'times' and 'concentrations' keys.")
    if len(times) < 3:
        raise ValueError(f"Simulation '{name}' must have at least 3 time points.")
    if len(times) != len(conc):
        raise ValueError(
            f"Simulation '{name}': times and concentrations must have the same length."
        )
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            raise ValueError(
                f"Simulation '{name}': times must be strictly monotonically increasing."
            )
    if any(c < 0 for c in conc):
        raise ValueError(f"Simulation '{name}': concentrations must be >= 0.")
    if dose is not None and dose <= 0:
        raise ValueError(f"Simulation '{name}': dose_mg must be > 0 if provided.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_pk_simulations(
    simulations: list[dict],
    metric: str,
) -> SimulationComparisonResult:
    """Compare multiple PK simulations on a single NCA metric.

    Parameters
    ----------
    simulations:
        List of dicts, each containing:
        - ``"name"`` (str): label for the simulation
        - ``"times"`` (list[float]): time points in hours
        - ``"concentrations"`` (list[float]): drug concentrations (mg/L)
        - ``"dose_mg"`` (float): administered dose in mg (required for
          ``"auc_normalized"``; ignored otherwise)
    metric:
        NCA metric to compare: ``"cmax"``, ``"auc"``, ``"tmax"``,
        ``"t_half"``, or ``"auc_normalized"``.

    Returns
    -------
    SimulationComparisonResult
        Metric values, ranking (best to worst), best/worst names, and notes.

    Raises
    ------
    ValueError
        If fewer than 2 simulations are provided, metric is unrecognised, or
        any simulation is malformed.
    """
    metric_lower = metric.strip().lower()
    if metric_lower not in _VALID_METRICS:
        raise ValueError(f"metric must be one of {_VALID_METRICS}, got '{metric}'")
    if len(simulations) < 2:
        raise ValueError(f"At least 2 simulations required, got {len(simulations)}.")

    for idx, sim in enumerate(simulations):
        _validate_simulation(sim, idx)

    names: list[str] = []
    values: list[float] = []
    notes_parts: list[str] = []

    for idx, sim in enumerate(simulations):
        name = sim.get("name", f"simulation_{idx}")
        times: list[float] = [float(x) for x in sim["times"]]
        conc: list[float] = [float(x) for x in sim["concentrations"]]
        dose_mg: float = float(sim.get("dose_mg", 1.0))

        try:
            val = _compute_metric(times, conc, dose_mg, metric_lower)
        except ValueError as exc:
            val = float("nan")
            notes_parts.append(f"'{name}': metric computation failed — {exc}")

        names.append(name)
        values.append(val)

    # Build ranking: filter out NaN, then rank
    valid_indices = [i for i, v in enumerate(values) if not math.isnan(v)]
    nan_indices = [i for i, v in enumerate(values) if math.isnan(v)]

    descending = metric_lower in _DESCENDING_METRICS
    valid_sorted = sorted(valid_indices, key=lambda i: values[i], reverse=descending)

    # tmax: ascending (lower tmax = faster absorption = typically better)
    # t_half: no clear best/worst — use ascending (shorter half-life = "best" in ranking)
    ranking = valid_sorted + nan_indices

    if not ranking:
        raise ValueError("All simulations produced NaN metric values; cannot rank.")

    best_name = names[ranking[0]]
    worst_name = names[ranking[-1]]

    if nan_indices:
        notes_parts.append(
            f"{len(nan_indices)} simulation(s) produced NaN for metric '{metric_lower}' "
            "and were ranked last."
        )

    if notes_parts:
        notes = "; ".join(notes_parts)
    else:
        notes = f"Ranked {len(simulations)} simulations by {metric_lower}."

    return SimulationComparisonResult(
        metric=metric_lower,
        n_simulations=len(simulations),
        simulation_names=names,
        metric_values=values,
        ranking=ranking,
        best_simulation_name=best_name,
        worst_simulation_name=worst_name,
        notes=notes,
    )


def fold_change_matrix(simulations: list[dict]) -> list[list[float]]:
    """Compute an n×n AUC fold-change matrix.

    Entry ``[i][j]`` = AUC_i / AUC_j.  Diagonal entries are 1.0.
    If a denominator AUC is zero or NaN, the entry is ``float('nan')``.

    Parameters
    ----------
    simulations:
        List of dicts with ``"name"``, ``"times"``, ``"concentrations"``,
        and ``"dose_mg"`` keys (same format as :func:`compare_pk_simulations`).

    Returns
    -------
    list[list[float]]
        n×n fold-change matrix.

    Raises
    ------
    ValueError
        If fewer than 2 simulations are provided or any simulation is malformed.
    """
    if len(simulations) < 2:
        raise ValueError(f"At least 2 simulations required, got {len(simulations)}.")

    for idx, sim in enumerate(simulations):
        _validate_simulation(sim, idx)

    aucs: list[float] = []
    for _idx, sim in enumerate(simulations):
        times: list[float] = [float(x) for x in sim["times"]]
        conc: list[float] = [float(x) for x in sim["concentrations"]]
        dose_mg: float = float(sim.get("dose_mg", 1.0))
        try:
            val = _compute_metric(times, conc, dose_mg, "auc")
        except ValueError:
            val = float("nan")
        aucs.append(val)

    n = len(aucs)
    matrix: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            auc_j = aucs[j]
            auc_i = aucs[i]
            if math.isnan(auc_i) or math.isnan(auc_j) or auc_j == 0.0:
                row.append(float("nan"))
            else:
                row.append(auc_i / auc_j)
        matrix.append(row)

    return matrix
