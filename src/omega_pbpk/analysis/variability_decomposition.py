"""Variability decomposition for pharmacokinetic data — Phase 647.

Decomposes total PK variability into between-subject (BSV), within-subject (WSV),
and residual (RUV) components using one-way random effects ANOVA.

All calculations use pure Python (no numpy/scipy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "VariabilityDecomposition",
    "decompose_variability",
    "compare_variability_sources",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariabilityDecomposition:
    """Results of one-way random effects ANOVA variability decomposition.

    Attributes
    ----------
    parameter_name : str
        Name of the PK parameter decomposed.
    n_subjects : int
        Number of unique subjects.
    n_occasions : int
        Number of repeated observations per subject (balanced assumption).
    bsv_variance : float
        Between-subject variance component (omega_BSV^2).
    wsv_variance : float
        Within-subject variance component (omega_WSV^2).
    ruv_variance : float
        Residual unexplained variance.
    total_variance : float
        Sum of all variance components.
    bsv_cv_pct : float
        Between-subject CV% (sqrt(bsv_variance) * 100, lognormal approximation).
    wsv_cv_pct : float
        Within-subject CV%.
    total_cv_pct : float
        Total CV% (sqrt(total_variance) * 100).
    bsv_fraction : float
        Fraction of total variance from BSV (bsv_variance / total_variance).
    wsv_fraction : float
        Fraction of total variance from WSV.
    icc : float
        Intraclass correlation coefficient = bsv / (bsv + wsv + ruv).
    interpretation : str
        Dominant source: "BSV-dominated", "WSV-dominated", or "RUV-dominated".
    notes : str
        Descriptive summary of the decomposition.
    """

    parameter_name: str
    n_subjects: int
    n_occasions: int
    bsv_variance: float
    wsv_variance: float
    ruv_variance: float
    total_variance: float
    bsv_cv_pct: float
    wsv_cv_pct: float
    total_cv_pct: float
    bsv_fraction: float
    wsv_fraction: float
    icc: float
    interpretation: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _group_by_subject(
    data: list[dict],
) -> dict[str, list[float]]:
    """Return {subject_id: [values]} preserving insertion order."""
    groups: dict[str, list[float]] = {}
    for row in data:
        sid = str(row["subject_id"])
        groups.setdefault(sid, []).append(float(row["value"]))
    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decompose_variability(
    data: list[dict],
    parameter_name: str = "PK_parameter",
    log_transform: bool = True,
) -> VariabilityDecomposition:
    """Decompose PK variability into BSV, WSV, and RUV components.

    Parameters
    ----------
    data : list[dict]
        List of observations, each with keys:
        ``{"subject_id": str, "occasion": int, "value": float}``.
    parameter_name : str
        Label for the parameter being decomposed.
    log_transform : bool
        If True, work on log(value) for lognormal assumption.

    Returns
    -------
    VariabilityDecomposition

    Raises
    ------
    ValueError
        If data is empty, fewer than 2 subjects, or non-positive values when
        log_transform=True.
    """
    # --- Validation ---
    if not data:
        raise ValueError("data must not be empty")

    # Validate values
    if log_transform:
        for row in data:
            if float(row["value"]) <= 0:
                raise ValueError(
                    "All values must be positive when log_transform=True; "
                    f"found value={row['value']}"
                )

    groups = _group_by_subject(data)
    n_subjects = len(groups)
    if n_subjects < 2:
        raise ValueError(f"At least 2 subjects required for decomposition; got {n_subjects}")

    # --- Optionally log-transform ---
    if log_transform:
        transformed: dict[str, list[float]] = {
            sid: [math.log(v) for v in vals] for sid, vals in groups.items()
        }
    else:
        transformed = {sid: list(vals) for sid, vals in groups.items()}

    # --- Derive n_occasions (use the mode or minimum across subjects) ---
    occasion_counts = [len(v) for v in transformed.values()]
    n_occasions = min(occasion_counts)  # conservative: use minimum

    # --- Grand mean ---
    all_values: list[float] = []
    for vals in transformed.values():
        all_values.extend(vals)
    n_total = len(all_values)
    grand_mean = _mean(all_values)

    # --- Group means ---
    group_means = {sid: _mean(vals) for sid, vals in transformed.items()}

    # --- ANOVA sums of squares ---
    # Between-subject SS: each group's deviation weighted by its size
    ss_between = sum(
        len(transformed[sid]) * (group_means[sid] - grand_mean) ** 2 for sid in transformed
    )
    # Within-subject SS: within each group
    ss_within = sum((v - group_means[sid]) ** 2 for sid, vals in transformed.items() for v in vals)

    df_between = n_subjects - 1
    df_within = n_total - n_subjects

    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    # --- Variance components ---
    # Average group size (harmonic mean for unbalanced designs)
    n0: float
    if n_subjects > 0:
        # Harmonic mean of group sizes
        harmonic_denom = sum(1.0 / len(transformed[sid]) for sid in transformed)
        # For one-way ANOVA with unequal groups, n0 = 1/(df_between) * (n_total - sum(ni^2/n_total))
        # Use simplified: n0 = n_total / n_subjects if balanced, else harmonic
        n0 = n_subjects / harmonic_denom
    else:
        n0 = 1.0

    bsv_variance = max(0.0, (ms_between - ms_within) / n0)

    # Single-occasion case: WSV is indistinguishable from RUV
    if n_occasions <= 1:
        wsv_variance = 0.0
        ruv_variance = ms_within
    else:
        wsv_variance = ms_within
        ruv_variance = 0.0

    total_variance = bsv_variance + wsv_variance + ruv_variance
    # Guard against degenerate case
    if total_variance <= 0.0:
        total_variance = max(bsv_variance + wsv_variance + ruv_variance, 1e-12)

    # --- CV% (lognormal approximation on log-scale variances) ---
    bsv_cv_pct = math.sqrt(bsv_variance) * 100.0
    wsv_cv_pct = math.sqrt(wsv_variance) * 100.0
    total_cv_pct = math.sqrt(total_variance) * 100.0

    # --- Fractions ---
    bsv_fraction = bsv_variance / total_variance
    wsv_fraction = wsv_variance / total_variance

    # --- ICC ---
    icc = bsv_variance / total_variance  # = bsv / (bsv + wsv + ruv)

    # --- Interpretation ---
    ruv_fraction = ruv_variance / total_variance
    max_frac = max(bsv_fraction, wsv_fraction, ruv_fraction)
    if max_frac == bsv_fraction:
        interpretation = "BSV-dominated"
    elif max_frac == wsv_fraction:
        interpretation = "WSV-dominated"
    else:
        interpretation = "RUV-dominated"

    # --- Notes ---
    log_label = " (log-scale)" if log_transform else ""
    notes = (
        f"{parameter_name}: n_subjects={n_subjects}, n_occasions={n_occasions}"
        f"{log_label}; "
        f"BSV CV%={bsv_cv_pct:.1f}, WSV CV%={wsv_cv_pct:.1f}, "
        f"Total CV%={total_cv_pct:.1f}; "
        f"BSV fraction={bsv_fraction:.3f}, ICC={icc:.3f}; "
        f"{interpretation}"
    )

    return VariabilityDecomposition(
        parameter_name=parameter_name,
        n_subjects=n_subjects,
        n_occasions=n_occasions,
        bsv_variance=bsv_variance,
        wsv_variance=wsv_variance,
        ruv_variance=ruv_variance,
        total_variance=total_variance,
        bsv_cv_pct=bsv_cv_pct,
        wsv_cv_pct=wsv_cv_pct,
        total_cv_pct=total_cv_pct,
        bsv_fraction=bsv_fraction,
        wsv_fraction=wsv_fraction,
        icc=icc,
        interpretation=interpretation,
        notes=notes,
    )


def compare_variability_sources(
    datasets: dict[str, list[dict]],
) -> list[VariabilityDecomposition]:
    """Decompose variability for multiple PK parameters and rank by total CV%.

    Parameters
    ----------
    datasets : dict[str, list[dict]]
        Mapping of parameter_name -> list of observation dicts.

    Returns
    -------
    list[VariabilityDecomposition]
        Sorted by total_cv_pct descending (highest variability first).
    """
    results: list[VariabilityDecomposition] = []
    for param_name, data in datasets.items():
        result = decompose_variability(data, parameter_name=param_name)
        results.append(result)
    results.sort(key=lambda r: r.total_cv_pct, reverse=True)
    return results
