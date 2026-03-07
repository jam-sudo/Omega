"""Biomarker-PK correlation analysis for PK/PD relationship modelling (Phase 336).

Correlates PK exposure metrics (AUC, Cmax, trough, time_above_threshold) with
biomarker response values to characterise PK/PD relationships.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

__all__ = [
    "BiomarkerCorrelationResult",
    "correlate_biomarker_pk",
    "screen_biomarker_correlations",
    "summarize_pkpd_relationship",
]

_VALID_CORRELATION_TYPES = {"auc", "cmax", "trough", "time_above_threshold"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiomarkerCorrelationResult:
    """Result of PK exposure vs biomarker response correlation analysis."""

    drug_name: str
    biomarker_name: str
    correlation_type: str  # "auc", "cmax", "trough", "time_above_threshold"
    n_subjects: int

    # Raw data
    pk_values: list[float]
    biomarker_values: list[float]

    # Correlation statistics
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float

    # Linear regression
    slope: float
    intercept: float
    r_squared: float

    # Emax model parameters
    emax: float
    ec50: float  # PK metric at 50% Emax

    best_model: str  # "linear" or "emax"
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hill_emax(c: np.ndarray, emax: float, ec50: float) -> np.ndarray:
    """Hill/Emax equation: E = Emax * C / (EC50 + C)."""
    return emax * c / (ec50 + c)


def _fit_emax(
    pk_vals: np.ndarray,
    bio_vals: np.ndarray,
) -> tuple[float, float, float]:
    """Fit Emax model and return (emax, ec50, r_squared_emax).

    Falls back to max(bio) / median(pk) if optimisation fails.
    """
    emax0 = float(np.max(bio_vals))
    ec50_0 = float(np.median(pk_vals))

    try:
        popt, _ = optimize.curve_fit(
            _hill_emax,
            pk_vals,
            bio_vals,
            p0=[emax0, ec50_0],
            bounds=([0.0, 1e-9], [np.inf, np.inf]),
            maxfev=5000,
        )
        emax_fit, ec50_fit = float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError, optimize.OptimizeWarning):
        emax_fit, ec50_fit = emax0, ec50_0

    # Compute R² for the Emax fit
    fitted = _hill_emax(pk_vals, emax_fit, ec50_fit)
    ss_res = float(np.sum((bio_vals - fitted) ** 2))
    ss_tot = float(np.sum((bio_vals - np.mean(bio_vals)) ** 2))
    r2_emax = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return emax_fit, ec50_fit, r2_emax


def _validate_pk_biomarker_inputs(
    pk_values: list[float],
    biomarker_values: list[float],
) -> None:
    """Validate that inputs are suitable for correlation analysis."""
    if len(pk_values) != len(biomarker_values):
        raise ValueError(
            f"pk_values (len={len(pk_values)}) and biomarker_values "
            f"(len={len(biomarker_values)}) must have the same length."
        )
    if len(pk_values) < 3:
        raise ValueError(f"At least 3 data points are required, got {len(pk_values)}.")
    for label, vals in [("pk_values", pk_values), ("biomarker_values", biomarker_values)]:
        for v in vals:
            if not math.isfinite(v):
                raise ValueError(f"All values in {label} must be finite; found {v}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def correlate_biomarker_pk(
    drug_name: str,
    biomarker_name: str,
    pk_values: list[float],
    biomarker_values: list[float],
    correlation_type: str = "auc",
) -> BiomarkerCorrelationResult:
    """Correlate PK exposure values with biomarker response.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    biomarker_name:
        Name of the biomarker (e.g. "IL-6", "FEV1", "tumour_size").
    pk_values:
        List of PK exposure values (one per subject).
    biomarker_values:
        List of biomarker response values (one per subject).
    correlation_type:
        Type of PK metric: "auc", "cmax", "trough", or "time_above_threshold".

    Returns
    -------
    BiomarkerCorrelationResult
    """
    _validate_pk_biomarker_inputs(pk_values, biomarker_values)

    pk_arr = np.array(pk_values, dtype=float)
    bio_arr = np.array(biomarker_values, dtype=float)
    n = len(pk_values)

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(pk_arr, bio_arr)
    pearson_r = float(pearson_r)
    pearson_p = float(pearson_p)

    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(pk_arr, bio_arr)
    spearman_r = float(spearman_r)
    spearman_p = float(spearman_p)

    # Linear regression via polyfit
    coeffs = np.polyfit(pk_arr, bio_arr, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    r_squared = pearson_r**2

    # Emax model fit
    emax, ec50, r2_emax = _fit_emax(pk_arr, bio_arr)

    # Choose best model
    best_model = "emax" if r2_emax > r_squared + 0.05 else "linear"

    notes = (
        f"{drug_name} / {biomarker_name}: n={n}, "
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.4f}), "
        f"Spearman r={spearman_r:.3f}, "
        f"best_model={best_model}"
    )

    return BiomarkerCorrelationResult(
        drug_name=drug_name,
        biomarker_name=biomarker_name,
        correlation_type=correlation_type,
        n_subjects=n,
        pk_values=list(pk_values),
        biomarker_values=list(biomarker_values),
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        emax=emax,
        ec50=ec50,
        best_model=best_model,
        notes=notes,
    )


def screen_biomarker_correlations(
    drug_name: str,
    pk_dict: dict[str, list[float]],
    biomarker_dict: dict[str, list[float]],
) -> list[BiomarkerCorrelationResult]:
    """Screen all PK metric × biomarker combinations and rank by |Pearson r|.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    pk_dict:
        Mapping of PK metric name → list of values (one per subject).
        E.g. {"auc": [...], "cmax": [...]}.
    biomarker_dict:
        Mapping of biomarker name → list of values (one per subject).

    Returns
    -------
    List of BiomarkerCorrelationResult sorted by |pearson_r| descending.
    """
    results: list[BiomarkerCorrelationResult] = []

    for pk_metric, pk_values in pk_dict.items():
        for bm_name, bm_values in biomarker_dict.items():
            try:
                result = correlate_biomarker_pk(
                    drug_name=drug_name,
                    biomarker_name=bm_name,
                    pk_values=pk_values,
                    biomarker_values=bm_values,
                    correlation_type=pk_metric,
                )
                results.append(result)
            except ValueError:
                # Skip pairs that fail validation
                continue

    return sorted(results, key=lambda r: abs(r.pearson_r), reverse=True)


def summarize_pkpd_relationship(result: BiomarkerCorrelationResult) -> str:
    """Return a formatted human-readable summary of a PK/PD correlation result.

    Parameters
    ----------
    result:
        A BiomarkerCorrelationResult from correlate_biomarker_pk.

    Returns
    -------
    Formatted summary string.
    """
    direction = "positive" if result.pearson_r >= 0 else "negative"
    strength = (
        "strong"
        if abs(result.pearson_r) >= 0.7
        else "moderate"
        if abs(result.pearson_r) >= 0.4
        else "weak"
    )
    significance = "significant" if result.pearson_p < 0.05 else "non-significant"

    lines = [
        "PK/PD Correlation Summary",
        f"  Drug:              {result.drug_name}",
        f"  Biomarker:         {result.biomarker_name}",
        f"  PK metric:         {result.correlation_type}",
        f"  N subjects:        {result.n_subjects}",
        f"  Pearson r:         {result.pearson_r:.3f} (p={result.pearson_p:.4f})",
        f"  Spearman r:        {result.spearman_r:.3f} (p={result.spearman_p:.4f})",
        f"  Relationship:      {strength} {direction} ({significance})",
        f"  Linear model:      slope={result.slope:.4f}, "
        f"intercept={result.intercept:.4f}, R²={result.r_squared:.3f}",
        f"  Emax model:        Emax={result.emax:.4f}, EC50={result.ec50:.4f}",
        f"  Best model:        {result.best_model}",
    ]
    return "\n".join(lines)
