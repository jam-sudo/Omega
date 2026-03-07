"""Phase 785 — Exposure-response analysis linking PK exposure metrics to PD outcomes.

Implements Emax, sigmoid Emax, and linear PD models with therapeutic threshold
assessment and batch analysis capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ExposureResponseResult",
    "analyze_exposure_response",
    "simulate_er_curve",
    "batch_analyze_er",
]

_VALID_PD_MODELS = ("emax", "sigmoid_emax", "linear")
_VALID_EXPOSURE_METRICS = ("auc", "cmax", "trough")


@dataclass(frozen=True)
class ExposureResponseResult:
    """Result of a single exposure-response analysis."""

    compound_name: str
    exposure_metric: str  # "auc", "cmax", "trough"
    exposure_value: float
    pd_effect_pct: float  # predicted % effect (0-100)
    emax_pct: float
    ec50: float
    hill_coef: float
    pd_model: str  # "emax", "sigmoid_emax", "linear"
    therapeutic_threshold_met: bool
    therapeutic_effect_pct: float  # threshold for clinical benefit (default 50)
    notes: str


def _compute_effect(
    exposure: float,
    emax_pct: float,
    ec50: float,
    hill_coef: float,
    pd_model: str,
) -> float:
    """Compute PD effect percentage for a given exposure value.

    Parameters
    ----------
    exposure:
        Drug exposure value (AUC, Cmax, or trough).
    emax_pct:
        Maximum effect as a percentage (0-100).
    ec50:
        Exposure producing 50% of maximum effect.
    hill_coef:
        Hill coefficient (steepness of sigmoid curve).
    pd_model:
        One of "emax", "sigmoid_emax", "linear".

    Returns
    -------
    float
        Predicted effect percentage clamped to [0, emax_pct].
    """
    if pd_model == "emax":
        # Standard Emax model (Hill coefficient = 1)
        effect = emax_pct * exposure / (ec50 + exposure)
    elif pd_model == "sigmoid_emax":
        # Sigmoid Emax (Hill) model
        e_n = exposure**hill_coef
        ec50_n = ec50**hill_coef
        effect = emax_pct * e_n / (ec50_n + e_n)
    elif pd_model == "linear":
        # Linear model capped at emax
        slope = emax_pct / ec50
        effect = min(emax_pct, slope * exposure)
    else:
        raise ValueError(f"Unknown pd_model '{pd_model}'. Must be one of {_VALID_PD_MODELS}.")

    # Clamp to [0, emax_pct]
    return max(0.0, min(emax_pct, effect))


def analyze_exposure_response(
    compound_name: str,
    exposure_value: float,
    emax_pct: float = 100.0,
    ec50: float = 1.0,
    hill_coef: float = 1.0,
    exposure_metric: str = "auc",
    pd_model: str = "sigmoid_emax",
    therapeutic_threshold_pct: float = 50.0,
) -> ExposureResponseResult:
    """Analyze exposure-response relationship for a single exposure value.

    Parameters
    ----------
    compound_name:
        Name of the compound being analyzed.
    exposure_value:
        Drug exposure measurement (must be >= 0).
    emax_pct:
        Maximum achievable PD effect as a percentage (must be > 0).
    ec50:
        Exposure producing 50% of Emax (must be > 0).
    hill_coef:
        Hill coefficient controlling sigmoidicity (must be > 0).
    exposure_metric:
        Type of exposure metric: "auc", "cmax", or "trough".
    pd_model:
        PD model: "emax", "sigmoid_emax", or "linear".
    therapeutic_threshold_pct:
        Minimum effect percentage considered clinically beneficial.

    Returns
    -------
    ExposureResponseResult
        Dataclass containing predicted effect and threshold assessment.

    Raises
    ------
    ValueError
        If exposure_value < 0, emax_pct <= 0, ec50 <= 0, hill_coef <= 0,
        invalid exposure_metric, or invalid pd_model.
    """
    if exposure_value < 0:
        raise ValueError(f"exposure_value must be >= 0, got {exposure_value}.")
    if emax_pct <= 0:
        raise ValueError(f"emax_pct must be > 0, got {emax_pct}.")
    if ec50 <= 0:
        raise ValueError(f"ec50 must be > 0, got {ec50}.")
    if hill_coef <= 0:
        raise ValueError(f"hill_coef must be > 0, got {hill_coef}.")
    if exposure_metric not in _VALID_EXPOSURE_METRICS:
        raise ValueError(
            f"Unknown exposure_metric '{exposure_metric}'. "
            f"Must be one of {_VALID_EXPOSURE_METRICS}."
        )
    if pd_model not in _VALID_PD_MODELS:
        raise ValueError(f"Unknown pd_model '{pd_model}'. Must be one of {_VALID_PD_MODELS}.")

    pd_effect_pct = _compute_effect(exposure_value, emax_pct, ec50, hill_coef, pd_model)
    therapeutic_threshold_met = pd_effect_pct >= therapeutic_threshold_pct

    # Build notes
    notes_parts = [
        f"Model: {pd_model}",
        f"Exposure ({exposure_metric}): {exposure_value:.4g}",
        f"Effect: {pd_effect_pct:.2f}%",
        f"Threshold ({therapeutic_threshold_pct:.1f}%): "
        + ("MET" if therapeutic_threshold_met else "NOT MET"),
    ]
    notes = "; ".join(notes_parts)

    return ExposureResponseResult(
        compound_name=compound_name,
        exposure_metric=exposure_metric,
        exposure_value=exposure_value,
        pd_effect_pct=pd_effect_pct,
        emax_pct=emax_pct,
        ec50=ec50,
        hill_coef=hill_coef,
        pd_model=pd_model,
        therapeutic_threshold_met=therapeutic_threshold_met,
        therapeutic_effect_pct=therapeutic_threshold_pct,
        notes=notes,
    )


def simulate_er_curve(
    compound_name: str,
    exposure_range: list[float],
    emax_pct: float = 100.0,
    ec50: float = 1.0,
    hill_coef: float = 1.0,
    pd_model: str = "sigmoid_emax",
) -> list[ExposureResponseResult]:
    """Simulate an exposure-response curve over a range of exposure values.

    Parameters
    ----------
    compound_name:
        Name of the compound.
    exposure_range:
        List of exposure values to evaluate.
    emax_pct:
        Maximum achievable PD effect as a percentage.
    ec50:
        Exposure producing 50% of Emax.
    hill_coef:
        Hill coefficient.
    pd_model:
        PD model: "emax", "sigmoid_emax", or "linear".

    Returns
    -------
    list[ExposureResponseResult]
        One result per exposure value, in the same order as exposure_range.
    """
    results: list[ExposureResponseResult] = []
    for exp_val in exposure_range:
        result = analyze_exposure_response(
            compound_name=compound_name,
            exposure_value=exp_val,
            emax_pct=emax_pct,
            ec50=ec50,
            hill_coef=hill_coef,
            pd_model=pd_model,
        )
        results.append(result)
    return results


def batch_analyze_er(subjects: list[dict]) -> list[ExposureResponseResult]:
    """Analyze exposure-response for a batch of subjects.

    Parameters
    ----------
    subjects:
        List of dicts, each containing keyword arguments for
        analyze_exposure_response. Required key: "compound_name",
        "exposure_value". All other keys are optional with defaults
        matching analyze_exposure_response defaults.

    Returns
    -------
    list[ExposureResponseResult]
        One result per subject dict.
    """
    results: list[ExposureResponseResult] = []
    for subj in subjects:
        result = analyze_exposure_response(
            compound_name=subj["compound_name"],
            exposure_value=subj["exposure_value"],
            emax_pct=subj.get("emax_pct", 100.0),
            ec50=subj.get("ec50", 1.0),
            hill_coef=subj.get("hill_coef", 1.0),
            exposure_metric=subj.get("exposure_metric", "auc"),
            pd_model=subj.get("pd_model", "sigmoid_emax"),
            therapeutic_threshold_pct=subj.get("therapeutic_threshold_pct", 50.0),
        )
        results.append(result)
    return results
