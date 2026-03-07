"""Phase 923 — Drug Pleural Fluid Distribution.

Predicts drug concentration in pleural fluid, relevant for malignant pleural
effusion treatment and intrapleural drug delivery.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PleuralDistributionResult:
    """Result of pleural fluid drug distribution prediction."""

    drug_name: str
    logp: float
    mw_Da: float
    condition: str
    dose_mg: float
    route: str
    plasma_to_pleural_ratio: float
    predicted_pleural_conc_mg_L: float
    plasma_conc_input_mg_L: float
    pleural_volume_L: float
    drug_amount_in_pleural_mg: float
    therapeutic_local_conc_achievable: bool
    notes: str


def predict_pleural_distribution(
    drug_name: str,
    dose_mg: float = 10.0,
    logp: float = 2.0,
    mw_Da: float = 300.0,
    route: str = "systemic",
    condition: str = "normal",
    plasma_conc_mg_L: float = 1.0,
) -> PleuralDistributionResult:
    """Predict drug distribution into pleural fluid.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    logp:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons.
    route:
        Administration route: "systemic" or "intrapleural".
    condition:
        Pleural condition: "normal" or "malignant_effusion".
    plasma_conc_mg_L:
        Plasma drug concentration in mg/L (used for systemic route).

    Returns
    -------
    PleuralDistributionResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if route not in {"intrapleural", "systemic"}:
        raise ValueError(f"route must be 'intrapleural' or 'systemic', got {route!r}")
    if condition not in {"normal", "malignant_effusion"}:
        raise ValueError(f"condition must be 'normal' or 'malignant_effusion', got {condition!r}")

    # Pleural volume based on condition
    if condition == "normal":
        pleural_volume_L = 0.020
    else:
        pleural_volume_L = 1.0

    if route == "systemic":
        # Plasma-to-pleural ratio driven by lipophilicity
        ratio = 0.3 + logp * 0.1
        ratio = max(0.05, min(2.0, ratio))
        predicted_pleural_conc = ratio * plasma_conc_mg_L
        plasma_to_pleural_ratio = ratio
    else:
        # Intrapleural: drug disperses into pleural volume
        predicted_pleural_conc = dose_mg / pleural_volume_L
        raw_ratio = predicted_pleural_conc / max(plasma_conc_mg_L, 0.001)
        plasma_to_pleural_ratio = max(1.0, min(500.0, raw_ratio))

    drug_amount_in_pleural_mg = predicted_pleural_conc * pleural_volume_L
    therapeutic_local_conc_achievable = predicted_pleural_conc >= 1.0

    # Notes selection (priority: therapeutic > malignant > default)
    if therapeutic_local_conc_achievable:
        notes = (
            "Therapeutic pleural concentrations achieved — suitable for intrapleural instillation"
        )
    elif condition == "malignant_effusion":
        notes = "Malignant effusion dilutes drug — higher doses or intrapleural route needed"
    else:
        notes = "Limited pleural penetration via systemic route"

    return PleuralDistributionResult(
        drug_name=drug_name,
        logp=logp,
        mw_Da=mw_Da,
        condition=condition,
        dose_mg=dose_mg,
        route=route,
        plasma_to_pleural_ratio=plasma_to_pleural_ratio,
        predicted_pleural_conc_mg_L=predicted_pleural_conc,
        plasma_conc_input_mg_L=plasma_conc_mg_L,
        pleural_volume_L=pleural_volume_L,
        drug_amount_in_pleural_mg=drug_amount_in_pleural_mg,
        therapeutic_local_conc_achievable=therapeutic_local_conc_achievable,
        notes=notes,
    )


def compare_pleural_conditions(
    drug_name: str,
    dose_mg: float,
    logp: float,
    **kwargs,
) -> list[PleuralDistributionResult]:
    """Compare drug distribution under normal vs. malignant effusion conditions.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    logp:
        Octanol-water partition coefficient.
    **kwargs:
        Additional keyword arguments passed to :func:`predict_pleural_distribution`.

    Returns
    -------
    list[PleuralDistributionResult]
        Two results (normal and malignant_effusion), sorted by
        ``predicted_pleural_conc_mg_L`` descending.
    """
    results = []
    for condition in ("normal", "malignant_effusion"):
        result = predict_pleural_distribution(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logp=logp,
            condition=condition,
            **kwargs,
        )
        results.append(result)

    results.sort(key=lambda r: r.predicted_pleural_conc_mg_L, reverse=True)
    return results
