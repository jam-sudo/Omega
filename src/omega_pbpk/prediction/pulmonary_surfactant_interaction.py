"""Pulmonary surfactant interaction prediction for inhaled drugs (Phase 880).

Predicts drug interaction with lung surfactant (DPPC/SP-B/SP-C), affecting
dissolution and absorption in alveoli. High logP drugs have stronger surfactant
binding, resulting in slower absorption but sustained lung levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "SurfactantInteractionResult",
    "predict_surfactant_interaction",
    "screen_inhaled_drugs",
]

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class SurfactantInteractionResult:
    """Result of pulmonary surfactant interaction prediction.

    Attributes
    ----------
    drug_name : Drug identifier.
    logp : Lipophilicity (log P).
    pka : Acid/base dissociation constant (None if not applicable).
    mw_Da : Molecular weight (Da).
    surfactant_binding_fraction : Fraction of drug bound to surfactant (0-1).
    effective_free_fraction : Free fraction available for absorption (0-1).
    dissolution_rate_factor : Factor modifying dissolution rate (>1 = faster).
    absorption_rate_modifier : Scales absorption rate constant (= free fraction).
    predicted_lung_t_half_h : Estimated drug half-life in lung (h), clamped [0.1, 72].
    retention_score : Lung retention score (0-100); higher = more retained.
    notes : Interpretation text.
    """

    drug_name: str
    logp: float
    pka: float | None
    mw_Da: float
    surfactant_binding_fraction: float
    effective_free_fraction: float
    dissolution_rate_factor: float
    absorption_rate_modifier: float
    predicted_lung_t_half_h: float
    retention_score: float
    notes: str


def predict_surfactant_interaction(
    drug_name: str,
    logp: float,
    pka: float | None = None,
    mw_Da: float = 300.0,
    ionization_type: str = "neutral",
    dose_mg: float = 1.0,
) -> SurfactantInteractionResult:
    """Predict lung surfactant binding and its impact on inhaled drug PK.

    Uses a sigmoid function of logP to estimate surfactant binding, with
    higher logP drugs binding more strongly to DPPC-based surfactant.

    Parameters
    ----------
    drug_name : Drug identifier string.
    logp : Lipophilicity (log P).
    pka : pKa of the drug (optional, for ionization context).
    mw_Da : Molecular weight in Daltons. Must be > 0.
    ionization_type : One of "acid", "base", "neutral".
    dose_mg : Inhaled dose (mg, informational only).
    """
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION_TYPES)}, "
            f"got '{ionization_type}'"
        )

    # Surfactant binding: sigmoid centered at logP = 2.0, steepness = 1.5
    # 50% binding at logP = 2, approaches 1 for high logP
    surfactant_binding_fraction = 1.0 / (1.0 + math.exp(-(logp - 2.0) * 1.5))

    effective_free_fraction = 1.0 - surfactant_binding_fraction

    # Surfactant aids dissolution for lipophilic drugs (up to 50% increase)
    dissolution_rate_factor = 1.0 + surfactant_binding_fraction * 0.5

    # Only free (unbound) drug is absorbed
    absorption_rate_modifier = effective_free_fraction

    # Lung half-life: base 2h, prolonged by binding
    base_t_half_h = 2.0
    if absorption_rate_modifier > 0:
        predicted_lung_t_half_h = base_t_half_h / absorption_rate_modifier
    else:
        predicted_lung_t_half_h = 72.0  # maximum if fully bound
    # Clamp to [0.1, 72.0]
    predicted_lung_t_half_h = max(0.1, min(72.0, predicted_lung_t_half_h))

    # Retention score: 0-100
    # Contribution from binding (up to 80) + MW contribution (up to 20)
    retention_score = surfactant_binding_fraction * 80.0 + (mw_Da / 500.0) * 20.0
    retention_score = max(0.0, min(100.0, retention_score))

    # Notes based on retention score
    if retention_score > 60.0:
        notes = "High lung retention — suitable for local pulmonary therapy"
    elif retention_score > 30.0:
        notes = "Moderate lung retention"
    else:
        notes = "Low lung retention — rapid systemic absorption"

    return SurfactantInteractionResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        mw_Da=mw_Da,
        surfactant_binding_fraction=surfactant_binding_fraction,
        effective_free_fraction=effective_free_fraction,
        dissolution_rate_factor=dissolution_rate_factor,
        absorption_rate_modifier=absorption_rate_modifier,
        predicted_lung_t_half_h=predicted_lung_t_half_h,
        retention_score=retention_score,
        notes=notes,
    )


def screen_inhaled_drugs(
    candidates: list[dict],
) -> list[SurfactantInteractionResult]:
    """Screen a list of drug candidates for pulmonary surfactant interaction.

    Parameters
    ----------
    candidates : List of dicts with keys:
        - "name" (str, required)
        - "logp" (float, required)
        - "pka" (float, optional)
        - "mw_Da" (float, optional, default 300.0)
        - "ionization_type" (str, optional, default "neutral")

    Returns
    -------
    list[SurfactantInteractionResult]
        Results sorted by retention_score descending (highest first).
    """
    results = []
    for candidate in candidates:
        name = candidate["name"]
        logp = float(candidate["logp"])
        pka = candidate.get("pka", None)
        mw_Da = float(candidate.get("mw_Da", 300.0))
        ionization_type = candidate.get("ionization_type", "neutral")

        result = predict_surfactant_interaction(
            drug_name=name,
            logp=logp,
            pka=pka,
            mw_Da=mw_Da,
            ionization_type=ionization_type,
        )
        results.append(result)

    results.sort(key=lambda r: r.retention_score, reverse=True)
    return results
