"""Phase 307 — Drug Solubility Enhancement by Cosolvents.

Predicts drug solubility enhancement in cosolvent systems using the
Yalkowsky log-linear model.
"""

import math
from dataclasses import dataclass

_VALID_COSOLVENTS = {"PEG400", "ethanol", "propylene_glycol", "glycerin"}

_SIGMA_FACTORS: dict[str, float] = {
    "PEG400": 1.8,
    "ethanol": 1.6,
    "propylene_glycol": 1.2,
    "glycerin": 0.8,
}

_TOXICITY_CONCERN_FRACTION: dict[str, float] = {
    "ethanol": 0.10,
    "PEG400": 0.50,
    "propylene_glycol": 0.30,
    "glycerin": 0.50,
}


@dataclass(frozen=True)
class CosolventSolubilityResult:
    drug_name: str
    logp: float
    intrinsic_solubility_mg_mL: float
    cosolvent_type: str
    cosolvent_fraction: float
    effective_sigma: float
    enhanced_solubility_mg_mL: float
    enhancement_ratio: float
    formulation_feasible: bool
    toxicity_concern_fraction: float
    solubility_class: str
    notes: str


def _validate_inputs(
    logp: float,
    intrinsic_solubility_mg_mL: float,
    cosolvent_fraction: float,
    cosolvent_type: str,
) -> None:
    if not (-5 < logp < 10):
        raise ValueError(f"logp must be in (-5, 10), got {logp}")
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if not (0 <= cosolvent_fraction <= 1):
        raise ValueError(f"cosolvent_fraction must be in [0, 1], got {cosolvent_fraction}")
    if cosolvent_type not in _VALID_COSOLVENTS:
        raise ValueError(
            f"cosolvent_type must be one of {_VALID_COSOLVENTS}, got {cosolvent_type!r}"
        )


def _solubility_class(sm: float) -> str:
    if sm > 10.0:
        return "excellent"
    elif sm >= 1.0:
        return "good"
    elif sm >= 0.1:
        return "moderate"
    else:
        return "poor"


def _compute_result(
    drug_name: str,
    logp: float,
    intrinsic_solubility_mg_mL: float,
    cosolvent_type: str,
    cosolvent_fraction: float,
) -> CosolventSolubilityResult:
    sigma_factor = _SIGMA_FACTORS[cosolvent_type]
    if logp <= 0:
        effective_sigma = sigma_factor * 0.1
    else:
        effective_sigma = sigma_factor * logp

    sm = intrinsic_solubility_mg_mL * (10 ** (effective_sigma * cosolvent_fraction))
    enhancement_ratio = sm / intrinsic_solubility_mg_mL

    tox_frac = _TOXICITY_CONCERN_FRACTION[cosolvent_type]
    feasible = cosolvent_fraction <= tox_frac
    sol_class = _solubility_class(sm)

    notes_parts = []
    if logp <= 0:
        notes_parts.append("Hydrophilic drug: minimal cosolvent effect expected.")
    if not feasible:
        notes_parts.append(
            f"Cosolvent fraction {cosolvent_fraction:.2f} exceeds safety limit"
            f" {tox_frac:.2f} for {cosolvent_type}."
        )
    notes_parts.append(f"Solubility class: {sol_class}.")
    notes = " ".join(notes_parts)

    return CosolventSolubilityResult(
        drug_name=drug_name,
        logp=logp,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        cosolvent_type=cosolvent_type,
        cosolvent_fraction=cosolvent_fraction,
        effective_sigma=effective_sigma,
        enhanced_solubility_mg_mL=sm,
        enhancement_ratio=enhancement_ratio,
        formulation_feasible=feasible,
        toxicity_concern_fraction=tox_frac,
        solubility_class=sol_class,
        notes=notes,
    )


def predict_cosolvent_solubility(
    drug_name: str,
    logp: float,
    intrinsic_solubility_mg_mL: float,
    cosolvent_type: str,
    cosolvent_fraction: float,
) -> CosolventSolubilityResult:
    """Predict drug solubility in a cosolvent mixture using the log-linear model.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logp:
        Octanol-water partition coefficient (must be in (-5, 10)).
    intrinsic_solubility_mg_mL:
        Aqueous (water) solubility in mg/mL (must be > 0).
    cosolvent_type:
        One of "PEG400", "ethanol", "propylene_glycol", "glycerin".
    cosolvent_fraction:
        Volume fraction of cosolvent in the mixture (0-1).

    Returns
    -------
    CosolventSolubilityResult
    """
    _validate_inputs(logp, intrinsic_solubility_mg_mL, cosolvent_fraction, cosolvent_type)
    return _compute_result(
        drug_name, logp, intrinsic_solubility_mg_mL, cosolvent_type, cosolvent_fraction
    )


def optimize_cosolvent_fraction(
    drug_name: str,
    logp: float,
    intrinsic_solubility_mg_mL: float,
    cosolvent_type: str,
    target_solubility_mg_mL: float,
    max_fraction: float = 1.0,
) -> CosolventSolubilityResult:
    """Find minimum cosolvent fraction to achieve target solubility.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logp:
        Octanol-water partition coefficient.
    intrinsic_solubility_mg_mL:
        Aqueous solubility in mg/mL.
    cosolvent_type:
        Cosolvent identifier.
    target_solubility_mg_mL:
        Desired enhanced solubility (must be > intrinsic_solubility_mg_mL).
    max_fraction:
        Upper bound for the search (0, 1].

    Returns
    -------
    CosolventSolubilityResult at the optimum (or max_fraction if target unreachable).
    """
    if target_solubility_mg_mL <= intrinsic_solubility_mg_mL:
        raise ValueError("target_solubility_mg_mL must be > intrinsic_solubility_mg_mL")
    if not (0 < max_fraction <= 1):
        raise ValueError("max_fraction must be in (0, 1]")
    _validate_inputs(logp, intrinsic_solubility_mg_mL, 0.0, cosolvent_type)

    sigma_factor = _SIGMA_FACTORS[cosolvent_type]
    effective_sigma = sigma_factor * (logp if logp > 0 else 0.1)

    # log10(target/Sw) = effective_sigma * f  => f = log10(target/Sw) / effective_sigma
    if effective_sigma == 0:
        optimal_fraction = max_fraction
    else:
        ratio = target_solubility_mg_mL / intrinsic_solubility_mg_mL
        optimal_fraction = math.log10(ratio) / effective_sigma

    fraction = min(optimal_fraction, max_fraction)
    fraction = max(0.0, fraction)

    return _compute_result(drug_name, logp, intrinsic_solubility_mg_mL, cosolvent_type, fraction)


def compare_cosolvents(
    drug_name: str,
    logp: float,
    intrinsic_solubility_mg_mL: float,
    cosolvent_fraction: float,
) -> list[CosolventSolubilityResult]:
    """Compare all four cosolvents at the given fraction.

    Returns results sorted by enhanced_solubility_mg_mL descending.
    """
    _validate_inputs(logp, intrinsic_solubility_mg_mL, cosolvent_fraction, "PEG400")
    results = [
        _compute_result(drug_name, logp, intrinsic_solubility_mg_mL, cs, cosolvent_fraction)
        for cs in _VALID_COSOLVENTS
    ]
    results.sort(key=lambda r: r.enhanced_solubility_mg_mL, reverse=True)
    return results
