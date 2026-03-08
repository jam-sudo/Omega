"""Phase 500 — Bile Salt Solubilization Model.

Model the effect of bile salts on poorly soluble drug solubilization
in the intestinal lumen.
"""

from __future__ import annotations

from dataclasses import dataclass

_FASTED_BILE_CONC_MM: float = 3.0
_FED_BILE_CONC_MM: float = 15.0
_REF_BILE_CONC_MM: float = 5.0
_BCS_SOLUBILITY_THRESHOLD: float = 0.1  # mg/mL

_VALID_FOOD_STATES = {"fasted", "fed"}


@dataclass(frozen=True)
class BileSaltSolubilizationResult:
    """Result of bile salt solubilization prediction."""

    drug_name: str
    logP: float
    intrinsic_solubility_mg_mL: float
    food_state: str
    bile_salt_conc_mM: float
    bile_enhancement_mg_mL: float
    total_solubility_mg_mL: float
    enhancement_fold: float
    bcs_relevant: bool
    notes: str


def _partition_factor(logP: float) -> float:
    """Compute bile partition factor from logP."""
    if logP > 1.0:
        return 0.1 * (logP - 1.0)
    return 0.0


def _compute_enhancement(
    intrinsic_solubility: float,
    partition_factor: float,
    bile_salt_conc: float,
) -> float:
    """Compute bile salt enhancement of solubility."""
    return max(
        0.0,
        intrinsic_solubility * partition_factor * bile_salt_conc / _REF_BILE_CONC_MM,
    )


def _build_notes(
    logP: float,
    food_state: str,
    partition_factor: float,
    enhancement_fold: float,
) -> str:
    """Build informative notes string."""
    parts: list[str] = []
    if partition_factor == 0.0:
        parts.append("Hydrophilic drug (logP <= 1): no bile salt enhancement expected.")
    else:
        parts.append(f"logP={logP:.2f}: bile partition factor={partition_factor:.3f}.")
    if food_state == "fed":
        parts.append("Fed state: higher bile salt concentration increases solubilization.")
    else:
        parts.append("Fasted state: lower bile salt concentration.")
    parts.append(f"Total enhancement: {enhancement_fold:.2f}-fold.")
    return " ".join(parts)


def predict_bile_solubilization(
    drug_name: str,
    logP: float,
    intrinsic_solubility_mg_mL: float,
    food_state: str,
) -> BileSaltSolubilizationResult:
    """Predict total intestinal solubility with bile salt enhancement.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol/water partition coefficient.
    intrinsic_solubility_mg_mL:
        Aqueous intrinsic solubility in mg/mL (must be > 0).
    food_state:
        Either "fasted" or "fed".

    Returns
    -------
    BileSaltSolubilizationResult
    """
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if food_state not in _VALID_FOOD_STATES:
        raise ValueError(f"food_state must be one of {_VALID_FOOD_STATES}, got '{food_state}'")

    bile_conc = _FASTED_BILE_CONC_MM if food_state == "fasted" else _FED_BILE_CONC_MM
    pf = _partition_factor(logP)
    enhancement = _compute_enhancement(intrinsic_solubility_mg_mL, pf, bile_conc)
    total_solubility = max(
        intrinsic_solubility_mg_mL,
        intrinsic_solubility_mg_mL + enhancement,
    )
    enhancement_fold = total_solubility / intrinsic_solubility_mg_mL
    bcs_relevant = intrinsic_solubility_mg_mL < _BCS_SOLUBILITY_THRESHOLD
    notes = _build_notes(logP, food_state, pf, enhancement_fold)

    return BileSaltSolubilizationResult(
        drug_name=drug_name,
        logP=logP,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        food_state=food_state,
        bile_salt_conc_mM=bile_conc,
        bile_enhancement_mg_mL=enhancement,
        total_solubility_mg_mL=total_solubility,
        enhancement_fold=enhancement_fold,
        bcs_relevant=bcs_relevant,
        notes=notes,
    )


def compare_food_states(
    drug_name: str,
    logP: float,
    intrinsic_solubility_mg_mL: float,
) -> list:
    """Compare fasted vs fed solubilization for a drug.

    Returns
    -------
    List of two BileSaltSolubilizationResult objects: [fasted, fed].
    """
    fasted = predict_bile_solubilization(
        drug_name=drug_name,
        logP=logP,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        food_state="fasted",
    )
    fed = predict_bile_solubilization(
        drug_name=drug_name,
        logP=logP,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        food_state="fed",
    )
    return [fasted, fed]
