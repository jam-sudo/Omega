"""Drug combination synergy — Bliss independence, Loewe additivity, CI method."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "SynergyResult",
    "dose_response",
    "compute_bliss_independence",
    "compute_loewe_ci",
    "assess_combination_synergy",
    "synergy_matrix",
]


def dose_response(dose: float, ic50: float, hill_n: float = 1.0) -> float:
    """Hill equation: E = dose^n / (ic50^n + dose^n). Returns float in [0, 1)."""
    dose = max(dose, 0.0)
    if ic50 <= 0:
        raise ValueError("ic50 must be > 0")
    d_n = dose**hill_n
    return d_n / (ic50**hill_n + d_n)


def compute_bliss_independence(effect_A: float, effect_B: float) -> float:
    """Bliss independence: E_bliss = E_A + E_B - E_A * E_B."""
    return effect_A + effect_B - effect_A * effect_B


def compute_loewe_ci(
    dose_A: float,
    dose_B: float,
    ic50_A: float,
    ic50_B: float,
    effect: float,
    hill_n_A: float = 1.0,
    hill_n_B: float = 1.0,
) -> float:
    """Combination Index (Chou-Talalay) for Loewe additivity."""
    if effect <= 0 or effect >= 1:
        return float("nan")
    ratio = effect / (1 - effect)
    Dx_A = ic50_A * ratio ** (1.0 / hill_n_A)
    Dx_B = ic50_B * ratio ** (1.0 / hill_n_B)
    if Dx_A <= 0 or Dx_B <= 0:
        return float("nan")
    return dose_A / Dx_A + dose_B / Dx_B


@dataclass(frozen=True)
class SynergyResult:
    drug_A: str
    drug_B: str
    dose_A: float
    dose_B: float
    effect_A_alone: float
    effect_B_alone: float
    combination_effect: float
    bliss_expected_effect: float
    bliss_delta: float
    loewe_ci: float
    ci_classification: str
    bliss_classification: str
    warnings: list[str] = field(default_factory=list)


def assess_combination_synergy(
    drug_A: str,
    drug_B: str,
    dose_A: float,
    dose_B: float,
    ic50_A: float,
    ic50_B: float,
    hill_n_A: float = 1.0,
    hill_n_B: float = 1.0,
    observed_effect: float | None = None,
) -> SynergyResult:
    """Assess drug combination synergy using Bliss and Loewe methods."""
    effect_A = dose_response(dose_A, ic50_A, hill_n_A)
    effect_B = dose_response(dose_B, ic50_B, hill_n_B)
    bliss_expected = compute_bliss_independence(effect_A, effect_B)
    combo_effect = observed_effect if observed_effect is not None else bliss_expected
    bliss_delta = combo_effect - bliss_expected

    loewe_ci = compute_loewe_ci(dose_A, dose_B, ic50_A, ic50_B, combo_effect, hill_n_A, hill_n_B)

    # CI classification
    if math.isnan(loewe_ci):
        ci_class = "undefined"
    elif loewe_ci < 0.9:
        ci_class = "synergy"
    elif loewe_ci <= 1.1:
        ci_class = "additivity"
    else:
        ci_class = "antagonism"

    # Bliss classification
    if bliss_delta > 0.1:
        bliss_class = "synergy"
    elif bliss_delta < -0.1:
        bliss_class = "antagonism"
    else:
        bliss_class = "additivity"

    warnings: list[str] = []
    if combo_effect > 0.95:
        warnings.append("Near-maximal effect — index unreliable")
    if math.isnan(loewe_ci):
        warnings.append("Loewe CI undefined (effect near 0 or 1)")

    return SynergyResult(
        drug_A=drug_A,
        drug_B=drug_B,
        dose_A=dose_A,
        dose_B=dose_B,
        effect_A_alone=effect_A,
        effect_B_alone=effect_B,
        combination_effect=combo_effect,
        bliss_expected_effect=bliss_expected,
        bliss_delta=bliss_delta,
        loewe_ci=loewe_ci,
        ci_classification=ci_class,
        bliss_classification=bliss_class,
        warnings=warnings,
    )


def synergy_matrix(
    drug_A: str,
    drug_B: str,
    doses_A: list[float],
    doses_B: list[float],
    ic50_A: float,
    ic50_B: float,
    hill_n_A: float = 1.0,
    hill_n_B: float = 1.0,
) -> list[list[SynergyResult]]:
    """Return 2D list [i][j] = SynergyResult for doses_A[i] x doses_B[j]."""
    return [
        [
            assess_combination_synergy(drug_A, drug_B, da, db, ic50_A, ic50_B, hill_n_A, hill_n_B)
            for db in doses_B
        ]
        for da in doses_A
    ]
