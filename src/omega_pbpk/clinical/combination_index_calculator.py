"""Phase 240: Combination Index Calculator using Chou-Talalay method.

Calculates combination index (CI) for drug combinations to classify
synergy, additivity, or antagonism.
"""

from dataclasses import dataclass

VALID_EFFECT_TYPES = ("cytotoxicity", "inhibition", "antimicrobial")


@dataclass(frozen=True)
class CombinationIndexResult:
    """Result of Chou-Talalay combination index calculation."""

    drug1_name: str
    drug2_name: str
    effect_type: str
    drug1_dose: float
    drug2_dose: float
    drug1_ic50: float
    drug2_ic50: float
    drug1_m: float
    drug2_m: float
    fa: float
    combination_index: float
    synergy_class: str
    dose_reduction_index_1: float
    dose_reduction_index_2: float
    notes: str


def _synergy_class(ci: float) -> str:
    """Classify synergy based on combination index."""
    if ci < 0.9:
        return "synergistic"
    elif ci <= 1.1:
        return "additive"
    else:
        return "antagonistic"


def _d_alone(ic50: float, m: float, fa: float) -> float:
    """Dose of a single drug needed to achieve effect fraction fa.

    Chou-Talalay: D = IC50 * (fa / (1 - fa))^(1/m)
    """
    return ic50 * ((fa / (1.0 - fa)) ** (1.0 / m))


def calculate_combination_index(
    drug1_name: str,
    drug2_name: str,
    drug1_dose: float,
    drug2_dose: float,
    drug1_ic50: float,
    drug2_ic50: float,
    drug1_m: float = 1.0,
    drug2_m: float = 1.0,
    fa: float = 0.5,
    effect_type: str = "inhibition",
) -> CombinationIndexResult:
    """Calculate combination index using the Chou-Talalay method.

    Parameters
    ----------
    drug1_name, drug2_name:
        Names of the two drugs.
    drug1_dose, drug2_dose:
        Doses used in combination.
    drug1_ic50, drug2_ic50:
        IC50 values for each drug alone.
    drug1_m, drug2_m:
        Hill (sigmoidicity) coefficients.
    fa:
        Fraction affected (0 < fa < 1).
    effect_type:
        One of 'cytotoxicity', 'inhibition', 'antimicrobial'.

    Returns
    -------
    CombinationIndexResult
    """
    # Validation
    if drug1_dose <= 0:
        raise ValueError("drug1_dose must be > 0")
    if drug2_dose <= 0:
        raise ValueError("drug2_dose must be > 0")
    if drug1_ic50 <= 0:
        raise ValueError("drug1_ic50 must be > 0")
    if drug2_ic50 <= 0:
        raise ValueError("drug2_ic50 must be > 0")
    if drug1_m <= 0:
        raise ValueError("drug1_m must be > 0")
    if drug2_m <= 0:
        raise ValueError("drug2_m must be > 0")
    if not (0.0 < fa < 1.0):
        raise ValueError("fa must be in (0, 1)")
    if effect_type not in VALID_EFFECT_TYPES:
        raise ValueError(f"effect_type must be one of {VALID_EFFECT_TYPES}; got '{effect_type}'")

    # Doses of each drug alone to achieve fa
    d1_alone = _d_alone(drug1_ic50, drug1_m, fa)
    d2_alone = _d_alone(drug2_ic50, drug2_m, fa)

    # Combination index: CI = D1/D1_alone + D2/D2_alone
    ci = drug1_dose / d1_alone + drug2_dose / d2_alone

    # Dose reduction indices
    dri1 = d1_alone / drug1_dose
    dri2 = d2_alone / drug2_dose

    synergy = _synergy_class(ci)

    notes_parts = [
        f"D1_alone={d1_alone:.4f}, D2_alone={d2_alone:.4f}",
        f"CI={ci:.4f} ({synergy})",
        f"DRI1={dri1:.2f}, DRI2={dri2:.2f}",
    ]
    notes = "; ".join(notes_parts)

    return CombinationIndexResult(
        drug1_name=drug1_name,
        drug2_name=drug2_name,
        effect_type=effect_type,
        drug1_dose=drug1_dose,
        drug2_dose=drug2_dose,
        drug1_ic50=drug1_ic50,
        drug2_ic50=drug2_ic50,
        drug1_m=drug1_m,
        drug2_m=drug2_m,
        fa=fa,
        combination_index=ci,
        synergy_class=synergy,
        dose_reduction_index_1=dri1,
        dose_reduction_index_2=dri2,
        notes=notes,
    )


def scan_fa_range(
    drug1_name: str,
    drug2_name: str,
    drug1_dose: float,
    drug2_dose: float,
    drug1_ic50: float,
    drug2_ic50: float,
    drug1_m: float = 1.0,
    drug2_m: float = 1.0,
    fa_values: list[float] | None = None,
) -> list[CombinationIndexResult]:
    """Compute CI across a range of effect fractions.

    Parameters
    ----------
    drug1_name, drug2_name:
        Names of the two drugs.
    drug1_dose, drug2_dose:
        Fixed doses used in combination for all fa values.
    drug1_ic50, drug2_ic50:
        IC50 values.
    drug1_m, drug2_m:
        Hill coefficients.
    fa_values:
        List of fa values to evaluate. Defaults to [0.25, 0.5, 0.75, 0.9].

    Returns
    -------
    List of CombinationIndexResult sorted by fa ascending.
    """
    if fa_values is None:
        fa_values = [0.25, 0.5, 0.75, 0.9]

    results = [
        calculate_combination_index(
            drug1_name=drug1_name,
            drug2_name=drug2_name,
            drug1_dose=drug1_dose,
            drug2_dose=drug2_dose,
            drug1_ic50=drug1_ic50,
            drug2_ic50=drug2_ic50,
            drug1_m=drug1_m,
            drug2_m=drug2_m,
            fa=fa,
        )
        for fa in fa_values
    ]
    results.sort(key=lambda r: r.fa)
    return results
