"""Drug degradation in biological matrices — Phase 1092.

Predicts pseudo-first-order degradation rates for drugs in:
  plasma, whole_blood, urine, liver_s9, brain_homogenate.

Degradation rate is modulated by:
  - Functional group multipliers (ester, amide, lactam, hydroxamate)
  - Arrhenius temperature correction (Ea = 60 kJ/mol, T_ref = 37°C)
  - Matrix-specific activity factors

Critical for IVIVE and bioanalytical method development.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Supported biological matrices
_VALID_MATRICES = {
    "plasma",
    "whole_blood",
    "urine",
    "liver_s9",
    "brain_homogenate",
}

# Matrix-specific base activity multipliers
_MATRIX_BASE_FACTOR: dict = {
    "plasma": 1.0,
    "whole_blood": 1.8,  # faster esterase + erythrocyte metabolism
    "urine": 0.7,  # pH-dependent hydrolysis, generally slower
    "liver_s9": 2.5,  # CYP + UGT activity
    "brain_homogenate": 1.5,  # MAO + esterases
}

# Functional group multipliers per matrix
# Keys: (has_ester, has_amide, has_lactam, has_hydroxamate)
# Values: dict {matrix: multiplier}
_FG_MULTIPLIERS: dict = {
    "has_ester": {
        "plasma": 3.0,
        "whole_blood": 5.0,
        "urine": 1.0,
        "liver_s9": 1.5,
        "brain_homogenate": 3.0,
    },
    "has_amide": {
        "plasma": 1.0,
        "whole_blood": 1.0,
        "urine": 1.0,
        "liver_s9": 1.5,
        "brain_homogenate": 1.0,
    },
    "has_lactam": {
        "plasma": 1.5,
        "whole_blood": 1.5,
        "urine": 2.0,
        "liver_s9": 1.5,
        "brain_homogenate": 1.5,
    },
    "has_hydroxamate": {
        "plasma": 4.0,
        "whole_blood": 4.0,
        "urine": 2.0,
        "liver_s9": 2.0,
        "brain_homogenate": 2.0,
    },
}

# Arrhenius parameters
_EA_J_PER_MOL = 60_000.0  # 60 kJ/mol
_R_J_PER_MOL_K = 8.314
_T_REF_K = 310.15  # 37°C in Kelvin

# Stability class thresholds (t½ in hours)
_STABLE_THR = 24.0
_LABILE_THR = 4.0


@dataclass(frozen=True)
class MatrixStabilityResult:
    """Result from predicting drug stability in a biological matrix."""

    drug_name: str
    matrix: str
    temperature_C: float
    k_degradation_per_h: float
    t_half_h: float
    stability_at_1h_pct: float
    stability_at_4h_pct: float
    stability_at_24h_pct: float
    stability_class: str
    bioanalytical_recommendation: str
    notes: str


def _temperature_factor(temp_C: float) -> float:
    """Arrhenius temperature correction relative to 37°C."""
    t_k = temp_C + 273.15
    exponent = (_EA_J_PER_MOL / _R_J_PER_MOL_K) * (1.0 / _T_REF_K - 1.0 / t_k)
    return math.exp(exponent)


def _stability_class(t_half_h: float) -> str:
    if t_half_h > _STABLE_THR:
        return "stable"
    if t_half_h >= _LABILE_THR:
        return "moderate"
    return "labile"


def _bioanalytical_recommendation(stability_class: str, matrix: str, t_half_h: float) -> str:
    if stability_class == "stable":
        return (
            f"Stable in {matrix} (t½={t_half_h:.1f} h). "
            "Standard storage at -80°C acceptable; samples can be processed at room temperature."
        )
    if stability_class == "moderate":
        return (
            f"Moderate stability in {matrix} (t½={t_half_h:.1f} h). "
            "Process samples on ice; store at -80°C within 4 h of collection; "
            "add protease inhibitors if peptide."
        )
    # labile
    return (
        f"Labile in {matrix} (t½={t_half_h:.1f} h). "
        "Immediate processing required; use stabilising additives (e.g., esterase inhibitors, "
        "NaF/KOx for blood); keep on dry ice during handling."
    )


def predict_matrix_stability(
    drug_name: str,
    matrix: str,
    has_ester: bool = False,
    has_amide: bool = False,
    has_lactam: bool = False,
    has_hydroxamate: bool = False,
    temperature_C: float = 37.0,
    base_k_per_h: float = 0.005,
) -> MatrixStabilityResult:
    """Predict pseudo-first-order degradation rate in a biological matrix.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    matrix:
        Biological matrix. One of: plasma, whole_blood, urine,
        liver_s9, brain_homogenate.
    has_ester:
        Drug contains an ester functional group.
    has_amide:
        Drug contains an amide functional group.
    has_lactam:
        Drug contains a lactam (cyclic amide) functional group.
    has_hydroxamate:
        Drug contains a hydroxamate functional group.
    temperature_C:
        Incubation/storage temperature (°C). Valid range: [0, 60].
    base_k_per_h:
        Base degradation rate constant at 37°C with no functional groups (1/h).

    Returns
    -------
    MatrixStabilityResult
    """
    if matrix not in _VALID_MATRICES:
        raise ValueError(f"matrix must be one of {sorted(_VALID_MATRICES)}, got '{matrix}'")
    if not (0.0 <= temperature_C <= 60.0):
        raise ValueError(f"temperature_C must be in [0, 60], got {temperature_C}")
    if base_k_per_h <= 0.0:
        raise ValueError("base_k_per_h must be > 0")

    # Functional group multiplier (additive on top of 1.0 baseline)
    fg_mult = 1.0
    if has_ester:
        fg_mult *= _FG_MULTIPLIERS["has_ester"][matrix]
    if has_amide:
        fg_mult *= _FG_MULTIPLIERS["has_amide"][matrix]
    if has_lactam:
        fg_mult *= _FG_MULTIPLIERS["has_lactam"][matrix]
    if has_hydroxamate:
        fg_mult *= _FG_MULTIPLIERS["has_hydroxamate"][matrix]

    matrix_factor = _MATRIX_BASE_FACTOR[matrix]
    temp_factor = _temperature_factor(temperature_C)

    k_deg = base_k_per_h * fg_mult * matrix_factor * temp_factor

    t_half = math.log(2.0) / k_deg

    stab_1h = 100.0 * math.exp(-k_deg * 1.0)
    stab_4h = 100.0 * math.exp(-k_deg * 4.0)
    stab_24h = 100.0 * math.exp(-k_deg * 24.0)

    s_class = _stability_class(t_half)
    recommendation = _bioanalytical_recommendation(s_class, matrix, t_half)

    fg_flags = []
    if has_ester:
        fg_flags.append("ester")
    if has_amide:
        fg_flags.append("amide")
    if has_lactam:
        fg_flags.append("lactam")
    if has_hydroxamate:
        fg_flags.append("hydroxamate")
    fg_str = ", ".join(fg_flags) if fg_flags else "none"

    notes = (
        f"Matrix={matrix}; T={temperature_C}°C; functional groups=[{fg_str}]; "
        f"base_k={base_k_per_h}/h; matrix_factor={matrix_factor}; "
        f"temp_factor={temp_factor:.3f}."
    )

    return MatrixStabilityResult(
        drug_name=drug_name,
        matrix=matrix,
        temperature_C=temperature_C,
        k_degradation_per_h=k_deg,
        t_half_h=t_half,
        stability_at_1h_pct=stab_1h,
        stability_at_4h_pct=stab_4h,
        stability_at_24h_pct=stab_24h,
        stability_class=s_class,
        bioanalytical_recommendation=recommendation,
        notes=notes,
    )


def screen_matrices(
    drug_name: str,
    has_ester: bool = False,
    has_amide: bool = False,
    has_lactam: bool = False,
    has_hydroxamate: bool = False,
    base_k_per_h: float = 0.005,
) -> list:
    """Screen drug stability across all 5 biological matrices at 37°C.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    has_ester, has_amide, has_lactam, has_hydroxamate:
        Functional group flags.
    base_k_per_h:
        Base degradation rate constant (1/h).

    Returns
    -------
    list[MatrixStabilityResult]
        Results for all 5 matrices, sorted by k_degradation_per_h ascending
        (most stable first).
    """
    results = []
    for matrix in _VALID_MATRICES:
        r = predict_matrix_stability(
            drug_name=drug_name,
            matrix=matrix,
            has_ester=has_ester,
            has_amide=has_amide,
            has_lactam=has_lactam,
            has_hydroxamate=has_hydroxamate,
            temperature_C=37.0,
            base_k_per_h=base_k_per_h,
        )
        results.append(r)

    results.sort(key=lambda r: r.k_degradation_per_h)
    return results
