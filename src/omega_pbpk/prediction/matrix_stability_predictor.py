"""Phase 310 — Drug Stability in Biological Matrices.

Predict drug stability in blood, plasma, urine, tissue homogenate, and CSF
matrices. Useful for bioanalytical method development.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base first-order degradation rate (per hour) at 37 °C for each matrix
_MATRIX_BASE_K: dict[str, float] = {
    "whole_blood": 0.02,
    "plasma": 0.01,
    "urine": 0.005,
    "tissue_homogenate": 0.05,
    "csf": 0.002,
}

_VALID_MATRICES = frozenset(_MATRIX_BASE_K.keys())

# Arrhenius parameters
_EA_KJ_PER_MOL = 50.0  # activation energy (kJ/mol)
_R_KJ = 8.314e-3  # gas constant (kJ / mol / K)
_T_REF_K = 310.15  # 37 °C in Kelvin (reference temperature)


# ---------------------------------------------------------------------------
# Result dataclass  (frozen=True because no list fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixStabilityResult:
    """Result from a matrix stability prediction."""

    drug_name: str
    matrix_type: str
    temperature_C: float
    duration_h: float

    k_deg_per_h: float
    stability_pct: float
    analyte_recovery_pct: float

    stability_class: str  # "stable", "acceptable", or "unstable"
    recommended_storage: str  # "-80C", "-20C", or "4C"
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chemical_liability_factor(
    is_ester: bool,
    is_amide: bool,
    has_oxidizable_group: bool,
    logp: float,
) -> float:
    """Return multiplicative factor for chemical liabilities."""
    k_factor = 1.0
    if is_ester:
        k_factor *= 3.0
    if is_amide:
        k_factor *= 1.5
    if has_oxidizable_group:
        k_factor *= 2.0
    if logp > 3.0:
        k_factor *= 1.2
    return k_factor


def _arrhenius_correction(temp_C: float) -> float:
    """Return k_temp/k_ref ratio using Arrhenius equation (Ea=50 kJ/mol)."""
    t_k = temp_C + 273.15
    exponent = (_EA_KJ_PER_MOL / _R_KJ) * (1.0 / _T_REF_K - 1.0 / t_k)
    return exp(exponent)


def _stability_class(pct: float) -> str:
    if pct > 95.0:
        return "stable"
    if pct >= 85.0:
        return "acceptable"
    return "unstable"


def _recommended_storage(stability_cls: str) -> str:
    mapping = {
        "stable": "4C",
        "acceptable": "-20C",
        "unstable": "-80C",
    }
    return mapping[stability_cls]


def _validate_matrix(matrix_type: str) -> None:
    if matrix_type not in _VALID_MATRICES:
        raise ValueError(
            f"matrix_type must be one of {sorted(_VALID_MATRICES)}, got '{matrix_type}'"
        )


def _validate_temperature(temperature_C: float) -> None:
    if not (-200.0 < temperature_C < 100.0):
        raise ValueError(f"temperature_C must be in (-200, 100), got {temperature_C}")


def _validate_duration(duration_h: float) -> None:
    if duration_h <= 0:
        raise ValueError(f"duration_h must be > 0, got {duration_h}")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def predict_matrix_stability(
    drug_name: str,
    matrix_type: str,
    temperature_C: float,
    duration_h: float,
    logp: float = 2.0,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    is_ester: bool = False,
    is_amide: bool = False,
    has_oxidizable_group: bool = False,
) -> MatrixStabilityResult:
    """Predict drug stability in a biological matrix.

    Parameters
    ----------
    drug_name:
        Name or identifier for the drug.
    matrix_type:
        One of: "whole_blood", "plasma", "urine", "tissue_homogenate", "csf".
    temperature_C:
        Storage / incubation temperature in degrees Celsius.
    duration_h:
        Duration of storage or incubation (h).
    logp:
        Octanol-water partition coefficient (log scale).
    pka:
        pKa of the ionizable group (informational).
    ionization_type:
        "acid", "base", or "neutral" (informational).
    is_ester:
        True if the drug contains an ester moiety (vulnerable to hydrolysis).
    is_amide:
        True if the drug contains an amide bond.
    has_oxidizable_group:
        True if the drug has an easily oxidizable functional group.

    Returns
    -------
    MatrixStabilityResult
    """
    _validate_matrix(matrix_type)
    _validate_temperature(temperature_C)
    _validate_duration(duration_h)

    k_base = _MATRIX_BASE_K[matrix_type]
    k_factor = _chemical_liability_factor(is_ester, is_amide, has_oxidizable_group, logp)
    k_temp_ratio = _arrhenius_correction(temperature_C)

    k_deg = k_base * k_factor * k_temp_ratio
    remaining = 100.0 * exp(-k_deg * duration_h)
    # Clamp to (0, 100]
    remaining = max(min(remaining, 100.0), 1e-6)

    cls = _stability_class(remaining)
    storage = _recommended_storage(cls)

    liability_notes = []
    if is_ester:
        liability_notes.append("ester(x3)")
    if is_amide:
        liability_notes.append("amide(x1.5)")
    if has_oxidizable_group:
        liability_notes.append("oxidizable(x2)")
    if logp > 3.0:
        liability_notes.append("lipophilic(x1.2)")

    notes = (
        f"k_base={k_base:.4f}/h; k_factor={k_factor:.3f}; "
        f"Arrhenius_ratio={k_temp_ratio:.4f}; liabilities=[{', '.join(liability_notes)}]; "
        f"pKa={pka:.2f}; ionization={ionization_type}"
    )

    return MatrixStabilityResult(
        drug_name=drug_name,
        matrix_type=matrix_type,
        temperature_C=temperature_C,
        duration_h=duration_h,
        k_deg_per_h=k_deg,
        stability_pct=remaining,
        analyte_recovery_pct=remaining,
        stability_class=cls,
        recommended_storage=storage,
        notes=notes,
    )


def assess_freeze_thaw_stability(
    drug_name: str,
    n_cycles: int,
    logp: float = 2.0,
    is_ester: bool = False,
    has_oxidizable_group: bool = False,
    matrix_type: str = "plasma",
) -> MatrixStabilityResult:
    """Assess stability over multiple freeze-thaw cycles.

    Each cycle consists of:
    - Warm phase: 25 °C for 1 h (degradation occurs)
    - Cold phase: -80 °C (no degradation)

    Parameters
    ----------
    drug_name:
        Name or identifier for the drug.
    n_cycles:
        Number of freeze-thaw cycles (>= 1).
    logp:
        logP value.
    is_ester:
        True if the drug contains an ester moiety.
    has_oxidizable_group:
        True if the drug has an oxidizable functional group.
    matrix_type:
        Matrix used; defaults to "plasma".

    Returns
    -------
    MatrixStabilityResult
    """
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
    _validate_matrix(matrix_type)

    k_base = _MATRIX_BASE_K[matrix_type]
    # Freeze-thaw uses only ester and oxidizable group liabilities
    k_factor = 1.0
    if is_ester:
        k_factor *= 3.0
    if has_oxidizable_group:
        k_factor *= 2.0
    if logp > 3.0:
        k_factor *= 1.2

    # Warm phase at 25 °C, 1 h per cycle
    warm_temp_C = 25.0
    k_temp_ratio = _arrhenius_correction(warm_temp_C)
    k_warm = k_base * k_factor * k_temp_ratio * 1.5  # 1.5 per spec

    # Total degradation: 1 h per warm cycle
    remaining = 100.0 * exp(-k_warm * n_cycles)
    remaining = max(min(remaining, 100.0), 1e-6)

    cls = _stability_class(remaining)
    storage = _recommended_storage(cls)

    notes = (
        f"Freeze-thaw: {n_cycles} cycles; k_warm={k_warm:.5f}/h; "
        f"25C warm phase 1h/cycle; k_factor={k_factor:.3f}"
    )

    return MatrixStabilityResult(
        drug_name=drug_name,
        matrix_type=matrix_type,
        temperature_C=warm_temp_C,
        duration_h=float(n_cycles),
        k_deg_per_h=k_warm,
        stability_pct=remaining,
        analyte_recovery_pct=remaining,
        stability_class=cls,
        recommended_storage=storage,
        notes=notes,
    )
