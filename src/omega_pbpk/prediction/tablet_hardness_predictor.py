"""Phase 231 — Tablet Hardness Predictor.

Predicts tablet hardness and friability from compaction properties using
empirical models based on compression force, excipient type, drug loading,
and particle size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Excipient lookup tables
_HMAX: dict[str, float] = {
    "MCC": 200.0,
    "DCPA": 120.0,
    "starch": 80.0,
    "spray_dried_lactose": 100.0,
    "mannitol": 110.0,
}

_EXCIPIENT_FACTOR: dict[str, float] = {
    "MCC": 1.0,
    "DCPA": 0.8,
    "starch": 0.6,
    "spray_dried_lactose": 0.75,
    "mannitol": 0.85,
}

_K_COMP = 0.1  # compressibility constant (1/kN)

VALID_EXCIPIENTS = frozenset(_HMAX.keys())


@dataclass(frozen=True)
class TabletHardnessResult:
    """Result of tablet hardness and friability prediction."""

    drug_name: str
    compression_force_kN: float
    excipient_type: str
    drug_loading_pct: float
    particle_size_um: float
    hardness_N: float
    friability_pct: float
    quality_status: str
    notes: str


def _validate_inputs(
    compression_force_kN: float,
    drug_loading_pct: float,
    excipient_type: str,
) -> None:
    """Validate inputs and raise ValueError on invalid parameters."""
    if compression_force_kN <= 0:
        raise ValueError(f"compression_force_kN must be > 0, got {compression_force_kN}")
    if not (0.0 <= drug_loading_pct <= 100.0):
        raise ValueError(f"drug_loading_pct must be in [0, 100], got {drug_loading_pct}")
    if excipient_type not in VALID_EXCIPIENTS:
        raise ValueError(
            f"excipient_type '{excipient_type}' not recognised. "
            f"Valid options: {sorted(VALID_EXCIPIENTS)}"
        )


def predict_tablet_hardness(
    drug_name: str,
    compression_force_kN: float,
    excipient_type: str,
    drug_loading_pct: float,
    particle_size_um: float = 100.0,
    notes: str = "",
) -> TabletHardnessResult:
    """Predict tablet hardness and friability from compaction properties.

    Parameters
    ----------
    drug_name:
        Identifier for the drug / formulation.
    compression_force_kN:
        Applied compression force in kN (must be > 0).
    excipient_type:
        Primary excipient: MCC, DCPA, starch, spray_dried_lactose, or mannitol.
    drug_loading_pct:
        Drug loading as weight percent in [0, 100].
    particle_size_um:
        Mean particle size in micrometres (default 100 µm).
    notes:
        Optional free-text notes appended to the result.

    Returns
    -------
    TabletHardnessResult
        Dataclass with hardness, friability, and quality status.
    """
    _validate_inputs(compression_force_kN, drug_loading_pct, excipient_type)

    h_max = _HMAX[excipient_type]
    excipient_factor = _EXCIPIENT_FACTOR[excipient_type]
    loading_factor = max(0.3, 1.0 - 0.005 * drug_loading_pct)
    particle_size_factor = max(0.7, 1.0 - 0.001 * particle_size_um)

    hardness_N = (
        h_max
        * (1.0 - math.exp(-_K_COMP * compression_force_kN))
        * excipient_factor
        * loading_factor
        * particle_size_factor
    )

    # Friability clamped to [0, 5] %
    friability_pct = max(0.0, min(5.0, 2.0 - hardness_N / 50.0))

    # Quality assessment
    if hardness_N >= 40.0 and friability_pct <= 1.0:
        quality_status = "acceptable"
    else:
        quality_status = "failing"

    # Build notes string
    result_notes = (
        notes
        if notes
        else (
            f"loading_factor={loading_factor:.3f}, particle_size_factor={particle_size_factor:.3f}"
        )
    )

    return TabletHardnessResult(
        drug_name=drug_name,
        compression_force_kN=compression_force_kN,
        excipient_type=excipient_type,
        drug_loading_pct=drug_loading_pct,
        particle_size_um=particle_size_um,
        hardness_N=round(hardness_N, 4),
        friability_pct=round(friability_pct, 4),
        quality_status=quality_status,
        notes=result_notes,
    )


def optimize_compression_force(
    drug_name: str,
    excipient_type: str,
    drug_loading_pct: float,
    target_hardness_N: float,
    particle_size_um: float = 100.0,
    max_force_kN: float = 50.0,
    tolerance_N: float = 0.1,
) -> dict:
    """Find the minimum compression force to achieve a target hardness.

    Inverts the hardness model analytically where possible and verifies
    with the forward prediction.

    Parameters
    ----------
    drug_name:
        Identifier for the drug / formulation.
    excipient_type:
        Primary excipient.
    drug_loading_pct:
        Drug loading weight percent [0, 100].
    target_hardness_N:
        Desired tablet hardness in Newtons.
    particle_size_um:
        Mean particle size in µm (default 100 µm).
    max_force_kN:
        Upper bound for compression force search (default 50 kN).
    tolerance_N:
        Acceptable deviation from target hardness (default 0.1 N).

    Returns
    -------
    dict with keys:
        min_force_kN, expected_hardness_N, expected_friability_pct,
        quality_status, achievable (bool)
    """
    _validate_inputs(1.0, drug_loading_pct, excipient_type)  # force dummy > 0

    h_max = _HMAX[excipient_type]
    excipient_factor = _EXCIPIENT_FACTOR[excipient_type]
    loading_factor = max(0.3, 1.0 - 0.005 * drug_loading_pct)
    particle_size_factor = max(0.7, 1.0 - 0.001 * particle_size_um)

    # effective maximum achievable hardness (force → ∞)
    h_effective_max = h_max * excipient_factor * loading_factor * particle_size_factor

    achievable = target_hardness_N <= h_effective_max

    if achievable:
        # Analytical inversion: H = H_eff_max * (1 - exp(-k * F))
        # => exp(-k*F) = 1 - H/H_eff_max
        ratio = target_hardness_N / h_effective_max
        ratio = min(ratio, 1.0 - 1e-9)  # guard against log(0)
        min_force_kN = -math.log(1.0 - ratio) / _K_COMP
    else:
        min_force_kN = max_force_kN

    # Compute actual expected hardness at the derived force
    result = predict_tablet_hardness(
        drug_name=drug_name,
        compression_force_kN=min_force_kN,
        excipient_type=excipient_type,
        drug_loading_pct=drug_loading_pct,
        particle_size_um=particle_size_um,
    )

    return {
        "min_force_kN": round(min_force_kN, 4),
        "expected_hardness_N": result.hardness_N,
        "expected_friability_pct": result.friability_pct,
        "quality_status": result.quality_status,
        "achievable": achievable,
    }
