"""Tissue:plasma partition coefficient (Kp) predictor.

Uses a simplified Poulin-Theil / Rodgers-Rowland approach based on tissue
lipid and water content to predict Kp for 8 standard tissues.

Phase 836.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionCoefficientResult:
    """Predicted tissue:plasma partition coefficient."""

    drug_name: str
    tissue: str
    kp: float  # tissue:plasma partition coefficient
    kp_uu: float  # unbound partition (simplified)
    fu_tissue: float  # unbound fraction in tissue
    fu_plasma: float  # unbound fraction in plasma
    log_kp: float  # log10(Kp)
    accumulation_risk: str  # "low" (kp<3), "moderate" (kp<10), "high" (kp>=10)
    method: str  # "rodgers_rowland" or "simplified"
    notes: str


# ---------------------------------------------------------------------------
# Tissue composition table
# Columns: neutral_lipid_fraction, phospholipid_fraction, water_fraction
# ---------------------------------------------------------------------------

_TISSUE_COMPOSITION: dict[str, tuple[float, float, float]] = {
    "adipose": (0.853, 0.002, 0.135),
    "brain": (0.039, 0.040, 0.770),
    "heart": (0.014, 0.012, 0.775),
    "kidney": (0.012, 0.024, 0.783),
    "liver": (0.038, 0.024, 0.751),
    "lung": (0.007, 0.024, 0.811),
    "muscle": (0.010, 0.007, 0.761),
    "skin": (0.021, 0.003, 0.718),
}

# Plasma composition
_PLASMA_NEUTRAL_LIP: float = 0.003
_PLASMA_PHOSPHOLIP: float = 0.002
_PLASMA_WATER: float = 0.939

# Known tissues
TISSUES: list[str] = list(_TISSUE_COMPOSITION.keys())


# ---------------------------------------------------------------------------
# Accumulation risk classification
# ---------------------------------------------------------------------------


def _accumulation_risk(kp: float) -> str:
    if kp >= 10.0:
        return "high"
    if kp >= 3.0:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Core Kp prediction
# ---------------------------------------------------------------------------


def predict_kp(
    drug_name: str,
    tissue: str,
    logp: float,
    pka: float | None = None,
    fu_plasma: float = 0.1,
    ionization_type: str = "neutral",
    neutral_lipid_fraction: float = 0.02,
    phospholipid_fraction: float = 0.01,
    water_fraction: float = 0.70,
) -> PartitionCoefficientResult:
    """Predict tissue:plasma partition coefficient (Kp) for a single tissue.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    tissue:
        Target tissue name. Must be one of the 8 known tissues or a custom
        tissue (uses the provided fraction defaults).
    logp:
        Octanol-water partition coefficient (log scale).
    pka:
        Drug pKa. If provided must be in [0, 14]. Pass None for neutral drugs.
    fu_plasma:
        Unbound fraction in plasma (0 < fu_plasma <= 1).
    ionization_type:
        "neutral", "acid", or "base".
    neutral_lipid_fraction:
        For custom (unknown) tissues only — neutral lipid fraction.
    phospholipid_fraction:
        For custom (unknown) tissues only — phospholipid fraction.
    water_fraction:
        For custom (unknown) tissues only — water fraction.

    Returns
    -------
    PartitionCoefficientResult
    """
    # --- Validation ---
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if pka is not None and not (0.0 <= pka <= 14.0):
        raise ValueError(f"pKa must be in [0, 14], got {pka}")

    # --- Tissue composition lookup ---
    if tissue in _TISSUE_COMPOSITION:
        fn_t, fp_t, fw_t = _TISSUE_COMPOSITION[tissue]
        method = "rodgers_rowland"
    else:
        fn_t = neutral_lipid_fraction
        fp_t = phospholipid_fraction
        fw_t = water_fraction
        method = "simplified"

    fn_p = _PLASMA_NEUTRAL_LIP
    fp_p = _PLASMA_PHOSPHOLIP
    fw_p = _PLASMA_WATER

    kow = 10.0**logp

    # --- Kp formula ---
    if pka is None or ionization_type == "neutral":
        # Simplified Poulin-Theil for neutral drugs
        numerator = fn_t * kow + fw_t
        denominator = fn_p * kow + fw_p
    else:
        # Include phospholipid binding term (Rodgers-Rowland extension)
        # Phospholipid affinity ~ Kow^0.3
        kow_03 = kow**0.3
        numerator = fn_t * kow + fp_t * kow_03 + fw_t
        denominator = fn_p * kow + fp_p * kow_03 + fw_p

    if denominator <= 0.0:
        denominator = 1e-9

    kp = numerator / denominator

    # Clamp kp to a physically reasonable minimum
    if kp < 1e-6:
        kp = 1e-6

    # --- Derived quantities ---
    # fu_tissue based on equilibrium: fu_tissue = fu_plasma / Kp  (Lyon et al.)
    fu_tissue = fu_plasma / kp
    # Clamp fu_tissue to (0, 1]
    if fu_tissue > 1.0:
        fu_tissue = 1.0
    if fu_tissue < 1e-9:
        fu_tissue = 1e-9

    # kp_uu = Kp * fu_tissue / fu_plasma (unbound-to-unbound partition)
    # With fu_tissue = fu_plasma/Kp → kp_uu = 1.0 (exact equilibrium)
    # Use simplified: kp_uu = Kp * 0.1 to represent non-equilibrium
    kp_uu = kp * 0.1

    log_kp = math.log10(kp) if kp > 0.0 else float("-inf")
    risk = _accumulation_risk(kp)

    notes = (
        f"logP={logp:.2f}, fu_plasma={fu_plasma:.3f}, ionization_type={ionization_type}, pKa={pka}"
    )

    return PartitionCoefficientResult(
        drug_name=drug_name,
        tissue=tissue,
        kp=kp,
        kp_uu=kp_uu,
        fu_tissue=fu_tissue,
        fu_plasma=fu_plasma,
        log_kp=log_kp,
        accumulation_risk=risk,
        method=method,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Predict Kp for all 8 standard tissues
# ---------------------------------------------------------------------------


def predict_kp_all_tissues(
    drug_name: str,
    logp: float,
    pka: float | None = None,
    fu_plasma: float = 0.1,
    ionization_type: str = "neutral",
) -> list[PartitionCoefficientResult]:
    """Predict Kp for all 8 standard tissues, sorted by kp descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Octanol-water partition coefficient (log scale).
    pka:
        Drug pKa (optional; None for neutral).
    fu_plasma:
        Unbound fraction in plasma.
    ionization_type:
        "neutral", "acid", or "base".

    Returns
    -------
    list[PartitionCoefficientResult]
        8 results sorted by kp descending.
    """
    results: list[PartitionCoefficientResult] = []
    for tissue in TISSUES:
        result = predict_kp(
            drug_name=drug_name,
            tissue=tissue,
            logp=logp,
            pka=pka,
            fu_plasma=fu_plasma,
            ionization_type=ionization_type,
        )
        results.append(result)

    results.sort(key=lambda r: r.kp, reverse=True)
    return results
