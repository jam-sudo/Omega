"""Phase 624 — Drug Tissue pH Effect on Distribution.

Models the effect of tissue pH on ionizable drug distribution into different
tissues using Henderson-Hasselbalch partitioning.

Pure Python stdlib only (math, dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass

# Standard tissue pH values
_TISSUE_PH: dict[str, float] = {
    "plasma": 7.4,
    "liver": 7.0,
    "muscle": 7.1,
    "kidney": 7.2,
    "lung": 7.4,
    "tumor": 6.5,
    "lysosome": 5.0,
    "stomach": 2.0,
    "colon": 6.5,
    "brain": 7.3,
}

_DEFAULT_TISSUE_PH = 7.2

# Standard tissues used for comparison
_STANDARD_TISSUES = [
    "plasma",
    "liver",
    "muscle",
    "kidney",
    "lung",
    "tumor",
    "lysosome",
    "brain",
]

_VALID_DRUG_TYPES = ("acid", "base", "neutral")


@dataclass(frozen=True)
class TissuePHDistributionResult:
    """Result of tissue pH-dependent drug distribution prediction."""

    drug_name: str
    drug_type: str  # "acid", "base", "neutral"
    pka: float
    ph_plasma: float
    ph_tissue: float
    tissue: str
    kp_ph_corrected: float  # pH-corrected Kp
    kp_neutral: float  # Kp without pH effect (logP-based)
    ph_partition_factor: float  # kp_ph_corrected / kp_neutral
    fu_tissue: float  # unbound fraction in tissue
    accumulation_risk: str  # "low"/"moderate"/"high"
    lysosomotropic: bool  # True if basic drug trapped in acidic lysosomes
    notes: str


def predict_tissue_ph_distribution(
    drug_name: str,
    logP: float,
    pka: float,
    drug_type: str,
    tissue: str,
    fu_plasma: float = 0.1,
) -> TissuePHDistributionResult:
    """Predict tissue pH effect on drug distribution.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logP:
        Octanol-water partition coefficient.
    pka:
        Acid dissociation constant of the ionizable group.
    drug_type:
        "acid", "base", or "neutral".
    tissue:
        Target tissue name. Must be one of the known tissues or defaults to pH 7.2.
    fu_plasma:
        Unbound fraction in plasma (0-1).

    Returns
    -------
    TissuePHDistributionResult
        Frozen dataclass with pH-corrected distribution parameters.
    """
    # Validation
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {_VALID_DRUG_TYPES}, got '{drug_type}'")

    ph_plasma = 7.4
    ph_tissue = _TISSUE_PH.get(tissue, _DEFAULT_TISSUE_PH)

    # --- pH-dependent partition factor (Henderson-Hasselbalch) ---
    if drug_type == "neutral":
        ph_partition_factor = 1.0
    elif drug_type == "acid":
        # Acid: ionized form increases with pH; tissue accumulation when pH_tissue < pH_plasma
        numerator = 1.0 + 10.0 ** (ph_plasma - pka)
        denominator = 1.0 + 10.0 ** (ph_tissue - pka)
        ph_partition_factor = numerator / max(1e-12, denominator)
    else:  # base
        # Base: ionized form increases with lower pH; tissue accumulation when pH_tissue < pH_plasma
        numerator = 1.0 + 10.0 ** (pka - ph_plasma)
        denominator = 1.0 + 10.0 ** (pka - ph_tissue)
        ph_partition_factor = numerator / max(1e-12, denominator)

    # --- Neutral Kp (logP-based) ---
    kp_neutral = max(0.01, logP * 0.5 + 0.3)

    # --- pH-corrected Kp ---
    kp_ph_corrected_raw = kp_neutral * ph_partition_factor
    kp_ph_corrected = max(0.001, min(1000.0, kp_ph_corrected_raw))

    # --- Unbound fraction in tissue ---
    fu_tissue_raw = fu_plasma / kp_ph_corrected
    fu_tissue = max(1e-6, min(1.0, fu_tissue_raw))

    # --- Accumulation risk ---
    if kp_ph_corrected < 2.0:
        accumulation_risk = "low"
    elif kp_ph_corrected < 10.0:
        accumulation_risk = "moderate"
    else:
        accumulation_risk = "high"

    # --- Lysosomotropic classification ---
    lysosomotropic = drug_type == "base" and pka > 6.0

    # --- Notes ---
    note_parts = []
    if lysosomotropic:
        note_parts.append(f"Lysosomotropic: basic drug (pKa={pka:.1f}) trapped in acidic lysosomes")
    note_parts.append(
        f"Tissue pH={ph_tissue:.1f} vs plasma pH={ph_plasma:.1f}; "
        f"pH partition factor={ph_partition_factor:.3f}"
    )
    if accumulation_risk == "high":
        note_parts.append(f"High tissue accumulation risk (Kp={kp_ph_corrected:.2f})")
    notes = "; ".join(note_parts)

    return TissuePHDistributionResult(
        drug_name=drug_name,
        drug_type=drug_type,
        pka=pka,
        ph_plasma=ph_plasma,
        ph_tissue=ph_tissue,
        tissue=tissue,
        kp_ph_corrected=kp_ph_corrected,
        kp_neutral=kp_neutral,
        ph_partition_factor=ph_partition_factor,
        fu_tissue=fu_tissue,
        accumulation_risk=accumulation_risk,
        lysosomotropic=lysosomotropic,
        notes=notes,
    )


def compare_tissues(
    drug_name: str,
    logP: float,
    pka: float,
    drug_type: str,
    fu_plasma: float = 0.1,
) -> list[TissuePHDistributionResult]:
    """Compare drug distribution across standard tissues.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logP:
        Octanol-water partition coefficient.
    pka:
        Acid dissociation constant.
    drug_type:
        "acid", "base", or "neutral".
    fu_plasma:
        Unbound fraction in plasma.

    Returns
    -------
    list[TissuePHDistributionResult]
        Results for standard tissues sorted by kp_ph_corrected descending.
    """
    results: list[TissuePHDistributionResult] = []
    for tissue in _STANDARD_TISSUES:
        result = predict_tissue_ph_distribution(
            drug_name=drug_name,
            logP=logP,
            pka=pka,
            drug_type=drug_type,
            tissue=tissue,
            fu_plasma=fu_plasma,
        )
        results.append(result)

    results.sort(key=lambda r: r.kp_ph_corrected, reverse=True)
    return results
