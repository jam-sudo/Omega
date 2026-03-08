"""Phase 992 — Red Blood Cell (RBC) Partitioning Model.

Models drug partitioning into red blood cells using logP/pKa descriptors
based on Hinderling 1997 simplified approach.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "RBCPartitioningResult",
    "predict_rbc_partitioning",
    "compare_hematocrit_effect",
]

_VALID_IONIZATION = {"acid", "base", "neutral"}
_PHYSIOLOGICAL_PH = 7.4


@dataclass(frozen=True)
class RBCPartitioningResult:
    """Result of RBC partitioning prediction."""

    drug_name: str
    logp: float
    pka: float
    ionization_type: str
    hematocrit: float
    kp_rbc: float
    blood_plasma_ratio: float
    f_in_rbc: float
    plasma_to_blood_correction: float
    rbc_partitioning_class: str
    hemolysis_risk: bool
    notes: str


def predict_rbc_partitioning(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str = "neutral",
    hematocrit: float = 0.45,
) -> RBCPartitioningResult:
    """Predict RBC partitioning and blood-to-plasma ratio.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    logp:
        Octanol-water partition coefficient.
    pka:
        Ionization constant (pKa).
    ionization_type:
        One of "acid", "base", "neutral".
    hematocrit:
        Fractional hematocrit (0 < hct < 1, typical 0.45).

    Returns
    -------
    RBCPartitioningResult
    """
    # --- Validation ---
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if not (0.0 < hematocrit < 1.0):
        raise ValueError(f"hematocrit must be in (0, 1), got {hematocrit}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got {ionization_type!r}"
        )

    # --- Base Kp_rbc from logP (Hinderling 1997 simplified) ---
    log_kp = 0.5 * logp - 0.3
    kp_rbc = max(0.1, 10.0**log_kp)

    # --- Ionization correction at pH 7.4 ---
    note_parts: list[str] = ["Hinderling 1997 logP-based Kp_rbc"]

    if ionization_type == "acid":
        # ionized fraction for acid: f_ion = 1 / (1 + 10^(pKa - pH))
        ionized_fraction = 1.0 / (1.0 + 10.0 ** (pka - _PHYSIOLOGICAL_PH))
        kp_rbc *= 1.0 - 0.3 * ionized_fraction
        kp_rbc = max(0.1, kp_rbc)
        note_parts.append(f"acid ionization correction (f_ion={ionized_fraction:.3f})")

    elif ionization_type == "base":
        # Basic drugs sequester in RBCs via amine trapping / carbonic anhydrase
        if pka > 6.0:
            enhancement = 1.0 + 0.5 * (pka - 7.0)
            enhancement = max(1.0, min(enhancement, 10.0))
            kp_rbc *= enhancement
            note_parts.append(f"base RBC sequestration (factor={enhancement:.2f})")
    # neutral: no correction

    # --- Blood-to-plasma ratio (Rb) ---
    hct = hematocrit
    blood_plasma_ratio = (1.0 - hct) + hct * kp_rbc

    # --- Fraction of drug in RBCs ---
    f_in_rbc = (hct * kp_rbc) / (hct * kp_rbc + (1.0 - hct))

    # --- Plasma-to-blood CL correction ---
    plasma_to_blood_correction = 1.0 / blood_plasma_ratio

    # --- Classification ---
    if blood_plasma_ratio > 2.0:
        rbc_partitioning_class = "RBC-avid"
    elif blood_plasma_ratio < 0.5:
        rbc_partitioning_class = "plasma-preferred"
    else:
        rbc_partitioning_class = "balanced"

    # --- Hemolysis risk ---
    hemolysis_risk = kp_rbc > 10.0

    notes = "; ".join(note_parts)

    return RBCPartitioningResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        hematocrit=hematocrit,
        kp_rbc=kp_rbc,
        blood_plasma_ratio=blood_plasma_ratio,
        f_in_rbc=f_in_rbc,
        plasma_to_blood_correction=plasma_to_blood_correction,
        rbc_partitioning_class=rbc_partitioning_class,
        hemolysis_risk=hemolysis_risk,
        notes=notes,
    )


def compare_hematocrit_effect(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    hct_values: list[float],
) -> list[RBCPartitioningResult]:
    """Return RBCPartitioningResult for each hematocrit value in hct_values."""
    return [
        predict_rbc_partitioning(
            drug_name=drug_name,
            logp=logp,
            pka=pka,
            ionization_type=ionization_type,
            hematocrit=hct,
        )
        for hct in hct_values
    ]
