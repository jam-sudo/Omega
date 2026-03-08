"""
Phase 510 — Drug Partitioning into Red Blood Cells

Predict blood-to-plasma ratio (B:P) and RBC partitioning from
physicochemical properties (logP, pKa, drug type).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RBCPartitioningResult:
    """Results of RBC partitioning prediction."""

    drug_name: str
    logP: float
    pKa: float
    drug_type: str  # "neutral", "acid", or "base"
    fu_plasma: float
    hematocrit: float

    kp_rbc: float  # RBC/plasma partition coefficient
    bp_ratio: float  # blood-to-plasma ratio
    fu_blood: float  # unbound fraction in blood

    rbc_accumulation: bool  # True if Kp_rbc > 2.0
    plasma_depleting: bool  # True if B:P > 2.0 (drug preferentially in RBC)
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_KP_RBC_MIN = 0.1
_KP_RBC_MAX = 20.0
_BP_MIN = 0.1
_BP_MAX = 10.0


def _estimate_kp_rbc(logP: float, pKa: float, drug_type: str) -> float:
    """
    Estimate RBC/plasma partition coefficient.

    Rules:
      Neutral / acid: Kp_rbc = 1 + 0.5 * max(0, logP - 1)
      Base:           Kp_rbc = 1 + 1.2 * max(0, logP - 1)
      Strong base (pKa > 8): Kp_rbc *= 1.5
    """
    logP_excess = max(0.0, logP - 1.0)

    if drug_type in {"neutral", "acid"}:
        kp_rbc = 1.0 + 0.5 * logP_excess
    else:  # base
        kp_rbc = 1.0 + 1.2 * logP_excess
        if pKa > 8.0:
            kp_rbc *= 1.5

    # Clamp
    kp_rbc = max(_KP_RBC_MIN, min(_KP_RBC_MAX, kp_rbc))
    return kp_rbc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_rbc_partitioning(
    drug_name: str,
    logP: float,
    pKa: float,
    drug_type: str,
    fu_plasma: float = 0.1,
    hematocrit: float = 0.45,
) -> RBCPartitioningResult:
    """
    Predict RBC partitioning and blood-to-plasma ratio.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logP : float
        Octanol-water partition coefficient (log scale).
    pKa : float
        Acid dissociation constant.
    drug_type : str
        "neutral", "acid", or "base".
    fu_plasma : float
        Unbound fraction in plasma (0 < fu_plasma <= 1).
    hematocrit : float
        Volume fraction of RBCs (0 < hematocrit < 1, default 0.45).

    Returns
    -------
    RBCPartitioningResult
    """
    # --- Validation ---
    if drug_type not in {"neutral", "acid", "base"}:
        raise ValueError("drug_type must be 'neutral', 'acid', or 'base'")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if not (0.0 < hematocrit < 1.0):
        raise ValueError("hematocrit must be in (0, 1)")

    # --- Kp_rbc estimation ---
    kp_rbc = _estimate_kp_rbc(logP, pKa, drug_type)

    # --- B:P ratio ---
    bp_ratio = 1.0 + hematocrit * (kp_rbc - 1.0)
    bp_ratio = max(_BP_MIN, min(_BP_MAX, bp_ratio))

    # --- fu_blood ---
    fu_blood = fu_plasma / bp_ratio

    # --- Flags ---
    rbc_accumulation = kp_rbc > 2.0
    plasma_depleting = bp_ratio > 2.0

    # --- Notes ---
    notes_parts: list[str] = []
    if rbc_accumulation:
        notes_parts.append(f"RBC accumulation: Kp_rbc={kp_rbc:.2f} (>2.0).")
    if plasma_depleting:
        notes_parts.append(f"Drug preferentially distributes into RBCs (B:P={bp_ratio:.2f}).")
    if drug_type == "base" and pKa > 8.0:
        notes_parts.append("Strong base: enhanced hemoglobin binding.")
    if not notes_parts:
        notes_parts.append("Normal RBC partitioning.")
    notes = " ".join(notes_parts)

    return RBCPartitioningResult(
        drug_name=drug_name,
        logP=logP,
        pKa=pKa,
        drug_type=drug_type,
        fu_plasma=fu_plasma,
        hematocrit=hematocrit,
        kp_rbc=kp_rbc,
        bp_ratio=bp_ratio,
        fu_blood=fu_blood,
        rbc_accumulation=rbc_accumulation,
        plasma_depleting=plasma_depleting,
        notes=notes,
    )


def screen_rbc_partitioning(compounds: list[dict]) -> list[RBCPartitioningResult]:
    """
    Screen a list of compounds for RBC partitioning.

    Each dict in compounds should have keys:
      drug_name, logP, pKa, drug_type
    Optional: fu_plasma, hematocrit

    Returns results sorted by bp_ratio descending.
    """
    results: list[RBCPartitioningResult] = []
    for compound in compounds:
        result = predict_rbc_partitioning(
            drug_name=compound["drug_name"],
            logP=compound["logP"],
            pKa=compound["pKa"],
            drug_type=compound["drug_type"],
            fu_plasma=compound.get("fu_plasma", 0.1),
            hematocrit=compound.get("hematocrit", 0.45),
        )
        results.append(result)

    results.sort(key=lambda r: r.bp_ratio, reverse=True)
    return results
