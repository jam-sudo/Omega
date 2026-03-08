"""Phase 1098 — Drug Erythrocyte Partitioning.

Predict drug partitioning into red blood cells (RBC) and its effect on
blood-to-plasma ratio (Rb). Critical for drugs with significant RBC
accumulation such as methotrexate, chloroquine, and immunosuppressants.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_IONIZATION_TYPES = {"acid", "base", "neutral", "amphoteric"}


@dataclass(frozen=True)
class ErythrocytePartitioningResult:
    """Result of RBC partitioning prediction."""

    drug_name: str
    logP: float
    pka: float
    ionization_type: str
    hematocrit: float
    kp_rbc: float
    rb_blood_plasma_ratio: float
    fu_plasma: float
    fu_blood: float
    cl_blood_correction_factor: float
    rbc_accumulation_risk: str
    clinical_implication: str
    notes: str


def _predict_kp_rbc(logP: float, ionization_type: str, pka: float) -> float:
    """Predict RBC-to-plasma partition coefficient.

    For neutral/acid drugs:
        Kp_rbc = max(0.1, min(10, 0.3 + 0.2 * logP))

    For basic drugs (pKa > 7.4): lysosomal trapping in RBC
        Kp_neutral = max(0.1, min(10, 0.3 + 0.2 * logP))
        Kp_rbc = min(50, Kp_neutral * 10^max(0, pKa - 7.2))

    For amphoteric: use the higher of acid/base values.
    """
    # Base lipophilicity-driven Kp (neutral/acid model)
    kp_neutral = max(0.1, min(10.0, 0.3 + 0.2 * logP))

    if ionization_type == "neutral":
        return kp_neutral

    elif ionization_type == "acid":
        return kp_neutral

    elif ionization_type == "base":
        # Lysosomal trapping: pKa raises Kp_rbc via proton trapping in RBC
        trapping_factor = 10.0 ** max(0.0, pka - 7.2)
        kp_base = min(50.0, kp_neutral * trapping_factor)
        return kp_base

    elif ionization_type == "amphoteric":
        # Use the higher of neutral (acid-like) and base calculations
        trapping_factor = 10.0 ** max(0.0, pka - 7.2)
        kp_base = min(50.0, kp_neutral * trapping_factor)
        return max(kp_neutral, kp_base)

    else:
        # Should be caught by validation but defensive fallback
        return kp_neutral


def _accumulation_risk(kp_rbc: float) -> str:
    """Classify RBC accumulation risk."""
    if kp_rbc < 1.0:
        return "low"
    elif kp_rbc <= 5.0:
        return "moderate"
    else:
        return "high"


def _clinical_implication(
    kp_rbc: float,
    rb: float,
    ionization_type: str,
) -> str:
    """Generate a clinical implication string."""
    if kp_rbc > 5.0:
        if ionization_type == "base":
            return (
                "Significant RBC accumulation via lysosomal trapping; "
                "blood clearance substantially exceeds plasma clearance"
            )
        return "RBC acts as drug reservoir; prolonged half-life expected"
    elif kp_rbc >= 1.0:
        if rb > 1.1:
            return "Moderate RBC partitioning; monitor blood vs plasma concentrations"
        return "Mild RBC partitioning; minor effect on PK parameters"
    else:
        return "Minimal RBC uptake; plasma concentrations reliably reflect blood exposure"


def predict_rbc_partitioning(
    drug_name: str,
    logP: float,
    ionization_type: str,
    pka: float = 7.0,
    fu_plasma: float = 0.1,
    hematocrit: float = 0.45,
) -> ErythrocytePartitioningResult:
    """Predict RBC partitioning and blood-to-plasma ratio for a drug.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logP:
        Octanol-water partition coefficient (log scale). Range: [-5, 10].
    ionization_type:
        Ionization class: 'acid', 'base', 'neutral', or 'amphoteric'.
    pka:
        pKa of the drug (relevant for basic drugs). Default 7.0.
    fu_plasma:
        Unbound fraction in plasma (0, 1]. Default 0.1.
    hematocrit:
        Blood hematocrit fraction (0, 1). Default 0.45.

    Returns
    -------
    ErythrocytePartitioningResult
    """
    # Validation
    if ionization_type not in VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {sorted(VALID_IONIZATION_TYPES)}, "
            f"got '{ionization_type}'"
        )
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if not (0.0 < hematocrit < 1.0):
        raise ValueError(f"hematocrit must be in (0, 1), got {hematocrit}")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")

    kp_rbc = _predict_kp_rbc(logP, ionization_type, pka)

    # Blood-to-plasma ratio: Rb = (1 - Hct) + Hct * Kp_rbc
    rb = (1.0 - hematocrit) + hematocrit * kp_rbc

    # fu_blood = fu_plasma / Rb
    fu_blood = fu_plasma / rb

    # CL correction factor (multiply CL_plasma to get CL_blood)
    cl_blood_correction_factor = rb

    risk = _accumulation_risk(kp_rbc)
    implication = _clinical_implication(kp_rbc, rb, ionization_type)

    # Build notes
    notes_parts = [
        f"Kp_rbc={kp_rbc:.3f}",
        f"Rb={rb:.3f}",
        f"Hct={hematocrit:.2f}",
        f"ionization={ionization_type}",
    ]
    if ionization_type in ("base", "amphoteric") and pka > 7.2:
        notes_parts.append(
            f"pKa={pka} triggers lysosomal trapping factor 10^{max(0.0, pka - 7.2):.2f}"
        )

    notes = "; ".join(notes_parts)

    return ErythrocytePartitioningResult(
        drug_name=drug_name,
        logP=logP,
        pka=pka,
        ionization_type=ionization_type,
        hematocrit=hematocrit,
        kp_rbc=kp_rbc,
        rb_blood_plasma_ratio=rb,
        fu_plasma=fu_plasma,
        fu_blood=fu_blood,
        cl_blood_correction_factor=cl_blood_correction_factor,
        rbc_accumulation_risk=risk,
        clinical_implication=implication,
        notes=notes,
    )


def screen_compounds(compounds: list) -> list:
    """Screen a list of compounds for RBC partitioning.

    Parameters
    ----------
    compounds:
        List of dicts, each with keys: 'name', 'logP', 'ionization_type',
        optionally 'pka', 'fu_plasma', 'hematocrit'.

    Returns
    -------
    List of ErythrocytePartitioningResult sorted by kp_rbc descending.
    """
    results = []
    for compound in compounds:
        name = compound["name"]
        logP = compound["logP"]
        ionization_type = compound["ionization_type"]
        pka = compound.get("pka", 7.0)
        fu_plasma = compound.get("fu_plasma", 0.1)
        hematocrit = compound.get("hematocrit", 0.45)

        result = predict_rbc_partitioning(
            drug_name=name,
            logP=logP,
            ionization_type=ionization_type,
            pka=pka,
            fu_plasma=fu_plasma,
            hematocrit=hematocrit,
        )
        results.append(result)

    # Sort by kp_rbc descending
    results.sort(key=lambda r: r.kp_rbc, reverse=True)
    return results
