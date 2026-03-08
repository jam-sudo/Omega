"""Phase 483 — Drug Binding to Red Blood Cells.

Models drug partitioning into red blood cells (RBCs) and blood-to-plasma
concentration ratio (B:P) determination.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RBCBindingResult:
    """Result of RBC drug binding prediction."""

    drug_name: str
    logP: float
    pKa: float
    ionization_type: str
    hematocrit: float
    fu_plasma: float
    kp_rbc: float
    blood_plasma_ratio: float
    fu_blood: float
    cl_blood_correction_factor: float
    rbc_binding_class: str
    clinical_relevance: str
    notes: str


def _compute_kp_rbc(logP: float, pKa: float, ionization_type: str) -> float:
    """Compute RBC/plasma partition coefficient."""
    if ionization_type == "neutral":
        kp = 0.57 + 0.3 * logP
    elif ionization_type == "acid":
        donnan = max(0.1, 1.0 - 0.3 * (7.4 - pKa))
        kp = 0.57 + 0.3 * logP * donnan
    else:  # base
        factor = 1.0 + 0.2 * (7.4 - pKa)
        kp = 0.57 + 0.3 * logP * factor

    # Clamp to physiologically plausible range
    return max(0.1, min(10.0, kp))


def _rbc_binding_class(kp_rbc: float) -> str:
    if kp_rbc < 0.7:
        return "low"
    if kp_rbc <= 2.0:
        return "moderate"
    return "high"


def _clinical_relevance(blood_plasma_ratio: float) -> str:
    diff = abs(blood_plasma_ratio - 1.0)
    if diff < 0.1:
        return "none"
    if diff < 0.3:
        return "minor"
    return "significant"


def predict_rbc_binding(
    drug_name: str,
    logP: float,
    pKa: float = 7.0,
    ionization_type: str = "neutral",
    hematocrit: float = 0.45,
    fu_plasma: float = 0.1,
) -> RBCBindingResult:
    """Predict RBC binding and blood-to-plasma ratio for a drug.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logP:
        Octanol-water partition coefficient (log scale).
    pKa:
        Acid dissociation constant.
    ionization_type:
        One of "neutral", "acid", "base".
    hematocrit:
        Volume fraction of red blood cells (0 < Hct < 1).
    fu_plasma:
        Fraction unbound in plasma (0 < fu_plasma <= 1).

    Returns
    -------
    RBCBindingResult
    """
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got '{ionization_type}'"
        )
    if not (0 < hematocrit < 1):
        raise ValueError(f"hematocrit must be between 0 and 1 (exclusive), got {hematocrit}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")

    kp_rbc = _compute_kp_rbc(logP, pKa, ionization_type)
    b_p = 1.0 + hematocrit * (kp_rbc - 1.0)
    fu_blood = fu_plasma / b_p
    cl_correction = 1.0 / b_p
    binding_class = _rbc_binding_class(kp_rbc)
    relevance = _clinical_relevance(b_p)

    notes_parts = [
        f"Kp_rbc={kp_rbc:.3f}",
        f"B:P={b_p:.3f}",
        f"Hct={hematocrit}",
        f"ionization={ionization_type}",
    ]
    if b_p > 1.3:
        notes_parts.append("Drug accumulates in RBCs; plasma-based PK may underestimate exposure.")
    elif b_p < 0.7:
        notes_parts.append("Drug excluded from RBCs; plasma-based PK may overestimate exposure.")
    else:
        notes_parts.append("B:P near unity; plasma PK is a reasonable surrogate for blood PK.")

    return RBCBindingResult(
        drug_name=drug_name,
        logP=logP,
        pKa=pKa,
        ionization_type=ionization_type,
        hematocrit=hematocrit,
        fu_plasma=fu_plasma,
        kp_rbc=kp_rbc,
        blood_plasma_ratio=b_p,
        fu_blood=fu_blood,
        cl_blood_correction_factor=cl_correction,
        rbc_binding_class=binding_class,
        clinical_relevance=relevance,
        notes="; ".join(notes_parts),
    )


def screen_rbc_binding(compounds: list) -> list:
    """Screen a list of compounds for RBC binding.

    Parameters
    ----------
    compounds:
        List of dicts, each containing keyword arguments for predict_rbc_binding.
        Required keys: drug_name, logP. Optional: pKa, ionization_type, hematocrit, fu_plasma.

    Returns
    -------
    List of RBCBindingResult sorted by blood_plasma_ratio descending.
    """
    results = []
    for compound in compounds:
        result = predict_rbc_binding(**compound)
        results.append(result)
    results.sort(key=lambda r: r.blood_plasma_ratio, reverse=True)
    return results
