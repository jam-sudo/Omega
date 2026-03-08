"""
Phase 456 — Pharmacophore Properties
Pharmacophore feature analysis for drug-receptor interaction prediction.
Pure Python, stdlib only, no external dependencies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PharmacophoreResult:
    """Result of pharmacophore feature analysis."""

    drug_name: str
    n_hbd: int  # H-bond donors
    n_hba: int  # H-bond acceptors
    n_aromatic: int  # aromatic rings
    n_hydrophobic: int  # hydrophobic groups
    n_positive_ionizable: int  # positively ionizable groups
    n_negative_ionizable: int  # negatively ionizable groups
    pharmacophore_complexity: float  # 0-100 score of feature diversity
    drug_receptor_complementarity: float  # 0-1
    predicted_ki_nM: float  # predicted binding affinity (nM)
    selectivity_score: float  # 0-1
    bioavailability_score: float  # 0-100
    pharmacophore_class: str  # class label
    activity_prediction: str  # potency label
    notes: str


def _compute_pharmacophore_complexity(
    n_hbd: int,
    n_hba: int,
    n_aromatic: int,
    n_hydrophobic: int,
    n_positive: int,
    n_negative: int,
) -> float:
    """Compute pharmacophore complexity score capped at 100."""
    raw = (
        10 * n_hbd
        + 8 * n_hba
        + 15 * n_aromatic
        + 5 * n_hydrophobic
        + 12 * n_positive
        + 12 * n_negative
    )
    return min(100.0, float(raw))


def _compute_complementarity(
    n_hbd: int,
    n_hba: int,
    n_aromatic: int,
    n_hydrophobic: int,
    n_positive: int,
    n_negative: int,
) -> float:
    """
    Compute drug-receptor complementarity [0, 1].

    Ideal profile: 2 HBD, 4 HBA, 2 aromatic, 1 hydrophobic, 1 ionizable.
    weights: hbd=0.2, hba=0.25, arom=0.25, hydro=0.1, ion=0.2
    """
    ideal_hbd = 2
    ideal_hba = 4
    ideal_arom = 2
    ideal_hydro = 1
    ideal_ion = 1  # sum of positive + negative vs 1

    w_hbd = 0.2
    w_hba = 0.25
    w_arom = 0.25
    w_hydro = 0.1
    w_ion = 0.2

    n_ion = n_positive + n_negative

    # Normalizer: max deviation * weight summed
    # Use a fixed normalizer so score stays in [0,1]
    # normalizer = sum of (max_reasonable_deviation * weight)
    # We use max_deviation = 5 per feature as a reasonable cap
    max_dev = 5.0
    normalizer = max_dev * (w_hbd + w_hba + w_arom + w_hydro + w_ion)

    deviation = (
        w_hbd * abs(n_hbd - ideal_hbd)
        + w_hba * abs(n_hba - ideal_hba)
        + w_arom * abs(n_aromatic - ideal_arom)
        + w_hydro * abs(n_hydrophobic - ideal_hydro)
        + w_ion * abs(n_ion - ideal_ion)
    )

    score = 1.0 - deviation / normalizer
    return max(0.0, min(1.0, score))


def _compute_predicted_ki(complementarity: float) -> float:
    """
    Deterministic predicted Ki (nM).
    ki = 10^(3 - 3 * complementarity), clamped [0.1, 100000].
    """
    exponent = 3.0 - 3.0 * complementarity
    ki = 10.0**exponent
    return max(0.1, min(100000.0, ki))


def _compute_selectivity(pharmacophore_complexity: float) -> float:
    """
    Selectivity score: low complexity -> high selectivity.
    score = max(0, 1 - complexity / 150)
    """
    return max(0.0, 1.0 - pharmacophore_complexity / 150.0)


def _compute_bioavailability(n_hbd: int, n_hba: int, n_aromatic: int) -> float:
    """
    Bioavailability score from pharmacophore perspective.
    base = 80; penalize HBD > 3, HBA > 7, aromatic > 4.
    """
    base = 80.0
    if n_hbd > 3:
        base -= 10.0 * (n_hbd - 3)
    if n_hba > 7:
        base -= 5.0 * (n_hba - 7)
    if n_aromatic > 4:
        base -= 8.0 * (n_aromatic - 4)
    return max(0.0, min(100.0, base))


def _assign_pharmacophore_class(
    n_hbd: int,
    n_hba: int,
    n_aromatic: int,
    n_hydrophobic: int,
    n_positive: int,
    n_negative: int,
) -> str:
    """Assign pharmacophore class based on dominant features."""
    # Priority rules (order matters)
    if n_aromatic >= 2 and n_positive > 0:
        return "CNS_friendly"
    if n_aromatic >= 2 and n_hbd < 2:
        return "kinase_like"
    if n_hbd > 3:
        return "HBD_rich"
    if n_hydrophobic > 3:
        return "lipophilic"
    return "GPCR_like"


def _assign_activity_prediction(ki_nM: float) -> str:
    """Assign activity label based on predicted Ki."""
    if ki_nM < 100.0:
        return "high_potency"
    if ki_nM <= 1000.0:
        return "moderate"
    return "weak"


def analyze_pharmacophore(
    drug_name: str,
    n_hbd: int,
    n_hba: int,
    n_aromatic: int,
    n_hydrophobic: int,
    n_positive_ionizable: int,
    n_negative_ionizable: int,
) -> PharmacophoreResult:
    """
    Analyze pharmacophore features and predict drug-receptor interaction properties.

    Parameters
    ----------
    drug_name : str
    n_hbd : int
        Number of H-bond donors (>= 0).
    n_hba : int
        Number of H-bond acceptors (>= 0).
    n_aromatic : int
        Number of aromatic rings (>= 0).
    n_hydrophobic : int
        Number of hydrophobic groups (>= 0).
    n_positive_ionizable : int
        Number of positively ionizable groups (>= 0).
    n_negative_ionizable : int
        Number of negatively ionizable groups (>= 0).

    Returns
    -------
    PharmacophoreResult
    """
    # --- Validation ---
    for name, val in [
        ("n_hbd", n_hbd),
        ("n_hba", n_hba),
        ("n_aromatic", n_aromatic),
        ("n_hydrophobic", n_hydrophobic),
        ("n_positive_ionizable", n_positive_ionizable),
        ("n_negative_ionizable", n_negative_ionizable),
    ]:
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")

    complexity = _compute_pharmacophore_complexity(
        n_hbd,
        n_hba,
        n_aromatic,
        n_hydrophobic,
        n_positive_ionizable,
        n_negative_ionizable,
    )
    complementarity = _compute_complementarity(
        n_hbd,
        n_hba,
        n_aromatic,
        n_hydrophobic,
        n_positive_ionizable,
        n_negative_ionizable,
    )
    ki_nM = _compute_predicted_ki(complementarity)
    selectivity = _compute_selectivity(complexity)
    bioavailability = _compute_bioavailability(n_hbd, n_hba, n_aromatic)
    pharm_class = _assign_pharmacophore_class(
        n_hbd,
        n_hba,
        n_aromatic,
        n_hydrophobic,
        n_positive_ionizable,
        n_negative_ionizable,
    )
    activity = _assign_activity_prediction(ki_nM)

    notes = (
        f"Class: {pharm_class}. "
        f"Complementarity: {complementarity:.3f}. "
        f"Predicted Ki: {ki_nM:.1f} nM ({activity})."
    )

    return PharmacophoreResult(
        drug_name=drug_name,
        n_hbd=n_hbd,
        n_hba=n_hba,
        n_aromatic=n_aromatic,
        n_hydrophobic=n_hydrophobic,
        n_positive_ionizable=n_positive_ionizable,
        n_negative_ionizable=n_negative_ionizable,
        pharmacophore_complexity=complexity,
        drug_receptor_complementarity=complementarity,
        predicted_ki_nM=ki_nM,
        selectivity_score=selectivity,
        bioavailability_score=bioavailability,
        pharmacophore_class=pharm_class,
        activity_prediction=activity,
        notes=notes,
    )


def screen_pharmacophores(compounds_list: list) -> list:
    """
    Screen a list of compound dicts and return PharmacophoreResult list
    sorted by drug_receptor_complementarity descending.

    Each dict must contain: drug_name, n_hbd, n_hba, n_aromatic,
    n_hydrophobic, n_positive_ionizable, n_negative_ionizable.
    """
    results = []
    for compound in compounds_list:
        res = analyze_pharmacophore(
            drug_name=compound["drug_name"],
            n_hbd=compound["n_hbd"],
            n_hba=compound["n_hba"],
            n_aromatic=compound["n_aromatic"],
            n_hydrophobic=compound["n_hydrophobic"],
            n_positive_ionizable=compound["n_positive_ionizable"],
            n_negative_ionizable=compound["n_negative_ionizable"],
        )
        results.append(res)
    results.sort(key=lambda r: r.drug_receptor_complementarity, reverse=True)
    return results
