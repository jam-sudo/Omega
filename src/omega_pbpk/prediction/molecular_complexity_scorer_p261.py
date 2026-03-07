"""Phase 261 — Molecular complexity scoring and synthetic accessibility prediction."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MolecularComplexityResult",
    "score_complexity",
    "compare_complexity",
]


@dataclass(frozen=True)
class MolecularComplexityResult:
    """Result of molecular complexity scoring."""

    drug_name: str
    mw: float
    logp: float
    psa_A2: float
    hbd: int
    hba: int
    rotatable_bonds: int
    rings: int
    chiral_centers: int
    stereocenters: int
    fragment_complexity_score: float  # 0-100
    synthetic_accessibility_score: float  # 1-10, 1=easy, 10=very hard
    molecular_complexity_class: str  # "simple", "moderate", "complex"
    development_risk: str  # "low", "moderate", "high"
    estimated_synthesis_steps: int
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def score_complexity(
    drug_name: str,
    mw: float,
    logp: float,
    psa_A2: float,
    hbd: int,
    hba: int,
    rotatable_bonds: int = 5,
    rings: int = 2,
    chiral_centers: int = 0,
    stereocenters: int = 0,
) -> MolecularComplexityResult:
    """Score molecular complexity and predict synthetic accessibility.

    Parameters
    ----------
    drug_name:
        Name of the drug or compound.
    mw:
        Molecular weight (Da). Must be > 0.
    logp:
        Calculated logP (octanol-water partition coefficient).
    psa_A2:
        Polar surface area (Angstrom^2). Must be >= 0.
    hbd:
        Number of hydrogen bond donors. Must be >= 0.
    hba:
        Number of hydrogen bond acceptors. Must be >= 0.
    rotatable_bonds:
        Number of rotatable bonds. Must be >= 0.
    rings:
        Number of ring systems. Must be >= 0.
    chiral_centers:
        Number of chiral centers. Must be >= 0.
    stereocenters:
        Number of stereocenters (including chiral). Must be >= 0.

    Returns
    -------
    MolecularComplexityResult
    """
    # Validation
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa_A2 < 0:
        raise ValueError("psa_A2 must be >= 0")
    if hbd < 0:
        raise ValueError("hbd must be >= 0")
    if hba < 0:
        raise ValueError("hba must be >= 0")
    if rotatable_bonds < 0:
        raise ValueError("rotatable_bonds must be >= 0")
    if rings < 0:
        raise ValueError("rings must be >= 0")
    if chiral_centers < 0:
        raise ValueError("chiral_centers must be >= 0")
    if stereocenters < 0:
        raise ValueError("stereocenters must be >= 0")

    # Fragment complexity score: 0-100
    fcs_raw = mw / 5.0 + psa_A2 / 2.0 + rotatable_bonds * 3.0 + rings * 5.0 + chiral_centers * 8.0
    fragment_complexity_score = min(100.0, fcs_raw)

    # Synthetic accessibility: 1-10
    sa_raw = (
        1.0
        + 0.01 * mw
        + 0.05 * chiral_centers * stereocenters
        + 0.1 * rings
        + 0.02 * hba
        - 0.2 * logp
    )
    synthetic_accessibility_score = _clamp(sa_raw, 1.0, 10.0)

    # Complexity class
    if synthetic_accessibility_score < 3.0:
        molecular_complexity_class = "simple"
    elif synthetic_accessibility_score <= 6.0:
        molecular_complexity_class = "moderate"
    else:
        molecular_complexity_class = "complex"

    # Development risk
    if synthetic_accessibility_score < 3.0:
        development_risk = "low"
    elif synthetic_accessibility_score < 6.0:
        development_risk = "moderate"
    else:
        development_risk = "high"

    # Estimated synthesis steps
    estimated_synthesis_steps = max(1, int(synthetic_accessibility_score * 2))

    notes = (
        f"Fragment complexity: {fragment_complexity_score:.1f}/100. "
        f"SA score: {synthetic_accessibility_score:.2f}/10 ({molecular_complexity_class}). "
        f"Development risk: {development_risk}. "
        f"Estimated synthesis steps: {estimated_synthesis_steps}."
    )

    return MolecularComplexityResult(
        drug_name=drug_name,
        mw=mw,
        logp=logp,
        psa_A2=psa_A2,
        hbd=hbd,
        hba=hba,
        rotatable_bonds=rotatable_bonds,
        rings=rings,
        chiral_centers=chiral_centers,
        stereocenters=stereocenters,
        fragment_complexity_score=fragment_complexity_score,
        synthetic_accessibility_score=synthetic_accessibility_score,
        molecular_complexity_class=molecular_complexity_class,
        development_risk=development_risk,
        estimated_synthesis_steps=estimated_synthesis_steps,
        notes=notes,
    )


def compare_complexity(compounds: list[dict]) -> list[MolecularComplexityResult]:
    """Score a list of compounds and sort by synthetic_accessibility_score ascending.

    Parameters
    ----------
    compounds:
        List of dicts. Each dict must contain keys: name (used as drug_name),
        mw, logp, psa_A2, hbd, hba. Optional keys: rotatable_bonds, rings,
        chiral_centers, stereocenters.

    Returns
    -------
    list[MolecularComplexityResult] sorted by SA score ascending (simplest first).
    """
    results = []
    for c in compounds:
        c2 = dict(c)
        # Support 'name' as alias for 'drug_name'
        if "name" in c2 and "drug_name" not in c2:
            c2["drug_name"] = c2.pop("name")
        results.append(score_complexity(**c2))
    return sorted(results, key=lambda r: r.synthetic_accessibility_score)
