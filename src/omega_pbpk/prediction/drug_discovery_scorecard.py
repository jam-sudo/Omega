"""Drug discovery scorecard — multi-parameter optimization (MPO) for early drug development."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MPOScoreResult",
    "compute_mpo_score",
    "screen_mpo",
    "compare_mpo_optimization",
    "generate_optimization_suggestions",
]


@dataclass(frozen=True)
class MPOScoreResult:
    """Result from MPO scoring of a drug candidate."""

    compound_name: str
    logP: float
    mw: float
    psa: float
    cns_mpo_score: float
    oral_mpo_score: float
    cns_mpo_class: str
    oral_mpo_class: str
    lipinski_pass: bool
    veber_pass: bool
    overall_recommendation: str
    optimization_suggestions: str
    notes: str


def _mpo_class(score: float) -> str:
    if score > 4.5:
        return "excellent"
    if score >= 3.0:
        return "acceptable"
    return "poor"


def _lipinski_pass(mw: float, logP: float, n_h_donors: int, n_h_acceptors: int) -> bool:
    """Lipinski Rule of Five: MW <= 500, logP <= 5, donors <= 5, acceptors <= 10."""
    violations = 0
    if mw > 500:
        violations += 1
    if logP > 5:
        violations += 1
    if n_h_donors > 5:
        violations += 1
    if n_h_acceptors > 10:
        violations += 1
    return violations <= 1


def _veber_pass(psa: float, n_rotatable_bonds: int) -> bool:
    """Veber criteria: PSA <= 140, rotatable bonds <= 10."""
    return psa <= 140 and n_rotatable_bonds <= 10


def _score_cns_mpo(
    logP: float,
    mw: float,
    psa: float,
    n_h_donors: int,
    pka_base: float | None,
    logD_74: float | None,
) -> float:
    """Compute CNS MPO score (0-6) based on Wager et al. 2010 adapted."""
    score = 0.0

    # 1. logP score
    if -1 <= logP <= 3:
        score += 1.0
    elif 3 < logP <= 5 or -2 <= logP < -1:
        score += 0.5
    # else 0

    # 2. logD_74 (use logD if provided, else logP)
    ld = logD_74 if logD_74 is not None else logP
    if -1 <= ld <= 2:
        score += 1.0
    elif 2 < ld <= 4:
        score += 0.5
    # else 0

    # 3. MW score
    if mw < 200:
        score += 1.0
    elif mw <= 360:
        score += 0.5
    # else 0

    # 4. PSA score
    if psa < 40:
        score += 1.0
    elif psa <= 90:
        score += 0.5
    # else 0

    # 5. H-donors score
    if n_h_donors == 0:
        score += 1.0
    elif 1 <= n_h_donors <= 3:
        score += 0.5
    # else 0

    # 6. pKa_base score (if provided)
    if pka_base is not None:
        if 7 <= pka_base <= 10:
            score += 1.0
        elif (10 < pka_base <= 11) or (5 <= pka_base < 7):
            score += 0.5
        # else 0
    else:
        # Not provided: give neutral 0.5 to keep max score achievable
        score += 0.5

    return score


def _score_oral_mpo(
    logP: float,
    mw: float,
    psa: float,
    n_h_donors: int,
    n_h_acceptors: int,
    n_rotatable_bonds: int,
) -> float:
    """Compute oral MPO score (0-6)."""
    score = 0.0

    # 1. logP score
    if 0 <= logP <= 3:
        score += 1.0
    elif (-1 <= logP < 0) or (3 < logP <= 5):
        score += 0.5
    # else 0

    # 2. MW score
    if mw < 350:
        score += 1.0
    elif mw <= 500:
        score += 0.5
    # else 0

    # 3. PSA score
    if 40 <= psa <= 120:
        score += 1.0
    elif (20 <= psa < 40) or (120 < psa <= 140):
        score += 0.5
    # else 0

    # 4. H-donors score
    if n_h_donors <= 2:
        score += 1.0
    elif n_h_donors <= 5:
        score += 0.5
    # else 0

    # 5. H-acceptors score
    if n_h_acceptors <= 5:
        score += 1.0
    elif n_h_acceptors <= 10:
        score += 0.5
    # else 0

    # 6. Rotatable bonds score
    if n_rotatable_bonds <= 5:
        score += 1.0
    elif n_rotatable_bonds <= 10:
        score += 0.5
    # else 0

    return score


def generate_optimization_suggestions(result: MPOScoreResult) -> list[str]:
    """Generate actionable optimization suggestions based on suboptimal properties."""
    suggestions = []

    if result.logP > 5:
        suggestions.append("Reduce lipophilicity (logP target: 1-3)")
    elif result.logP < -1:
        suggestions.append("Increase lipophilicity (logP target: 1-3)")

    if result.mw > 500:
        suggestions.append("Reduce molecular weight (target: <500 Da)")
    elif result.mw > 350:
        suggestions.append("Consider reducing molecular weight (target: <350 Da)")

    if result.psa > 140:
        suggestions.append("Reduce PSA by removing H-bond donors/acceptors")
    elif result.psa < 20:
        suggestions.append("Increase PSA for improved aqueous solubility (target: 40-120)")

    # Additional logP range check
    logP = result.logP
    if logP <= 5 and not (-1 <= logP <= 5):
        if "lipophilicity" not in " ".join(suggestions):
            suggestions.append("Adjust logP into acceptable range (-1 to 5)")

    if not suggestions and result.oral_mpo_score >= 4.5:
        suggestions.append("Compound has good oral MPO profile; consider further ADMET profiling")

    if not suggestions:
        if result.oral_mpo_score < 3.0:
            suggestions.append(
                "Multiple properties require optimization; consult medicinal chemist"
            )
        elif result.oral_mpo_score < 4.5:
            suggestions.append(
                "Profile is acceptable; incremental optimization may improve developability"
            )

    return suggestions


def _build_suggestions_str(
    logP: float,
    mw: float,
    psa: float,
    n_h_donors: int,
    n_h_acceptors: int,
    n_rotatable_bonds: int,
    oral_score: float,
) -> str:
    """Build optimization suggestions string for MPOScoreResult."""
    parts = []

    if logP > 5:
        parts.append("Reduce lipophilicity (logP target: 1-3)")
    elif logP < -1:
        parts.append("Increase lipophilicity (logP target: 1-3)")

    if mw > 500:
        parts.append("Reduce molecular weight (target: <500 Da)")
    elif mw > 350:
        parts.append("Consider reducing molecular weight (target: <350 Da)")

    if psa > 140:
        parts.append("Reduce PSA by removing H-bond donors/acceptors")
    elif psa < 20:
        parts.append("Increase PSA for improved aqueous solubility (target: 40-120)")

    if n_h_donors > 5:
        parts.append("Reduce H-bond donors (target: ≤2)")
    elif n_h_donors > 2:
        parts.append("Consider reducing H-bond donors (target: ≤2)")

    if n_h_acceptors > 10:
        parts.append("Reduce H-bond acceptors (target: ≤5)")
    elif n_h_acceptors > 5:
        parts.append("Consider reducing H-bond acceptors (target: ≤5)")

    if n_rotatable_bonds > 10:
        parts.append("Reduce rotatable bonds (target: ≤5)")
    elif n_rotatable_bonds > 5:
        parts.append("Consider reducing rotatable bonds (target: ≤5)")

    if not parts:
        if oral_score >= 4.5:
            parts.append("Compound has good oral MPO profile; consider further ADMET profiling")
        else:
            parts.append(
                "Profile is acceptable; incremental optimization may improve developability"
            )

    return "; ".join(parts)


def _overall_recommendation(oral_mpo: float, lipinski: bool) -> str:
    if oral_mpo >= 4.5 and lipinski:
        return "proceed"
    if oral_mpo < 2.0 or not lipinski:
        return "deprioritize"
    return "optimize"


def _validate_inputs(mw: float, psa: float, n_h_donors: int, n_h_acceptors: int) -> None:
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if n_h_donors < 0:
        raise ValueError(f"n_h_donors must be >= 0, got {n_h_donors}")
    if n_h_acceptors < 0:
        raise ValueError(f"n_h_acceptors must be >= 0, got {n_h_acceptors}")


def compute_mpo_score(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_h_donors: int,
    n_h_acceptors: int,
    n_rotatable_bonds: int,
    pka_base: float | None = None,
    logD_74: float | None = None,
) -> MPOScoreResult:
    """Compute CNS and oral MPO scores for a drug candidate.

    Args:
        compound_name: Name of the compound
        logP: Octanol-water partition coefficient
        mw: Molecular weight (Da)
        psa: Polar surface area (Angstrom^2)
        n_h_donors: Number of H-bond donors
        n_h_acceptors: Number of H-bond acceptors
        n_rotatable_bonds: Number of rotatable bonds
        pka_base: Basic pKa (optional)
        logD_74: logD at pH 7.4 (optional; uses logP if not provided)

    Returns:
        MPOScoreResult with CNS and oral MPO scores and recommendations
    """
    _validate_inputs(mw, psa, n_h_donors, n_h_acceptors)

    cns_score = _score_cns_mpo(logP, mw, psa, n_h_donors, pka_base, logD_74)
    oral_score = _score_oral_mpo(logP, mw, psa, n_h_donors, n_h_acceptors, n_rotatable_bonds)

    lipinski = _lipinski_pass(mw, logP, n_h_donors, n_h_acceptors)
    veber = _veber_pass(psa, n_rotatable_bonds)

    cns_cls = _mpo_class(cns_score)
    oral_cls = _mpo_class(oral_score)

    recommendation = _overall_recommendation(oral_score, lipinski)

    suggestions = _build_suggestions_str(
        logP, mw, psa, n_h_donors, n_h_acceptors, n_rotatable_bonds, oral_score
    )

    notes = (
        f"MPO analysis | logP={logP}, MW={mw}, PSA={psa}, "
        f"HBD={n_h_donors}, HBA={n_h_acceptors}, RotBonds={n_rotatable_bonds} | "
        f"CNS MPO={cns_score:.1f} ({cns_cls}), Oral MPO={oral_score:.1f} ({oral_cls}) | "
        f"Lipinski: {'pass' if lipinski else 'fail'}, Veber: {'pass' if veber else 'fail'}"
    )

    return MPOScoreResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        psa=psa,
        cns_mpo_score=cns_score,
        oral_mpo_score=oral_score,
        cns_mpo_class=cns_cls,
        oral_mpo_class=oral_cls,
        lipinski_pass=lipinski,
        veber_pass=veber,
        overall_recommendation=recommendation,
        optimization_suggestions=suggestions,
        notes=notes,
    )


def screen_mpo(compounds: list[dict]) -> list[MPOScoreResult]:
    """Screen a list of compounds using MPO scoring, sorted by oral_mpo_score descending.

    Each compound dict must contain: compound_name, logP, mw, psa,
    n_h_donors, n_h_acceptors, n_rotatable_bonds.
    Optional: pka_base, logD_74.
    """
    results = []
    for cpd in compounds:
        result = compute_mpo_score(
            compound_name=cpd["compound_name"],
            logP=cpd["logP"],
            mw=cpd["mw"],
            psa=cpd["psa"],
            n_h_donors=cpd["n_h_donors"],
            n_h_acceptors=cpd["n_h_acceptors"],
            n_rotatable_bonds=cpd["n_rotatable_bonds"],
            pka_base=cpd.get("pka_base"),
            logD_74=cpd.get("logD_74"),
        )
        results.append(result)

    results.sort(key=lambda r: r.oral_mpo_score, reverse=True)
    return results


def compare_mpo_optimization(base: dict, variants: list[dict]) -> list[MPOScoreResult]:
    """Compare a base compound to optimization variants.

    Returns a list with base compound first, followed by variants sorted
    by oral_mpo_score descending.
    """
    base_result = compute_mpo_score(
        compound_name=base["compound_name"],
        logP=base["logP"],
        mw=base["mw"],
        psa=base["psa"],
        n_h_donors=base["n_h_donors"],
        n_h_acceptors=base["n_h_acceptors"],
        n_rotatable_bonds=base["n_rotatable_bonds"],
        pka_base=base.get("pka_base"),
        logD_74=base.get("logD_74"),
    )

    variant_results = []
    for v in variants:
        r = compute_mpo_score(
            compound_name=v["compound_name"],
            logP=v["logP"],
            mw=v["mw"],
            psa=v["psa"],
            n_h_donors=v["n_h_donors"],
            n_h_acceptors=v["n_h_acceptors"],
            n_rotatable_bonds=v["n_rotatable_bonds"],
            pka_base=v.get("pka_base"),
            logD_74=v.get("logD_74"),
        )
        variant_results.append(r)

    variant_results.sort(key=lambda r: r.oral_mpo_score, reverse=True)
    return [base_result] + variant_results
