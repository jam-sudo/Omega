"""Phase 716 — Molecular complexity scoring for synthetic accessibility and drug-likeness."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ComplexityResult",
    "score_molecular_complexity",
    "screen_complexity",
    "compare_complexity_optimization",
]


@dataclass(frozen=True)
class ComplexityResult:
    """Result of molecular complexity scoring."""

    compound_name: str
    mw: float
    n_rings: int
    n_stereocenters: int
    complexity_score: float  # 0-100
    sas: float  # 1-10 synthetic accessibility score
    sas_class: str  # easy / moderate / difficult / very_difficult
    lipinski_pass: bool
    veber_pass: bool
    egan_pass: bool
    lead_like: bool
    development_stage_fit: str  # lead / candidate / advanced
    optimization_priority: str
    notes: str


def _compute_complexity(
    mw: float,
    n_rings: int,
    n_stereocenters: int,
    n_rotatable_bonds: int,
    n_aromatic_rings: int,
    n_bridgeheads: int,
) -> float:
    """Sum complexity contributions, capped at 100."""
    mw_contrib = min((mw / 500.0) * 20.0, 20.0)
    ring_contrib = min(n_rings * 5.0, 25.0)
    stereo_contrib = min(n_stereocenters * 8.0, 32.0)
    rot_contrib = min(n_rotatable_bonds * 1.5, 15.0)
    arom_contrib = min(n_aromatic_rings * 3.0, 12.0)
    bridge_contrib = min(n_bridgeheads * 10.0, 20.0)
    total = mw_contrib + ring_contrib + stereo_contrib + rot_contrib + arom_contrib + bridge_contrib
    return min(total, 100.0)


def _sas_class(sas: float) -> str:
    if sas < 3.0:
        return "easy"
    if sas <= 5.0:
        return "moderate"
    if sas <= 7.0:
        return "difficult"
    return "very_difficult"


def _optimization_priority(
    mw: float,
    n_rings: int,
    n_stereocenters: int,
    n_rotatable_bonds: int,
    n_aromatic_rings: int,
    n_bridgeheads: int,
) -> str:
    """Return the property that contributes most to complexity."""
    contributions = {
        "molecular_weight": min((mw / 500.0) * 20.0, 20.0),
        "ring_count": min(n_rings * 5.0, 25.0),
        "stereocenters": min(n_stereocenters * 8.0, 32.0),
        "rotatable_bonds": min(n_rotatable_bonds * 1.5, 15.0),
        "aromatic_rings": min(n_aromatic_rings * 3.0, 12.0),
        "bridgeheads": min(n_bridgeheads * 10.0, 20.0),
    }
    return max(contributions, key=lambda k: contributions[k])


def score_molecular_complexity(
    compound_name: str,
    mw: float,
    n_rings: int,
    n_stereocenters: int,
    n_rotatable_bonds: int,
    n_h_donors: int,
    n_h_acceptors: int,
    n_aromatic_rings: int = 0,
    n_bridgeheads: int = 0,
) -> ComplexityResult:
    """Score molecular complexity and drug-likeness."""
    # Validation
    if mw <= 0:
        raise ValueError("mw must be positive")
    if n_rings < 0:
        raise ValueError("n_rings must be >= 0")
    if n_stereocenters < 0:
        raise ValueError("n_stereocenters must be >= 0")
    if n_rotatable_bonds < 0:
        raise ValueError("n_rotatable_bonds must be >= 0")

    complexity_score = _compute_complexity(
        mw, n_rings, n_stereocenters, n_rotatable_bonds, n_aromatic_rings, n_bridgeheads
    )

    sas = 1.0 + (complexity_score / 100.0) * 9.0
    sas_cls = _sas_class(sas)

    # Drug-likeness rules
    lipinski_pass = mw <= 500 and n_h_donors <= 5 and n_h_acceptors <= 10
    veber_pass = n_rotatable_bonds <= 10 and (n_h_donors + n_h_acceptors) <= 12
    egan_pass = n_h_donors <= 5 and n_rotatable_bonds <= 10 and mw <= 450
    lead_like = mw <= 350 and n_rings <= 3 and n_stereocenters <= 1

    if lead_like:
        dev_stage = "lead"
    elif lipinski_pass and veber_pass:
        dev_stage = "candidate"
    else:
        dev_stage = "advanced"

    opt_priority = _optimization_priority(
        mw, n_rings, n_stereocenters, n_rotatable_bonds, n_aromatic_rings, n_bridgeheads
    )

    notes_parts = [
        f"Complexity score: {complexity_score:.1f}/100.",
        f"SAS: {sas:.2f} ({sas_cls}).",
        f"Lipinski: {'pass' if lipinski_pass else 'fail'}.",
        f"Veber: {'pass' if veber_pass else 'fail'}.",
        f"Lead-like: {'yes' if lead_like else 'no'}.",
        f"Primary complexity driver: {opt_priority}.",
    ]
    notes = " ".join(notes_parts)

    return ComplexityResult(
        compound_name=compound_name,
        mw=mw,
        n_rings=n_rings,
        n_stereocenters=n_stereocenters,
        complexity_score=complexity_score,
        sas=sas,
        sas_class=sas_cls,
        lipinski_pass=lipinski_pass,
        veber_pass=veber_pass,
        egan_pass=egan_pass,
        lead_like=lead_like,
        development_stage_fit=dev_stage,
        optimization_priority=opt_priority,
        notes=notes,
    )


def screen_complexity(compounds: list[dict]) -> list[ComplexityResult]:
    """Score a list of compound dicts and return sorted by complexity_score ascending."""
    results = [score_molecular_complexity(**c) for c in compounds]
    return sorted(results, key=lambda r: r.complexity_score)


def compare_complexity_optimization(
    base_compound: dict, variants: list[dict]
) -> list[ComplexityResult]:
    """Return base compound first, then variants sorted by complexity_score ascending."""
    base_result = score_molecular_complexity(**base_compound)
    variant_results = sorted(
        [score_molecular_complexity(**v) for v in variants],
        key=lambda r: r.complexity_score,
    )
    return [base_result, *variant_results]
