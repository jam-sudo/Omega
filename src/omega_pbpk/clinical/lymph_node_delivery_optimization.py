"""
Phase 1063 — Drug Lymph Node Delivery Optimization

Prediction/optimization module for designing formulations to maximize
lymph node delivery. Distinct from core/lymph_node_pk.py (simulation).
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_DRUG_CLASSES = {"small_molecule", "peptide", "protein", "nanoparticle"}
_VALID_FORMULATIONS = {"solution", "nanoparticle", "liposome", "emulsion", "depot"}
_VALID_ROUTES = {"SC", "IM", "intradermal"}

_ROUTE_PARAMS: dict[str, dict[str, float]] = {
    "SC": {"lymph_fraction_max": 0.5, "plasma_fraction": 0.5},
    "IM": {"lymph_fraction_max": 0.3, "plasma_fraction": 0.7},
    "intradermal": {"lymph_fraction_max": 0.7, "plasma_fraction": 0.3},
}

_FORMULATION_MULT: dict[str, float] = {
    "solution": 0.5,
    "nanoparticle": 1.0,
    "liposome": 0.9,
    "emulsion": 0.7,
    "depot": 0.6,
}

_FORMULATION_SCORE_BASE: dict[str, float] = {
    "solution": 40.0,
    "nanoparticle": 90.0,
    "liposome": 85.0,
    "emulsion": 70.0,
    "depot": 60.0,
}

_OPTIMAL_ROUTE_BY_CLASS: dict[str, str] = {
    "small_molecule": "SC",
    "peptide": "SC",
    "protein": "SC",
    "nanoparticle": "intradermal",
}


@dataclass(frozen=True)
class LymphDeliveryOptimizationResult:
    """Result of lymph node delivery optimization."""

    drug_name: str
    formulation_type: str
    particle_size_nm: float
    f_lymph: float  # fraction delivered to lymph nodes (0-1)
    lymph_node_auc_ratio: float  # lymph/plasma AUC ratio
    systemic_exposure_reduction: float  # 0-1 (reduce systemic to minimize side effects)
    targeting_score: float  # 0-100
    optimal_injection_route: str  # "SC", "IM", "intradermal"
    recommended_particle_size: float  # nm (optimal for this drug class)
    formulation_score: float  # 0-100
    recommended: bool  # True if targeting_score >= 70
    notes: str


def _calc_size_score(eff_size: float) -> float:
    """Peak at 50 nm, clamped to [0, 100]."""
    score = 100.0 - abs(eff_size - 50.0) * 2.0
    return max(0.0, min(100.0, score))


def _calc_recommended_particle_size(drug_class: str, mw_Da: float) -> float:
    """Optimal particle size (nm) by drug class."""
    if drug_class == "small_molecule":
        return max(1.0, mw_Da / 500.0)
    if drug_class in ("nanoparticle", "liposome", "emulsion"):
        return 30.0
    if drug_class == "peptide":
        return 5.0
    if drug_class == "protein":
        return 10.0
    # depot
    return 100.0


def optimize_lymph_node_delivery(
    drug_name: str,
    mw_Da: float,
    logp: float,
    drug_class: str = "small_molecule",
    formulation_type: str = "solution",
    particle_size_nm: float = 0.0,
    injection_route: str = "SC",
    target_lymph_nodes: str = "draining",
    peg_coated: bool = False,
) -> LymphDeliveryOptimizationResult:
    """Optimize formulation parameters for maximal lymph node delivery.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    mw_Da:
        Molecular weight in Daltons (must be > 0).
    logp:
        Octanol-water partition coefficient.
    drug_class:
        One of "small_molecule", "peptide", "protein", "nanoparticle".
    formulation_type:
        One of "solution", "nanoparticle", "liposome", "emulsion", "depot".
    particle_size_nm:
        Particle size in nm; 0 means free drug (size estimated from MW).
    injection_route:
        One of "SC", "IM", "intradermal".
    target_lymph_nodes:
        "draining" or "distant" (informational).
    peg_coated:
        Whether the formulation has PEG coating.

    Returns
    -------
    LymphDeliveryOptimizationResult
    """
    # Validation
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if particle_size_nm < 0:
        raise ValueError(f"particle_size_nm must be >= 0, got {particle_size_nm}")
    if drug_class not in _VALID_DRUG_CLASSES:
        raise ValueError(
            f"drug_class must be one of {sorted(_VALID_DRUG_CLASSES)}, got '{drug_class}'"
        )
    if formulation_type not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation_type must be one of {sorted(_VALID_FORMULATIONS)}, got '{formulation_type}'"
        )
    if injection_route not in _VALID_ROUTES:
        raise ValueError(
            f"injection_route must be one of {sorted(_VALID_ROUTES)}, got '{injection_route}'"
        )

    # Effective size for scoring
    if particle_size_nm == 0.0:
        eff_size = max(1.0, mw_Da / 1000.0)
    else:
        eff_size = particle_size_nm

    size_score = _calc_size_score(eff_size)

    # Route parameters
    route_p = _ROUTE_PARAMS[injection_route]
    lymph_fraction_max = route_p["lymph_fraction_max"]

    # Formulation modifier
    f_lymph_mult = _FORMULATION_MULT[formulation_type]

    # PEG boost
    peg_boost = 1.2 if peg_coated else 1.0

    # Lymph fraction
    f_lymph = lymph_fraction_max * f_lymph_mult * peg_boost * (size_score / 100.0 + 0.1)
    f_lymph = min(0.85, f_lymph)

    # Systemic exposure reduction
    systemic_exposure_reduction = f_lymph

    # Lymph node AUC ratio
    if f_lymph < 1.0:
        lymph_node_auc_ratio = f_lymph / (1.0 - f_lymph)
    else:
        lymph_node_auc_ratio = 10.0

    # Formulation score (base)
    formulation_score_base = _FORMULATION_SCORE_BASE[formulation_type]
    formulation_score = formulation_score_base

    # Targeting score
    targeting_score = size_score * 0.4 + f_lymph * 100.0 * 0.4 + formulation_score_base * 0.2
    targeting_score = max(0.0, min(100.0, targeting_score))

    # Optimal injection route
    optimal_injection_route = _OPTIMAL_ROUTE_BY_CLASS.get(drug_class, "SC")

    # Recommended particle size
    recommended_particle_size = _calc_recommended_particle_size(drug_class, mw_Da)

    # Recommendation flag
    recommended = targeting_score >= 70.0

    # Notes
    note_parts: list[str] = []
    note_parts.append(f"Effective size {eff_size:.1f} nm (size_score={size_score:.1f})")
    note_parts.append(f"Route '{injection_route}': lymph_fraction_max={lymph_fraction_max}")
    note_parts.append(
        f"Formulation '{formulation_type}': mult={f_lymph_mult}, base_score={formulation_score_base}"
    )
    if peg_coated:
        note_parts.append("PEG coating applied (1.2x boost)")
    note_parts.append(f"f_lymph={f_lymph:.3f}, targeting_score={targeting_score:.1f}")
    if recommended:
        note_parts.append("RECOMMENDED: targeting score >= 70")
    else:
        note_parts.append(
            f"Not recommended (score {targeting_score:.1f} < 70); consider nanoparticle or intradermal route"
        )
    note_parts.append(f"Optimal particle size for {drug_class}: {recommended_particle_size:.1f} nm")
    if target_lymph_nodes == "distant":
        note_parts.append("Distant lymph node targeting requires systemic circulation")
    notes = "; ".join(note_parts)

    return LymphDeliveryOptimizationResult(
        drug_name=drug_name,
        formulation_type=formulation_type,
        particle_size_nm=particle_size_nm,
        f_lymph=f_lymph,
        lymph_node_auc_ratio=lymph_node_auc_ratio,
        systemic_exposure_reduction=systemic_exposure_reduction,
        targeting_score=targeting_score,
        optimal_injection_route=optimal_injection_route,
        recommended_particle_size=recommended_particle_size,
        formulation_score=formulation_score,
        recommended=recommended,
        notes=notes,
    )
