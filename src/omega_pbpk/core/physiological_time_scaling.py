"""Phase 346 — Physiological Time Scaling (Species Translation).

Models species-to-human translation using allometric scaling with
optional MLP (maximum lifespan potential) correction for t_half.
"""

from __future__ import annotations

from dataclasses import dataclass

# Body weights in kg
_BODY_WEIGHTS: dict[str, float] = {
    "mouse": 0.025,
    "rat": 0.25,
    "monkey": 5.0,
    "dog": 10.0,
    "human": 70.0,
}

# Maximum lifespan potential in years
_LIFESPAN_YEARS: dict[str, float] = {
    "mouse": 2.0,
    "rat": 3.0,
    "monkey": 30.0,
    "dog": 12.0,
    "human": 80.0,
}

# Species for which MLP correction should be applied (MLP < 30 years)
_MLP_CORRECTION_SPECIES: frozenset[str] = frozenset({"mouse", "rat", "dog"})

_VALID_SPECIES: frozenset[str] = frozenset(_BODY_WEIGHTS.keys())

_CL_EXPONENT: float = 0.75
_VD_EXPONENT: float = 1.0


@dataclass(frozen=True)
class PhysiologicalTimeScalingResult:
    """Result of physiological time scaling / allometric species translation."""

    drug_name: str
    species_from: str
    species_to: str
    bw_from_kg: float
    bw_to_kg: float
    cl_from_L_per_h: float
    cl_predicted_L_per_h: float
    vd_from_L: float
    vd_predicted_L: float
    t_half_from_h: float
    t_half_predicted_h: float
    cl_scaling_exponent: float
    vd_scaling_exponent: float
    mlp_correction_applied: bool
    confidence_class: str
    notes: str


def _confidence_class(species: str) -> str:
    """Return confidence class based on source species."""
    if species == "human":
        return "reference"
    if species in ("rat", "dog", "monkey"):
        return "high"
    return "moderate"  # mouse


def _scale_cl(cl_from: float, bw_from: float, bw_to: float) -> float:
    """Scale clearance using allometric exponent 0.75."""
    return cl_from * (bw_to / bw_from) ** _CL_EXPONENT


def _scale_vd(vd_from: float, bw_from: float, bw_to: float) -> float:
    """Scale volume of distribution linearly."""
    return vd_from * (bw_to / bw_from) ** _VD_EXPONENT


def _t_half(vd_L: float, cl_L_per_h: float) -> float:
    """Compute half-life from Vd and CL."""
    if cl_L_per_h <= 0:
        return 0.0
    return 0.693 * vd_L / cl_L_per_h


def _mlp_correction_factor(species_from: str) -> tuple[bool, float]:
    """Return (applied, factor) for MLP correction."""
    mlp_from = _LIFESPAN_YEARS[species_from]
    mlp_human = _LIFESPAN_YEARS["human"]
    if mlp_from < 30.0:
        factor = (mlp_human / mlp_from) ** (1.0 / 3.0)
        return True, factor
    return False, 1.0


def translate_species(
    drug_name: str,
    species_from: str,
    species_to: str,
    cl_from_L_per_h: float,
    vd_from_L: float,
    t_half_from_h: float | None = None,
) -> PhysiologicalTimeScalingResult:
    """Translate PK parameters from one species to another using allometric scaling.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    species_from:
        Source species ('mouse', 'rat', 'monkey', 'dog', 'human').
    species_to:
        Target species ('mouse', 'rat', 'monkey', 'dog', 'human').
    cl_from_L_per_h:
        Clearance in the source species (L/h).
    vd_from_L:
        Volume of distribution in the source species (L).
    t_half_from_h:
        Half-life in source species (h). If None, derived from cl/vd.

    Returns
    -------
    PhysiologicalTimeScalingResult
    """
    if species_from not in _VALID_SPECIES:
        raise ValueError(
            f"species_from must be one of {sorted(_VALID_SPECIES)}, got {species_from!r}"
        )
    if species_to not in _VALID_SPECIES:
        raise ValueError(f"species_to must be one of {sorted(_VALID_SPECIES)}, got {species_to!r}")
    if cl_from_L_per_h <= 0:
        raise ValueError(f"cl_from_L_per_h must be > 0, got {cl_from_L_per_h}")
    if vd_from_L <= 0:
        raise ValueError(f"vd_from_L must be > 0, got {vd_from_L}")

    bw_from = _BODY_WEIGHTS[species_from]
    bw_to = _BODY_WEIGHTS[species_to]

    cl_pred = _scale_cl(cl_from_L_per_h, bw_from, bw_to)
    vd_pred = _scale_vd(vd_from_L, bw_from, bw_to)

    # t_half from source species
    if t_half_from_h is None:
        t_half_from = _t_half(vd_from_L, cl_from_L_per_h)
    else:
        t_half_from = t_half_from_h

    # MLP correction applies only when scaling to human from small species
    mlp_applied = False
    mlp_factor = 1.0
    if species_to == "human" and species_from in _MLP_CORRECTION_SPECIES:
        mlp_applied, mlp_factor = _mlp_correction_factor(species_from)

    # Predicted half-life from scaled values, then apply MLP correction
    t_half_pred_base = _t_half(vd_pred, cl_pred)
    t_half_pred = t_half_pred_base * mlp_factor

    confidence = _confidence_class(species_from)

    notes_parts = [
        f"Allometric CL exponent: {_CL_EXPONENT}",
        f"Allometric Vd exponent: {_VD_EXPONENT}",
        f"BW ratio: {bw_to / bw_from:.2f}",
    ]
    if mlp_applied:
        notes_parts.append(f"MLP correction factor: {mlp_factor:.3f}")
    if species_from == species_to:
        notes_parts.append("Trivial case: same species")

    return PhysiologicalTimeScalingResult(
        drug_name=drug_name,
        species_from=species_from,
        species_to=species_to,
        bw_from_kg=bw_from,
        bw_to_kg=bw_to,
        cl_from_L_per_h=cl_from_L_per_h,
        cl_predicted_L_per_h=cl_pred,
        vd_from_L=vd_from_L,
        vd_predicted_L=vd_pred,
        t_half_from_h=t_half_from,
        t_half_predicted_h=t_half_pred,
        cl_scaling_exponent=_CL_EXPONENT,
        vd_scaling_exponent=_VD_EXPONENT,
        mlp_correction_applied=mlp_applied,
        confidence_class=confidence,
        notes="; ".join(notes_parts),
    )


def translate_across_species(
    drug_name: str,
    cl_dict: dict[str, float],
    vd_dict: dict[str, float],
    species_to: str = "human",
) -> list[PhysiologicalTimeScalingResult]:
    """Translate PK parameters from multiple species to a target species.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    cl_dict:
        Dict mapping species name to clearance (L/h).
    vd_dict:
        Dict mapping species name to volume of distribution (L).
    species_to:
        Target species for all translations (default 'human').

    Returns
    -------
    List of PhysiologicalTimeScalingResult, one per source species.
    """
    results: list[PhysiologicalTimeScalingResult] = []
    for species, cl_val in cl_dict.items():
        vd_val = vd_dict.get(species)
        if vd_val is None:
            continue
        result = translate_species(
            drug_name=drug_name,
            species_from=species,
            species_to=species_to,
            cl_from_L_per_h=cl_val,
            vd_from_L=vd_val,
        )
        results.append(result)
    return results
