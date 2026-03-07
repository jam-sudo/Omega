"""Phase 308 — Intrinsic Clearance Scaling (In Vitro to In Vivo).

Scales in vitro intrinsic clearance (microsomal or hepatocyte) to in vivo
hepatic clearance using the well-stirred model. Handles species differences
and fu_mic corrections.
"""

from dataclasses import dataclass

_VALID_SYSTEMS = {"microsomes", "hepatocytes"}
_VALID_SPECIES = {"human", "rat", "mouse", "dog", "monkey"}

_SPECIES_PARAMS: dict[str, dict[str, float]] = {
    "human": {
        "liver_weight_g": 1500.0,
        "q_hepatic_L_per_h": 90.0,
        "microsomal_protein_mg_per_g_liver": 45.0,
        "hepatocytes_million_per_g": 120.0,
    },
    "rat": {
        "liver_weight_g": 10.0,
        "q_hepatic_L_per_h": 4.2,
        "microsomal_protein_mg_per_g_liver": 48.0,
        "hepatocytes_million_per_g": 135.0,
    },
    "mouse": {
        "liver_weight_g": 1.75,
        "q_hepatic_L_per_h": 0.8,
        "microsomal_protein_mg_per_g_liver": 45.0,
        "hepatocytes_million_per_g": 130.0,
    },
    "dog": {
        "liver_weight_g": 400.0,
        "q_hepatic_L_per_h": 30.0,
        "microsomal_protein_mg_per_g_liver": 37.0,
        "hepatocytes_million_per_g": 100.0,
    },
    "monkey": {
        "liver_weight_g": 120.0,
        "q_hepatic_L_per_h": 22.0,
        "microsomal_protein_mg_per_g_liver": 44.0,
        "hepatocytes_million_per_g": 110.0,
    },
}


@dataclass(frozen=True)
class ClintScalingResult:
    drug_name: str
    species: str
    in_vitro_system: str
    fu_mic: float
    fu_plasma: float
    cl_int_total_L_per_h: float
    extraction_ratio: float
    cl_hepatic_L_per_h: float
    f_oral_estimate: float
    extraction_class: str
    notes: str


def _validate_inputs(
    cl_int_uL_per_min_per_mg_protein: float,
    in_vitro_system: str,
    species: str,
    fu_mic: float,
    fu_plasma: float,
    incubation_protein_mg_per_mL: float,
) -> None:
    if cl_int_uL_per_min_per_mg_protein <= 0:
        raise ValueError(
            f"cl_int_uL_per_min_per_mg_protein must be > 0, got {cl_int_uL_per_min_per_mg_protein}"
        )
    if in_vitro_system not in _VALID_SYSTEMS:
        raise ValueError(
            f"in_vitro_system must be one of {_VALID_SYSTEMS}, got {in_vitro_system!r}"
        )
    if species not in _VALID_SPECIES:
        raise ValueError(f"species must be one of {_VALID_SPECIES}, got {species!r}")
    if not (0 < fu_mic <= 1):
        raise ValueError(f"fu_mic must be in (0, 1], got {fu_mic}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if incubation_protein_mg_per_mL <= 0:
        raise ValueError(
            f"incubation_protein_mg_per_mL must be > 0, got {incubation_protein_mg_per_mL}"
        )


def _extraction_class(eh: float) -> str:
    if eh > 0.7:
        return "high"
    elif eh >= 0.3:
        return "intermediate"
    else:
        return "low"


def _compute_result(
    drug_name: str,
    cl_int_uL_per_min_per_mg_protein: float,
    in_vitro_system: str,
    species: str,
    fu_mic: float,
    fu_plasma: float,
) -> ClintScalingResult:
    params = _SPECIES_PARAMS[species]
    liver_weight_g = params["liver_weight_g"]
    q = params["q_hepatic_L_per_h"]

    # Convert raw assay value to L/h per unit (mg protein or million cells)
    cl_int_per_unit = cl_int_uL_per_min_per_mg_protein * 60.0 / 1000.0  # L/h per unit

    if in_vitro_system == "microsomes":
        scale_factor = params["microsomal_protein_mg_per_g_liver"] * liver_weight_g
        cl_int_total = cl_int_per_unit * scale_factor  # L/h
    else:
        # hepatocytes: cl_int expressed per million cells
        scale_factor = params["hepatocytes_million_per_g"] * liver_weight_g
        cl_int_total = cl_int_per_unit * scale_factor  # L/h

    # Correct for fu_mic (bound fraction not metabolized)
    cl_int_scaled = cl_int_total * fu_mic

    # Well-stirred model: EH = fu_plasma * CLint_scaled / (Q + fu_plasma * CLint_scaled)
    numerator = fu_plasma * cl_int_scaled
    eh = numerator / (q + numerator)
    eh = max(0.0, min(1.0, eh))

    cl_hepatic = eh * q
    f_oral = 1.0 - eh
    ext_class = _extraction_class(eh)

    notes_parts = [
        f"CL_int_total (unbound-corrected): {cl_int_scaled:.3f} L/h.",
        f"Hepatic blood flow (Q): {q:.1f} L/h.",
        f"Extraction class: {ext_class}.",
    ]
    if eh > 0.7:
        notes_parts.append("High extraction: sensitive to hepatic blood flow changes.")
    elif eh < 0.3:
        notes_parts.append("Low extraction: sensitive to fu_plasma and CLint changes.")

    return ClintScalingResult(
        drug_name=drug_name,
        species=species,
        in_vitro_system=in_vitro_system,
        fu_mic=fu_mic,
        fu_plasma=fu_plasma,
        cl_int_total_L_per_h=cl_int_total,
        extraction_ratio=eh,
        cl_hepatic_L_per_h=cl_hepatic,
        f_oral_estimate=f_oral,
        extraction_class=ext_class,
        notes=" ".join(notes_parts),
    )


def scale_clint_to_in_vivo(
    drug_name: str,
    cl_int_uL_per_min_per_mg_protein: float,
    in_vitro_system: str,
    species: str,
    fu_mic: float,
    fu_plasma: float,
    incubation_protein_mg_per_mL: float = 1.0,
) -> ClintScalingResult:
    """Scale in vitro CLint to in vivo hepatic clearance.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    cl_int_uL_per_min_per_mg_protein:
        In vitro intrinsic clearance in uL/min/mg protein (or uL/min/million cells
        for hepatocytes). Must be > 0.
    in_vitro_system:
        "microsomes" or "hepatocytes".
    species:
        "human", "rat", "mouse", "dog", or "monkey".
    fu_mic:
        Fraction unbound in the incubation (0, 1].
    fu_plasma:
        Fraction unbound in plasma (0, 1].
    incubation_protein_mg_per_mL:
        Protein concentration in incubation (mg/mL). Must be > 0.

    Returns
    -------
    ClintScalingResult
    """
    _validate_inputs(
        cl_int_uL_per_min_per_mg_protein,
        in_vitro_system,
        species,
        fu_mic,
        fu_plasma,
        incubation_protein_mg_per_mL,
    )
    return _compute_result(
        drug_name,
        cl_int_uL_per_min_per_mg_protein,
        in_vitro_system,
        species,
        fu_mic,
        fu_plasma,
    )


def compare_species(
    drug_name: str,
    cl_int_uL_per_min_per_mg_protein: float,
    in_vitro_system: str,
    fu_mic: float,
    fu_plasma: float,
) -> list[ClintScalingResult]:
    """Compare CLint scaling across all five species.

    Returns results sorted by cl_hepatic_L_per_h descending.
    """
    _validate_inputs(
        cl_int_uL_per_min_per_mg_protein,
        in_vitro_system,
        "human",
        fu_mic,
        fu_plasma,
        1.0,
    )
    results = [
        _compute_result(
            drug_name,
            cl_int_uL_per_min_per_mg_protein,
            in_vitro_system,
            sp,
            fu_mic,
            fu_plasma,
        )
        for sp in _VALID_SPECIES
    ]
    results.sort(key=lambda r: r.cl_hepatic_L_per_h, reverse=True)
    return results
