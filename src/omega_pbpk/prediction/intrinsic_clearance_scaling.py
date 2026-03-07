"""Phase 181 — Intrinsic clearance scaling from in vitro to in vivo.

Uses the well-stirred liver model to predict hepatic clearance from
microsomal intrinsic clearance data.
"""

from __future__ import annotations

from dataclasses import dataclass

# Species-specific parameters: (Q_h L/h, liver_weight_g/kg_BW, typical_BW_kg)
_SPECIES_PARAMS: dict = {
    "mouse": (0.09, 88.0, 0.025),
    "rat": (4.2, 40.0, 0.25),
    "dog": (30.9, 32.0, 10.0),
    "human": (90.0, 21.4, 70.0),
}

_DEFAULT_HPGL = 45.0  # mg microsomal protein / g liver
_DEFAULT_QH_HUMAN = 90.0  # L/h
_DEFAULT_LIVER_WEIGHT_PER_KG = 21.4  # g/kg body weight
_DEFAULT_BW_KG = 70.0


@dataclass(frozen=True)
class ClintScalingResult:
    """Result of CLint in vitro → in vivo scaling."""

    drug_name: str
    species: str
    clint_invitro_uL_min_mg: float
    clint_invivo_mL_per_min: float
    cl_hepatic_L_per_h: float
    extraction_ratio: float
    fu_mic: float
    hpgl: float
    notes: str


def scale_clint(
    drug_name: str,
    clint_uL_per_min_per_mg: float,
    fu_mic: float,
    hpgl: float = _DEFAULT_HPGL,
    liver_weight_g: float | None = None,
    body_weight_kg: float = _DEFAULT_BW_KG,
    q_h_L_per_h: float = _DEFAULT_QH_HUMAN,
    fu_plasma: float | None = None,
    species: str = "human",
) -> ClintScalingResult:
    """Scale microsomal CLint to in vivo hepatic CL using well-stirred model.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    clint_uL_per_min_per_mg:
        Measured in vitro intrinsic clearance (uL/min/mg microsomal protein).
    fu_mic:
        Fraction unbound in microsomal incubation (0 < fu_mic <= 1).
    hpgl:
        Hepatocellularity / microsomal protein per gram liver
        (mg/g liver, default 45).
    liver_weight_g:
        Total liver weight in grams.  If None, calculated from
        ``liver_weight_per_kg * body_weight_kg`` using 21.4 g/kg for humans.
    body_weight_kg:
        Body weight used to estimate liver weight when ``liver_weight_g`` is None.
    q_h_L_per_h:
        Hepatic blood/plasma flow (L/h, default 90 for humans).
    fu_plasma:
        Fraction unbound in plasma.  Defaults to ``fu_mic`` when not provided.
    species:
        Label stored in the result (informational).

    Returns
    -------
    ClintScalingResult
    """
    # --- validation ---
    if clint_uL_per_min_per_mg <= 0:
        raise ValueError("clint_uL_per_min_per_mg must be > 0")
    if not (0 < fu_mic <= 1):
        raise ValueError("fu_mic must be in (0, 1]")
    if body_weight_kg <= 0:
        raise ValueError("body_weight_kg must be > 0")

    if liver_weight_g is None:
        liver_weight_g = _DEFAULT_LIVER_WEIGHT_PER_KG * body_weight_kg

    if fu_plasma is None:
        fu_plasma = fu_mic

    # Step 1: scale to whole-liver CLint (mL/min)
    # CLint_invitro is in uL/min/mg → convert to mL/min/mg (* 0.001)
    clint_mL_per_min_per_mg = clint_uL_per_min_per_mg * 1e-3
    clint_invivo_mL_per_min = clint_mL_per_min_per_mg * hpgl * liver_weight_g

    # Step 2: well-stirred model
    # CL_h = Q_h * fu_p * CLint / (Q_h + fu_p * CLint)
    # Convert Q_h from L/h to mL/min for consistent units
    q_h_mL_per_min = q_h_L_per_h * 1000.0 / 60.0

    fu_clint = fu_plasma * clint_invivo_mL_per_min
    cl_h_mL_per_min = q_h_mL_per_min * fu_clint / (q_h_mL_per_min + fu_clint)
    cl_h_L_per_h = cl_h_mL_per_min * 60.0 / 1000.0

    extraction_ratio = cl_h_mL_per_min / q_h_mL_per_min

    notes_parts: list = []
    if extraction_ratio > 0.7:
        notes_parts.append("high extraction (>0.7)")
    elif extraction_ratio < 0.3:
        notes_parts.append("low extraction (<0.3)")
    else:
        notes_parts.append("intermediate extraction")
    notes = "; ".join(notes_parts)

    return ClintScalingResult(
        drug_name=drug_name,
        species=species,
        clint_invitro_uL_min_mg=clint_uL_per_min_per_mg,
        clint_invivo_mL_per_min=clint_invivo_mL_per_min,
        cl_hepatic_L_per_h=cl_h_L_per_h,
        extraction_ratio=extraction_ratio,
        fu_mic=fu_mic,
        hpgl=hpgl,
        notes=notes,
    )


def compare_species(
    drug_name: str,
    clint_uL_per_min_per_mg: float,
    fu_mic: float,
    hpgl: float = _DEFAULT_HPGL,
) -> list[ClintScalingResult]:
    """Compare CLint scaling across mouse, rat, dog, and human.

    Parameters
    ----------
    drug_name:
        Compound identifier.
    clint_uL_per_min_per_mg:
        In vitro CLint (uL/min/mg microsomal protein).
    fu_mic:
        Fraction unbound in the microsomal incubation.
    hpgl:
        Microsomal protein per gram liver (mg/g).

    Returns
    -------
    list of ClintScalingResult, one per species (mouse, rat, dog, human).
    """
    results: list = []
    for species, (q_h, lw_per_kg, bw) in _SPECIES_PARAMS.items():
        liver_weight_g = lw_per_kg * bw
        result = scale_clint(
            drug_name=drug_name,
            clint_uL_per_min_per_mg=clint_uL_per_min_per_mg,
            fu_mic=fu_mic,
            hpgl=hpgl,
            liver_weight_g=liver_weight_g,
            body_weight_kg=bw,
            q_h_L_per_h=q_h,
            species=species,
        )
        results.append(result)
    return results
