"""Phase 357 — Hepatic Clearance Scaling: In Vitro to In Vivo (IVIVE).

Scales in vitro hepatic clearance measurements (microsomal, hepatocyte, s9)
to in vivo human clearance using well-stirred liver model.
"""

from __future__ import annotations

from dataclasses import dataclass

# Scaling constants
MPPGL_MG_PER_G = 45.0  # microsomal protein per gram liver
HPGL_CELLS_PER_G = 120e6  # hepatocytes per gram liver
S9PPGL_MG_PER_G = 120.0  # S9 protein per gram liver

VALID_SYSTEMS = {"microsomal", "hepatocyte", "s9"}


@dataclass(frozen=True)
class IVIVEHepaticScalingResult:
    """Result of in vitro to in vivo hepatic clearance scaling."""

    drug_name: str
    in_vitro_system: str
    clint_in_vitro: float
    fu_inc: float
    fu_plasma: float
    clint_liver_L_per_h: float
    clint_corrected_L_per_h: float
    cl_hepatic_L_per_h: float
    extraction_ratio: float
    f_hepatic: float
    ivive_success: bool
    scaling_factor: float
    notes: str


def scale_hepatic_clearance(
    drug_name: str,
    clint_in_vitro: float,
    in_vitro_system: str = "microsomal",
    incubation_protein_mg_mL: float = 1.0,
    cell_density: float = 1e6,
    fu_inc: float = 1.0,
    fu_plasma: float = 1.0,
    body_weight_kg: float = 70.0,
    liver_weight_g: float = 1500.0,
    q_hepatic_L_per_h: float = 90.0,
) -> IVIVEHepaticScalingResult:
    """Scale in vitro intrinsic clearance to in vivo hepatic clearance.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    clint_in_vitro:
        In vitro intrinsic clearance (mL/h/mg protein for microsomal/s9,
        or mL/h/million cells for hepatocyte).
    in_vitro_system:
        One of "microsomal", "hepatocyte", or "s9".
    incubation_protein_mg_mL:
        Protein concentration in incubation (mg/mL) — used for microsomal/s9.
    cell_density:
        Cell density (cells/mL) — used for hepatocyte system.
    fu_inc:
        Fraction unbound in incubation medium (0 < fu_inc <= 1).
    fu_plasma:
        Fraction unbound in plasma (0 < fu_plasma <= 1).
    body_weight_kg:
        Subject body weight in kg (default 70).
    liver_weight_g:
        Liver weight in grams (default 1500).
    q_hepatic_L_per_h:
        Hepatic blood flow in L/h (default 90).

    Returns
    -------
    IVIVEHepaticScalingResult
    """
    # --- Input validation ---
    if clint_in_vitro <= 0:
        raise ValueError(f"clint_in_vitro must be > 0, got {clint_in_vitro}")
    if not (0 < fu_inc <= 1):
        raise ValueError(f"fu_inc must be in (0, 1], got {fu_inc}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if in_vitro_system not in VALID_SYSTEMS:
        raise ValueError(f"in_vitro_system must be one of {VALID_SYSTEMS}, got {in_vitro_system!r}")
    if liver_weight_g <= 0:
        raise ValueError(f"liver_weight_g must be > 0, got {liver_weight_g}")
    if q_hepatic_L_per_h <= 0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0, got {q_hepatic_L_per_h}")

    # --- Step 1: Scale to whole-liver CLint (mL/h) then convert to L/h ---
    if in_vitro_system == "microsomal":
        # clint_in_vitro is in mL/min/mg protein
        # CLint_liver (mL/h) = CLint_iv / protein_conc * MPPGL * liver_wt
        clint_liver_mL_per_h = (
            clint_in_vitro / incubation_protein_mg_mL * MPPGL_MG_PER_G * liver_weight_g
        )
        scaling_factor = MPPGL_MG_PER_G * liver_weight_g / incubation_protein_mg_mL
    elif in_vitro_system == "hepatocyte":
        # clint_in_vitro is in mL/h/million cells; cell_density in cells/mL
        # Normalise: divide by (cell_density / 1e6) to get per-mL-incubation basis,
        # then multiply by HPGL * liver_wt
        cell_density_M = cell_density / 1e6  # convert to million cells/mL
        clint_liver_mL_per_h = (
            clint_in_vitro / cell_density_M * (HPGL_CELLS_PER_G / 1e6) * liver_weight_g
        )
        scaling_factor = (HPGL_CELLS_PER_G / 1e6) * liver_weight_g / cell_density_M
    else:  # s9
        clint_liver_mL_per_h = (
            clint_in_vitro / incubation_protein_mg_mL * S9PPGL_MG_PER_G * liver_weight_g
        )
        scaling_factor = S9PPGL_MG_PER_G * liver_weight_g / incubation_protein_mg_mL

    # Convert mL/h → L/h
    clint_liver_L_per_h = clint_liver_mL_per_h / 1000.0

    # --- Step 2: Binding correction ---
    clint_corrected_L_per_h = clint_liver_L_per_h * fu_inc / fu_plasma

    # --- Step 3: Well-stirred model ---
    q_h = q_hepatic_L_per_h
    numerator = q_h * fu_plasma * clint_corrected_L_per_h
    denominator = q_h + fu_plasma * clint_corrected_L_per_h
    cl_hepatic_L_per_h = numerator / denominator if denominator > 0 else 0.0

    # --- Step 4: Derived metrics ---
    extraction_ratio = cl_hepatic_L_per_h / q_h
    f_hepatic = 1.0 - extraction_ratio
    ivive_success = 0.0 <= extraction_ratio <= 1.0

    notes_parts = [f"System: {in_vitro_system}"]
    if extraction_ratio > 0.7:
        notes_parts.append("high-extraction compound")
    elif extraction_ratio < 0.3:
        notes_parts.append("low-extraction compound")
    else:
        notes_parts.append("intermediate-extraction compound")

    notes = "; ".join(notes_parts)

    return IVIVEHepaticScalingResult(
        drug_name=drug_name,
        in_vitro_system=in_vitro_system,
        clint_in_vitro=clint_in_vitro,
        fu_inc=fu_inc,
        fu_plasma=fu_plasma,
        clint_liver_L_per_h=clint_liver_L_per_h,
        clint_corrected_L_per_h=clint_corrected_L_per_h,
        cl_hepatic_L_per_h=cl_hepatic_L_per_h,
        extraction_ratio=extraction_ratio,
        f_hepatic=f_hepatic,
        ivive_success=ivive_success,
        scaling_factor=scaling_factor,
        notes=notes,
    )
