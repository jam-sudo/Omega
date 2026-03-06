"""Metabolic stability — intrinsic clearance from in vitro assays."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# In vitro system parameters
# ---------------------------------------------------------------------------
MICROSOME_SCALING: dict[str, dict[str, float]] = {
    "human_liver": {"mppgl": 45.0, "liver_weight_g": 1500.0},
    "rat_liver": {"mppgl": 48.0, "liver_weight_g": 10.0},
    "mouse_liver": {"mppgl": 52.0, "liver_weight_g": 1.5},
}

HEPATOCYTE_SCALING: dict[str, dict[str, float]] = {
    "human": {"cells_per_g_liver": 120e6, "liver_weight_g": 1500.0},
    "rat": {"cells_per_g_liver": 110e6, "liver_weight_g": 10.0},
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetabolicStabilityResult:
    drug_name: str
    assay_type: str  # "microsome" or "hepatocyte"
    species: str
    clint_uL_per_min_per_mg: float  # intrinsic CL (microsomes) or per million cells
    clint_scaled_L_per_h: float  # scaled to whole liver
    cl_hepatic_L_per_h: float  # well-stirred hepatic CL
    extraction_ratio: float  # hepatic E
    bioavailability_fh: float  # 1 - E
    half_life_in_vitro_min: float  # t1/2 from in vitro data
    stability_class: str  # "stable" (>60min), "moderate" (30-60min), "labile" (<30min)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Function 1: CLint from in vitro half-life
# ---------------------------------------------------------------------------
def clint_from_half_life(
    t_half_min: float,
    protein_mg_mL: float = 0.5,
    cells_per_mL: float = 1e6,
    assay_type: str = "microsome",
) -> float:
    """Derive intrinsic clearance from in vitro half-life.

    Returns CLint in uL/min/mg protein (microsomes) or uL/min/10^6 cells
    (hepatocytes).
    """
    if assay_type == "microsome":
        return (0.693 / t_half_min) * (1000.0 / protein_mg_mL)
    elif assay_type == "hepatocyte":
        return (0.693 / t_half_min) * (1e6 / cells_per_mL)
    else:
        raise ValueError(f"Unknown assay_type: {assay_type!r}")


# ---------------------------------------------------------------------------
# Function 2: Scale CLint to whole liver
# ---------------------------------------------------------------------------
def scale_clint_to_liver(
    clint_invitro: float,
    assay_type: str,
    species: str,
) -> float:
    """Scale in vitro CLint to whole-liver CLint (L/h).

    Microsomes: uL/min/mg -> mL/min (via mppgl * liver_weight / 1000) -> L/h
    Hepatocytes: uL/min/10^6 cells -> mL/min (via cells/g * liver / 1e6 / 1000) -> L/h
    """
    if assay_type == "microsome":
        params = MICROSOME_SCALING[species]
        mppgl = params["mppgl"]
        liver_wt = params["liver_weight_g"]
        # clint_invitro (uL/min/mg) * mppgl (mg/g) * liver_wt (g) = uL/min
        # uL/min -> mL/min: / 1000
        cl_mL_min = clint_invitro * mppgl * liver_wt / 1000.0
        return cl_mL_min * 60.0 / 1000.0  # L/h
    elif assay_type == "hepatocyte":
        params = HEPATOCYTE_SCALING[species]
        cells_per_g = params["cells_per_g_liver"]
        liver_wt = params["liver_weight_g"]
        # clint_invitro (uL/min/10^6 cells) * total_cells/10^6 = uL/min
        total_cells_millions = cells_per_g * liver_wt / 1e6
        cl_uL_min = clint_invitro * total_cells_millions
        return cl_uL_min / 1000.0 * 60.0 / 1000.0  # uL->mL->L, min->h
    else:
        raise ValueError(f"Unknown assay_type: {assay_type!r}")


# ---------------------------------------------------------------------------
# Function 3: Predict metabolic stability
# ---------------------------------------------------------------------------
def predict_metabolic_stability(
    drug_name: str,
    t_half_min: float,
    assay_type: str = "microsome",
    species: str = "human_liver",
    fup: float = 1.0,
    protein_mg_mL: float = 0.5,
    cells_per_mL: float = 1e6,
    q_liver_L_per_h: float = 90.0,
    fu_mic: float = 1.0,
) -> MetabolicStabilityResult:
    """Predict hepatic clearance and stability from in vitro half-life data."""
    warnings: list[str] = []

    # In vitro CLint
    clint_invitro = clint_from_half_life(t_half_min, protein_mg_mL, cells_per_mL, assay_type)

    # Correct for microsomal binding
    clint_corrected = clint_invitro / fu_mic

    # Scale to whole liver
    scaling_species = species.split("_")[0] if assay_type == "hepatocyte" else species
    clint_scaled = scale_clint_to_liver(clint_corrected, assay_type, scaling_species)

    # Well-stirred model
    e = fup * clint_scaled / (q_liver_L_per_h + fup * clint_scaled)
    cl_hepatic = q_liver_L_per_h * e

    # Stability classification
    if t_half_min < 30.0:
        stability_class = "labile"
    elif t_half_min <= 60.0:
        stability_class = "moderate"
    else:
        stability_class = "stable"

    # Warnings
    if e > 0.7:
        warnings.append("High extraction — CL insensitive to protein binding changes")
    if fu_mic < 0.3:
        warnings.append("High microsomal binding correction applied")

    return MetabolicStabilityResult(
        drug_name=drug_name,
        assay_type=assay_type,
        species=species,
        clint_uL_per_min_per_mg=clint_invitro,
        clint_scaled_L_per_h=clint_scaled,
        cl_hepatic_L_per_h=cl_hepatic,
        extraction_ratio=e,
        bioavailability_fh=1.0 - e,
        half_life_in_vitro_min=t_half_min,
        stability_class=stability_class,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Function 4: CLint from substrate depletion data
# ---------------------------------------------------------------------------
def in_vitro_clearance_from_depletion(
    drug_name: str,
    times_min: list[float],
    concentrations: list[float],
    protein_mg_mL: float = 0.5,
) -> dict:
    """Fit log-linear decay to substrate depletion data and extract CLint.

    Returns dict with t_half_min, ke_per_min, r_squared, and
    clint_uL_per_min_per_mg.
    """
    t = np.asarray(times_min, dtype=float)
    c = np.asarray(concentrations, dtype=float)
    ln_c = np.log(c)

    # Linear regression: ln(C) = ln(C0) - ke * t
    n = len(t)
    sum_t = np.sum(t)
    sum_lnc = np.sum(ln_c)
    sum_t_lnc = np.sum(t * ln_c)
    sum_t2 = np.sum(t * t)

    slope = (n * sum_t_lnc - sum_t * sum_lnc) / (n * sum_t2 - sum_t**2)
    intercept = (sum_lnc - slope * sum_t) / n

    ke = -slope
    t_half = 0.693 / ke

    # R-squared
    ss_res = np.sum((ln_c - (intercept + slope * t)) ** 2)
    ss_tot = np.sum((ln_c - np.mean(ln_c)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    clint = clint_from_half_life(t_half, protein_mg_mL=protein_mg_mL)

    return {
        "drug_name": drug_name,
        "t_half_min": float(t_half),
        "ke_per_min": float(ke),
        "r_squared": float(r_squared),
        "clint_uL_per_min_per_mg": float(clint),
    }


__all__ = [
    "MetabolicStabilityResult",
    "predict_metabolic_stability",
    "clint_from_half_life",
    "in_vitro_clearance_from_depletion",
]
