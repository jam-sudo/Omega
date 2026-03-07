"""Intrinsic clearance scaling from in vitro to in vivo hepatic clearance.

Scales microsomal or hepatocyte CLint measurements to in vivo hepatic CL using
standard IVIVE (in vitro to in vivo extrapolation) approaches consistent with
FDA and EMA guidance.

Key references:
- Houston (1994) Biochem Pharmacol 47:1469-1479 (microsomal scaling)
- Iwatsubo et al. (1997) Pharmacol Ther 73:147-171 (well-stirred model)
- Barter et al. (2007) Drug Metab Dispos 35:1841-1849 (MPPGL values)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IVIVEResult:
    """Result of a full IVIVE pipeline calculation.

    Attributes
    ----------
    compound_name : str
        Name of the compound being scaled.
    clint_obs : float
        Observed (measured) intrinsic clearance (µL/min/mg protein).
    clint_corrected : float
        CLint after fu_mic correction (µL/min/mg protein).
    clint_liver_L_per_h : float
        Scaled intrinsic clearance for the whole liver (L/h).
    cl_hepatic_L_per_h : float
        Predicted hepatic clearance via well-stirred model (L/h).
    extraction_ratio : float
        Hepatic extraction ratio E = CL_H / QH (dimensionless, 0-1).
    fu_plasma : float
        Unbound fraction in plasma (dimensionless).
    fu_mic : float
        Unbound fraction in microsomal incubation (dimensionless).
    notes : str
        Descriptive notes about the calculation.
    """

    compound_name: str
    clint_obs: float
    clint_corrected: float
    clint_liver_L_per_h: float
    cl_hepatic_L_per_h: float
    extraction_ratio: float
    fu_plasma: float
    fu_mic: float
    notes: str = field(default="")


# ---------------------------------------------------------------------------
# Individual scaling functions
# ---------------------------------------------------------------------------


def microsomal_clint_scale(
    clint_uL_per_min_per_mg: float,
    protein_conc_mg_per_mL: float = 0.5,
    liver_weight_g: float = 1500.0,
    mppgl: float = 45.0,
) -> float:
    """Scale microsomal CLint to whole-liver intrinsic clearance.

    Applies the standard Houston (1994) microsomal scaling equation:

        CLint_hep (L/h) = CLint_mic (µL/min/mg)
                          * protein_conc (mg/mL)
                          * liver_weight (g)
                          * MPPGL (mg/g)
                          * 60 (min/h)
                          / 1e6 (µL → L)

    The protein_conc term appears because CLint_mic is per mg of added
    microsomal protein, and we divide by the incubation protein concentration
    to get per-mL CLint, then multiply by total protein in the liver.

    Parameters
    ----------
    clint_uL_per_min_per_mg : float
        Observed microsomal CLint in µL/min/mg microsomal protein.
    protein_conc_mg_per_mL : float
        Microsomal protein concentration in incubation (mg/mL).  Default 0.5.
    liver_weight_g : float
        Human liver weight in grams.  Default 1500 g.
    mppgl : float
        Microsomal protein per gram liver (mg/g).  Default 45 mg/g (human).

    Returns
    -------
    float
        Whole-liver intrinsic clearance in L/h.

    Raises
    ------
    ValueError
        If any parameter is non-positive.
    """
    if clint_uL_per_min_per_mg <= 0:
        raise ValueError("clint_uL_per_min_per_mg must be positive.")
    if protein_conc_mg_per_mL <= 0:
        raise ValueError("protein_conc_mg_per_mL must be positive.")
    if liver_weight_g <= 0:
        raise ValueError("liver_weight_g must be positive.")
    if mppgl <= 0:
        raise ValueError("mppgl must be positive.")

    clint_liver = (
        clint_uL_per_min_per_mg * protein_conc_mg_per_mL * liver_weight_g * mppgl * 60.0 / 1e6
    )
    return float(clint_liver)


def hepatocyte_clint_scale(
    clint_uL_per_min_per_1e6cells: float,
    liver_weight_g: float = 1500.0,
    hpgl: float = 120e6,
) -> float:
    """Scale hepatocyte CLint to whole-liver intrinsic clearance.

    Applies the hepatocyte scaling equation:

        CLint_hep (L/h) = CLint_cells (µL/min/10^6 cells)
                          * HPGL (cells/g) / 1e6
                          * liver_weight (g)
                          * 60 (min/h)
                          / 1e6 (µL → L)

    Parameters
    ----------
    clint_uL_per_min_per_1e6cells : float
        Observed hepatocyte CLint in µL/min per 10^6 cells.
    liver_weight_g : float
        Human liver weight in grams.  Default 1500 g.
    hpgl : float
        Hepatocytes per gram liver.  Default 120e6 cells/g (human).

    Returns
    -------
    float
        Whole-liver intrinsic clearance in L/h.

    Raises
    ------
    ValueError
        If any parameter is non-positive.
    """
    if clint_uL_per_min_per_1e6cells <= 0:
        raise ValueError("clint_uL_per_min_per_1e6cells must be positive.")
    if liver_weight_g <= 0:
        raise ValueError("liver_weight_g must be positive.")
    if hpgl <= 0:
        raise ValueError("hpgl must be positive.")

    # hpgl is in cells/g; divide by 1e6 to convert to 10^6-cells/g
    clint_liver = clint_uL_per_min_per_1e6cells * (hpgl / 1e6) * liver_weight_g * 60.0 / 1e6
    return float(clint_liver)


def fu_mic_correction(clint_obs: float, fu_mic: float) -> float:
    """Correct observed CLint for nonspecific microsomal binding.

    Nonspecific binding to microsomal protein means that only the free
    (unbound) fraction of drug is available for metabolism.  The true
    free-drug CLint is therefore higher:

        CLint_free = CLint_obs / fu_mic

    Parameters
    ----------
    clint_obs : float
        Observed intrinsic clearance (any consistent units).
    fu_mic : float
        Unbound fraction in microsomal incubation (0 < fu_mic <= 1).

    Returns
    -------
    float
        fu_mic-corrected intrinsic clearance (same units as clint_obs).

    Raises
    ------
    ValueError
        If clint_obs <= 0 or fu_mic not in (0, 1].
    """
    if clint_obs <= 0:
        raise ValueError("clint_obs must be positive.")
    if not (0 < fu_mic <= 1.0):
        raise ValueError("fu_mic must be in (0, 1].")

    return float(clint_obs / fu_mic)


def well_stirred_cl(
    clint_L_per_h: float,
    fu_blood: float,
    qh_L_per_h: float = 90.0,
) -> float:
    """Predict hepatic clearance using the well-stirred (venous equilibrium) model.

    The well-stirred model assumes the liver is a single well-mixed compartment:

        CL_H = QH * fu * CLint / (QH + fu * CLint)

    Parameters
    ----------
    clint_L_per_h : float
        Intrinsic hepatic clearance (L/h).
    fu_blood : float
        Unbound fraction in blood (0 < fu_blood <= 1).  Use fu_plasma if
        blood-to-plasma ratio is assumed to be 1.
    qh_L_per_h : float
        Hepatic blood flow (L/h).  Default 90 L/h (human).

    Returns
    -------
    float
        Predicted hepatic clearance (L/h).

    Raises
    ------
    ValueError
        If clint_L_per_h <= 0, fu_blood not in (0, 1], or qh_L_per_h <= 0.
    """
    if clint_L_per_h <= 0:
        raise ValueError("clint_L_per_h must be positive.")
    if not (0 < fu_blood <= 1.0):
        raise ValueError("fu_blood must be in (0, 1].")
    if qh_L_per_h <= 0:
        raise ValueError("qh_L_per_h must be positive.")

    numerator = qh_L_per_h * fu_blood * clint_L_per_h
    denominator = qh_L_per_h + fu_blood * clint_L_per_h
    return float(numerator / denominator)


def intrinsic_clearance_from_t_half(
    t_half_microsomal_min: float,
    protein_mg_per_mL: float = 0.5,
    incubation_volume_mL: float = 0.5,
) -> float:
    """Estimate CLint from a microsomal half-life experiment.

    In a microsomal half-life depletion assay the substrate disappearance
    follows first-order kinetics:

        ke = 0.693 / t_half
        CLint = ke * V_incubation / protein_amount

    where protein_amount = protein_mg_per_mL * incubation_volume_mL.

    Parameters
    ----------
    t_half_microsomal_min : float
        Microsomal half-life in minutes (> 0).
    protein_mg_per_mL : float
        Microsomal protein concentration (mg/mL).  Default 0.5 mg/mL.
    incubation_volume_mL : float
        Total incubation volume (mL).  Default 0.5 mL.

    Returns
    -------
    float
        Intrinsic clearance in µL/min/mg microsomal protein.

    Raises
    ------
    ValueError
        If any parameter is non-positive.
    """
    if t_half_microsomal_min <= 0:
        raise ValueError("t_half_microsomal_min must be positive.")
    if protein_mg_per_mL <= 0:
        raise ValueError("protein_mg_per_mL must be positive.")
    if incubation_volume_mL <= 0:
        raise ValueError("incubation_volume_mL must be positive.")

    ke = 0.693 / t_half_microsomal_min  # min^-1
    protein_amount_mg = protein_mg_per_mL * incubation_volume_mL  # mg
    # CLint in mL/min/mg → convert to µL/min/mg (* 1000)
    clint_mL = ke * incubation_volume_mL / protein_amount_mg
    return float(clint_mL * 1000.0)


def ivive_pipeline(
    compound_name: str,
    clint_uL_per_min_per_mg: float,
    fu_plasma: float,
    fu_mic: float = 1.0,
    protein_conc: float = 0.5,
    liver_weight_g: float = 1500.0,
    mppgl: float = 45.0,
    qh_L_per_h: float = 90.0,
) -> IVIVEResult:
    """Full IVIVE pipeline: microsomal CLint to predicted hepatic clearance.

    Steps:
    1. fu_mic correction: CLint_corrected = CLint_obs / fu_mic
    2. Scale to liver:   CLint_liver = microsomal_clint_scale(CLint_corrected)
    3. Well-stirred CL:  CL_H = well_stirred_cl(CLint_liver, fu_plasma)

    Parameters
    ----------
    compound_name : str
        Identifier for the compound.
    clint_uL_per_min_per_mg : float
        Observed microsomal CLint (µL/min/mg protein).
    fu_plasma : float
        Unbound fraction in plasma (0 < fu_plasma <= 1).
    fu_mic : float
        Unbound fraction in microsomal incubation (0 < fu_mic <= 1).
        Default 1.0 (no correction applied).
    protein_conc : float
        Microsomal protein concentration (mg/mL).  Default 0.5.
    liver_weight_g : float
        Liver weight (g).  Default 1500.
    mppgl : float
        Microsomal protein per gram liver (mg/g).  Default 45.
    qh_L_per_h : float
        Hepatic blood flow (L/h).  Default 90.

    Returns
    -------
    IVIVEResult
        Full results including intermediate values and extraction ratio.

    Raises
    ------
    ValueError
        If inputs fail validation.
    """
    # Validate inputs
    if clint_uL_per_min_per_mg <= 0:
        raise ValueError("clint_uL_per_min_per_mg must be positive.")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1].")
    if not (0 < fu_mic <= 1.0):
        raise ValueError("fu_mic must be in (0, 1].")
    if protein_conc <= 0:
        raise ValueError("protein_conc must be positive.")
    if liver_weight_g <= 0:
        raise ValueError("liver_weight_g must be positive.")
    if qh_L_per_h <= 0:
        raise ValueError("qh_L_per_h must be positive.")

    # Step 1: fu_mic correction
    clint_corrected = fu_mic_correction(clint_uL_per_min_per_mg, fu_mic)

    # Step 2: Scale to liver
    clint_liver = microsomal_clint_scale(
        clint_corrected,
        protein_conc_mg_per_mL=protein_conc,
        liver_weight_g=liver_weight_g,
        mppgl=mppgl,
    )

    # Step 3: Well-stirred hepatic CL
    cl_h = well_stirred_cl(clint_liver, fu_blood=fu_plasma, qh_L_per_h=qh_L_per_h)

    extraction_ratio = cl_h / qh_L_per_h

    # Build notes
    er_class = (
        "high extraction (E>0.7)"
        if extraction_ratio > 0.7
        else "intermediate extraction (0.3<E<0.7)"
        if extraction_ratio > 0.3
        else "low extraction (E<0.3)"
    )
    notes = (
        f"fu_mic correction factor: {1.0 / fu_mic:.2f}x; "
        f"CLint_liver={clint_liver:.4f} L/h; {er_class}"
    )

    return IVIVEResult(
        compound_name=compound_name,
        clint_obs=clint_uL_per_min_per_mg,
        clint_corrected=clint_corrected,
        clint_liver_L_per_h=clint_liver,
        cl_hepatic_L_per_h=cl_h,
        extraction_ratio=extraction_ratio,
        fu_plasma=fu_plasma,
        fu_mic=fu_mic,
        notes=notes,
    )


__all__ = [
    "IVIVEResult",
    "microsomal_clint_scale",
    "hepatocyte_clint_scale",
    "fu_mic_correction",
    "well_stirred_cl",
    "intrinsic_clearance_from_t_half",
    "ivive_pipeline",
]
