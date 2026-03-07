"""Drug intrinsic clearance scaling: in vitro to in vivo (IVIVE).

Scales microsomal or hepatocyte CLint to hepatic clearance using the
well-stirred liver model (WSLM).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClintScalingResult:
    """Result of intrinsic clearance scaling to in vivo hepatic clearance."""

    drug_name: str
    clint_in_vitro: float
    assay_type: str
    clint_liver_L_per_h: float
    cl_hepatic_L_per_h: float
    extraction_ratio: float
    extraction_class: str
    fup_sensitivity: str
    bioavailability_first_pass: float
    notes: str


def _scale_microsomal(
    clint_mic: float,
    mppgl: float,
    liver_weight_g: float,
) -> float:
    """Scale microsomal CLint to whole-liver CLint in L/h.

    Parameters
    ----------
    clint_mic:
        In vitro CLint in µL/min/mg protein.
    mppgl:
        Microsomal protein per gram liver in mg/g (default 45).
    liver_weight_g:
        Liver weight in grams (default 1500).

    Returns
    -------
    CLint_liver in L/h.
    """
    # CLint_liver (L/h) = CLint_mic (µL/min/mg) × MPPGL (mg/g) × liver_weight (g) × 60 min/h / 1e9
    # 1e9 converts µL to L (µL/min/mg * mg/g * g = µL/min; × 60 = µL/h; / 1e6 = L/h;
    #   but actually: µL/min/mg × mg_per_g × g = µL/min, then × 60 = µL/h, / 1e6 = L/h)
    # Combined: × 60 / 1e6 = × 6e-5
    return clint_mic * mppgl * liver_weight_g * 60.0 / 1.0e9


def _scale_hepatocyte(
    clint_hep: float,
    hpgl: float,
    liver_weight_g: float,
) -> float:
    """Scale hepatocyte CLint to whole-liver CLint in L/h.

    Parameters
    ----------
    clint_hep:
        In vitro CLint in µL/min/10^6 cells.
    hpgl:
        Hepatocytes per gram liver (cells/g; default 1.1e8).
    liver_weight_g:
        Liver weight in grams (default 1500).

    Returns
    -------
    CLint_liver in L/h.
    """
    # CLint_liver (L/h) = CLint_hep (µL/min/1e6 cells) × HPGL (cells/g) × liver_weight (g)
    #                     × 60 min/h / 1e12
    # 1e12 accounts for µL→L (1e6) and 1e6 cells denominator (1e6)
    return clint_hep * hpgl * liver_weight_g * 60.0 / 1.0e12


def _well_stirred_model(
    clint_liver_L_per_h: float,
    fup: float,
    q_liver_L_per_h: float,
) -> float:
    """Apply well-stirred liver model to get hepatic clearance.

    CL_hepatic = Q_liver × fup × CLint_liver / (Q_liver + fup × CLint_liver)

    Returns CL_hepatic in L/h.
    """
    fup_clint = fup * clint_liver_L_per_h
    return q_liver_L_per_h * fup_clint / (q_liver_L_per_h + fup_clint)


def _extraction_class(er: float) -> str:
    """Classify extraction ratio."""
    if er < 0.3:
        return "low"
    elif er <= 0.7:
        return "intermediate"
    else:
        return "high"


def _fup_sensitivity(er: float) -> str:
    """Determine sensitivity of hepatic CL to plasma protein binding."""
    if er < 0.3:
        return "high sensitivity"
    elif er > 0.7:
        return "low sensitivity"
    else:
        return "moderate"


def scale_clint_to_human(
    drug_name: str,
    clint_in_vitro: float,
    assay_type: str = "microsomal",
    fup: float = 0.5,
    q_liver_L_per_h: float = 90.0,
    mppgl: float = 45.0,
    hpgl: float = 1.1e8,
    liver_weight_g: float = 1500.0,
) -> ClintScalingResult:
    """Scale in vitro CLint to in vivo hepatic clearance using WSLM.

    Parameters
    ----------
    drug_name:
        Drug identifier string.
    clint_in_vitro:
        In vitro CLint value. Units depend on assay_type:
        - microsomal: µL/min/mg protein
        - hepatocyte: µL/min/10^6 cells
    assay_type:
        "microsomal" or "hepatocyte".
    fup:
        Fraction unbound in plasma (0 < fup <= 1).
    q_liver_L_per_h:
        Hepatic blood flow in L/h (default 90).
    mppgl:
        Microsomal protein per gram liver in mg/g (for microsomal assay).
    hpgl:
        Hepatocytes per gram liver (cells/g; for hepatocyte assay).
    liver_weight_g:
        Liver weight in grams (default 1500).

    Returns
    -------
    ClintScalingResult
    """
    # Validation
    if clint_in_vitro <= 0:
        raise ValueError(f"clint_in_vitro must be > 0, got {clint_in_vitro}")
    if fup <= 0:
        raise ValueError(f"fup must be > 0, got {fup}")
    if q_liver_L_per_h <= 0:
        raise ValueError(f"q_liver_L_per_h must be > 0, got {q_liver_L_per_h}")

    assay_type = assay_type.lower()
    if assay_type not in ("microsomal", "hepatocyte"):
        raise ValueError(f"assay_type must be 'microsomal' or 'hepatocyte', got {assay_type}")

    # Scale in vitro to whole-liver CLint
    if assay_type == "microsomal":
        clint_liver = _scale_microsomal(clint_in_vitro, mppgl, liver_weight_g)
    else:
        clint_liver = _scale_hepatocyte(clint_in_vitro, hpgl, liver_weight_g)

    # Well-stirred model
    cl_hepatic = _well_stirred_model(clint_liver, fup, q_liver_L_per_h)

    # Derived quantities
    er = cl_hepatic / q_liver_L_per_h
    er = min(er, 1.0)  # cap at 1 for numerical safety
    ext_class = _extraction_class(er)
    fup_sens = _fup_sensitivity(er)
    fh = 1.0 - er  # first-pass bioavailability fraction

    # Build notes
    notes_parts = [
        f"Assay: {assay_type}",
        f"CLint_liver = {clint_liver:.4f} L/h",
        f"ER = {er:.3f} ({ext_class} extraction)",
        f"FUP sensitivity: {fup_sens}",
        f"First-pass F = {fh:.3f}",
    ]
    if ext_class == "high":
        notes_parts.append("Flow-limited: CL_hepatic ≈ Q_liver; fup changes have minimal effect.")
    elif ext_class == "low":
        notes_parts.append(
            "Binding-limited: CL_hepatic ≈ fup × CLint; protein binding is critical."
        )
    notes = "; ".join(notes_parts)

    return ClintScalingResult(
        drug_name=drug_name,
        clint_in_vitro=clint_in_vitro,
        assay_type=assay_type,
        clint_liver_L_per_h=clint_liver,
        cl_hepatic_L_per_h=cl_hepatic,
        extraction_ratio=er,
        extraction_class=ext_class,
        fup_sensitivity=fup_sens,
        bioavailability_first_pass=fh,
        notes=notes,
    )


def compare_assay_types(
    drug_name: str,
    clint_mic: float,
    clint_hep: float,
    fup: float = 0.5,
    q_liver_L_per_h: float = 90.0,
    mppgl: float = 45.0,
    hpgl: float = 1.1e8,
    liver_weight_g: float = 1500.0,
) -> dict:
    """Compare microsomal and hepatocyte CLint scaling for the same drug.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    clint_mic:
        Microsomal CLint in µL/min/mg protein.
    clint_hep:
        Hepatocyte CLint in µL/min/10^6 cells.
    fup:
        Fraction unbound in plasma.
    q_liver_L_per_h:
        Hepatic blood flow in L/h.
    mppgl:
        Microsomal protein per gram liver.
    hpgl:
        Hepatocytes per gram liver.
    liver_weight_g:
        Liver weight in grams.

    Returns
    -------
    dict with keys:
        microsomal_result (ClintScalingResult)
        hepatocyte_result (ClintScalingResult)
        cl_ratio (float): cl_hepatic_mic / cl_hepatic_hep
    """
    mic_result = scale_clint_to_human(
        drug_name=drug_name,
        clint_in_vitro=clint_mic,
        assay_type="microsomal",
        fup=fup,
        q_liver_L_per_h=q_liver_L_per_h,
        mppgl=mppgl,
        hpgl=hpgl,
        liver_weight_g=liver_weight_g,
    )
    hep_result = scale_clint_to_human(
        drug_name=drug_name,
        clint_in_vitro=clint_hep,
        assay_type="hepatocyte",
        fup=fup,
        q_liver_L_per_h=q_liver_L_per_h,
        mppgl=mppgl,
        hpgl=hpgl,
        liver_weight_g=liver_weight_g,
    )

    cl_ratio = (
        mic_result.cl_hepatic_L_per_h / hep_result.cl_hepatic_L_per_h
        if hep_result.cl_hepatic_L_per_h > 0
        else float("inf")
    )

    return {
        "microsomal_result": mic_result,
        "hepatocyte_result": hep_result,
        "cl_ratio": cl_ratio,
    }
