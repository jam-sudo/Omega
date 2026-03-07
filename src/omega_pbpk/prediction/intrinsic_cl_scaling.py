"""In vitro to in vivo CLint scaling — microsomal, hepatocyte, and S9 assays.

Scales in vitro intrinsic clearance (CLint) to hepatic clearance (CL_H) using
standard IVIVE approaches: well-stirred, parallel-tube, and dispersion models.

References
----------
- Houston JB. Biochem Pharmacol 1994; microsomal CLint scaling
- Obach RS. Drug Metab Dispos 1999; human liver microsomal scaling
- Davies B & Morris T. Pharm Res 1993; physiological parameters
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Species physiological parameters
# ---------------------------------------------------------------------------

# Scaling factors: mg protein/kg BW (microsomal) or 10^6 cells/kg BW (hepatocyte)
_SCALING_FACTORS: dict[str, dict[str, float]] = {
    "microsomal": {
        "human": 963.0,
        "rat": 2100.0,
        "mouse": 3960.0,
        "dog": 1240.0,
    },
    "hepatocyte": {
        "human": 2119.0,
        "rat": 4680.0,
        "mouse": 11880.0,
        "dog": 6665.0,
    },
    "s9": {
        "human": 482.0,
        "rat": 1050.0,
        "mouse": 1980.0,  # ~50% of microsomal
        "dog": 620.0,
    },
}

# Hepatic blood flow (mL/min/kg)
_Q_HEPATIC: dict[str, float] = {
    "human": 20.7,
    "rat": 70.0,
    "mouse": 90.0,
    "dog": 30.0,
}

_SUPPORTED_SPECIES = {"human", "rat", "mouse", "dog"}
_SUPPORTED_ASSAYS = {"microsomal", "hepatocyte", "s9"}
_SUPPORTED_METHODS = {"well_stirred", "parallel_tube", "dispersion"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CLintScalingResult:
    """Result of CLint scaling from in vitro to in vivo.

    Attributes
    ----------
    drug_name : Drug identifier.
    assay_type : "microsomal", "hepatocyte", or "s9".
    species : Target species.
    clint_in_vitro : In vitro CLint (mL/min/mg protein or mL/min/10^6 cells).
    clint_units : Unit string for the in vitro CLint.
    scaling_factor : ISEF scaling factor (mg protein/kg BW or cells/kg BW).
    clint_in_vivo_mL_per_min_per_kg : Scaled CLint corrected for protein binding.
    cl_hepatic_mL_per_min_per_kg : Hepatic CL (mL/min/kg).
    cl_hepatic_L_per_h_per_kg : Hepatic CL (L/h/kg).
    extraction_ratio : CL_H / Q_H (clamped to [0, 0.9999]).
    hepatic_bioavailability : 1 - extraction_ratio.
    fu_plasma : Unbound fraction in plasma.
    fu_mic_or_hep : Unbound fraction in the in vitro assay.
    predicted_f_oral : Estimated oral bioavailability (hepatic_ba * 0.8).
    prediction_method : "well_stirred", "parallel_tube", or "dispersion".
    notes : Summary string.
    """

    drug_name: str
    assay_type: str
    species: str
    clint_in_vitro: float
    clint_units: str
    scaling_factor: float
    clint_in_vivo_mL_per_min_per_kg: float
    cl_hepatic_mL_per_min_per_kg: float
    cl_hepatic_L_per_h_per_kg: float
    extraction_ratio: float
    hepatic_bioavailability: float
    fu_plasma: float
    fu_mic_or_hep: float
    predicted_f_oral: float
    prediction_method: str
    notes: str


# ---------------------------------------------------------------------------
# Core scaling function
# ---------------------------------------------------------------------------


def scale_clint(
    drug_name: str,
    assay_type: str = "microsomal",
    clint_in_vitro: float = 1.0,
    species: str = "human",
    fu_plasma: float = 1.0,
    fu_mic_or_hep: float = 1.0,
    prediction_method: str = "well_stirred",
) -> CLintScalingResult:
    """Scale in vitro CLint to in vivo hepatic clearance.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    assay_type : str
        "microsomal" (mL/min/mg protein), "hepatocyte" (mL/min/10^6 cells),
        or "s9" (mL/min/mg protein).
    clint_in_vitro : float
        In vitro CLint value. Must be > 0.
    species : str
        "human", "rat", "mouse", or "dog".
    fu_plasma : float
        Unbound fraction in plasma (0, 1]. Must be in (0, 1].
    fu_mic_or_hep : float
        Unbound fraction in the in vitro assay (0, 1]. Must be in (0, 1].
    prediction_method : str
        Hepatic model: "well_stirred", "parallel_tube", or "dispersion".

    Returns
    -------
    CLintScalingResult
    """
    # --- Validation ---
    if clint_in_vitro <= 0.0:
        raise ValueError("clint_in_vitro must be > 0")
    if fu_plasma <= 0.0 or fu_plasma > 1.0:
        raise ValueError("fu_plasma must be in (0, 1]")
    if fu_mic_or_hep <= 0.0 or fu_mic_or_hep > 1.0:
        raise ValueError("fu_mic_or_hep must be in (0, 1]")
    if species.lower() not in _SUPPORTED_SPECIES:
        raise ValueError(f"Unknown species '{species}'. Choose from {sorted(_SUPPORTED_SPECIES)}")
    if assay_type.lower() not in _SUPPORTED_ASSAYS:
        raise ValueError(
            f"Unknown assay_type '{assay_type}'. Choose from {sorted(_SUPPORTED_ASSAYS)}"
        )

    sp = species.lower()
    at = assay_type.lower()

    # --- Scaling factor ---
    sf = _SCALING_FACTORS[at][sp]

    # CLint unit labels
    if at == "hepatocyte":
        clint_units = "mL/min/10^6 cells"
    else:
        clint_units = "mL/min/mg protein"

    # --- In vivo CLint (mL/min/kg) ---
    # CLint_in_vivo = clint_in_vitro * scaling_factor * (fu_plasma / fu_mic_or_hep)
    clint_iv = clint_in_vitro * sf * (fu_plasma / fu_mic_or_hep)

    # --- Hepatic blood flow ---
    q_h = _Q_HEPATIC[sp]

    # --- Hepatic CL model ---
    pm = prediction_method.lower()
    fu_clint = fu_plasma * clint_iv

    if pm == "well_stirred":
        cl_h = q_h * fu_clint / (q_h + fu_clint)
    elif pm == "parallel_tube":
        exponent = -fu_clint / q_h
        cl_h = q_h * (1.0 - math.exp(exponent))
    else:
        # Dispersion: use well-stirred as simplified approximation
        cl_h = q_h * fu_clint / (q_h + fu_clint)

    # Clamp extraction ratio to [0, 0.9999]
    er = min(cl_h / q_h, 0.9999)
    er = max(er, 0.0)
    cl_h = er * q_h

    hepatic_ba = 1.0 - er
    cl_h_l_per_h_per_kg = cl_h / 1000.0 * 60.0
    predicted_f_oral = hepatic_ba * 0.8

    notes = (
        f"method={prediction_method}; CLint_iv={clint_iv:.3f} mL/min/kg; "
        f"Q_H={q_h} mL/min/kg; ER={er:.3f}; FH={hepatic_ba:.3f}"
    )

    return CLintScalingResult(
        drug_name=drug_name,
        assay_type=at,
        species=sp,
        clint_in_vitro=clint_in_vitro,
        clint_units=clint_units,
        scaling_factor=sf,
        clint_in_vivo_mL_per_min_per_kg=clint_iv,
        cl_hepatic_mL_per_min_per_kg=cl_h,
        cl_hepatic_L_per_h_per_kg=cl_h_l_per_h_per_kg,
        extraction_ratio=er,
        hepatic_bioavailability=hepatic_ba,
        fu_plasma=fu_plasma,
        fu_mic_or_hep=fu_mic_or_hep,
        predicted_f_oral=predicted_f_oral,
        prediction_method=prediction_method,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Compare models
# ---------------------------------------------------------------------------


def compare_models(
    drug_name: str,
    assay_type: str = "microsomal",
    clint_in_vitro: float = 1.0,
    species: str = "human",
    fu_plasma: float = 1.0,
    fu_mic: float = 1.0,
) -> list[CLintScalingResult]:
    """Run all three prediction models and return their results.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    assay_type : str
        "microsomal", "hepatocyte", or "s9".
    clint_in_vitro : float
        In vitro CLint value.
    species : str
        Target species.
    fu_plasma : float
        Unbound fraction in plasma.
    fu_mic : float
        Unbound fraction in the in vitro assay.

    Returns
    -------
    list[CLintScalingResult]
        Three results: well_stirred, parallel_tube, dispersion.
    """
    return [
        scale_clint(
            drug_name=drug_name,
            assay_type=assay_type,
            clint_in_vitro=clint_in_vitro,
            species=species,
            fu_plasma=fu_plasma,
            fu_mic_or_hep=fu_mic,
            prediction_method=method,
        )
        for method in ["well_stirred", "parallel_tube", "dispersion"]
    ]


__all__ = [
    "CLintScalingResult",
    "scale_clint",
    "compare_models",
]
