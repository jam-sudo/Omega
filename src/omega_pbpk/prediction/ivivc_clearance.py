"""Phase 386: In Vitro-In Vivo Correlation for Clearance.

Predict in vivo hepatic clearance from in vitro microsomal intrinsic clearance
using well-stirred, parallel-tube, and dispersion models.

References:
    Houston JB, Carlile DJ. Drug Metab Rev 1997
    Obach RS. Drug Metab Dispos 1999
    Poulin P, Theil F-P. J Pharm Sci 2000
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Species parameters
# ---------------------------------------------------------------------------

_SPECIES_PARAMS: dict[str, dict] = {
    "human": {
        "liver_weight_g_per_kg": 25.7,
        "mppgl": 45.0,
        "qh_L_per_h_per_kg": 1.26,
        "fu_mic_default": 1.0,
    },
    "rat": {
        "liver_weight_g_per_kg": 40.0,
        "mppgl": 60.0,
        "qh_L_per_h_per_kg": 4.2,
        "fu_mic_default": 1.0,
    },
    "dog": {
        "liver_weight_g_per_kg": 32.0,
        "mppgl": 55.0,
        "qh_L_per_h_per_kg": 2.1,
        "fu_mic_default": 1.0,
    },
    "monkey": {
        "liver_weight_g_per_kg": 26.8,
        "mppgl": 52.0,
        "qh_L_per_h_per_kg": 1.7,
        "fu_mic_default": 1.0,
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IVIVCClearanceResult:
    """Result of IVIVC clearance prediction."""

    drug_name: str
    species: str
    cl_int_in_vitro_uL_per_min_per_mg: float  # microsomal CLint input
    cl_int_scaled_mL_per_min_per_kg: float  # scaled to in vivo per kg
    cl_hepatic_well_stirred_L_per_h_per_kg: float
    cl_hepatic_parallel_tube_L_per_h_per_kg: float
    cl_hepatic_dispersion_L_per_h_per_kg: float  # dispersion model (intermediate)
    extraction_ratio_well_stirred: float  # 0-1
    fu_plasma: float
    fu_mic: float  # fraction unbound in microsomes
    microsomal_protein_mg_per_mL: float
    hepatic_blood_flow_L_per_h_per_kg: float
    scaling_factor_used: float  # MPPGL * liver_weight_factor
    bioavailability_fg: float  # gut wall bioavailability (1 for IV/non-oral)
    predicted_f_hepatic: float  # first-pass extraction (1 - ER)
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scale_cl_int(
    cl_int_uL_per_min_per_mg: float,
    fu_plasma: float,
    fu_mic: float,
    mppgl: float,
    liver_weight_g_per_kg: float,
) -> tuple[float, float]:
    """Scale in vitro CLint to in vivo CLint (mL/min/kg).

    Returns
    -------
    (cl_int_scaled_mL_per_min_per_kg, scaling_factor)
    """
    # scaling_factor = mg microsomal protein per kg body weight
    scaling_factor = mppgl * liver_weight_g_per_kg
    # fu correction: bound fraction in microsomes reduces effective CLint
    fu_ratio = fu_plasma / fu_mic if fu_mic > 0.0 else fu_plasma
    # uL/min/mg_prot * mg_prot/kg_bw / 1000 = mL/min/kg_bw
    cl_int_scaled = cl_int_uL_per_min_per_mg * fu_ratio * scaling_factor / 1000.0
    return cl_int_scaled, scaling_factor


def _well_stirred(
    cl_int_scaled_mL_per_min_per_kg: float,
    qh_mL_per_min_per_kg: float,
) -> float:
    """Well-stirred (venous equilibration) hepatic CL model.

    Returns CLh in mL/min/kg.
    """
    return (
        qh_mL_per_min_per_kg
        * cl_int_scaled_mL_per_min_per_kg
        / (qh_mL_per_min_per_kg + cl_int_scaled_mL_per_min_per_kg)
    )


def _parallel_tube(
    cl_int_scaled_mL_per_min_per_kg: float,
    qh_mL_per_min_per_kg: float,
) -> float:
    """Parallel-tube (undistributed sinusoidal) hepatic CL model.

    Returns CLh in mL/min/kg.
    """
    ratio = (
        cl_int_scaled_mL_per_min_per_kg / qh_mL_per_min_per_kg
        if qh_mL_per_min_per_kg > 0.0
        else 0.0
    )
    return qh_mL_per_min_per_kg * (1.0 - math.exp(-ratio))


def _mL_per_min_to_L_per_h(value_mL_per_min_per_kg: float) -> float:
    """Convert mL/min/kg to L/h/kg."""
    return value_mL_per_min_per_kg * 60.0 / 1000.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_clearance_ivivc(
    drug_name: str,
    cl_int_uL_per_min_per_mg_mic: float,
    fu_plasma: float,
    fu_mic: float | None = None,
    species: str = "human",
    microsomal_protein_mg_per_mL: float = 0.5,
) -> IVIVCClearanceResult:
    """Predict in vivo hepatic clearance from in vitro microsomal CLint.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    cl_int_uL_per_min_per_mg_mic:
        In vitro microsomal intrinsic clearance (uL/min/mg protein).
    fu_plasma:
        Fraction unbound in plasma (0 < fu_plasma <= 1).
    fu_mic:
        Fraction unbound in microsomes (0 < fu_mic <= 1).
        If None, uses species default (1.0 for all species).
    species:
        Species identifier: "human", "rat", "dog", or "monkey".
    microsomal_protein_mg_per_mL:
        Microsomal protein concentration in the assay (mg/mL).

    Returns
    -------
    IVIVCClearanceResult
    """
    # Validation
    if cl_int_uL_per_min_per_mg_mic <= 0.0:
        raise ValueError(
            f"cl_int_uL_per_min_per_mg_mic must be > 0, got {cl_int_uL_per_min_per_mg_mic}"
        )
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if species not in _SPECIES_PARAMS:
        raise ValueError(f"Unknown species '{species}'. Choose from {list(_SPECIES_PARAMS)}")
    if microsomal_protein_mg_per_mL <= 0.0:
        raise ValueError(
            f"microsomal_protein_mg_per_mL must be > 0, got {microsomal_protein_mg_per_mL}"
        )

    params = _SPECIES_PARAMS[species]

    # Resolve fu_mic
    if fu_mic is None:
        fu_mic_used = params["fu_mic_default"]
    else:
        if not (0.0 < fu_mic <= 1.0):
            raise ValueError(f"fu_mic must be in (0, 1], got {fu_mic}")
        fu_mic_used = fu_mic

    mppgl: float = params["mppgl"]
    liver_weight_g_per_kg: float = params["liver_weight_g_per_kg"]
    qh_L_per_h_per_kg: float = params["qh_L_per_h_per_kg"]

    # Scale CLint
    cl_int_scaled, scaling_factor = _scale_cl_int(
        cl_int_uL_per_min_per_mg_mic,
        fu_plasma,
        fu_mic_used,
        mppgl,
        liver_weight_g_per_kg,
    )

    # Hepatic blood flow in mL/min/kg
    qh_mL_per_min_per_kg = qh_L_per_h_per_kg * 1000.0 / 60.0

    # Three hepatic models (mL/min/kg)
    clh_ws_mL = _well_stirred(cl_int_scaled, qh_mL_per_min_per_kg)
    clh_pt_mL = _parallel_tube(cl_int_scaled, qh_mL_per_min_per_kg)

    # Dispersion model: approximate as average of WS and PT
    clh_disp_mL = 0.5 * (clh_ws_mL + clh_pt_mL)

    # Convert to L/h/kg
    clh_ws = _mL_per_min_to_L_per_h(clh_ws_mL)
    clh_pt = _mL_per_min_to_L_per_h(clh_pt_mL)
    clh_disp = _mL_per_min_to_L_per_h(clh_disp_mL)

    # Extraction ratio (well-stirred)
    extraction_ratio = clh_ws / qh_L_per_h_per_kg if qh_L_per_h_per_kg > 0.0 else 0.0
    extraction_ratio = min(extraction_ratio, 1.0)

    # First-pass hepatic bioavailability
    predicted_f_hepatic = 1.0 - extraction_ratio

    # Gut wall bioavailability (not modelled here — 1.0 placeholder)
    bioavailability_fg = 1.0

    notes = (
        f"Scaling: MPPGL={mppgl:.0f} mg/g * liver_wt={liver_weight_g_per_kg:.1f} g/kg"
        f" = {scaling_factor:.0f} mg_prot/kg; "
        f"fu correction: fu_p/fu_mic={fu_plasma:.3f}/{fu_mic_used:.3f}={fu_plasma / fu_mic_used:.3f}; "
        f"WS ER={extraction_ratio:.3f}"
    )

    return IVIVCClearanceResult(
        drug_name=drug_name,
        species=species,
        cl_int_in_vitro_uL_per_min_per_mg=cl_int_uL_per_min_per_mg_mic,
        cl_int_scaled_mL_per_min_per_kg=cl_int_scaled,
        cl_hepatic_well_stirred_L_per_h_per_kg=clh_ws,
        cl_hepatic_parallel_tube_L_per_h_per_kg=clh_pt,
        cl_hepatic_dispersion_L_per_h_per_kg=clh_disp,
        extraction_ratio_well_stirred=extraction_ratio,
        fu_plasma=fu_plasma,
        fu_mic=fu_mic_used,
        microsomal_protein_mg_per_mL=microsomal_protein_mg_per_mL,
        hepatic_blood_flow_L_per_h_per_kg=qh_L_per_h_per_kg,
        scaling_factor_used=scaling_factor,
        bioavailability_fg=bioavailability_fg,
        predicted_f_hepatic=predicted_f_hepatic,
        notes=notes,
    )


def compare_species(
    drug_name: str,
    cl_int_uL_per_min_per_mg_mic: float,
    fu_plasma: float,
) -> list[IVIVCClearanceResult]:
    """Compare predicted CLh across all 4 species.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    cl_int_uL_per_min_per_mg_mic:
        In vitro microsomal CLint (uL/min/mg protein).
    fu_plasma:
        Fraction unbound in plasma (0 < fu_plasma <= 1).

    Returns
    -------
    list[IVIVCClearanceResult]
        Sorted by cl_hepatic_well_stirred_L_per_h_per_kg descending.
    """
    results = [
        predict_clearance_ivivc(drug_name, cl_int_uL_per_min_per_mg_mic, fu_plasma, species=sp)
        for sp in _SPECIES_PARAMS
    ]
    results.sort(key=lambda r: r.cl_hepatic_well_stirred_L_per_h_per_kg, reverse=True)
    return results


__all__ = [
    "IVIVCClearanceResult",
    "predict_clearance_ivivc",
    "compare_species",
]
