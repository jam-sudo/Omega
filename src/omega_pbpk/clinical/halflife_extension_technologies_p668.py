"""Phase 668 - Drug Half-life Extension Technologies.

Models PEGylation, Fc fusion, albumin binding, nanoparticle encapsulation,
and HES conjugation for extending drug half-life.
"""

import math
from dataclasses import dataclass

VALID_TECHNOLOGIES = {
    "PEGylation",
    "Fc_fusion",
    "albumin_binding",
    "nanoparticle",
    "HES_conjugation",
}

COMMON_INTERVALS = [6, 12, 24, 48, 168]


@dataclass(frozen=True)
class HalflifeExtensionResult:
    """Result of half-life extension technology prediction."""

    drug_name: str
    base_t_half_h: float
    technology: str
    extended_t_half_h: float
    extension_factor: float
    mechanism: str
    new_cl_L_per_h: float
    new_vd_L: float
    base_cl_L_per_h: float
    base_vd_L: float
    dosing_interval_recommendation_h: int
    auc_ratio: float
    peak_trough_improvement: float
    notes: str


def _get_technology_factors(technology: str, mw_kDa: float):
    """Return (cl_factor, vd_factor, mechanism) for a given technology."""
    if technology == "PEGylation":
        cl_factor = 1.0 / (1.0 + 0.5 * min(mw_kDa, 100.0) / 10.0)
        vd_factor = 1.0 + 0.1 * min(mw_kDa, 100.0) / 10.0
        mechanism = "reduced_renal_filtration_and_proteolysis"
    elif technology == "Fc_fusion":
        cl_factor = 0.05
        vd_factor = 1.5
        mechanism = "FcRn_recycling_endosomal_protection"
    elif technology == "albumin_binding":
        cl_factor = 0.1
        vd_factor = 2.0
        mechanism = "albumin_carrier_extended_circulation"
    elif technology == "nanoparticle":
        cl_factor = 0.3
        vd_factor = 0.5
        mechanism = "MPS_evasion_controlled_release"
    elif technology == "HES_conjugation":
        cl_factor = 1.0 / (1.0 + 0.3 * min(mw_kDa, 50.0) / 10.0)
        vd_factor = 1.0
        mechanism = "hydroxyethyl_starch_reduced_renal_clearance"
    else:
        raise ValueError(f"Unknown technology '{technology}'. Valid: {sorted(VALID_TECHNOLOGIES)}")
    return cl_factor, vd_factor, mechanism


def _nearest_interval(target_h: float) -> int:
    """Find the nearest common dosing interval."""
    return min(COMMON_INTERVALS, key=lambda x: abs(x - target_h))


def predict_halflife_extension(
    drug_name: str,
    base_cl_L_per_h: float,
    base_vd_L: float,
    mw_kDa: float,
    technology: str,
) -> HalflifeExtensionResult:
    """Predict half-life extension for a given technology.

    Args:
        drug_name: Name of the drug.
        base_cl_L_per_h: Baseline clearance (L/h).
        base_vd_L: Baseline volume of distribution (L).
        mw_kDa: Molecular weight (kDa).
        technology: One of the valid extension technologies.

    Returns:
        HalflifeExtensionResult with extended PK parameters.
    """
    if base_cl_L_per_h <= 0:
        raise ValueError("base_cl_L_per_h must be > 0")
    if base_vd_L <= 0:
        raise ValueError("base_vd_L must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if technology not in VALID_TECHNOLOGIES:
        raise ValueError(f"Unknown technology '{technology}'. Valid: {sorted(VALID_TECHNOLOGIES)}")

    base_t_half = math.log(2) * base_vd_L / base_cl_L_per_h

    cl_factor, vd_factor, mechanism = _get_technology_factors(technology, mw_kDa)

    new_cl = base_cl_L_per_h * cl_factor
    new_vd = base_vd_L * vd_factor
    extended_t_half = math.log(2) * new_vd / new_cl
    extension_factor = extended_t_half / base_t_half

    target_interval = 2.0 * extended_t_half
    dosing_interval = _nearest_interval(target_interval)

    auc_ratio = base_cl_L_per_h / new_cl
    peak_trough_improvement = extension_factor

    notes = (
        f"{technology} applied to {drug_name} (MW={mw_kDa} kDa). "
        f"CL reduced by {(1 - cl_factor) * 100:.1f}%, "
        f"Vd changed by {(vd_factor - 1) * 100:+.1f}%. "
        f"Half-life extended {extension_factor:.1f}x."
    )

    return HalflifeExtensionResult(
        drug_name=drug_name,
        base_t_half_h=base_t_half,
        technology=technology,
        extended_t_half_h=extended_t_half,
        extension_factor=extension_factor,
        mechanism=mechanism,
        new_cl_L_per_h=new_cl,
        new_vd_L=new_vd,
        base_cl_L_per_h=base_cl_L_per_h,
        base_vd_L=base_vd_L,
        dosing_interval_recommendation_h=dosing_interval,
        auc_ratio=auc_ratio,
        peak_trough_improvement=peak_trough_improvement,
        notes=notes,
    )


def compare_technologies(
    drug_name: str,
    base_cl_L_per_h: float,
    base_vd_L: float,
    mw_kDa: float,
) -> list:
    """Compare all 5 technologies sorted by extension_factor descending.

    Args:
        drug_name: Name of the drug.
        base_cl_L_per_h: Baseline clearance (L/h).
        base_vd_L: Baseline volume of distribution (L).
        mw_kDa: Molecular weight (kDa).

    Returns:
        List of HalflifeExtensionResult sorted by extension_factor descending.
    """
    results = []
    for tech in sorted(VALID_TECHNOLOGIES):
        result = predict_halflife_extension(
            drug_name=drug_name,
            base_cl_L_per_h=base_cl_L_per_h,
            base_vd_L=base_vd_L,
            mw_kDa=mw_kDa,
            technology=tech,
        )
        results.append(result)
    results.sort(key=lambda r: r.extension_factor, reverse=True)
    return results
