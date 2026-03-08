"""Phase 533 — Metabolite Safety Monitoring.

Predict circulating metabolite exposure and assess safety based on FDA MIST guidance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetaboliteSafetyResult:
    """Result of metabolite safety assessment per FDA MIST guidance."""

    drug_name: str
    metabolite_name: str
    dose_mg: float
    fm: float
    mw_parent_Da: float
    mw_metabolite_Da: float
    cmax_parent_mg_L: float
    auc_parent_mg_h_per_L: float
    cmax_metabolite_mg_L: float
    auc_metabolite_mg_h_per_L: float
    auc_ratio_met_to_parent: float
    mist_threshold_exceeded: bool
    major_metabolite: bool
    safety_testing_required: str  # "none" / "characterization" / "full_ich_m3"
    notes: str


def assess_metabolite_safety(
    drug_name: str,
    metabolite_name: str,
    dose_mg: float,
    cl_parent_L_per_h: float,
    vd_L: float,
    cl_metabolite_L_per_h: float,
    fm: float = 0.8,
    mw_parent_Da: float = 400.0,
    mw_metabolite_Da: float = 370.0,
) -> MetaboliteSafetyResult:
    """Assess metabolite safety using FDA MIST guidance.

    Parameters
    ----------
    drug_name:
        Name of the parent drug.
    metabolite_name:
        Name of the metabolite.
    dose_mg:
        Dose in mg.
    cl_parent_L_per_h:
        Parent drug clearance in L/h.
    vd_L:
        Volume of distribution in L.
    cl_metabolite_L_per_h:
        Metabolite clearance in L/h.
    fm:
        Fraction metabolized via the pathway (0 < fm < 1).
    mw_parent_Da:
        Molecular weight of parent in Daltons.
    mw_metabolite_Da:
        Molecular weight of metabolite in Daltons.

    Returns
    -------
    MetaboliteSafetyResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_parent_L_per_h <= 0:
        raise ValueError("cl_parent_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")
    if cl_metabolite_L_per_h <= 0:
        raise ValueError("cl_metabolite_L_per_h must be positive")
    if not (0 < fm < 1):
        raise ValueError("fm must be between 0 and 1 (exclusive)")

    # Parent PK
    cmax_parent = dose_mg / vd_L  # mg/L
    auc_parent = dose_mg / cl_parent_L_per_h  # mg·h/L

    # Molar ratio accounts for MW difference
    molar_ratio = mw_parent_Da / mw_metabolite_Da

    # Metabolite exposure using Dain's approximation
    # Cmax_metabolite ≈ Cmax_parent * fm * (CL_parent / CL_metabolite) * molar_ratio
    cmax_metabolite = cmax_parent * fm * (cl_parent_L_per_h / cl_metabolite_L_per_h) * molar_ratio

    # AUC_metabolite = AUC_parent * fm * (molar_ratio * CL_parent / CL_metabolite)
    auc_metabolite = auc_parent * fm * (molar_ratio * cl_parent_L_per_h / cl_metabolite_L_per_h)

    # AUC ratio
    auc_ratio = auc_metabolite / auc_parent

    # MIST thresholds (FDA 2020 guidance)
    mist_exceeded = auc_ratio > 0.10
    major_met = auc_ratio > 0.25

    # Safety testing classification
    if major_met:
        testing = "full_ich_m3"
    elif mist_exceeded:
        testing = "characterization"
    else:
        testing = "none"

    # Build notes
    note_parts = []
    note_parts.append(f"AUC ratio {auc_ratio:.3f} (threshold 0.10)")
    if major_met:
        note_parts.append(
            "Major metabolite (>25% of parent AUC); full ICH M3 safety testing required."
        )
    elif mist_exceeded:
        note_parts.append(
            "MIST threshold exceeded; metabolite characterization in safety studies required."
        )
    else:
        note_parts.append(
            "Below MIST threshold; no additional metabolite-specific safety studies required."
        )

    notes = " ".join(note_parts)

    return MetaboliteSafetyResult(
        drug_name=drug_name,
        metabolite_name=metabolite_name,
        dose_mg=dose_mg,
        fm=fm,
        mw_parent_Da=mw_parent_Da,
        mw_metabolite_Da=mw_metabolite_Da,
        cmax_parent_mg_L=cmax_parent,
        auc_parent_mg_h_per_L=auc_parent,
        cmax_metabolite_mg_L=cmax_metabolite,
        auc_metabolite_mg_h_per_L=auc_metabolite,
        auc_ratio_met_to_parent=auc_ratio,
        mist_threshold_exceeded=mist_exceeded,
        major_metabolite=major_met,
        safety_testing_required=testing,
        notes=notes,
    )


def screen_metabolites(
    drug_name: str,
    dose_mg: float,
    cl_parent_L_per_h: float,
    vd_L: float,
    metabolites: list,
) -> list:
    """Screen multiple metabolites and return sorted by AUC ratio descending.

    Parameters
    ----------
    drug_name:
        Name of the parent drug.
    dose_mg:
        Dose in mg.
    cl_parent_L_per_h:
        Parent clearance in L/h.
    vd_L:
        Volume of distribution in L.
    metabolites:
        List of dicts with keys: metabolite_name, cl_metabolite_L_per_h, fm, mw_metabolite_Da.

    Returns
    -------
    list of MetaboliteSafetyResult sorted by auc_ratio_met_to_parent descending.
    """
    results = []
    for m in metabolites:
        result = assess_metabolite_safety(
            drug_name=drug_name,
            metabolite_name=m["metabolite_name"],
            dose_mg=dose_mg,
            cl_parent_L_per_h=cl_parent_L_per_h,
            vd_L=vd_L,
            cl_metabolite_L_per_h=m["cl_metabolite_L_per_h"],
            fm=m["fm"],
            mw_metabolite_Da=m.get("mw_metabolite_Da", 370.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_ratio_met_to_parent, reverse=True)
    return results
