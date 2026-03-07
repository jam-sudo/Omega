"""Phase 752 — Clinical PK Summary Generator.

Generates a structured clinical PK summary report from key pharmacokinetic
parameters, providing human-readable assessment for clinical decision-making.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicalPKSummary:
    """Structured clinical PK summary for a drug."""

    drug_name: str
    t_half_h: float
    dosing_interval_recommendation: str  # "QD" / "BID" / "TID" / "QID"
    vd_classification: str  # "low" / "moderate" / "high"
    cl_classification: str  # "low" / "moderate" / "high"
    bioavailability_category: str  # "excellent" / "good" / "moderate" / "poor"
    protein_binding_category: str  # "low" / "moderate" / "high"
    therapeutic_window_width: float  # (mtc-mec)/mec, or -1.0 if unavailable
    narrow_therapeutic_index: bool
    elimination_route_primary: str  # "renal-dominant" / "hepatic-dominant" / "mixed"
    accumulation_risk: str  # "low" / "moderate" / "high"
    clinical_notes: str


def _dosing_interval(t_half_h: float) -> str:
    if t_half_h > 18.0:
        return "QD"
    if t_half_h >= 8.0:
        return "BID"
    if t_half_h >= 4.0:
        return "TID"
    return "QID"


def _vd_classification(vd_L_per_kg: float) -> str:
    if vd_L_per_kg < 0.5:
        return "low"
    if vd_L_per_kg <= 3.0:
        return "moderate"
    return "high"


def _cl_classification(cl_mL_per_min_per_kg: float) -> str:
    if cl_mL_per_min_per_kg < 2.0:
        return "low"
    if cl_mL_per_min_per_kg <= 10.0:
        return "moderate"
    return "high"


def _bioavailability_category(f_oral: float) -> str:
    pct = f_oral * 100.0
    if pct > 80.0:
        return "excellent"
    if pct >= 50.0:
        return "good"
    if pct >= 20.0:
        return "moderate"
    return "poor"


def _protein_binding_category(ppb_pct: float) -> str:
    if ppb_pct > 85.0:
        return "high"
    if ppb_pct >= 50.0:
        return "moderate"
    return "low"


def _elimination_route(fe_renal: float) -> str:
    if fe_renal > 0.6:
        return "renal-dominant"
    if fe_renal < 0.3:
        return "hepatic-dominant"
    return "mixed"


def _accumulation_risk(t_half_h: float, dosing_interval_str: str) -> str:
    freq_map = {"QD": 1, "BID": 2, "TID": 3, "QID": 4}
    freq = freq_map[dosing_interval_str]
    tau_h = 24.0 / freq
    ke = math.log(2.0) / t_half_h
    exponent = -ke * tau_h
    # Clamp to avoid exp underflow/overflow edge cases
    exp_val = math.exp(max(-700.0, exponent))
    denom = 1.0 - exp_val
    if denom <= 1e-9:
        ratio = float("inf")
    else:
        ratio = 1.0 / denom
    if ratio < 1.5:
        return "low"
    if ratio <= 3.0:
        return "moderate"
    return "high"


def generate_clinical_pk_summary(
    drug_name: str,
    t_half_h: float,
    vd_L_per_kg: float,
    cl_mL_per_min_per_kg: float,
    f_oral: float,
    protein_binding_pct: float,
    mec_mg_L: float | None = None,
    mtc_mg_L: float | None = None,
    fe_renal: float = 0.4,
) -> ClinicalPKSummary:
    """Generate a structured clinical PK summary.

    Parameters
    ----------
    drug_name: str
        Drug identifier.
    t_half_h: float
        Terminal elimination half-life (h). Must be > 0.
    vd_L_per_kg: float
        Apparent volume of distribution per kg body weight (L/kg). Must be > 0.
    cl_mL_per_min_per_kg: float
        Systemic clearance normalised to body weight (mL/min/kg). Must be > 0.
    f_oral: float
        Oral bioavailability fraction (0–1).
    protein_binding_pct: float
        Plasma protein binding percentage (0–100).
    mec_mg_L: float | None
        Minimum effective concentration (mg/L). Optional.
    mtc_mg_L: float | None
        Minimum toxic concentration (mg/L). Optional.
    fe_renal: float
        Fraction of dose eliminated renally (0–1). Default 0.4.

    Returns
    -------
    ClinicalPKSummary (frozen dataclass)
    """
    # --- Validation ---
    if t_half_h <= 0.0:
        raise ValueError(f"t_half_h must be > 0, got {t_half_h}")
    if vd_L_per_kg <= 0.0:
        raise ValueError(f"vd_L_per_kg must be > 0, got {vd_L_per_kg}")
    if cl_mL_per_min_per_kg <= 0.0:
        raise ValueError(f"cl_mL_per_min_per_kg must be > 0, got {cl_mL_per_min_per_kg}")

    # --- Classifications ---
    dosing = _dosing_interval(t_half_h)
    vd_class = _vd_classification(vd_L_per_kg)
    cl_class = _cl_classification(cl_mL_per_min_per_kg)
    ba_cat = _bioavailability_category(f_oral)
    ppb_cat = _protein_binding_category(protein_binding_pct)
    elim_route = _elimination_route(fe_renal)
    accum_risk = _accumulation_risk(t_half_h, dosing)

    # --- Therapeutic window ---
    if mec_mg_L is not None and mtc_mg_L is not None and mec_mg_L > 0.0:
        tw_width = (mtc_mg_L - mec_mg_L) / mec_mg_L
        nti = tw_width > 0.0 and tw_width < 2.0
    else:
        tw_width = -1.0
        nti = False

    # --- Clinical notes ---
    notes_parts = [
        f"Drug: {drug_name}.",
        f"t½={t_half_h:.1f}h → recommended dosing: {dosing}.",
        f"Vd={vd_L_per_kg:.2f}L/kg ({vd_class}), "
        f"CL={cl_mL_per_min_per_kg:.2f}mL/min/kg ({cl_class}).",
        f"Oral bioavailability: {f_oral * 100:.0f}% ({ba_cat}).",
        f"Protein binding: {protein_binding_pct:.0f}% ({ppb_cat}).",
        f"Elimination: {elim_route} (fe_renal={fe_renal:.2f}).",
        f"Accumulation risk: {accum_risk}.",
    ]
    if tw_width >= 0.0:
        nti_label = "NTI" if nti else "wide TI"
        notes_parts.append(
            f"Therapeutic window width={(tw_width):.2f} ({nti_label}); "
            f"MEC={mec_mg_L}mg/L, MTC={mtc_mg_L}mg/L."
        )
    else:
        notes_parts.append("Therapeutic window: not specified.")

    clinical_notes = " ".join(notes_parts)

    return ClinicalPKSummary(
        drug_name=drug_name,
        t_half_h=t_half_h,
        dosing_interval_recommendation=dosing,
        vd_classification=vd_class,
        cl_classification=cl_class,
        bioavailability_category=ba_cat,
        protein_binding_category=ppb_cat,
        therapeutic_window_width=tw_width,
        narrow_therapeutic_index=nti,
        elimination_route_primary=elim_route,
        accumulation_risk=accum_risk,
        clinical_notes=clinical_notes,
    )


def screen_pk_profiles(drugs_list: list[dict]) -> list[ClinicalPKSummary]:
    """Screen multiple drug PK profiles and return summaries sorted by t_half descending.

    Parameters
    ----------
    drugs_list: list[dict]
        Each dict must contain the same keys as generate_clinical_pk_summary parameters.

    Returns
    -------
    list[ClinicalPKSummary] sorted by t_half_h descending.
    """
    summaries = [generate_clinical_pk_summary(**d) for d in drugs_list]
    summaries.sort(key=lambda s: s.t_half_h, reverse=True)
    return summaries
