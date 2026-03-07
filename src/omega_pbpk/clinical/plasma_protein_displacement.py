"""Phase 296 — Plasma Protein Binding Displacement.

Model DDI due to plasma protein binding displacement: a displacing drug
competes for albumin binding sites, increasing the free fraction of the
victim drug.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PPBDisplacementResult:
    """Result of a PPB displacement assessment."""

    victim_drug: str
    displacing_drug: str
    victim_fu_baseline: float
    fu_displaced: float
    fu_change_fold: float
    cl_displaced_L_per_h: float
    vd_displaced_L: float
    auc_ratio: float
    cmax_ratio: float
    restrictive_clearance: bool
    clinical_significance: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_RESTRICTIVE_CL_THRESHOLD = 30.0  # L/h


def _validate_inputs(
    victim_fu_baseline: float,
    victim_cl_L_per_h: float,
    victim_vd_L: float,
    victim_dose_mg: float,
    displacing_concentration_mg_L: float,
    ic50_displacement_mg_L: float,
) -> None:
    if not (0 < victim_fu_baseline <= 1):
        raise ValueError("victim_fu_baseline must be in (0, 1]")
    if victim_cl_L_per_h <= 0:
        raise ValueError("victim_cl_L_per_h must be > 0")
    if victim_vd_L <= 0:
        raise ValueError("victim_vd_L must be > 0")
    if victim_dose_mg <= 0:
        raise ValueError("victim_dose_mg must be > 0")
    if displacing_concentration_mg_L < 0:
        raise ValueError("displacing_concentration_mg_L must be >= 0")
    if ic50_displacement_mg_L <= 0:
        raise ValueError("ic50_displacement_mg_L must be > 0")


def _classify_significance(fu_change_fold: float) -> str:
    if fu_change_fold >= 2.0:
        return "significant"
    if fu_change_fold >= 1.5:
        return "moderate"
    return "minor"


def _compute_displacement(
    victim_drug: str,
    displacing_drug: str,
    victim_fu_baseline: float,
    victim_cl_L_per_h: float,
    victim_vd_L: float,
    displacing_concentration_mg_L: float,
    ic50_displacement_mg_L: float,
) -> PPBDisplacementResult:
    """Core computation shared by assess and screen functions."""
    # Emax-style displacement fraction
    disp_fraction = displacing_concentration_mg_L / (
        ic50_displacement_mg_L + displacing_concentration_mg_L
    )

    # New free fraction (capped at 1.0)
    fu_displaced = min(
        1.0,
        victim_fu_baseline + (1.0 - victim_fu_baseline) * disp_fraction,
    )

    fu_change_fold = fu_displaced / victim_fu_baseline
    restrictive = victim_cl_L_per_h < _RESTRICTIVE_CL_THRESHOLD

    if restrictive:
        # Restrictive clearance: CL scales with fu
        cl_displaced = victim_cl_L_per_h * fu_change_fold
        auc_ratio = 1.0  # fu/baseline cancels with CL scaling
    else:
        # Non-restrictive (high-extraction): CL unchanged
        cl_displaced = victim_cl_L_per_h
        auc_ratio = fu_change_fold

    # Vd proportional to fu ratio (simplified)
    vd_displaced = victim_vd_L * fu_change_fold

    # Transient Cmax driven by free fraction change
    cmax_ratio = fu_change_fold

    significance = _classify_significance(fu_change_fold)

    notes_parts = [
        f"Displacement fraction = {disp_fraction:.3f}",
        f"fu: {victim_fu_baseline:.3f} -> {fu_displaced:.3f}",
        "Restrictive CL: AUC unchanged" if restrictive else "Non-restrictive CL: AUC increases",
    ]

    return PPBDisplacementResult(
        victim_drug=victim_drug,
        displacing_drug=displacing_drug,
        victim_fu_baseline=victim_fu_baseline,
        fu_displaced=fu_displaced,
        fu_change_fold=fu_change_fold,
        cl_displaced_L_per_h=cl_displaced,
        vd_displaced_L=vd_displaced,
        auc_ratio=auc_ratio,
        cmax_ratio=cmax_ratio,
        restrictive_clearance=restrictive,
        clinical_significance=significance,
        notes="; ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_ppb_displacement(
    victim_drug: str,
    displacing_drug: str,
    victim_fu_baseline: float,
    victim_cl_L_per_h: float,
    victim_vd_L: float,
    victim_dose_mg: float,
    displacing_concentration_mg_L: float,
    ic50_displacement_mg_L: float,
) -> PPBDisplacementResult:
    """Assess PPB displacement DDI between a victim and displacing drug.

    Parameters
    ----------
    victim_drug:
        Name of the victim (bound) drug.
    displacing_drug:
        Name of the displacing drug.
    victim_fu_baseline:
        Unbound fraction of victim drug at baseline (0 < fu <= 1).
    victim_cl_L_per_h:
        Total (apparent) clearance of victim drug, L/h.
    victim_vd_L:
        Volume of distribution of victim drug, L.
    victim_dose_mg:
        Dose of victim drug, mg (used for context/validation).
    displacing_concentration_mg_L:
        Plasma concentration of displacing drug, mg/L.
    ic50_displacement_mg_L:
        IC50 for displacement of victim from albumin, mg/L.

    Returns
    -------
    PPBDisplacementResult
    """
    _validate_inputs(
        victim_fu_baseline,
        victim_cl_L_per_h,
        victim_vd_L,
        victim_dose_mg,
        displacing_concentration_mg_L,
        ic50_displacement_mg_L,
    )
    return _compute_displacement(
        victim_drug=victim_drug,
        displacing_drug=displacing_drug,
        victim_fu_baseline=victim_fu_baseline,
        victim_cl_L_per_h=victim_cl_L_per_h,
        victim_vd_L=victim_vd_L,
        displacing_concentration_mg_L=displacing_concentration_mg_L,
        ic50_displacement_mg_L=ic50_displacement_mg_L,
    )


def screen_displacement_pairs(
    victim_drug: str,
    victim_fu_baseline: float,
    victim_cl_L_per_h: float,
    victim_vd_L: float,
    victim_dose_mg: float,
    displacing_drugs_data: list[dict],
) -> list[PPBDisplacementResult]:
    """Screen multiple displacing drugs against a single victim.

    Parameters
    ----------
    victim_drug:
        Name of the victim drug.
    victim_fu_baseline:
        Unbound fraction of victim at baseline.
    victim_cl_L_per_h:
        Clearance of victim drug, L/h.
    victim_vd_L:
        Volume of distribution of victim drug, L.
    victim_dose_mg:
        Dose of victim drug, mg.
    displacing_drugs_data:
        List of dicts, each with keys:
          - drug_name (str)
          - concentration_mg_L (float)
          - ic50_mg_L (float)

    Returns
    -------
    list[PPBDisplacementResult] sorted by fu_change_fold descending.
    """
    if not displacing_drugs_data:
        return []

    results: list[PPBDisplacementResult] = []
    for entry in displacing_drugs_data:
        drug_name = entry["drug_name"]
        concentration = float(entry["concentration_mg_L"])
        ic50 = float(entry["ic50_mg_L"])
        _validate_inputs(
            victim_fu_baseline,
            victim_cl_L_per_h,
            victim_vd_L,
            victim_dose_mg,
            concentration,
            ic50,
        )
        result = _compute_displacement(
            victim_drug=victim_drug,
            displacing_drug=drug_name,
            victim_fu_baseline=victim_fu_baseline,
            victim_cl_L_per_h=victim_cl_L_per_h,
            victim_vd_L=victim_vd_L,
            displacing_concentration_mg_L=concentration,
            ic50_displacement_mg_L=ic50,
        )
        results.append(result)

    results.sort(key=lambda r: r.fu_change_fold, reverse=True)
    return results
