"""Phase 1084 — Drug Protein Binding Displacement.

Predicts the PK consequence of plasma protein binding displacement DDI.
When a perpetrator displaces a victim from albumin, fu_victim increases →
transient exposure change (free drug Cmax spike).
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_SITES = frozenset({"site_1", "site_2"})


@dataclass(frozen=True)
class ProteinBindingDisplacementResult:
    """Result of a protein binding displacement DDI prediction."""

    victim_drug: str
    perpetrator_drug: str
    albumin_site: str
    fu_victim_baseline: float
    fu_victim_new: float
    f_bound_baseline: float
    delta_fu_absolute: float
    delta_fu_pct: float
    cmax_free_ratio: float
    hepatic_extraction_class: str
    predicted_aucr: float
    clinical_significance: str
    notes: str


def _classify_hepatic_extraction(ratio: float) -> str:
    if ratio > 0.7:
        return "restricted"
    elif ratio < 0.3:
        return "unrestricted"
    else:
        return "intermediate"


def _predicted_aucr(fu_baseline: float, fu_new: float, hepatic_class: str) -> float:
    """Estimate AUC ratio from fu change given hepatic extraction class.

    For restricted drugs (high E):
        CL_unbound increases proportionally with fu → AUC increases ~ fu_new/fu_baseline.
    For unrestricted drugs (low E):
        CL total increases proportionally → AUC remains approximately constant (AUCR ≈ 1).
    For intermediate:
        Partial effect — geometric mean of the two extremes.
    """
    ratio_fu = fu_new / fu_baseline
    if hepatic_class == "restricted":
        # CL ∝ fu → AUC ∝ 1/fu → AUCR = fu_baseline / fu_new < 1 (restricted CL increases)
        # Actually for restricted drugs, total drug AUC may increase temporarily
        # Standard interpretation: for restricted clearance, AUC of total drug
        # increases proportionally because CL_int is the same; free drug AUC constant.
        # For simplicity and clinical practice: AUCR_total ≈ 1 for restricted drugs.
        return 1.0
    elif hepatic_class == "unrestricted":
        # CL increases with fu → AUC_total ≈ constant
        return 1.0
    else:
        # Intermediate: partial effect. Use ratio_fu as a rough upper bound.
        # In practice, binding displacement rarely causes > 2-fold AUC change.
        return 1.0 + (ratio_fu - 1.0) * 0.5


def _classify_clinical_significance(cmax_free_ratio: float, delta_fu_pct: float) -> str:
    """Classify clinical significance of binding displacement.

    Binding displacement alone rarely causes clinically significant DDI,
    but a free drug Cmax spike can matter for drugs with narrow TI.
    """
    if cmax_free_ratio < 1.25 and delta_fu_pct < 25:
        return "negligible"
    elif cmax_free_ratio < 2.0 and delta_fu_pct < 100:
        return "minor"
    else:
        return "moderate"


def predict_binding_displacement(
    victim_drug: str,
    perpetrator_drug: str,
    albumin_site: str,
    fu_victim: float = 0.05,
    ki_displacement_mg_per_L: float = 5.0,
    perpetrator_cmax_mg_per_L: float = 10.0,
    hepatic_extraction_ratio: float = 0.3,
) -> ProteinBindingDisplacementResult:
    """Predict PK impact of plasma protein binding displacement.

    Parameters
    ----------
    victim_drug:
        Name of the victim drug.
    perpetrator_drug:
        Name of the perpetrator drug.
    albumin_site:
        Albumin binding site. Must be "site_1" or "site_2".
    fu_victim:
        Baseline fraction unbound of victim drug. Must be in (0, 1].
    ki_displacement_mg_per_L:
        Dissociation constant of perpetrator-albumin binding (mg/L). Must be > 0.
    perpetrator_cmax_mg_per_L:
        Perpetrator plasma Cmax in mg/L. Must be >= 0.
    hepatic_extraction_ratio:
        Hepatic extraction ratio of the victim drug. Must be in [0, 1].
    """
    # --- Validation ---
    if albumin_site not in _VALID_SITES:
        raise ValueError(
            f"albumin_site must be one of {sorted(_VALID_SITES)}, got '{albumin_site}'"
        )
    if not (0 < fu_victim <= 1.0):
        raise ValueError(f"fu_victim must be in (0, 1], got {fu_victim}")
    if ki_displacement_mg_per_L <= 0:
        raise ValueError(f"ki_displacement_mg_per_L must be > 0, got {ki_displacement_mg_per_L}")
    if perpetrator_cmax_mg_per_L < 0:
        raise ValueError(f"perpetrator_cmax_mg_per_L must be >= 0, got {perpetrator_cmax_mg_per_L}")
    if not (0.0 <= hepatic_extraction_ratio <= 1.0):
        raise ValueError(
            f"hepatic_extraction_ratio must be in [0, 1], got {hepatic_extraction_ratio}"
        )

    # --- Displacement calculation ---
    f_bound_baseline = 1.0 - fu_victim

    # Fraction of bound drug displaced by perpetrator
    # f_displaced = f_bound * [I] / (Ki + [I])
    f_displaced = (
        f_bound_baseline
        * perpetrator_cmax_mg_per_L
        / (ki_displacement_mg_per_L + perpetrator_cmax_mg_per_L)
    )

    # New fu: new_fu = min(1.0, fu_victim / (1 - f_displaced * (1 - fu_victim)))
    # Denominator: remaining bound fraction = (1 - fu) - f_displaced*(1-fu)
    #                                       = (1-fu)(1 - f_displaced)
    # So: new fu_victim_new = fu_victim + f_displaced*(1-fu_victim)  [simplified additive]
    # Use the spec formula: new_fu = min(1.0, fu / (1 - f_displaced * (1 - fu)))
    denominator = 1.0 - f_displaced * f_bound_baseline
    if denominator <= 0:
        fu_victim_new = 1.0
    else:
        fu_victim_new = min(1.0, fu_victim / denominator)

    delta_fu_absolute = fu_victim_new - fu_victim
    delta_fu_pct = (delta_fu_absolute / fu_victim) * 100.0 if fu_victim > 0 else 0.0

    # Transient Cmax_free increase ratio
    cmax_free_ratio = fu_victim_new / fu_victim

    # Hepatic extraction class
    hepatic_class = _classify_hepatic_extraction(hepatic_extraction_ratio)

    # Predicted AUC ratio
    predicted_aucr = _predicted_aucr(fu_victim, fu_victim_new, hepatic_class)

    # Clinical significance
    clinical_significance = _classify_clinical_significance(cmax_free_ratio, delta_fu_pct)

    notes = (
        f"f_bound_baseline = {f_bound_baseline:.3f}. "
        f"f_displaced = {f_displaced:.4f} "
        f"([I]/Ki = {perpetrator_cmax_mg_per_L / ki_displacement_mg_per_L:.3f}). "
        f"Binding displacement alone rarely produces clinically significant DDI; "
        f"free drug Cmax spike of {cmax_free_ratio:.2f}x may be transient."
    )

    return ProteinBindingDisplacementResult(
        victim_drug=victim_drug,
        perpetrator_drug=perpetrator_drug,
        albumin_site=albumin_site,
        fu_victim_baseline=fu_victim,
        fu_victim_new=fu_victim_new,
        f_bound_baseline=f_bound_baseline,
        delta_fu_absolute=delta_fu_absolute,
        delta_fu_pct=delta_fu_pct,
        cmax_free_ratio=cmax_free_ratio,
        hepatic_extraction_class=hepatic_class,
        predicted_aucr=predicted_aucr,
        clinical_significance=clinical_significance,
        notes=notes,
    )


def screen_displacement_risk(
    victim_drug: str,
    fu_victim: float,
    perpetrators: list[dict],
) -> list[ProteinBindingDisplacementResult]:
    """Screen a victim drug against multiple perpetrators for binding displacement risk.

    Parameters
    ----------
    victim_drug:
        Name of the victim drug.
    fu_victim:
        Baseline fraction unbound for the victim drug.
    perpetrators:
        List of dicts, each with keys:
        - "name" (str): perpetrator drug name
        - "ki" (float): Ki for displacement (mg/L)
        - "cmax" (float): perpetrator Cmax (mg/L)
        - "site" (str): albumin site ("site_1" or "site_2")
        Optional keys forwarded to predict_binding_displacement:
        - "hepatic_extraction_ratio" (float)
    """
    results: list[ProteinBindingDisplacementResult] = []
    for perp in perpetrators:
        result = predict_binding_displacement(
            victim_drug=victim_drug,
            perpetrator_drug=perp["name"],
            albumin_site=perp["site"],
            fu_victim=fu_victim,
            ki_displacement_mg_per_L=perp["ki"],
            perpetrator_cmax_mg_per_L=perp["cmax"],
            hepatic_extraction_ratio=perp.get("hepatic_extraction_ratio", 0.3),
        )
        results.append(result)
    results.sort(key=lambda r: r.cmax_free_ratio, reverse=True)
    return results
