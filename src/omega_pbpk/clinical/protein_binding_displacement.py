"""
Phase 876: Drug Protein Binding Displacement Model
Models drug-drug interactions via plasma protein binding displacement.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProteinBindingDisplacementResult:
    """Result from protein binding displacement prediction."""

    drug_name_victim: str
    drug_name_displacer: str
    fup_victim_baseline: float
    fup_victim_displaced: float
    delta_fup: float
    delta_fup_pct: float
    same_binding_site: bool
    hepatic_er_victim: str
    auc_total_change_pct: float
    cmax_free_change_pct: float
    clinical_significance: str
    notes: str


def _classify_er(hepatic_er: float) -> str:
    """Classify extraction ratio as low/intermediate/high."""
    if hepatic_er < 0.3:
        return "low"
    elif hepatic_er > 0.7:
        return "high"
    else:
        return "intermediate"


def _clinical_significance(delta_fup_pct: float) -> str:
    """Return clinical significance based on % change in free fraction."""
    if delta_fup_pct < 10.0:
        return "none"
    elif delta_fup_pct < 30.0:
        return "minor"
    else:
        return "significant"


def predict_protein_binding_displacement(
    drug_name_victim: str,
    drug_name_displacer: str,
    fup_victim: float,
    kd_victim_uM: float,
    c_displacer_uM: float,
    kd_displacer_uM: float,
    hepatic_er_victim: float = 0.2,
    same_binding_site: bool = True,
) -> ProteinBindingDisplacementResult:
    """
    Predict protein binding displacement interaction.

    Parameters
    ----------
    drug_name_victim : str
    drug_name_displacer : str
    fup_victim : float
        Baseline fraction unbound for victim drug (0, 1].
    kd_victim_uM : float
        Dissociation constant of victim for albumin (µM).
    c_displacer_uM : float
        Plasma concentration of displacer (µM).
    kd_displacer_uM : float
        Dissociation constant of displacer for albumin (µM).
    hepatic_er_victim : float
        Hepatic extraction ratio of victim (0–1).
    same_binding_site : bool
        True if drugs compete for same binding site.

    Returns
    -------
    ProteinBindingDisplacementResult
    """
    # Validation
    if not (0 < fup_victim <= 1.0):
        raise ValueError("fup_victim must be in (0, 1]")
    if kd_victim_uM <= 0:
        raise ValueError("kd_victim_uM must be > 0")
    if c_displacer_uM < 0:
        raise ValueError("c_displacer_uM must be >= 0")

    # Displacement factor
    displacement_factor = 0.5 if same_binding_site else 0.1

    # Scatchard-type displacement: increase in fup
    # Saturability of displacer at its binding site: C_B / (Kd_B + C_B)
    displacer_saturation = c_displacer_uM / (kd_displacer_uM + c_displacer_uM)

    # Amount of bound fraction available to be displaced
    bound_fraction = 1.0 - fup_victim

    # New fup after displacement
    fup_displaced = fup_victim + bound_fraction * displacer_saturation * displacement_factor

    # Clamp to [0, 1]
    if fup_displaced > 1.0:
        fup_displaced = 1.0

    delta_fup = fup_displaced - fup_victim
    # Percent increase relative to baseline
    delta_fup_pct = (delta_fup / fup_victim) * 100.0

    # Classify extraction ratio
    er_class = _classify_er(hepatic_er_victim)

    # AUC total change
    # Low ER: CL_total = fup * CL_int (unbound clearance)
    #   AUC_total = Dose / CL_total  ∝ 1/fup
    #   AUC_total_new / AUC_total_old = fup_old / fup_new
    #   % change = (fup_old/fup_new - 1) * 100 (negative = decrease)
    # High ER: CL ≈ Q (flow-limited), AUC_total ≈ unchanged
    # Intermediate: partial effect
    if er_class == "low":
        auc_ratio = fup_victim / fup_displaced if fup_displaced > 0 else 1.0
        auc_total_change_pct = (auc_ratio - 1.0) * 100.0
    elif er_class == "high":
        auc_total_change_pct = 0.0
    else:
        # Intermediate: scale between 0 (high) and full low-ER effect
        # Linear interpolation: ER=0.3 → full effect, ER=0.7 → no effect
        fraction_low = 1.0 - (hepatic_er_victim - 0.3) / 0.4
        auc_ratio = fup_victim / fup_displaced if fup_displaced > 0 else 1.0
        auc_total_change_pct = (auc_ratio - 1.0) * 100.0 * fraction_low

    # Free Cmax change: proportional to increase in free fraction
    # Cmax_free_new / Cmax_free_old = fup_displaced / fup_victim
    cmax_free_ratio = fup_displaced / fup_victim if fup_victim > 0 else 1.0
    cmax_free_change_pct = (cmax_free_ratio - 1.0) * 100.0

    significance = _clinical_significance(delta_fup_pct)

    notes_parts = []
    if er_class == "high":
        notes_parts.append("High hepatic ER: total AUC unchanged (flow-limited clearance).")
    elif er_class == "low":
        notes_parts.append("Low hepatic ER: total AUC may decrease due to higher unbound CL.")
    else:
        notes_parts.append("Intermediate hepatic ER: partial AUC effect.")

    if same_binding_site:
        notes_parts.append("Competitive binding at same albumin site.")
    else:
        notes_parts.append("Independent binding sites; minor displacement expected.")

    if significance == "significant":
        notes_parts.append("Clinically significant displacement (delta fup >= 30%).")
    elif significance == "minor":
        notes_parts.append("Minor clinical significance (delta fup 10-30%).")

    notes = " ".join(notes_parts)

    return ProteinBindingDisplacementResult(
        drug_name_victim=drug_name_victim,
        drug_name_displacer=drug_name_displacer,
        fup_victim_baseline=fup_victim,
        fup_victim_displaced=fup_displaced,
        delta_fup=delta_fup,
        delta_fup_pct=delta_fup_pct,
        same_binding_site=same_binding_site,
        hepatic_er_victim=er_class,
        auc_total_change_pct=auc_total_change_pct,
        cmax_free_change_pct=cmax_free_change_pct,
        clinical_significance=significance,
        notes=notes,
    )


def screen_displacers(
    drug_name_victim: str,
    fup_victim: float,
    kd_victim_uM: float,
    displacers: list[dict],
) -> list[ProteinBindingDisplacementResult]:
    """
    Screen multiple displacer drugs for protein binding displacement.

    Parameters
    ----------
    drug_name_victim : str
    fup_victim : float
    kd_victim_uM : float
    displacers : list[dict]
        Each dict: {"name", "c_uM", "kd_uM", "same_site" (bool, default True)}

    Returns
    -------
    list[ProteinBindingDisplacementResult] sorted by delta_fup descending
    """
    results = []
    for d in displacers:
        name = d.get("name", "Unknown")
        c_uM = float(d.get("c_uM", 0.0))
        kd_uM = float(d.get("kd_uM", 1.0))
        same_site = bool(d.get("same_site", True))

        result = predict_protein_binding_displacement(
            drug_name_victim=drug_name_victim,
            drug_name_displacer=name,
            fup_victim=fup_victim,
            kd_victim_uM=kd_victim_uM,
            c_displacer_uM=c_uM,
            kd_displacer_uM=kd_uM,
            same_binding_site=same_site,
        )
        results.append(result)

    # Sort by delta_fup descending (most concerning first)
    results.sort(key=lambda r: r.delta_fup, reverse=True)
    return results
