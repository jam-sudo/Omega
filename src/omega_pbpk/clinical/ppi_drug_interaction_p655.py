"""Proton pump inhibitor effect on drug absorption.

Models the effect of PPIs (e.g., omeprazole) on drug absorption by increasing
gastric pH.  Relevant for weakly basic drugs (reduced absorption) and
enteric-coated formulations.  Uses Henderson-Hasselbalch solubility and a
dose-number fraction-absorbed model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPIDrugInteractionResult:
    """Result of PPI-drug interaction prediction."""

    drug_name: str
    ppi_drug: str
    baseline_gastric_ph: float
    ppi_gastric_ph: float
    baseline_solubility_mg_mL: float
    ppi_solubility_mg_mL: float
    baseline_fa_pct: float
    ppi_fa_pct: float
    delta_fa_pct: float
    baseline_auc_ratio: float
    ppi_auc_ratio: float
    ddi_magnitude: str
    clinical_significance: str
    notes: str


def _compute_solubility(
    solubility_at_ph1_mg_mL: float,
    pka: float,
    ph: float,
    is_base: bool,
) -> float:
    """Henderson-Hasselbalch solubility at a given pH."""
    if is_base:
        return solubility_at_ph1_mg_mL * (1 + 10 ** (pka - ph))
    return solubility_at_ph1_mg_mL * (1 + 10 ** (ph - pka))


def _fraction_absorbed(dose_mg: float, solubility_mg_mL: float) -> float:
    """Dose-number model for fraction absorbed."""
    do = dose_mg / (solubility_mg_mL * 250)
    if do <= 1.0:
        return 1.0
    return min(1.0, 1.0 / do * 0.9)


def predict_ppi_interaction(
    drug_name: str,
    logP: float,
    mw_Da: float,
    pka: float,
    is_base: bool,
    dose_mg: float = 100.0,
    ppi_drug: str = "omeprazole",
    solubility_at_ph1_mg_mL: float = 1.0,
) -> PPIDrugInteractionResult:
    """Predict PPI effect on drug absorption."""
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if solubility_at_ph1_mg_mL <= 0:
        raise ValueError("solubility_at_ph1_mg_mL must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")

    baseline_ph = 1.5
    ppi_ph = 4.5

    s_baseline = _compute_solubility(solubility_at_ph1_mg_mL, pka, baseline_ph, is_base)
    s_ppi = _compute_solubility(solubility_at_ph1_mg_mL, pka, ppi_ph, is_base)

    fa_baseline = _fraction_absorbed(dose_mg, s_baseline)
    fa_ppi = _fraction_absorbed(dose_mg, s_ppi)

    baseline_fa_pct = fa_baseline * 100.0
    ppi_fa_pct = fa_ppi * 100.0
    delta_fa_pct = ppi_fa_pct - baseline_fa_pct

    baseline_auc_ratio = 1.0
    if baseline_fa_pct > 0:
        ppi_auc_ratio = ppi_fa_pct / baseline_fa_pct
    else:
        ppi_auc_ratio = 1.0
    ppi_auc_ratio = max(0.1, min(5.0, ppi_auc_ratio))

    abs_diff = abs(ppi_auc_ratio - 1.0)
    if abs_diff > 0.8:
        ddi_magnitude = "major"
    elif abs_diff > 0.3:
        ddi_magnitude = "moderate"
    else:
        ddi_magnitude = "minor"

    if ddi_magnitude == "major":
        clinical_significance = "avoid"
    elif ddi_magnitude == "moderate":
        clinical_significance = "monitor"
    else:
        clinical_significance = "no_action"

    notes = (
        f"PPI ({ppi_drug}) raises gastric pH from {baseline_ph} to {ppi_ph}. "
        f"{'Basic' if is_base else 'Acidic'} drug (pKa={pka})."
    )

    return PPIDrugInteractionResult(
        drug_name=drug_name,
        ppi_drug=ppi_drug,
        baseline_gastric_ph=baseline_ph,
        ppi_gastric_ph=ppi_ph,
        baseline_solubility_mg_mL=s_baseline,
        ppi_solubility_mg_mL=s_ppi,
        baseline_fa_pct=baseline_fa_pct,
        ppi_fa_pct=ppi_fa_pct,
        delta_fa_pct=delta_fa_pct,
        baseline_auc_ratio=baseline_auc_ratio,
        ppi_auc_ratio=ppi_auc_ratio,
        ddi_magnitude=ddi_magnitude,
        clinical_significance=clinical_significance,
        notes=notes,
    )


def screen_ppi_interactions(
    compounds: list[dict],
) -> list[PPIDrugInteractionResult]:
    """Screen multiple compounds and return sorted by abs(delta_fa_pct) desc."""
    results = []
    for c in compounds:
        r = predict_ppi_interaction(**c)
        results.append(r)
    results.sort(key=lambda x: abs(x.delta_fa_pct), reverse=True)
    return results
