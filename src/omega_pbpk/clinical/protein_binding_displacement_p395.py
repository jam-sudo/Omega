"""
Phase 395: Protein Binding Displacement Interaction
Models pharmacokinetic consequences of drug-drug interactions involving
plasma protein binding displacement.
"""

from dataclasses import dataclass

_VALID_PROTEINS = {"albumin", "alpha1_agp", "both"}


@dataclass(frozen=True)
class ProteinBindingDisplacementResult:
    victim_drug: str
    displacer_drug: str
    binding_protein: str
    victim_fup_baseline: float
    displacer_conc_uM: float
    victim_fup_displaced: float
    fup_increase_fold: float
    cl_change_pct: float
    vd_change_pct: float
    auc_change_pct: float
    cmax_change_pct: float
    clinical_significance: str
    narrow_therapeutic_index: bool
    monitoring_required: bool
    notes: str


def _validate_inputs(
    victim_fup_baseline,
    ki_displacement_uM,
    displacer_conc_uM,
    victim_total_bound_fraction,
    binding_protein,
):
    if not (0 < victim_fup_baseline <= 1.0):
        raise ValueError(f"victim_fup_baseline must be in (0, 1], got {victim_fup_baseline}")
    if ki_displacement_uM <= 0:
        raise ValueError(f"ki_displacement_uM must be > 0, got {ki_displacement_uM}")
    if displacer_conc_uM < 0:
        raise ValueError(f"displacer_conc_uM must be >= 0, got {displacer_conc_uM}")
    if not (0 <= victim_total_bound_fraction <= 1.0):
        raise ValueError(
            f"victim_total_bound_fraction must be in [0, 1], got {victim_total_bound_fraction}"
        )
    if binding_protein not in _VALID_PROTEINS:
        raise ValueError(
            f"binding_protein must be one of {_VALID_PROTEINS}, got '{binding_protein}'"
        )


def _classify_significance(fup_increase_fold):
    if fup_increase_fold < 1.1:
        return "negligible"
    elif fup_increase_fold < 1.5:
        return "minor"
    elif fup_increase_fold < 2.0:
        return "moderate"
    else:
        return "major"


def _build_notes(
    victim_drug,
    displacer_drug,
    binding_protein,
    fup_increase_fold,
    clinical_significance,
    monitoring_required,
    narrow_ti,
):
    lines = [
        f"{displacer_drug} displaces {victim_drug} from {binding_protein} binding sites.",
        f"Unbound fraction increases {fup_increase_fold:.2f}-fold.",
        f"Interaction classified as '{clinical_significance}'.",
    ]
    if narrow_ti:
        lines.append(
            f"{victim_drug} has a narrow therapeutic index; enhanced monitoring is essential."
        )
    if monitoring_required:
        lines.append("Therapeutic drug monitoring and clinical observation are recommended.")
    if clinical_significance == "negligible":
        lines.append("No dose adjustment anticipated.")
    elif clinical_significance == "minor":
        lines.append("Unlikely to require dose adjustment; routine monitoring advised.")
    elif clinical_significance == "moderate":
        lines.append("Consider dose reduction or increased monitoring interval.")
    else:
        lines.append(
            "Significant interaction; consider alternative therapy or careful dose titration."
        )
    return " ".join(lines)


def predict_displacement_interaction(
    victim_drug,
    displacer_drug,
    binding_protein,
    victim_fup_baseline,
    displacer_conc_uM,
    ki_displacement_uM,
    victim_total_bound_fraction,
    narrow_ti=False,
):
    """Predict pharmacokinetic consequences of protein binding displacement.

    Parameters
    ----------
    victim_drug : str
        Name of the drug being displaced.
    displacer_drug : str
        Name of the displacing drug.
    binding_protein : str
        Protein involved: "albumin", "alpha1_agp", or "both".
    victim_fup_baseline : float
        Baseline unbound fraction of victim drug (0, 1].
    displacer_conc_uM : float
        Displacer plasma concentration in uM (>= 0).
    ki_displacement_uM : float
        Inhibition constant for displacement in uM (> 0).
    victim_total_bound_fraction : float
        Fraction of victim drug that is protein-bound [0, 1].
    narrow_ti : bool
        Whether victim drug has a narrow therapeutic index.

    Returns
    -------
    ProteinBindingDisplacementResult
    """
    _validate_inputs(
        victim_fup_baseline,
        ki_displacement_uM,
        displacer_conc_uM,
        victim_total_bound_fraction,
        binding_protein,
    )

    # Competitive displacement model
    fraction_displaced = displacer_conc_uM / (displacer_conc_uM + ki_displacement_uM)

    # Additional unbound drug released from protein
    new_unbound_fraction = victim_total_bound_fraction * fraction_displaced

    # New unbound fraction (capped at 1)
    victim_fup_displaced = min(1.0, victim_fup_baseline + new_unbound_fraction)

    fup_increase_fold = victim_fup_displaced / victim_fup_baseline

    # High-extraction worst-case scenario
    cl_change_pct = (fup_increase_fold - 1.0) * 100.0
    vd_change_pct = (fup_increase_fold - 1.0) * 50.0
    auc_change_pct = 0.0  # high ER: fu*CL constant, AUC unchanged
    cmax_change_pct = (fup_increase_fold - 1.0) * 50.0  # transient increase

    clinical_significance = _classify_significance(fup_increase_fold)
    monitoring_required = narrow_ti or fup_increase_fold > 1.5

    notes = _build_notes(
        victim_drug,
        displacer_drug,
        binding_protein,
        fup_increase_fold,
        clinical_significance,
        monitoring_required,
        narrow_ti,
    )

    return ProteinBindingDisplacementResult(
        victim_drug=victim_drug,
        displacer_drug=displacer_drug,
        binding_protein=binding_protein,
        victim_fup_baseline=victim_fup_baseline,
        displacer_conc_uM=displacer_conc_uM,
        victim_fup_displaced=victim_fup_displaced,
        fup_increase_fold=fup_increase_fold,
        cl_change_pct=cl_change_pct,
        vd_change_pct=vd_change_pct,
        auc_change_pct=auc_change_pct,
        cmax_change_pct=cmax_change_pct,
        clinical_significance=clinical_significance,
        narrow_therapeutic_index=narrow_ti,
        monitoring_required=monitoring_required,
        notes=notes,
    )


def screen_displacers(
    victim_drug, victim_fup_baseline, victim_total_bound_fraction, displacers_list
):
    """Screen multiple potential displacers for a victim drug.

    Parameters
    ----------
    victim_drug : str
        Name of victim drug.
    victim_fup_baseline : float
        Baseline unbound fraction of victim.
    victim_total_bound_fraction : float
        Fraction of victim drug that is protein-bound.
    displacers_list : list of dict
        Each dict must have keys: 'name', 'conc_uM', 'ki_uM', 'binding_protein'.

    Returns
    -------
    list of ProteinBindingDisplacementResult
        Sorted by fup_increase_fold descending (largest displacement first).
    """
    results = []
    for d in displacers_list:
        result = predict_displacement_interaction(
            victim_drug=victim_drug,
            displacer_drug=d["name"],
            binding_protein=d["binding_protein"],
            victim_fup_baseline=victim_fup_baseline,
            displacer_conc_uM=d["conc_uM"],
            ki_displacement_uM=d["ki_uM"],
            victim_total_bound_fraction=victim_total_bound_fraction,
        )
        results.append(result)

    results.sort(key=lambda r: r.fup_increase_fold, reverse=True)
    return results
