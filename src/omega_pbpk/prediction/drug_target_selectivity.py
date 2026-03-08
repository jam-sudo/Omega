"""Drug target selectivity prediction.

Predicts drug selectivity across related biological targets using selectivity
index (SI = Ki_off / Ki_primary) analysis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetSelectivityResult:
    """Result of a drug target selectivity analysis."""

    drug_name: str
    primary_target: str
    primary_ki_nM: float
    off_targets: tuple
    off_target_ki_nM: tuple
    selectivity_indices: tuple
    minimum_selectivity_index: float
    mean_selectivity_index: float
    selectivity_class: str
    therapeutic_window_factor: float
    notes: str


def predict_target_selectivity(
    drug_name: str,
    primary_target: str,
    primary_ki_nM: float,
    off_target_names: list,
    off_target_ki_nM: list,
) -> TargetSelectivityResult:
    """Predict selectivity of a drug across off-target proteins.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    primary_target:
        Name of the primary intended target.
    primary_ki_nM:
        Binding affinity for the primary target in nM.
    off_target_names:
        List of off-target protein names.
    off_target_ki_nM:
        List of Ki values (nM) for each off-target (same order as names).

    Returns
    -------
    TargetSelectivityResult
    """
    # --- Validation ---
    if primary_ki_nM <= 0:
        raise ValueError(f"primary_ki_nM must be > 0, got {primary_ki_nM}")
    if len(off_target_names) == 0:
        raise ValueError("At least one off-target must be provided")
    if len(off_target_names) != len(off_target_ki_nM):
        raise ValueError(
            f"off_target_names (len={len(off_target_names)}) and "
            f"off_target_ki_nM (len={len(off_target_ki_nM)}) must have equal length"
        )
    for i, ki in enumerate(off_target_ki_nM):
        if ki <= 0:
            raise ValueError(f"off_target_ki_nM[{i}] ({off_target_names[i]}) must be > 0, got {ki}")

    # --- Selectivity indices ---
    si_list = [ki / primary_ki_nM for ki in off_target_ki_nM]
    min_si = min(si_list)
    mean_si = sum(si_list) / len(si_list)

    if min_si > 100:
        selectivity_class = "highly_selective"
    elif min_si > 10:
        selectivity_class = "selective"
    else:
        selectivity_class = "non_selective"

    therapeutic_window_factor = min_si / 10.0

    notes = (
        f"primary_ki={primary_ki_nM:.2f} nM; "
        f"{len(off_target_names)} off-targets; "
        f"min SI={min_si:.1f} ({selectivity_class})"
    )

    return TargetSelectivityResult(
        drug_name=drug_name,
        primary_target=primary_target,
        primary_ki_nM=primary_ki_nM,
        off_targets=tuple(off_target_names),
        off_target_ki_nM=tuple(off_target_ki_nM),
        selectivity_indices=tuple(si_list),
        minimum_selectivity_index=min_si,
        mean_selectivity_index=mean_si,
        selectivity_class=selectivity_class,
        therapeutic_window_factor=therapeutic_window_factor,
        notes=notes,
    )


def screen_selectivity_panel(
    drug_name: str,
    primary_target: str,
    primary_ki_nM: float,
    panel: dict,
) -> TargetSelectivityResult:
    """Screen a drug against a panel of off-targets provided as a dict.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    primary_target:
        Name of the primary intended target.
    primary_ki_nM:
        Binding affinity for the primary target in nM.
    panel:
        Dictionary mapping off-target name to Ki in nM.

    Returns
    -------
    TargetSelectivityResult
    """
    names = list(panel.keys())
    kis = list(panel.values())
    return predict_target_selectivity(drug_name, primary_target, primary_ki_nM, names, kis)


def compare_drug_selectivity(drugs: list) -> list:
    """Compare selectivity of multiple drugs and sort by minimum SI descending.

    Parameters
    ----------
    drugs:
        List of dicts, each with keys:
          - drug_name (str)
          - primary_target (str)
          - primary_ki_nM (float)
          - off_targets (list of str)
          - off_target_ki_nM (list of float)

    Returns
    -------
    list of TargetSelectivityResult, sorted by minimum_selectivity_index descending.
    """
    results = []
    for d in drugs:
        result = predict_target_selectivity(
            drug_name=d["drug_name"],
            primary_target=d["primary_target"],
            primary_ki_nM=d["primary_ki_nM"],
            off_target_names=d["off_targets"],
            off_target_ki_nM=d["off_target_ki_nM"],
        )
        results.append(result)

    results.sort(key=lambda r: r.minimum_selectivity_index, reverse=True)
    return results
