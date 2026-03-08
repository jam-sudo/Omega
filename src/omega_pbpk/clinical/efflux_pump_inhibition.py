"""Phase 492 — Drug Efflux Pump Inhibition Effect.

Models the effect of efflux pump inhibitors (P-gp, BCRP, MRP) on drug PK.
P-gp inhibitors increase oral bioavailability of P-gp substrates and BBB penetration.
BCRP inhibition increases intestinal absorption.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EffluxPumpInhibitionResult:
    """Result of efflux pump inhibition prediction."""

    drug_name: str
    inhibitor: str
    pump_target: str
    inhibitor_conc_uM: float
    ki_uM: float
    inhibition_fraction: float
    fa_baseline: float
    fa_inhibited: float
    auc_ratio: float
    bbb_penetration_ratio: float
    clinical_benefit: str
    combination_recommended: bool
    notes: str


# Efflux pump inhibitor database
_INHIBITOR_DB: dict = {
    "verapamil": {"target": "P-gp", "Ki_uM": 2.0, "max_fold_increase_fa": 3.0},
    "elacridar": {"target": "P-gp_BCRP", "Ki_uM": 0.1, "max_fold_increase_fa": 8.0},
    "tariquidar": {"target": "P-gp", "Ki_uM": 0.01, "max_fold_increase_fa": 10.0},
    "Ko143": {"target": "BCRP", "Ki_uM": 0.01, "max_fold_increase_fa": 5.0},
    "MK571": {"target": "MRP", "Ki_uM": 1.0, "max_fold_increase_fa": 2.0},
}

_VALID_INHIBITORS = set(_INHIBITOR_DB.keys())


def _is_pgp_target(target: str) -> bool:
    """Return True if the pump target includes P-gp."""
    return "P-gp" in target


def _clinical_benefit(auc_ratio: float) -> str:
    """Classify clinical benefit from AUC ratio."""
    increase_pct = (auc_ratio - 1.0) * 100.0
    if increase_pct < 10.0:
        return "none"
    if increase_pct <= 50.0:
        return "minor"
    return "significant"


def predict_efflux_pump_inhibition(
    drug_name: str,
    inhibitor: str,
    base_fa: float,
    inhibitor_conc_uM: float,
) -> EffluxPumpInhibitionResult:
    """Predict the effect of an efflux pump inhibitor on drug PK.

    Parameters
    ----------
    drug_name:
        Identifier for the substrate drug.
    inhibitor:
        Name of the efflux pump inhibitor. Must be one of: verapamil, elacridar,
        tariquidar, Ko143, MK571.
    base_fa:
        Baseline fraction absorbed (0 < base_fa <= 1).
    inhibitor_conc_uM:
        Inhibitor concentration at the membrane (uM). Must be >= 0.

    Returns
    -------
    EffluxPumpInhibitionResult
    """
    if inhibitor not in _VALID_INHIBITORS:
        raise ValueError(f"inhibitor must be one of {sorted(_VALID_INHIBITORS)}, got {inhibitor!r}")
    if not (0 < base_fa <= 1.0):
        raise ValueError(f"base_fa must be in (0, 1], got {base_fa}")
    if inhibitor_conc_uM < 0:
        raise ValueError(f"inhibitor_conc_uM must be >= 0, got {inhibitor_conc_uM}")

    entry = _INHIBITOR_DB[inhibitor]
    ki_uM: float = entry["Ki_uM"]
    max_fold: float = entry["max_fold_increase_fa"]
    pump_target: str = entry["target"]

    # Inhibition fraction
    if inhibitor_conc_uM == 0.0:
        fi = 0.0
    else:
        fi = inhibitor_conc_uM / (inhibitor_conc_uM + ki_uM)

    fa_inhibited = min(1.0, base_fa * (1.0 + fi * (max_fold - 1.0)))
    auc_ratio = fa_inhibited / base_fa

    # BBB penetration increase (only for P-gp targets)
    if _is_pgp_target(pump_target):
        bbb_penetration_ratio = 1.0 + fi * 2.0
    else:
        bbb_penetration_ratio = 1.0

    benefit = _clinical_benefit(auc_ratio)
    combination_recommended = auc_ratio > 1.5 or bbb_penetration_ratio > 1.5

    notes_parts = []
    notes_parts.append(f"Pump target: {pump_target}; Ki={ki_uM} uM")
    notes_parts.append(f"Inhibition fraction: {fi:.3f}")
    if fa_inhibited == 1.0 and base_fa * (1.0 + fi * (max_fold - 1.0)) > 1.0:
        notes_parts.append("fa_inhibited capped at 1.0 (complete absorption)")
    notes = "; ".join(notes_parts)

    return EffluxPumpInhibitionResult(
        drug_name=drug_name,
        inhibitor=inhibitor,
        pump_target=pump_target,
        inhibitor_conc_uM=inhibitor_conc_uM,
        ki_uM=ki_uM,
        inhibition_fraction=fi,
        fa_baseline=base_fa,
        fa_inhibited=fa_inhibited,
        auc_ratio=auc_ratio,
        bbb_penetration_ratio=bbb_penetration_ratio,
        clinical_benefit=benefit,
        combination_recommended=combination_recommended,
        notes=notes,
    )


def compare_inhibitors(
    drug_name: str,
    base_fa: float,
    inhibitor_conc_uM: float = 1.0,
) -> list:
    """Compare all 5 efflux pump inhibitors for a given drug and concentration.

    Returns list of EffluxPumpInhibitionResult sorted by auc_ratio descending.
    """
    results = []
    for inhibitor_name in _INHIBITOR_DB:
        result = predict_efflux_pump_inhibition(
            drug_name=drug_name,
            inhibitor=inhibitor_name,
            base_fa=base_fa,
            inhibitor_conc_uM=inhibitor_conc_uM,
        )
        results.append(result)
    results.sort(key=lambda r: r.auc_ratio, reverse=True)
    return results
