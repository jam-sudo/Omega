"""
Phase 589 — Bile Acid Drug Interaction
Model drug interactions with bile acid transporters: NTCP, BSEP, ASBT.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class BileAcidDrugInteractionResult:
    """Result of bile acid transporter interaction prediction."""

    drug_name: str
    ic50_ntcp_uM: float
    ic50_bsep_uM: float
    ic50_asbt_uM: float
    inhibition_risk: str
    fold_increase_bile_acids: float
    cholestasis_risk_score: float
    dili_risk_class: str
    notes: str


def predict_bile_acid_interaction(
    drug_name: str,
    mw_Da: float,
    logP: float,
    dose_uM: float,
    ki_ntcp_uM: float | None = None,
    ki_bsep_uM: float | None = None,
    ki_asbt_uM: float | None = None,
) -> BileAcidDrugInteractionResult:
    """
    Predict drug interaction with bile acid transporters.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    mw_Da : float
        Molecular weight in Daltons (must be > 0).
    logP : float
        Lipophilicity (log partition coefficient).
    dose_uM : float
        Drug concentration at the transporter site in micromolar (must be > 0).
    ki_ntcp_uM : float, optional
        Known Ki/IC50 for NTCP inhibition (uM). If None, estimated from logP/MW.
    ki_bsep_uM : float, optional
        Known Ki/IC50 for BSEP inhibition (uM). If None, estimated from logP/MW.
    ki_asbt_uM : float, optional
        Known Ki/IC50 for ASBT inhibition (uM). If None, estimated from logP/MW.

    Returns
    -------
    BileAcidDrugInteractionResult
    """
    if dose_uM <= 0:
        raise ValueError(f"dose_uM must be > 0, got {dose_uM}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")

    # IC50 estimation from physicochemical properties if not provided
    if ki_ntcp_uM is not None:
        ic50_ntcp = ki_ntcp_uM
    else:
        ic50_ntcp = max(0.1, 50.0 / (1.0 + logP * mw_Da / 10000.0))

    if ki_bsep_uM is not None:
        ic50_bsep = ki_bsep_uM
    else:
        ic50_bsep = max(0.1, 30.0 / (1.0 + logP * mw_Da / 8000.0))

    if ki_asbt_uM is not None:
        ic50_asbt = ki_asbt_uM
    else:
        ic50_asbt = max(0.1, 80.0 / (1.0 + logP * mw_Da / 12000.0))

    # Inhibition fractions at given dose
    fi_ntcp = dose_uM / (dose_uM + ic50_ntcp)
    fi_bsep = dose_uM / (dose_uM + ic50_bsep)
    fi_asbt = dose_uM / (dose_uM + ic50_asbt)

    # Risk classification based on BSEP inhibition fraction
    if fi_bsep < 0.1:
        inhibition_risk = "low"
    elif fi_bsep < 0.3:
        inhibition_risk = "moderate"
    elif fi_bsep < 0.6:
        inhibition_risk = "high"
    else:
        inhibition_risk = "critical"

    # Fold increase in serum bile acids (capped at 10)
    fold_increase = 1.0 + fi_ntcp * 2.0 + fi_bsep * 3.0
    if fold_increase > 10.0:
        fold_increase = 10.0

    # Cholestasis risk score (0–100)
    cholestasis_risk_score = min(100.0, fi_bsep * 50.0 + fi_ntcp * 30.0 + fi_asbt * 20.0)

    # DILI risk class from cholestasis score
    if cholestasis_risk_score < 20.0:
        dili_risk_class = "negligible"
    elif cholestasis_risk_score < 40.0:
        dili_risk_class = "low"
    elif cholestasis_risk_score < 70.0:
        dili_risk_class = "moderate"
    else:
        dili_risk_class = "high"

    notes = (
        f"NTCP fi={fi_ntcp:.3f}, BSEP fi={fi_bsep:.3f}, ASBT fi={fi_asbt:.3f}. "
        f"Cholestasis risk score={cholestasis_risk_score:.1f}."
    )

    return BileAcidDrugInteractionResult(
        drug_name=drug_name,
        ic50_ntcp_uM=ic50_ntcp,
        ic50_bsep_uM=ic50_bsep,
        ic50_asbt_uM=ic50_asbt,
        inhibition_risk=inhibition_risk,
        fold_increase_bile_acids=fold_increase,
        cholestasis_risk_score=cholestasis_risk_score,
        dili_risk_class=dili_risk_class,
        notes=notes,
    )


def screen_bile_acid_interactions(
    compounds: list[dict[str, Any]],
) -> list[BileAcidDrugInteractionResult]:
    """
    Screen a list of compounds for bile acid transporter interactions.

    Parameters
    ----------
    compounds : list of dict
        Each dict must have keys matching ``predict_bile_acid_interaction`` parameters:
        drug_name, mw_Da, logP, dose_uM, and optionally ki_ntcp_uM, ki_bsep_uM, ki_asbt_uM.

    Returns
    -------
    list of BileAcidDrugInteractionResult
        Sorted by cholestasis_risk_score descending.
    """
    results = []
    for compound in compounds:
        result = predict_bile_acid_interaction(
            drug_name=compound["drug_name"],
            mw_Da=compound["mw_Da"],
            logP=compound["logP"],
            dose_uM=compound["dose_uM"],
            ki_ntcp_uM=compound.get("ki_ntcp_uM"),
            ki_bsep_uM=compound.get("ki_bsep_uM"),
            ki_asbt_uM=compound.get("ki_asbt_uM"),
        )
        results.append(result)

    results.sort(key=lambda r: r.cholestasis_risk_score, reverse=True)
    return results
