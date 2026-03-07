"""Phase 312 — Drug-Induced QT Prolongation Risk Calculator.

Calculates QT prolongation risk from hERG binding affinity, clinical exposure,
and cardiac safety margins for regulatory submissions.

Note: A prior qt_prolongation.py (Phase 64 hERG module) exists in this package;
this module focuses on integrated risk scoring per ICH E14 / regulatory guidance.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_RISK_CATEGORIES = {"high", "moderate", "low"}
VALID_REGULATORY_CONCERNS = {"black_box_warning", "precaution", "none"}


@dataclass(frozen=True)
class QTRiskResult:
    """Result of QT prolongation risk assessment."""

    drug_name: str
    ic50_herg_uM: float
    cmax_free_nM: float
    logp: float
    mw: float
    clinical_dose_mg: float
    ic50_herg_nM: float
    safety_margin: float
    herg_inhibition_pct: float
    delta_qtc_ms_estimate: float
    risk_category: str
    regulatory_concern: str
    thorough_qt_study_required: bool
    notes: str


def _validate_qt_inputs(
    ic50_herg_uM: float,
    cmax_free_nM: float,
    mw: float,
    clinical_dose_mg: float,
) -> None:
    """Validate inputs, raising ValueError on invalid values."""
    if ic50_herg_uM <= 0:
        raise ValueError(f"ic50_herg_uM must be > 0, got {ic50_herg_uM}")
    if cmax_free_nM <= 0:
        raise ValueError(f"cmax_free_nM must be > 0, got {cmax_free_nM}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if clinical_dose_mg <= 0:
        raise ValueError(f"clinical_dose_mg must be > 0, got {clinical_dose_mg}")


def calculate_qt_risk(
    drug_name: str,
    ic50_herg_uM: float,
    cmax_free_nM: float,
    logp: float,
    mw: float,
    clinical_dose_mg: float,
    has_sodium_channel_block: bool,
) -> QTRiskResult:
    """Calculate QT prolongation risk for a drug compound.

    Parameters
    ----------
    drug_name:
        Identifier for the drug compound.
    ic50_herg_uM:
        IC50 for hERG channel inhibition in micromolar (uM). Must be > 0.
    cmax_free_nM:
        Free (unbound) Cmax at clinical dose in nanomolar (nM). Must be > 0.
    logp:
        Octanol-water partition coefficient.
    mw:
        Molecular weight in g/mol. Must be > 0.
    clinical_dose_mg:
        Clinical dose in milligrams. Must be > 0.
    has_sodium_channel_block:
        True if the drug also blocks sodium channels (INa block is cardioprotective).

    Returns
    -------
    QTRiskResult
        Dataclass containing safety margin, estimated delta QTc, risk category,
        regulatory concern flag, and TQT study recommendation.
    """
    _validate_qt_inputs(ic50_herg_uM, cmax_free_nM, mw, clinical_dose_mg)

    # Convert IC50 from uM to nM
    ic50_herg_nM = ic50_herg_uM * 1000.0

    # Safety margin: IC50_nM / Cmax_free_nM (dimensionless)
    safety_margin = ic50_herg_nM / cmax_free_nM

    # hERG inhibition percentage (Hill equation, n=1)
    herg_inhibition_pct = cmax_free_nM / (cmax_free_nM + ic50_herg_nM) * 100.0

    # Baseline delta QTc estimate: 100% inhibition ≈ 50 ms prolongation
    delta_qtc = herg_inhibition_pct * 0.5

    # Apply modifiers
    if has_sodium_channel_block:
        # INa block is protective; reduce delta_qtc by 30%
        delta_qtc *= 0.70

    if logp > 3:
        # High cardiac tissue distribution amplifies effect
        delta_qtc *= 1.2

    if mw < 300:
        # Small molecules penetrate cardiac tissue rapidly
        delta_qtc *= 1.1

    # Risk category determination
    if safety_margin < 10 or delta_qtc > 20:
        risk_category = "high"
    elif safety_margin < 30 or delta_qtc > 10:
        risk_category = "moderate"
    else:
        risk_category = "low"

    # Regulatory concern
    if delta_qtc > 30:
        regulatory_concern = "black_box_warning"
    elif delta_qtc > 10:
        regulatory_concern = "precaution"
    else:
        regulatory_concern = "none"

    # Thorough QT (TQT) study required if any flag triggered
    thorough_qt_study_required = delta_qtc > 10 or safety_margin < 30

    # Build notes
    notes_parts: list[str] = []
    notes_parts.append(f"IC50 hERG = {ic50_herg_nM:.1f} nM; Cmax_free = {cmax_free_nM:.1f} nM.")
    notes_parts.append(f"Safety margin = {safety_margin:.1f}x.")
    notes_parts.append(f"Estimated delta QTc = {delta_qtc:.1f} ms.")
    if has_sodium_channel_block:
        notes_parts.append("INa block applied (30% reduction in delta QTc).")
    if logp > 3:
        notes_parts.append("High logP: cardiac distribution modifier applied (+20%).")
    if mw < 300:
        notes_parts.append("Low MW: rapid penetration modifier applied (+10%).")
    if thorough_qt_study_required:
        notes_parts.append("Thorough QT/QTc study recommended per ICH E14.")
    notes = " ".join(notes_parts)

    return QTRiskResult(
        drug_name=drug_name,
        ic50_herg_uM=ic50_herg_uM,
        cmax_free_nM=cmax_free_nM,
        logp=logp,
        mw=mw,
        clinical_dose_mg=clinical_dose_mg,
        ic50_herg_nM=ic50_herg_nM,
        safety_margin=safety_margin,
        herg_inhibition_pct=herg_inhibition_pct,
        delta_qtc_ms_estimate=delta_qtc,
        risk_category=risk_category,
        regulatory_concern=regulatory_concern,
        thorough_qt_study_required=thorough_qt_study_required,
        notes=notes,
    )


def compare_dose_levels(
    drug_name: str,
    ic50_herg_uM: float,
    cmax_free_list_nM: list[float],
    logp: float,
    mw: float,
    has_sodium_channel_block: bool,
) -> list[QTRiskResult]:
    """Compare QT risk across multiple dose levels (Cmax_free values).

    Returns results sorted ascending by safety_margin (most concerning first).

    Parameters
    ----------
    drug_name:
        Identifier for the drug compound.
    ic50_herg_uM:
        IC50 for hERG channel inhibition in micromolar.
    cmax_free_list_nM:
        List of free Cmax values (nM) corresponding to different dose levels.
    logp:
        Octanol-water partition coefficient.
    mw:
        Molecular weight in g/mol.
    has_sodium_channel_block:
        True if the drug also blocks sodium channels.

    Returns
    -------
    list[QTRiskResult]
        Results sorted ascending by safety_margin (lowest margin = highest concern first).
    """
    results: list[QTRiskResult] = []
    for i, cmax_nM in enumerate(cmax_free_list_nM):
        # Use index+1 as a proxy clinical_dose_mg since actual doses not provided
        result = calculate_qt_risk(
            drug_name=drug_name,
            ic50_herg_uM=ic50_herg_uM,
            cmax_free_nM=cmax_nM,
            logp=logp,
            mw=mw,
            clinical_dose_mg=float(i + 1),
            has_sodium_channel_block=has_sodium_channel_block,
        )
        results.append(result)

    results.sort(key=lambda r: r.safety_margin)
    return results
