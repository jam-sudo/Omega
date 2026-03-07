"""Drug safety margin computation relative to therapeutic concentrations."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SafetyMarginResult",
    "compute_safety_margin",
    "classify_therapeutic_window",
    "screen_safety_margins",
]


@dataclass(frozen=True)
class SafetyMarginResult:
    drug_name: str
    therapeutic_cmax_mg_L: float
    toxic_cmax_mg_L: float
    therapeutic_index: float
    safety_margin_fold: float
    auc_ratio: float
    mec_mg_L: float
    mtc_mg_L: float
    therapeutic_window_width: float
    narrow_therapeutic_window: bool
    risk_category: str
    dose_adjustment_sensitivity: str
    notes: str


def _risk_category(ti: float) -> str:
    if ti < 2.0:
        return "very_high"
    if ti < 5.0:
        return "high"
    if ti < 10.0:
        return "moderate"
    return "low"


def _dose_sensitivity(ti: float) -> str:
    if ti < 2.0:
        return "very_sensitive"
    if ti < 5.0:
        return "sensitive"
    if ti < 10.0:
        return "moderate"
    return "robust"


def compute_safety_margin(
    drug_name: str,
    therapeutic_cmax_mg_L: float,
    toxic_cmax_mg_L: float,
    mec_mg_L: float,
    mtc_mg_L: float,
    therapeutic_auc: float | None = None,
    toxic_auc: float | None = None,
) -> SafetyMarginResult:
    """Compute safety margin metrics for a drug.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    therapeutic_cmax_mg_L:
        Peak concentration at therapeutic dose (mg/L).
    toxic_cmax_mg_L:
        Concentration causing adverse effects (mg/L). Must exceed therapeutic.
    mec_mg_L:
        Minimum effective concentration (mg/L).
    mtc_mg_L:
        Maximum tolerated concentration (mg/L). Must exceed mec.
    therapeutic_auc:
        AUC at therapeutic dose (optional).
    toxic_auc:
        AUC at toxic dose (optional).

    Returns
    -------
    SafetyMarginResult

    Raises
    ------
    ValueError
        On invalid inputs.
    """
    if therapeutic_cmax_mg_L <= 0:
        raise ValueError("therapeutic_cmax_mg_L must be > 0")
    if toxic_cmax_mg_L <= therapeutic_cmax_mg_L:
        raise ValueError("toxic_cmax_mg_L must be > therapeutic_cmax_mg_L")
    if mec_mg_L <= 0:
        raise ValueError("mec_mg_L must be > 0")
    if mtc_mg_L <= mec_mg_L:
        raise ValueError("mtc_mg_L must be > mec_mg_L")

    ti = toxic_cmax_mg_L / therapeutic_cmax_mg_L
    safety_margin_fold = toxic_cmax_mg_L / therapeutic_cmax_mg_L

    if therapeutic_auc is not None and toxic_auc is not None:
        auc_ratio = toxic_auc / therapeutic_auc
    else:
        auc_ratio = 1.0

    window_width = (mtc_mg_L - mec_mg_L) / mec_mg_L * 100.0
    narrow = ti < 2.0 or window_width < 100.0

    risk = _risk_category(ti)
    sensitivity = _dose_sensitivity(ti)

    notes_parts = [
        f"TI={ti:.2f}",
        f"window={window_width:.1f}%",
        f"risk={risk}",
    ]
    if narrow:
        notes_parts.append("NARROW therapeutic window — close monitoring required")
    notes = "; ".join(notes_parts)

    return SafetyMarginResult(
        drug_name=drug_name,
        therapeutic_cmax_mg_L=therapeutic_cmax_mg_L,
        toxic_cmax_mg_L=toxic_cmax_mg_L,
        therapeutic_index=ti,
        safety_margin_fold=safety_margin_fold,
        auc_ratio=auc_ratio,
        mec_mg_L=mec_mg_L,
        mtc_mg_L=mtc_mg_L,
        therapeutic_window_width=window_width,
        narrow_therapeutic_window=narrow,
        risk_category=risk,
        dose_adjustment_sensitivity=sensitivity,
        notes=notes,
    )


def classify_therapeutic_window(
    mec_mg_L: float,
    mtc_mg_L: float,
) -> dict:
    """Classify the therapeutic window width.

    Parameters
    ----------
    mec_mg_L:
        Minimum effective concentration (mg/L).
    mtc_mg_L:
        Maximum tolerated concentration (mg/L).

    Returns
    -------
    dict with keys: window_width_pct, classification, notes
    """
    window_width_pct = (mtc_mg_L - mec_mg_L) / mec_mg_L * 100.0

    if window_width_pct < 100.0:
        classification = "narrow"
        notes = "Narrow therapeutic window; careful dose titration required."
    elif window_width_pct <= 400.0:
        classification = "moderate"
        notes = "Moderate therapeutic window; standard monitoring recommended."
    else:
        classification = "wide"
        notes = "Wide therapeutic window; dose flexibility is high."

    return {
        "window_width_pct": window_width_pct,
        "classification": classification,
        "notes": notes,
    }


def screen_safety_margins(compounds: list[dict]) -> list[SafetyMarginResult]:
    """Screen multiple compounds and sort by therapeutic index ascending (most dangerous first).

    Each dict in compounds must contain keys accepted by compute_safety_margin:
    drug_name, therapeutic_cmax_mg_L, toxic_cmax_mg_L, mec_mg_L, mtc_mg_L,
    and optionally therapeutic_auc, toxic_auc.

    Parameters
    ----------
    compounds:
        List of parameter dicts.

    Returns
    -------
    list[SafetyMarginResult] sorted by therapeutic_index ascending.
    """
    if not compounds:
        return []

    results = []
    for c in compounds:
        result = compute_safety_margin(
            drug_name=c["drug_name"],
            therapeutic_cmax_mg_L=c["therapeutic_cmax_mg_L"],
            toxic_cmax_mg_L=c["toxic_cmax_mg_L"],
            mec_mg_L=c["mec_mg_L"],
            mtc_mg_L=c["mtc_mg_L"],
            therapeutic_auc=c.get("therapeutic_auc"),
            toxic_auc=c.get("toxic_auc"),
        )
        results.append(result)

    results.sort(key=lambda r: r.therapeutic_index)
    return results
