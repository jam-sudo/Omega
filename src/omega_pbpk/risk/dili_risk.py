"""Drug-Induced Liver Injury (DILI) Risk Score.

Predicts DILI risk using structural alerts, dose, and metabolic activation
burden, inspired by the DILIrank classification scheme.

References:
    Chen M et al. (2016) Drug-induced liver injury: Interactions between
    drug properties and host factors. J Hepatol 64(2):422-31.
    FDA DILI guidance and DILIrank dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Structural alert definitions: (description, weight, list_of_substrings)
# ---------------------------------------------------------------------------
_ALERT_DEFINITIONS: list[tuple[str, float, list[str]]] = [
    # furan ring: "c1ccoc1" or "o1cccc1"
    ("furan_ring", 3.0, ["c1ccoc1", "o1cccc1"]),
    # hydrazine: "nn" present but not amide-hydrazide "c(=o)nn"
    ("hydrazine", 2.5, ["nn"]),
    # nitro group
    ("nitro_group", 2.0, ["[n+](=o)[o-]", "n(=o)=o", "no2"]),
    # michael acceptor
    ("michael_acceptor", 2.0, ["c=cc=o"]),
    # quinone
    ("quinone", 2.0, ["c1(=o)ccc(=o)"]),
    # epoxide
    ("epoxide", 1.5, ["c1co1"]),
    # aniline
    ("aniline", 1.5, ["nc1ccccc1"]),
    # acyl halide
    ("acyl_halide", 3.0, ["c(=o)cl", "c(=o)br"]),
    # alkyl halide (not acyl)
    ("alkyl_halide", 1.0, ["ccl", "cbr"]),
]

# High-weight alerts that indicate reactive metabolite risk
_RM_WEIGHT_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DILIRiskResult:
    """Result of DILI risk assessment for a single compound."""

    compound_name: str
    smiles: str
    daily_dose_mg: float

    # Structural alerts
    alerts_triggered: list[str]
    n_alerts: int

    # Risk factors
    dose_risk_score: float
    mitochondrial_risk: bool
    bile_acid_transport_risk: bool
    reactive_metabolite_risk: bool

    # Combined score
    dili_risk_score: float
    dili_category: str  # "no_concern", "less_concern", "most_concern"

    # Dose threshold analysis
    dose_threshold_mg: float
    safe_dose_fraction: float

    recommendation: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_alerts(smiles_lower: str) -> list[tuple[str, float]]:
    """Return list of (alert_name, weight) for alerts found in SMILES."""
    triggered: list[tuple[str, float]] = []
    for name, weight, patterns in _ALERT_DEFINITIONS:
        if name == "hydrazine":
            # hydrazine: "nn" present but not an amide-hydrazide context
            if "nn" in smiles_lower and "c(=o)nn" not in smiles_lower:
                triggered.append((name, weight))
        elif name == "alkyl_halide":
            # alkyl halide: ccl or cbr present but NOT acyl context
            has_pattern = any(p in smiles_lower for p in patterns)
            is_acyl = "c(=o)cl" in smiles_lower or "c(=o)br" in smiles_lower
            if has_pattern and not is_acyl:
                triggered.append((name, weight))
        else:
            if any(p in smiles_lower for p in patterns):
                triggered.append((name, weight))
    return triggered


def _dose_threshold_for_category(category: str) -> float:
    if category == "most_concern":
        return 50.0
    if category == "less_concern":
        return 100.0
    return 500.0


def _build_recommendation(
    category: str,
    safe_dose_fraction: float,
    n_alerts: int,
    mitochondrial_risk: bool,
    reactive_metabolite_risk: bool,
) -> str:
    if category == "most_concern":
        parts = ["HIGH DILI concern — close hepatic monitoring required."]
        if safe_dose_fraction > 1.0:
            parts.append(
                f"Daily dose exceeds the {_dose_threshold_for_category(category):.0f} mg "
                "threshold for this risk tier."
            )
        if reactive_metabolite_risk:
            parts.append("Reactive metabolite risk detected; consider trapping studies.")
        if mitochondrial_risk:
            parts.append("Amphiphilic profile suggests mitochondrial liability.")
        return " ".join(parts)

    if category == "less_concern":
        parts = ["Moderate DILI concern — periodic LFT monitoring recommended."]
        if n_alerts > 0:
            parts.append(f"{n_alerts} structural alert(s) detected.")
        return " ".join(parts)

    return "Low DILI concern. Standard safety monitoring is sufficient."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_dili_risk(
    compound_name: str,
    smiles: str,
    daily_dose_mg: float,
    mw: float,
    logP: float,
    fu_plasma: float = 0.5,
    fm_cyp: float = 0.7,
) -> DILIRiskResult:
    """Assess DILI risk for a single compound.

    Parameters
    ----------
    compound_name:
        Human-readable drug name.
    smiles:
        SMILES string (any case; will be lowercased internally).
    daily_dose_mg:
        Total daily dose in milligrams.
    mw:
        Molecular weight (Da).
    logP:
        Octanol-water partition coefficient.
    fu_plasma:
        Fraction unbound in plasma (0-1).
    fm_cyp:
        Fraction metabolised by CYP enzymes (0-1).

    Returns
    -------
    DILIRiskResult
    """
    if daily_dose_mg <= 0:
        raise ValueError(f"daily_dose_mg must be > 0, got {daily_dose_mg}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    smiles_lower = smiles.lower()

    # Structural alerts
    alert_pairs = _detect_alerts(smiles_lower)
    alert_names = [a[0] for a in alert_pairs]
    alert_weights = [a[1] for a in alert_pairs]
    alert_score = sum(alert_weights)

    # Mechanistic risk flags
    mitochondrial_risk: bool = logP > 3 and mw > 300
    bile_acid_transport_risk: bool = logP > 2 and mw > 400
    reactive_metabolite_risk: bool = any(w >= _RM_WEIGHT_THRESHOLD for w in alert_weights)

    # Dose contribution
    dose_risk_score = math.log10(daily_dose_mg)

    # Composite score
    raw_score = (
        alert_score
        + dose_risk_score
        + (1.0 if mitochondrial_risk else 0.0)
        + (0.5 if bile_acid_transport_risk else 0.0)
    )
    dili_risk_score = max(0.0, min(10.0, raw_score))

    # Category
    if dili_risk_score < 2.0:
        dili_category = "no_concern"
    elif dili_risk_score < 5.0:
        dili_category = "less_concern"
    else:
        dili_category = "most_concern"

    # Dose threshold
    dose_threshold_mg = _dose_threshold_for_category(dili_category)
    safe_dose_fraction = daily_dose_mg / dose_threshold_mg

    recommendation = _build_recommendation(
        dili_category,
        safe_dose_fraction,
        len(alert_names),
        mitochondrial_risk,
        reactive_metabolite_risk,
    )

    notes = (
        f"alert_score={alert_score:.2f}, dose_risk={dose_risk_score:.2f}, "
        f"mito={mitochondrial_risk}, bsep={bile_acid_transport_risk}, "
        f"rm={reactive_metabolite_risk}"
    )

    return DILIRiskResult(
        compound_name=compound_name,
        smiles=smiles,
        daily_dose_mg=daily_dose_mg,
        alerts_triggered=alert_names,
        n_alerts=len(alert_names),
        dose_risk_score=dose_risk_score,
        mitochondrial_risk=mitochondrial_risk,
        bile_acid_transport_risk=bile_acid_transport_risk,
        reactive_metabolite_risk=reactive_metabolite_risk,
        dili_risk_score=dili_risk_score,
        dili_category=dili_category,
        dose_threshold_mg=dose_threshold_mg,
        safe_dose_fraction=safe_dose_fraction,
        recommendation=recommendation,
        notes=notes,
    )


def rank_dili_risk(
    compounds: list[dict],
) -> list[DILIRiskResult]:
    """Rank a list of compounds by DILI risk score (descending).

    Parameters
    ----------
    compounds:
        List of dicts with keys: name, smiles, daily_dose_mg, mw, logP.
        Optional keys: fu_plasma, fm_cyp.

    Returns
    -------
    List of DILIRiskResult sorted by dili_risk_score descending.
    """
    results: list[DILIRiskResult] = []
    for c in compounds:
        result = assess_dili_risk(
            compound_name=c["name"],
            smiles=c["smiles"],
            daily_dose_mg=c["daily_dose_mg"],
            mw=c["mw"],
            logP=c["logP"],
            fu_plasma=c.get("fu_plasma", 0.5),
            fm_cyp=c.get("fm_cyp", 0.7),
        )
        results.append(result)

    results.sort(key=lambda r: r.dili_risk_score, reverse=True)
    return results
