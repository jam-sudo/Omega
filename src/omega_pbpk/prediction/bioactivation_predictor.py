"""Drug bioactivation and reactive metabolite prediction — Phase 483."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "BioactivationResult",
    "predict_bioactivation_risk",
    "calculate_daily_bioactivation_burden",
    "screen_compound_series",
]

# ---------------------------------------------------------------------------
# Alert definitions: (alert_name, substrings_to_match, weight)
# ---------------------------------------------------------------------------

_ALERT_DEFINITIONS: list[tuple[str, list[str], float]] = [
    ("furan", ["c1ccoc1", "o1cccc1"], 3.0),
    ("thiophene", ["c1ccsc1", "c1csc"], 2.0),
    ("aniline", ["nc1ccc"], 2.0),
    ("nitro_aromatic", ["c[n+](=o)[o-]", "[n+](=o)"], 3.0),
    ("michael_acceptor", ["c=cc=o", "c=cc(=o)"], 2.0),
    ("quinone_precursor", ["c1cc(o)cc", "oc1ccc"], 2.0),
]

_EPOXIDE_PRECURSOR_ALERT = "epoxide_precursor"
_EPOXIDE_WEIGHT = 1.0

# Risk score → category thresholds
_CATEGORY_THRESHOLDS = [
    (75.0, "very_high"),
    (50.0, "high"),
    (25.0, "moderate"),
    (0.0, "low"),
]

_MITIGATION_BY_ALERT: dict[str, list[str]] = {
    "furan": [
        "Consider replacing furan with less-reactive isosteres (e.g., pyridine, thiazole)",
        "Monitor for covalent adduct formation in vitro",
    ],
    "thiophene": [
        "Evaluate S-oxidation pathway and thiolactone formation",
        "Test for time-dependent CYP inhibition",
    ],
    "aniline": [
        "Screen for methemoglobin formation",
        "Consider N-acetylation or N-methylation to block hydroxylamine formation",
    ],
    "nitro_aromatic": [
        "Assess nitroreductase activity across species",
        "Test for DNA reactivity (Ames assay)",
    ],
    "michael_acceptor": [
        "Measure GSH adduct formation in microsomal incubations",
        "Consider saturating the reactive double bond",
    ],
    "quinone_precursor": [
        "Test for NADPH-dependent covalent binding",
        "Consider methylation or fluorination of the catechol/phenol",
    ],
    "epoxide_precursor": [
        "Measure epoxide hydrolase-mediated detoxification",
        "Assess arene oxide formation for aromatic systems",
    ],
}

_GENERIC_LOW_RISK_MITIGATION = ["Routine safety monitoring during phase I"]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class BioactivationResult:
    """Result from bioactivation risk prediction."""

    compound_name: str
    smiles: str
    daily_dose_mg: float
    structural_alerts: list[str] = field(default_factory=list)
    n_alerts: int = 0
    total_fm_cyp: float = 0.0
    bioactivation_fraction: float = 0.0
    bioactivation_burden: float = 0.0
    risk_score: float = 0.0
    risk_category: str = "low"
    mitigation_strategies: list[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_alerts(
    smiles_lower: str,
    molecular_features: dict | None,
) -> tuple[list[str], float]:
    """Return (alert_names, total_alert_weight) from SMILES string."""
    alerts: list[str] = []
    total_weight = 0.0

    for alert_name, substrings, weight in _ALERT_DEFINITIONS:
        for sub in substrings:
            if sub in smiles_lower:
                alerts.append(alert_name)
                total_weight += weight
                break  # only count each alert once

    # Epoxide precursor: structural-feature-based
    if molecular_features is not None:
        mw = float(molecular_features.get("mw", molecular_features.get("molecular_weight", 500.0)))
        logp = float(molecular_features.get("logp", molecular_features.get("logP", 0.0)))
        if mw < 400.0 and logp > 1.5:
            alerts.append(_EPOXIDE_PRECURSOR_ALERT)
            total_weight += _EPOXIDE_WEIGHT

    return alerts, total_weight


def _categorise(risk_score: float) -> str:
    for threshold, category in _CATEGORY_THRESHOLDS:
        if risk_score >= threshold:
            return category
    return "low"


def _build_mitigation(alerts: list[str]) -> list[str]:
    strategies: list[str] = []
    seen: set[str] = set()
    for alert in alerts:
        for s in _MITIGATION_BY_ALERT.get(alert, []):
            if s not in seen:
                strategies.append(s)
                seen.add(s)
    if not strategies:
        strategies = list(_GENERIC_LOW_RISK_MITIGATION)
    return strategies


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_daily_bioactivation_burden(
    daily_dose_mg: float,
    bioactivation_fraction: float,
    fm_cyp: float,
) -> float:
    """Calculate the daily bioactivation burden score.

    Parameters
    ----------
    daily_dose_mg:
        Total daily dose in milligrams.
    bioactivation_fraction:
        Fraction of dose that undergoes bioactivation (0–1).
    fm_cyp:
        Fraction metabolised by the relevant CYP enzyme (0–1).

    Returns
    -------
    float
        Dimensionless burden score (higher = greater concern).
    """
    if daily_dose_mg <= 0:
        raise ValueError("daily_dose_mg must be > 0")
    if not (0.0 <= bioactivation_fraction <= 1.0):
        raise ValueError("bioactivation_fraction must be in [0, 1]")
    if not (0.0 <= fm_cyp <= 1.0):
        raise ValueError("fm_cyp must be in [0, 1]")

    return bioactivation_fraction * math.log10(daily_dose_mg + 1.0) * fm_cyp


def predict_bioactivation_risk(
    compound_name: str,
    smiles: str,
    daily_dose_mg: float,
    fm_cyp3a4: float = 0.0,
    fm_cyp2d6: float = 0.0,
    fm_cyp1a2: float = 0.0,
    fm_mao: float = 0.0,
    molecular_features: dict | None = None,
) -> BioactivationResult:
    """Predict the bioactivation risk for a compound.

    Parameters
    ----------
    compound_name:
        Human-readable name of the compound.
    smiles:
        SMILES string (case-insensitive matching is applied internally).
    daily_dose_mg:
        Total daily dose in milligrams (must be > 0).
    fm_cyp3a4, fm_cyp2d6, fm_cyp1a2, fm_mao:
        Fraction of dose metabolised by each enzyme (0–1 each).
    molecular_features:
        Optional dict containing ``'mw'``/``'molecular_weight'`` and
        ``'logp'``/``'logP'`` for epoxide-precursor detection.

    Returns
    -------
    BioactivationResult
    """
    # --- Input validation ---
    if daily_dose_mg <= 0:
        raise ValueError("daily_dose_mg must be > 0")
    for name, val in [
        ("fm_cyp3a4", fm_cyp3a4),
        ("fm_cyp2d6", fm_cyp2d6),
        ("fm_cyp1a2", fm_cyp1a2),
        ("fm_mao", fm_mao),
    ]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0, 1]")

    smiles_lower = smiles.lower()
    alerts, alert_weight = _detect_alerts(smiles_lower, molecular_features)

    total_fm_cyp = min(1.0, fm_cyp3a4 + fm_cyp2d6 + fm_cyp1a2 + fm_mao)

    # bioactivation_fraction is a normalised measure of structural alert load
    # capped at 1 so it behaves as a fraction
    bioactivation_fraction = min(1.0, alert_weight / 10.0)

    # burden = alert_weight * log10(dose+1) * total_fm_cyp
    bioactivation_burden = alert_weight * math.log10(daily_dose_mg + 1.0) * total_fm_cyp

    risk_score = min(100.0, bioactivation_burden * 10.0)
    risk_category = _categorise(risk_score)
    mitigation_strategies = _build_mitigation(alerts)

    notes_parts: list[str] = []
    if total_fm_cyp == 0.0 and alerts:
        notes_parts.append(
            "Structural alerts detected but no CYP fm provided; burden may be underestimated."
        )
    if total_fm_cyp > 0.9:
        notes_parts.append("High CYP fraction suggests significant metabolic liability.")
    notes = " ".join(notes_parts)

    return BioactivationResult(
        compound_name=compound_name,
        smiles=smiles,
        daily_dose_mg=daily_dose_mg,
        structural_alerts=alerts,
        n_alerts=len(alerts),
        total_fm_cyp=total_fm_cyp,
        bioactivation_fraction=bioactivation_fraction,
        bioactivation_burden=bioactivation_burden,
        risk_score=risk_score,
        risk_category=risk_category,
        mitigation_strategies=mitigation_strategies,
        notes=notes,
    )


def screen_compound_series(
    compounds: list[dict],
) -> list[BioactivationResult]:
    """Screen a series of compounds and return results sorted by risk_score descending.

    Each dict in ``compounds`` must contain at minimum:
    - ``'compound_name'`` (str)
    - ``'smiles'`` (str)
    - ``'daily_dose_mg'`` (float)

    Optional keys mirror the keyword arguments of :func:`predict_bioactivation_risk`.

    Returns
    -------
    list[BioactivationResult]
        Sorted from highest to lowest risk_score.
    """
    results: list[BioactivationResult] = []
    for c in compounds:
        result = predict_bioactivation_risk(
            compound_name=c["compound_name"],
            smiles=c["smiles"],
            daily_dose_mg=c["daily_dose_mg"],
            fm_cyp3a4=c.get("fm_cyp3a4", 0.0),
            fm_cyp2d6=c.get("fm_cyp2d6", 0.0),
            fm_cyp1a2=c.get("fm_cyp1a2", 0.0),
            fm_mao=c.get("fm_mao", 0.0),
            molecular_features=c.get("molecular_features"),
        )
        results.append(result)
    results.sort(key=lambda r: r.risk_score, reverse=True)
    return results
