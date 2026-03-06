"""Comprehensive cardiac safety scoring beyond hERG channel inhibition.

Covers:
- QTc prolongation risk (hERG IC50, concentration, logP, dose)
- Negative inotropy (cardiac contractility reduction)
- Structural alerts from SMILES pattern matching
- Overall cardiac risk classification with mitigation advice
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CardiacSafetyResult", "assess_cardiac_safety"]


@dataclass(frozen=True)
class CardiacSafetyResult:
    """Result of a comprehensive cardiac safety assessment."""

    drug_name: str
    smiles: str
    herg_ic50_uM: float
    concentration_uM: float
    qt_risk_score: float       # 0–100
    qt_risk_category: str      # low / moderate / high / critical
    delta_qtc_ms: float        # estimated QTc prolongation (ms)
    inotropy_risk: str         # low / moderate / high
    overall_cardiac_risk: str  # low / moderate / high / critical
    structural_alerts: list[str]
    mitigation: list[str]
    notes: str


# ── Structural alert patterns (substring match against SMILES) ────────────────

_STRUCTURAL_ALERT_PATTERNS: list[tuple[str, str]] = [
    ("anthraquinone", "C(=O)c1"),          # quinone carbonyl attached to aromatic ring
    ("tertiary_amine_basic", "N(C)(C)C"),  # fully methylated tertiary amine
    ("biguanide", "NC(=N)N"),              # biguanide scaffold
]


def _detect_structural_alerts(smiles: str) -> list[str]:
    """Return list of structural alert names found in *smiles*."""
    alerts: list[str] = []
    for alert_name, pattern in _STRUCTURAL_ALERT_PATTERNS:
        if pattern in smiles:
            alerts.append(alert_name)
    return alerts


def _qt_risk_category(qt_risk_score: float) -> str:
    """Map 0-100 qt_risk_score to a risk category string."""
    if qt_risk_score < 20:
        return "low"
    if qt_risk_score < 40:
        return "moderate"
    if qt_risk_score < 70:
        return "high"
    return "critical"


def _inotropy_risk(logP: float, mw: float) -> str:
    """Estimate negative inotropy risk from physicochemical properties."""
    if logP > 3 and mw > 350:
        return "moderate"
    return "low"


def _overall_risk(
    qt_risk_category: str,
    inotropy_risk: str,
    structural_alerts: list[str],
) -> str:
    """Aggregate QT, inotropy, and structural alert risks into one category."""
    _order = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
    _invert = {v: k for k, v in _order.items()}

    level = _order[qt_risk_category]

    # Inotropy bump
    inot_level = _order[inotropy_risk]
    level = max(level, inot_level)

    # Each structural alert bumps risk by one tier (capped at critical)
    level = min(level + len(structural_alerts), 3)

    return _invert[level]


def assess_cardiac_safety(
    drug_name: str,
    smiles: str,
    herg_ic50_uM: float,
    concentration_uM: float,
    logP: float = 2.0,
    mw: float = 400.0,
    daily_dose_mg: float = 100.0,
) -> CardiacSafetyResult:
    """Perform a comprehensive cardiac safety assessment.

    Parameters
    ----------
    drug_name : str
        Human-readable drug identifier.
    smiles : str
        SMILES string of the compound (used for structural alert detection).
    herg_ic50_uM : float
        hERG channel IC50 (µM). Must be > 0.
    concentration_uM : float
        Free plasma concentration (µM) for inhibition calculation.
    logP : float
        Octanol/water partition coefficient.
    mw : float
        Molecular weight (g/mol).
    daily_dose_mg : float
        Total daily dose (mg), used in QT risk composite score.

    Returns
    -------
    CardiacSafetyResult
    """
    # ── Validation ───────────────────────────────────────────────────────────
    if herg_ic50_uM <= 0:
        raise ValueError(f"herg_ic50_uM must be > 0, got {herg_ic50_uM}")
    if concentration_uM < 0:
        raise ValueError(f"concentration_uM must be >= 0, got {concentration_uM}")
    if daily_dose_mg < 0:
        raise ValueError(f"daily_dose_mg must be >= 0, got {daily_dose_mg}")

    # ── hERG inhibition (Hill equation, Hill coefficient = 1) ────────────────
    herg_inhibition_pct = 100.0 * concentration_uM / (concentration_uM + herg_ic50_uM)

    # ── QTc model ────────────────────────────────────────────────────────────
    # Rough empirical: 80% inhibition → ~64 ms prolongation
    delta_qtc_ms = herg_inhibition_pct * 0.8

    qt_risk_score = min(100.0, herg_inhibition_pct + logP * 2.0 + daily_dose_mg / 100.0)
    qt_risk_cat = _qt_risk_category(qt_risk_score)

    # ── Inotropy ─────────────────────────────────────────────────────────────
    inot_risk = _inotropy_risk(logP, mw)

    # ── Structural alerts ─────────────────────────────────────────────────────
    alerts = _detect_structural_alerts(smiles)

    # ── Overall risk ─────────────────────────────────────────────────────────
    overall = _overall_risk(qt_risk_cat, inot_risk, alerts)

    # ── Mitigation ───────────────────────────────────────────────────────────
    mitigation: list[str] = []
    if qt_risk_cat in ("high", "critical"):
        mitigation.append("Comprehensive cardiac monitoring required")
    if herg_inhibition_pct > 50:
        mitigation.append("Evaluate safety pharmacology studies")

    notes = (
        f"hERG inhibition = {herg_inhibition_pct:.1f}% at {concentration_uM} µM "
        f"(IC50 = {herg_ic50_uM} µM). "
        f"delta_QTc = {delta_qtc_ms:.1f} ms. "
        f"logP={logP}, MW={mw}, daily_dose={daily_dose_mg} mg."
    )

    return CardiacSafetyResult(
        drug_name=drug_name,
        smiles=smiles,
        herg_ic50_uM=herg_ic50_uM,
        concentration_uM=concentration_uM,
        qt_risk_score=qt_risk_score,
        qt_risk_category=qt_risk_cat,
        delta_qtc_ms=delta_qtc_ms,
        inotropy_risk=inot_risk,
        overall_cardiac_risk=overall,
        structural_alerts=alerts,
        mitigation=mitigation,
        notes=notes,
    )
