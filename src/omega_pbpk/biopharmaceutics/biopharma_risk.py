"""Biopharmaceutics risk assessment for oral drug development — Phase 506."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BiopharmRiskResult",
    "assess_biopharmaceutics_risk",
    "solubility_risk_score",
    "permeability_risk_score",
    "absorption_risk_index",
    "formulation_strategy_from_risk",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Permeability thresholds (cm/s) — based on Caco-2 Papp
_PAPP_HIGH_THRESHOLD = 2e-5  # > 2e-5 → high permeability / low risk
_PAPP_MOD_THRESHOLD = 0.5e-5  # 0.5e-5 – 2e-5 → moderate risk

# Dose-number thresholds
_DN_LOW_MAX = 1.0
_DN_MOD_MAX = 5.0

# GI volume default
_GI_VOLUME_DEFAULT_ML = 250.0

# Risk score breakpoints
_RISK_LOW_MAX = 30.0
_RISK_MOD_MAX = 55.0
_RISK_HIGH_MAX = 75.0


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BiopharmRiskResult:
    """Biopharmaceutics risk assessment result for an oral drug candidate."""

    drug_name: str
    bcs_class: str  # 'I', 'II', 'III', 'IV'
    solubility_risk_score: float  # 0–100, higher = riskier
    permeability_risk_score: float  # 0–100
    absorption_risk_index: float  # weighted composite 0–100
    dose_number: float
    overall_risk: str  # 'low', 'moderate', 'high', 'very_high'
    formulation_strategy: str
    clinical_relevance: str
    notes: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _bcs_class(high_sol: bool, high_perm: bool) -> str:
    """Return BCS class string from solubility and permeability flags."""
    if high_sol and high_perm:
        return "I"
    if not high_sol and high_perm:
        return "II"
    if high_sol and not high_perm:
        return "III"
    return "IV"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solubility_risk_score(
    solubility_mg_mL: float,
    dose_mg: float,
    gi_volume_mL: float = _GI_VOLUME_DEFAULT_ML,
) -> float:
    """Calculate solubility-based biopharmaceutics risk score (0–100).

    Uses the Dose Number: Dn = dose_mg / (solubility_mg_mL * gi_volume_mL).

    Risk mapping:
      Dn < 1   → score 0–20   (low risk)
      1 ≤ Dn < 5 → score 20–60 (moderate risk, linear interpolation)
      Dn ≥ 5   → score 60–100 (high risk, capped at 100)

    Parameters
    ----------
    solubility_mg_mL:
        Equilibrium solubility at intestinal pH (mg/mL).
    dose_mg:
        Administered dose (mg).
    gi_volume_mL:
        GI luminal volume for dissolution (mL); default 250 mL.

    Returns
    -------
    float
        Risk score 0–100.
    """
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if gi_volume_mL <= 0:
        raise ValueError("gi_volume_mL must be > 0")

    dn = dose_mg / (solubility_mg_mL * gi_volume_mL)

    if dn < _DN_LOW_MAX:
        # Linear: 0 → 20 as Dn goes 0 → 1
        return 20.0 * dn
    if dn < _DN_MOD_MAX:
        # Linear: 20 → 60 as Dn goes 1 → 5
        return 20.0 + 40.0 * (dn - 1.0) / (_DN_MOD_MAX - _DN_LOW_MAX)
    # High risk: 60 → 100 as Dn goes 5 → 20, capped at 100
    excess = dn - _DN_MOD_MAX
    score = 60.0 + 40.0 * excess / 15.0
    return min(100.0, score)


def permeability_risk_score(papp_cm_s: float) -> float:
    """Calculate permeability-based biopharmaceutics risk score (0–100).

    Risk mapping:
      Papp > 2e-5 cm/s    → low risk   (score 0–20)
      0.5e-5 – 2e-5 cm/s → moderate   (score 20–60, linear)
      < 0.5e-5 cm/s      → high risk  (score 60–100)

    Parameters
    ----------
    papp_cm_s:
        Apparent permeability coefficient (cm/s).

    Returns
    -------
    float
        Risk score 0–100.
    """
    if papp_cm_s < 0:
        raise ValueError("papp_cm_s must be >= 0")
    if papp_cm_s == 0:
        return 100.0

    if papp_cm_s >= _PAPP_HIGH_THRESHOLD:
        # Linear: 0 → 20 as Papp decreases from some high value to 2e-5
        # At 2e-5 score = 20; at 4e-5+ score approaches 0
        ratio = _PAPP_HIGH_THRESHOLD / papp_cm_s
        return 20.0 * ratio
    if papp_cm_s >= _PAPP_MOD_THRESHOLD:
        # Linear: 20 → 60 as Papp drops from 2e-5 to 0.5e-5
        span = _PAPP_HIGH_THRESHOLD - _PAPP_MOD_THRESHOLD
        return 20.0 + 40.0 * (_PAPP_HIGH_THRESHOLD - papp_cm_s) / span
    # Below 0.5e-5: 60 → 100 as Papp → 0
    ratio = papp_cm_s / _PAPP_MOD_THRESHOLD
    return 100.0 - 40.0 * ratio


def absorption_risk_index(
    sol_score: float,
    perm_score: float,
    weights: tuple[float, float] = (0.6, 0.4),
) -> float:
    """Weighted composite absorption risk index.

    Parameters
    ----------
    sol_score:
        Solubility risk score (0–100).
    perm_score:
        Permeability risk score (0–100).
    weights:
        (w_sol, w_perm) weights that must sum to 1.0.

    Returns
    -------
    float
        Weighted composite risk index (0–100).
    """
    w_sol, w_perm = weights
    if abs(w_sol + w_perm - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1.0")
    return w_sol * sol_score + w_perm * perm_score


def formulation_strategy_from_risk(
    risk_score: float,
    logP: float,
    papp_cm_s: float,
    solubility_mg_mL: float,
) -> str:
    """Recommend a formulation strategy based on risk profile.

    Parameters
    ----------
    risk_score:
        Overall absorption risk index (0–100).
    logP:
        Lipophilicity (log partition coefficient).
    papp_cm_s:
        Apparent permeability (cm/s).
    solubility_mg_mL:
        Equilibrium solubility (mg/mL).

    Returns
    -------
    str
        Recommended formulation strategy.
    """
    high_perm = papp_cm_s >= _PAPP_HIGH_THRESHOLD
    low_sol = solubility_mg_mL < 0.1

    if risk_score < _RISK_LOW_MAX:
        return "Standard oral formulation (tablet/capsule); no special strategy required"

    if risk_score < _RISK_MOD_MAX:
        if not high_perm:
            return "Permeation enhancers or tight-junction modulators; consider prodrug approach"
        return (
            "Optimise particle size (micronisation); controlled-release to extend absorption window"
        )

    if risk_score < _RISK_HIGH_MAX:
        if low_sol and logP > 2:
            return (
                "Lipid-based drug delivery system (LBDDS/SEDDS); "
                "solid dispersion or amorphous formulation"
            )
        if low_sol:
            return (
                "Salt selection; co-crystal engineering; nanosuspension or nanocrystal technology"
            )
        return "Absorption enhancement; consider bioadhesive or mucoadhesive delivery system"

    # Very high risk
    if not high_perm and low_sol:
        return (
            "Comprehensive re-engineering required: combinatorial approach — "
            "nanocrystal + permeation enhancer + solubiliser; "
            "evaluate prodrug or alternative route of administration"
        )
    if not high_perm:
        return (
            "Permeation enhancer + mucoadhesive system; "
            "consider cyclic peptide or cell-penetrating peptide carrier"
        )
    return (
        "Advanced solubilisation: amorphous solid dispersion (ASD) + "
        "polymer carrier (HPMC-AS/PVP-VA); hot-melt extrusion or spray-drying"
    )


def assess_biopharmaceutics_risk(
    drug_name: str,
    logP: float,
    mw: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    solubility_mg_mL: float = 1.0,
    papp_cm_s: float = 1e-5,
    dose_mg: float = 100.0,
    target_dose_number_max: float = 1.0,
) -> BiopharmRiskResult:
    """Full biopharmaceutics risk assessment for an oral drug candidate.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logP:
        Lipophilicity (log octanol-water partition coefficient).
    mw:
        Molecular weight (Da).
    pka:
        Ionisation pKa; None for neutral compounds.
    drug_type:
        'acid', 'base', or 'neutral'.
    solubility_mg_mL:
        Equilibrium aqueous solubility at intestinal pH (mg/mL).
    papp_cm_s:
        Caco-2 apparent permeability coefficient (cm/s).
    dose_mg:
        Administered dose (mg).
    target_dose_number_max:
        Acceptable maximum dose number (dimensionless); used in notes.

    Returns
    -------
    BiopharmRiskResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if papp_cm_s < 0:
        raise ValueError("papp_cm_s must be >= 0")
    if mw <= 0:
        raise ValueError("mw must be > 0")

    # Dose number
    dn = dose_mg / (solubility_mg_mL * _GI_VOLUME_DEFAULT_ML)

    # BCS classification
    high_sol = dn < 1.0
    high_perm = papp_cm_s >= _PAPP_HIGH_THRESHOLD
    bcs = _bcs_class(high_sol, high_perm)

    # Risk scores
    sol_score = solubility_risk_score(solubility_mg_mL, dose_mg)
    perm_score = permeability_risk_score(papp_cm_s)
    ari = absorption_risk_index(sol_score, perm_score)

    # Overall risk classification
    if ari < _RISK_LOW_MAX:
        overall = "low"
    elif ari < _RISK_MOD_MAX:
        overall = "moderate"
    elif ari < _RISK_HIGH_MAX:
        overall = "high"
    else:
        overall = "very_high"

    # Formulation strategy
    strategy = formulation_strategy_from_risk(ari, logP, papp_cm_s, solubility_mg_mL)

    # Clinical relevance
    _clinical_map = {
        "I": (
            "BCS Class I: high solubility & permeability — "
            "good oral bioavailability expected; biowaiver likely eligible"
        ),
        "II": (
            "BCS Class II: low solubility, high permeability — "
            "dissolution-rate limited absorption; food effect likely"
        ),
        "III": (
            "BCS Class III: high solubility, low permeability — "
            "permeability-limited absorption; efflux transporters may restrict bioavailability"
        ),
        "IV": (
            "BCS Class IV: low solubility & permeability — "
            "poor and variable oral bioavailability; high development risk"
        ),
    }
    clinical_relevance = _clinical_map[bcs]

    # Notes
    notes_parts = [
        f"Dose Number (Dn) = {dn:.2f}",
        f"BCS class = {bcs}",
        f"logP = {logP:.1f}",
        f"MW = {mw:.1f} Da",
    ]
    if pka is not None:
        notes_parts.append(f"pKa = {pka:.1f} ({drug_type})")
    if dn > target_dose_number_max:
        notes_parts.append(
            f"Dn exceeds target threshold ({target_dose_number_max:.1f}); "
            "solubility enhancement needed"
        )
    notes = "; ".join(notes_parts)

    return BiopharmRiskResult(
        drug_name=drug_name,
        bcs_class=bcs,
        solubility_risk_score=sol_score,
        permeability_risk_score=perm_score,
        absorption_risk_index=ari,
        dose_number=dn,
        overall_risk=overall,
        formulation_strategy=strategy,
        clinical_relevance=clinical_relevance,
        notes=notes,
    )
