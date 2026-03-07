"""Absorption enhancement prediction for poorly permeable drugs — Phase 495."""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Permeability threshold for "good" absorption (cm/s)
_HIGH_PERM_THRESHOLD = 2.0e-4

# Enhancer catalogue: (base_fold_low, base_fold_high, mechanism, reversibility,
#                      concern_base, notes)
_ENHANCER_DATA: dict[str, tuple[float, float, str, str, str, str]] = {
    "SDS": (
        3.0,
        8.0,
        "surfactant_membrane_fluidization",
        "partially_reversible",
        "high",
        "Sodium dodecyl sulfate; strong membrane disruptor; systemic absorption risk.",
    ),
    "sodium_caprate": (
        2.0,
        4.0,
        "tight_junction_opening_lipid_based",
        "reversible",
        "moderate",
        "C10 fatty acid; GRAS status; opens tight junctions transiently.",
    ),
    "chitosan": (
        1.5,
        3.0,
        "bioadhesive_tight_junction_modulation",
        "reversible",
        "low",
        "Cationic polymer; pH-dependent activity; mucoadhesive.",
    ),
    "EDTA": (
        2.0,
        5.0,
        "calcium_chelation_tight_junction_disruption",
        "reversible",
        "moderate",
        "Divalent cation chelator; disrupts paracellular Ca2+-dependent junctions.",
    ),
    "zonula_occludens_toxin": (
        3.0,
        10.0,
        "tight_junction_protein_disassembly",
        "reversible",
        "high",
        "Bacterial toxin; potent TJ opener; immunogenicity risk.",
    ),
    "labrasol": (
        2.0,
        3.0,
        "surfactant_membrane_fluidization",
        "reversible",
        "low",
        "PEG-8 caprylic/capric glycerides; GRAS; mild surfactant.",
    ),
    "bile_salts": (
        1.5,
        2.5,
        "mixed_micelle_solubilization_membrane_perturbation",
        "reversible",
        "low",
        "Endogenous surfactants; improve solubilization and membrane fluidity.",
    ),
}

# Safety concern thresholds by concentration (concentration_pct modifiers)
_CONCERN_CONCENTRATION_THRESHOLDS: dict[str, tuple[float, float]] = {
    # (moderate_threshold_pct, high_threshold_pct)
    "SDS": (0.5, 1.0),
    "sodium_caprate": (1.0, 3.0),
    "chitosan": (2.0, 5.0),
    "EDTA": (0.5, 1.5),
    "zonula_occludens_toxin": (0.1, 0.5),
    "labrasol": (3.0, 8.0),
    "bile_salts": (2.0, 5.0),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnhancerEffectResult:
    """Result from absorption enhancer effect prediction.

    Attributes:
        enhancer: Name of the absorption enhancer.
        concentration_pct: Applied concentration (% w/v).
        papp_enhancement_fold: Fold-increase in apparent permeability.
        predicted_fa_increase_pct: Absolute increase in fraction absorbed (%).
        mechanism: Mechanistic description of enhancement.
        reversibility: Reversibility classification.
        safety_concern: Risk level ('low', 'moderate', 'high').
        notes: Additional clinical or formulation notes.
    """

    enhancer: str
    concentration_pct: float
    papp_enhancement_fold: float
    predicted_fa_increase_pct: float
    mechanism: str
    reversibility: str
    safety_concern: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_enhancer(enhancer: str) -> None:
    """Raise ValueError if enhancer is not in catalogue."""
    if enhancer not in _ENHANCER_DATA:
        valid = ", ".join(sorted(_ENHANCER_DATA))
        raise ValueError(f"Unknown enhancer '{enhancer}'. Valid options: {valid}")


def _drug_mw_modifier(drug_mw: float) -> float:
    """Return a fold-modifier based on drug MW (larger MW → more paracellular benefit).

    Higher MW drugs benefit more from tight-junction openers because they have
    limited transcellular permeability.

    Returns a multiplier in [0.7, 1.4].
    """
    # Sigmoid centred at MW=500; MW<200 → 0.7, MW>900 → 1.4
    mw_norm = (drug_mw - 500.0) / 200.0
    sigmoid = 1.0 / (1.0 + math.exp(-mw_norm))
    return 0.7 + 0.7 * sigmoid


def _drug_logP_modifier(drug_logP: float, mechanism: str) -> float:
    """Return a fold-modifier based on logP and mechanism.

    Hydrophilic drugs (low logP) benefit more from paracellular / surfactant
    enhancers because their transcellular route is already rate-limiting.
    Lipophilic drugs respond less to TJ openers but may respond to membrane
    fluidizers.

    Returns a multiplier in [0.8, 1.3].
    """
    is_paracellular = "tight_junction" in mechanism or "chelation" in mechanism
    is_surfactant = "surfactant" in mechanism or "micelle" in mechanism or "lipid" in mechanism

    if is_paracellular:
        # Low logP → hydrophilic → more paracellular benefit
        # logP < 1 → 1.3, logP > 4 → 0.8
        modifier = 1.3 - 0.5 * min(1.0, max(0.0, (drug_logP - 1.0) / 3.0))
    elif is_surfactant:
        # Moderate logP benefits most from membrane fluidizers
        # logP 1-3 → 1.2, outside → 0.9
        dev = abs(drug_logP - 2.0) / 2.0
        modifier = 1.2 - 0.3 * min(1.0, dev)
    else:
        modifier = 1.0

    return modifier


def _concentration_fold_modifier(concentration_pct: float) -> float:
    """Return fold-modifier for concentration (0.5–2× over concentration range 0.1–5%).

    Uses a square-root relationship (diminishing returns at high concentration).
    """
    ref_conc = 1.0  # % reference
    ratio = max(0.01, concentration_pct) / ref_conc
    return math.sqrt(ratio)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_enhancer_effect(
    drug_papp_cm_s: float,
    drug_mw: float,
    drug_logP: float,
    enhancer: str,
    concentration_pct: float = 1.0,
) -> EnhancerEffectResult:
    """Predict fold-enhancement in apparent permeability from an absorption enhancer.

    Parameters
    ----------
    drug_papp_cm_s:
        Baseline Caco-2 / PAMPA apparent permeability (cm/s).
    drug_mw:
        Drug molecular weight (g/mol).
    drug_logP:
        Drug lipophilicity (logP).
    enhancer:
        Enhancer name (must be one of the supported enhancers).
    concentration_pct:
        Enhancer concentration (% w/v); default 1.0.

    Returns
    -------
    EnhancerEffectResult
        Contains predicted permeability enhancement and safety information.
    """
    if drug_papp_cm_s <= 0:
        raise ValueError("drug_papp_cm_s must be > 0")
    if drug_mw <= 0:
        raise ValueError("drug_mw must be > 0")
    if concentration_pct <= 0:
        raise ValueError("concentration_pct must be > 0")

    _validate_enhancer(enhancer)

    fold_low, fold_high, mechanism, reversibility, concern_base, notes = _ENHANCER_DATA[enhancer]

    # Weighted midpoint enhancement adjusted for MW and logP
    base_fold = (fold_low + fold_high) / 2.0

    mw_mod = _drug_mw_modifier(drug_mw)
    logp_mod = _drug_logP_modifier(drug_logP, mechanism)
    conc_mod = _concentration_fold_modifier(concentration_pct)

    fold = max(1.0, base_fold * mw_mod * logp_mod * conc_mod)

    # Clamp fold to plausible physiological range
    fold = min(fold, fold_high * 2.5)

    # Calculate baseline fa and enhanced fa
    baseline_fa = calculate_enhanced_fa(drug_papp_cm_s, 1.0)
    enhanced_fa = calculate_enhanced_fa(drug_papp_cm_s, fold)
    fa_increase_pct = max(0.0, (enhanced_fa - baseline_fa) * 100.0)

    concern = safety_concern(enhancer, concentration_pct)

    return EnhancerEffectResult(
        enhancer=enhancer,
        concentration_pct=concentration_pct,
        papp_enhancement_fold=round(fold, 4),
        predicted_fa_increase_pct=round(fa_increase_pct, 2),
        mechanism=mechanism,
        reversibility=reversibility,
        safety_concern=concern,
        notes=notes,
    )


def calculate_enhanced_fa(
    baseline_papp_cm_s: float,
    enhancement_fold: float,
    transit_time_h: float = 1.5,
    intestinal_radius_cm: float = 1.75,
) -> float:
    """Estimate fraction absorbed from enhanced Papp using the tube model.

    Uses the cylindrical tube model:
        fa = 1 - exp(-2 * P_eff * t / R)

    where P_eff = baseline_papp_cm_s * enhancement_fold.

    Parameters
    ----------
    baseline_papp_cm_s:
        Unenhanced Papp (cm/s).
    enhancement_fold:
        Fold increase in permeability from enhancer (≥ 1.0).
    transit_time_h:
        Small intestinal transit time (h); default 1.5 h.
    intestinal_radius_cm:
        Effective intestinal radius (cm); default 1.75 cm.

    Returns
    -------
    float
        Fraction absorbed in [0, 1].
    """
    if baseline_papp_cm_s < 0:
        raise ValueError("baseline_papp_cm_s must be >= 0")
    if enhancement_fold < 1.0:
        raise ValueError("enhancement_fold must be >= 1.0")
    if transit_time_h <= 0:
        raise ValueError("transit_time_h must be > 0")
    if intestinal_radius_cm <= 0:
        raise ValueError("intestinal_radius_cm must be > 0")

    p_eff = baseline_papp_cm_s * enhancement_fold
    transit_s = transit_time_h * 3600.0
    exponent = 2.0 * p_eff * transit_s / intestinal_radius_cm
    fa = 1.0 - math.exp(-exponent)
    return min(1.0, max(0.0, fa))


def rank_enhancers(
    drug_papp_cm_s: float,
    drug_mw: float,
    drug_logP: float,
    enhancers: list[str] | None = None,
) -> list[EnhancerEffectResult]:
    """Rank all (or specified) enhancers by predicted fold-enhancement.

    Parameters
    ----------
    drug_papp_cm_s:
        Baseline Papp (cm/s).
    drug_mw:
        Drug MW (g/mol).
    drug_logP:
        Drug logP.
    enhancers:
        List of enhancer names to evaluate; defaults to all 7 catalogue entries.

    Returns
    -------
    list[EnhancerEffectResult]
        Sorted descending by papp_enhancement_fold.
    """
    if enhancers is None:
        enhancers = list(_ENHANCER_DATA.keys())

    results = []
    for enh in enhancers:
        result = predict_enhancer_effect(drug_papp_cm_s, drug_mw, drug_logP, enh)
        results.append(result)

    results.sort(key=lambda r: r.papp_enhancement_fold, reverse=True)
    return results


def safety_concern(enhancer: str, concentration_pct: float) -> str:
    """Return safety concern level ('low', 'moderate', 'high') for an enhancer.

    Concern escalates with concentration per enhancer-specific thresholds.

    Parameters
    ----------
    enhancer:
        Enhancer name.
    concentration_pct:
        Concentration (% w/v).

    Returns
    -------
    str
        'low', 'moderate', or 'high'.
    """
    _validate_enhancer(enhancer)
    _, _, _, _, concern_base, _ = _ENHANCER_DATA[enhancer]
    moderate_thresh, high_thresh = _CONCERN_CONCENTRATION_THRESHOLDS[enhancer]

    # Escalate based on concentration
    if concern_base == "high":
        # Already high at base; escalation doesn't help
        if concentration_pct >= high_thresh:
            return "high"
        elif concentration_pct >= moderate_thresh:
            return "high"
        else:
            return "moderate"
    elif concern_base == "moderate":
        if concentration_pct >= high_thresh:
            return "high"
        elif concentration_pct >= moderate_thresh:
            return "moderate"
        else:
            return "low"
    else:  # base = "low"
        if concentration_pct >= high_thresh:
            return "high"
        elif concentration_pct >= moderate_thresh:
            return "moderate"
        else:
            return "low"
