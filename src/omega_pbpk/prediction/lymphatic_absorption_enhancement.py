"""Phase 1035 — Drug Lymphatic Absorption Enhancement Prediction."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class LymphaticEnhancementResult:
    drug_name: str
    strategy: str
    logp: float
    mw_Da: float
    f_lymph_baseline: float  # fraction via lymphatics without enhancement (0-1)
    f_lymph_enhanced: float  # fraction via lymphatics with strategy (0-1)
    enhancement_ratio: float  # f_enhanced / f_baseline
    portal_bypass_fraction: float  # fraction avoiding portal/first-pass (0-1)
    first_pass_avoidance: bool  # True if portal_bypass_fraction > 0.3
    bioavailability_gain: float  # absolute gain in F (0-1)
    recommended: bool  # True if enhancement_ratio > 2.0
    notes: str


_VALID_STRATEGIES = {"lipid_formulation", "self_emulsifying", "nanoparticle", "prodrug", "none"}
_VALID_IONIZATION = {"neutral", "acidic", "basic", "zwitterion", "amphoteric"}


def predict_lymphatic_enhancement(
    drug_name: str,
    logp: float,
    mw_Da: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    strategy: str = "lipid_formulation",
    long_chain_triglyceride_pct: float = 0.0,
    f_hepatic: float = 0.7,
) -> LymphaticEnhancementResult:
    """Predict the extent to which a lymphatic absorption strategy enhances drug exposure."""
    # Validation
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if not (0.0 <= f_hepatic <= 1.0):
        raise ValueError(f"f_hepatic must be in [0, 1], got {f_hepatic}")
    if not (0.0 <= long_chain_triglyceride_pct <= 100.0):
        raise ValueError(
            f"long_chain_triglyceride_pct must be in [0, 100], got {long_chain_triglyceride_pct}"
        )
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {_VALID_STRATEGIES}, got '{strategy}'")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got '{ionization_type}'"
        )

    # Baseline lymphatic fraction — logP-driven sigmoid
    f_lymph_base = 1.0 / (1.0 + exp(-(logp - 3.0)))
    f_lymph_base = min(0.5, f_lymph_base)

    # Strategy enhancement factor
    if strategy == "lipid_formulation":
        lct_factor = min(2.0, 1.0 + long_chain_triglyceride_pct * 0.02)
        enhancement = 2.0 * lct_factor
    elif strategy == "self_emulsifying":
        enhancement = 3.0 * (1.0 + logp * 0.1)
    elif strategy == "nanoparticle":
        enhancement = 1.5
    elif strategy == "prodrug":
        enhancement = 2.5
    else:  # none
        enhancement = 1.0

    f_lymph_enhanced = min(0.80, f_lymph_base * enhancement)

    # Enhancement ratio
    if f_lymph_base == 0.0:
        enhancement_ratio = 1.0
    else:
        enhancement_ratio = f_lymph_enhanced / f_lymph_base

    # Portal bypass and first-pass avoidance
    portal_bypass_fraction = f_lymph_enhanced
    first_pass_avoidance = portal_bypass_fraction > 0.3

    # Bioavailability gain
    bioavailability_gain = portal_bypass_fraction * (1.0 - f_hepatic)

    recommended = enhancement_ratio > 2.0

    # Build notes
    notes_parts = [
        f"Strategy '{strategy}' applied to {drug_name} (logP={logp:.2f}, MW={mw_Da:.1f} Da).",
        f"Baseline lymphatic fraction: {f_lymph_base:.3f}.",
        f"Enhanced lymphatic fraction: {f_lymph_enhanced:.3f} (ratio {enhancement_ratio:.2f}x).",
    ]
    if first_pass_avoidance:
        notes_parts.append("Significant portal bypass achieved (>30%).")
    if recommended:
        notes_parts.append("Enhancement recommended (>2x improvement).")
    else:
        notes_parts.append("Enhancement may not be clinically meaningful (<2x improvement).")

    return LymphaticEnhancementResult(
        drug_name=drug_name,
        strategy=strategy,
        logp=logp,
        mw_Da=mw_Da,
        f_lymph_baseline=f_lymph_base,
        f_lymph_enhanced=f_lymph_enhanced,
        enhancement_ratio=enhancement_ratio,
        portal_bypass_fraction=portal_bypass_fraction,
        first_pass_avoidance=first_pass_avoidance,
        bioavailability_gain=bioavailability_gain,
        recommended=recommended,
        notes=" ".join(notes_parts),
    )


def compare_strategies(
    drug_name: str,
    logp: float,
    mw_Da: float,
    pka: float = 7.0,
    ionization_type: str = "neutral",
    f_hepatic: float = 0.7,
) -> list:
    """Return LymphaticEnhancementResult for all 5 strategies, sorted by f_lymph_enhanced desc."""
    results = []
    for strategy in _VALID_STRATEGIES:
        result = predict_lymphatic_enhancement(
            drug_name=drug_name,
            logp=logp,
            mw_Da=mw_Da,
            pka=pka,
            ionization_type=ionization_type,
            strategy=strategy,
            long_chain_triglyceride_pct=0.0,
            f_hepatic=f_hepatic,
        )
        results.append(result)

    results.sort(key=lambda r: r.f_lymph_enhanced, reverse=True)
    return results
