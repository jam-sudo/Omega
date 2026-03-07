"""Oral absorption enhancement prediction for BCS II/IV drugs.

Predicts the fold-improvement in oral absorption achievable through formulation
strategies:
  - Nanosizing: particle size reduction increases dissolution rate
  - Solid dispersion (ASD): amorphous solid dispersion increases apparent solubility
  - Lipid formulation: SEDDS/SMEDDS enhances solubility and lymphatic uptake
  - Cyclodextrin complexation: inclusion complexes increase apparent solubility
  - Cocrystal: pharmaceutical cocrystals improve solubility

References:
  - Butler & Dressman (2010). The developability classification system. J Pharm Sci.
  - Pouton (2006). Formulation of poorly water-soluble drugs for oral administration.
    Eur J Pharm Sci.
  - Sun et al. (2004). Peff-solubility relationship for oral absorption.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PEFF_HIGH_THRESHOLD = 2.0e-4  # cm/s — BCS high permeability cutoff
_DN_HIGH_SOL_THRESHOLD = 1.0   # dimensionless dose number cutoff

# BCS-based baseline fa values
_BASELINE_FA = {
    "I": 0.90,
    "III": 0.40,
}

# Strategy names (canonical)
_STRATEGY_NANOSIZING = "nanosizing"
_STRATEGY_SOLID_DISPERSION = "solid_dispersion"
_STRATEGY_LIPID = "lipid_formulation"
_STRATEGY_CYCLODEXTRIN = "cyclodextrin"
_STRATEGY_COCRYSTAL = "cocrystal"

_ALL_STRATEGIES = [
    _STRATEGY_NANOSIZING,
    _STRATEGY_SOLID_DISPERSION,
    _STRATEGY_LIPID,
    _STRATEGY_CYCLODEXTRIN,
    _STRATEGY_COCRYSTAL,
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbsorptionEnhancementResult:
    """Results of absorption enhancement strategy prediction."""

    drug_name: str
    bcs_class: str  # "I", "II", "III", "IV"

    baseline_fa: float  # Fraction absorbed with conventional formulation

    # Per-strategy enhanced fa
    fa_nanosizing: float
    fa_solid_dispersion: float
    fa_lipid_formulation: float
    fa_cyclodextrin: float
    fa_cocrystal: float

    # Best strategy
    best_strategy: str
    best_fa: float
    fold_improvement: float  # best_fa / baseline_fa

    # Solubility enhancement factors
    solubility_enhancement_nanosizing: float  # Fold improvement from nanosizing
    solubility_enhancement_asd: float  # ASD solubility enhancement

    recommendation: str
    notes: str


# ---------------------------------------------------------------------------
# BCS classification helpers
# ---------------------------------------------------------------------------


def _classify_bcs(
    peff_cm_per_s: float,
    dose_mg: float,
    solubility_mg_mL: float,
    dose_volume_mL: float,
) -> tuple[str, float]:
    """Classify BCS class and return (class_str, dose_number)."""
    dose_number = dose_mg / (solubility_mg_mL * dose_volume_mL)
    high_perm = peff_cm_per_s > _PEFF_HIGH_THRESHOLD
    high_sol = dose_number < _DN_HIGH_SOL_THRESHOLD

    if high_perm and high_sol:
        bcs = "I"
    elif high_perm and not high_sol:
        bcs = "II"
    elif not high_perm and high_sol:
        bcs = "III"
    else:
        bcs = "IV"

    return bcs, dose_number


def _baseline_fa(bcs_class: str, dose_number: float) -> float:
    """Compute baseline fraction absorbed for given BCS class and Dn."""
    if bcs_class == "I":
        return 0.90
    if bcs_class == "II":
        return min(0.85, max(0.05, 0.5 / dose_number))
    if bcs_class == "III":
        return 0.40
    # BCS IV
    return min(0.30, max(0.03, 0.15 / dose_number))


# ---------------------------------------------------------------------------
# Enhancement calculators
# ---------------------------------------------------------------------------


def _calc_nanosizing(
    bcs_class: str,
    logP: float,
    baseline: float,
) -> tuple[float, float]:
    """Return (fa_nanosizing, sol_enhancement_factor)."""
    # Empirical solubility enhancement factor based on logP
    sol_factor = min(10.0, 2.0 + logP * 0.5)
    sol_factor = max(1.0, sol_factor)  # at least 1x

    if bcs_class in ("II", "IV"):
        fa = min(0.95, baseline * sol_factor)
    else:
        # Permeability limited: nanosizing doesn't help
        fa = baseline

    return fa, sol_factor


def _calc_solid_dispersion(
    bcs_class: str,
    logP: float,
    baseline: float,
) -> tuple[float, float]:
    """Return (fa_asd, sol_enhancement_factor)."""
    # ASD: solubility enhancement 10-100x for lipophilic drugs
    sol_factor = min(100.0, 10.0 ** (logP * 0.5))
    sol_factor = max(1.0, sol_factor)

    if bcs_class == "II":
        fa = min(0.90, baseline + 0.30)
    elif bcs_class == "IV":
        fa = min(0.60, baseline + 0.20)
    else:
        # BCS I/III: minimal benefit for permeability-limited
        fa = baseline

    return fa, sol_factor


def _calc_lipid(bcs_class: str, baseline: float) -> float:
    """Return fa_lipid."""
    if bcs_class in ("II", "IV"):
        return min(0.90, baseline + 0.25)
    else:
        return min(1.0, baseline + 0.05)


def _calc_cyclodextrin(bcs_class: str, baseline: float) -> float:
    """Return fa_cyclodextrin."""
    if bcs_class in ("II", "IV"):
        return min(0.85, baseline + 0.20)
    else:
        return baseline


def _calc_cocrystal(bcs_class: str, baseline: float) -> float:
    """Return fa_cocrystal."""
    if bcs_class in ("II", "IV"):
        return min(0.80, baseline + 0.15)
    else:
        return baseline


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------


def predict_absorption_enhancement(
    drug_name: str,
    logP: float,
    mw: float,
    peff_cm_per_s: float,
    solubility_mg_mL: float,
    dose_mg: float,
    dose_volume_mL: float = 250.0,
) -> AbsorptionEnhancementResult:
    """Predict the fold-improvement in oral absorption via formulation strategies.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    logP:
        Octanol-water partition coefficient (can be any float).
    mw:
        Molecular weight (g/mol). Must be > 0.
    peff_cm_per_s:
        Effective intestinal permeability (cm/s). Must be > 0.
    solubility_mg_mL:
        Aqueous solubility (mg/mL). Must be > 0.
    dose_mg:
        Administered dose (mg). Must be > 0.
    dose_volume_mL:
        Reference GI volume for dose number calculation (mL). Default 250 mL.

    Returns
    -------
    AbsorptionEnhancementResult
    """
    # Input validation
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if peff_cm_per_s <= 0:
        raise ValueError(f"peff_cm_per_s must be > 0, got {peff_cm_per_s}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")

    # BCS classification
    bcs_class, dose_number = _classify_bcs(
        peff_cm_per_s, dose_mg, solubility_mg_mL, dose_volume_mL
    )

    # Baseline fa
    baseline = _baseline_fa(bcs_class, dose_number)

    # Per-strategy enhanced fa
    fa_nano, sol_nano = _calc_nanosizing(bcs_class, logP, baseline)
    fa_asd, sol_asd = _calc_solid_dispersion(bcs_class, logP, baseline)
    fa_lipid = _calc_lipid(bcs_class, baseline)
    fa_cd = _calc_cyclodextrin(bcs_class, baseline)
    fa_cc = _calc_cocrystal(bcs_class, baseline)

    # Best strategy
    strategy_fas = {
        _STRATEGY_NANOSIZING: fa_nano,
        _STRATEGY_SOLID_DISPERSION: fa_asd,
        _STRATEGY_LIPID: fa_lipid,
        _STRATEGY_CYCLODEXTRIN: fa_cd,
        _STRATEGY_COCRYSTAL: fa_cc,
    }
    best_strategy = max(strategy_fas, key=lambda s: strategy_fas[s])
    best_fa = strategy_fas[best_strategy]

    fold_improvement = best_fa / max(0.01, baseline)
    fold_improvement = max(1.0, fold_improvement)

    # Recommendation
    if bcs_class == "I":
        recommendation = (
            f"{drug_name} is BCS Class I (high solubility, high permeability). "
            "Formulation strategies provide minimal benefit; conventional tablet/capsule adequate."
        )
    elif bcs_class == "II":
        recommendation = (
            f"{drug_name} is BCS Class II (dissolution-limited). "
            f"Recommended strategy: {best_strategy} (predicted fa {best_fa:.2f}). "
            "ASD or lipid formulation typically most effective for high-logP BCS II drugs."
        )
    elif bcs_class == "III":
        recommendation = (
            f"{drug_name} is BCS Class III (permeability-limited). "
            "Absorption enhancement strategies have limited impact. "
            "Consider permeation enhancers or prodrug approaches."
        )
    else:
        recommendation = (
            f"{drug_name} is BCS Class IV (low solubility, low permeability). "
            f"Recommended strategy: {best_strategy} (predicted fa {best_fa:.2f}). "
            "Combination approaches (lipid + permeation enhancer) may be needed."
        )

    notes = (
        f"{drug_name}: BCS {bcs_class}, Dn={dose_number:.2f}, "
        f"logP={logP:.1f}, MW={mw:.0f}, "
        f"baseline_fa={baseline:.3f}, best={best_strategy} fa={best_fa:.3f}, "
        f"fold={fold_improvement:.2f}x"
    )

    return AbsorptionEnhancementResult(
        drug_name=drug_name,
        bcs_class=bcs_class,
        baseline_fa=baseline,
        fa_nanosizing=fa_nano,
        fa_solid_dispersion=fa_asd,
        fa_lipid_formulation=fa_lipid,
        fa_cyclodextrin=fa_cd,
        fa_cocrystal=fa_cc,
        best_strategy=best_strategy,
        best_fa=best_fa,
        fold_improvement=fold_improvement,
        solubility_enhancement_nanosizing=sol_nano,
        solubility_enhancement_asd=sol_asd,
        recommendation=recommendation,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Batch screening
# ---------------------------------------------------------------------------


def screen_formulation_strategies(
    compounds: list[dict],
) -> list[AbsorptionEnhancementResult]:
    """Screen multiple compounds and rank by absorption enhancement potential.

    Parameters
    ----------
    compounds:
        List of dicts, each with keys:
        "name" (str), "logP" (float), "mw" (float), "peff" (float),
        "solubility_mg_mL" (float), "dose_mg" (float).
        Optionally "dose_volume_mL" (float, default 250.0).

    Returns
    -------
    list[AbsorptionEnhancementResult]
        Sorted by fold_improvement descending.
    """
    results = []
    for cmpd in compounds:
        result = predict_absorption_enhancement(
            drug_name=cmpd["name"],
            logP=cmpd["logP"],
            mw=cmpd["mw"],
            peff_cm_per_s=cmpd["peff"],
            solubility_mg_mL=cmpd["solubility_mg_mL"],
            dose_mg=cmpd["dose_mg"],
            dose_volume_mL=cmpd.get("dose_volume_mL", 250.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.fold_improvement, reverse=True)
    return results
