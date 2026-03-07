"""Dissolution enhancement strategy predictor — Phase 454.

Predicts dissolution enhancement factors for different formulation strategies
based on physicochemical properties (logP, MW, solubility) and BCS class.

Reference: Loftsson & Brewster, J Pharm Pharmacol 2010; Kawabata et al.,
Int J Pharm 2011; FDA Biopharmaceutics Classification System guidance.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DissolutionEnhancementResult",
    "predict_enhancement",
    "rank_strategies",
    "cyclodextrin_complexation",
]

# Supported strategies
_STRATEGIES = frozenset(
    {
        "micronization",
        "nanosuspension",
        "solid_dispersion",
        "cyclodextrin",
        "self_emulsifying",
        "cocrystal",
    }
)

# Mechanism descriptions
_MECHANISMS = {
    "micronization": (
        "Particle size reduction increases surface area (Noyes-Whitney), "
        "improving dissolution rate for BCS II/IV drugs."
    ),
    "nanosuspension": (
        "Nano-scale particles (<1 µm) provide large surface area and "
        "saturated solubility enhancement via Ostwald-Freundlich effect."
    ),
    "solid_dispersion": (
        "Amorphous drug in polymer matrix eliminates crystal lattice energy, "
        "dramatically increasing apparent solubility for lipophilic drugs."
    ),
    "cyclodextrin": (
        "Host-guest inclusion complexation with cyclodextrin cavity increases "
        "aqueous solubility via molecular encapsulation."
    ),
    "self_emulsifying": (
        "SEDDS/SMEDDS pre-dissolve drug in lipid/surfactant mixture; "
        "in vivo emulsification maintains drug in solution in GI tract."
    ),
    "cocrystal": (
        "Pharmaceutical cocrystal with co-former modifies crystal packing, "
        "providing solubility advantage without losing crystallinity."
    ),
}


@dataclass
class DissolutionEnhancementResult:
    """Result of dissolution enhancement prediction for a given strategy."""

    strategy: str
    drug_logP: float
    drug_mw: float
    enhancement_factor: float  # fold increase in dissolution rate
    solubility_improvement: float  # fold increase in solubility
    absorption_improvement: float  # fraction improvement (0-1)
    recommended: bool
    mechanism: str
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _bcs_solubility_limited(bcs_class: str) -> bool:
    """Return True if BCS class has solubility limitation (II or IV)."""
    return bcs_class.upper() in ("II", "BCS II", "2", "IV", "BCS IV", "4")


def _bcs_permeability_limited(bcs_class: str) -> bool:
    """Return True if BCS class has permeability limitation (III or IV)."""
    return bcs_class.upper() in ("III", "BCS III", "3", "IV", "BCS IV", "4")


def _is_bcs_one(bcs_class: str) -> bool:
    """Return True if BCS class I (high solubility, high permeability)."""
    return bcs_class.upper() in ("I", "BCS I", "1")


def _micronization(
    drug_logP: float, drug_mw: float, drug_solubility_mg_mL: float, bcs_class: str
) -> tuple[float, float, float, bool, str]:
    """Micronization: effective for BCS II with moderate logP."""
    sol_limited = _bcs_solubility_limited(bcs_class)
    bcs1 = _is_bcs_one(bcs_class)

    if bcs1 or drug_solubility_mg_mL > 1.0:
        # Already soluble — minimal benefit
        enhancement = 1.2
        sol_imp = 1.0
        abs_imp = 0.05
        recommended = False
        notes = "Minimal benefit for already-soluble drug (BCS I or high solubility)."
    elif sol_limited and drug_logP <= 4.0:
        # Moderate lipophilicity — good candidate
        enhancement = _clamp(2.0 + (4.0 - drug_logP) * 0.5, 1.5, 5.0)
        sol_imp = 1.5
        abs_imp = 0.30
        recommended = True
        notes = "Effective for BCS II with moderate lipophilicity."
    elif sol_limited and drug_logP > 4.0:
        # High lipophilicity — limited by wettability
        enhancement = _clamp(1.5 + (5.0 - min(drug_logP, 7.0)) * 0.3, 1.2, 3.0)
        sol_imp = 1.2
        abs_imp = 0.15
        recommended = False
        notes = "Limited by poor wettability at high logP; consider nanosuspension."
    else:
        enhancement = 1.3
        sol_imp = 1.0
        abs_imp = 0.05
        recommended = False
        notes = "Low benefit expected for this BCS class."
    return enhancement, sol_imp, abs_imp, recommended, notes


def _nanosuspension(
    drug_logP: float, drug_mw: float, drug_solubility_mg_mL: float, bcs_class: str
) -> tuple[float, float, float, bool, str]:
    """Nanosuspension: best for poorly soluble, high-MW drugs."""
    sol_limited = _bcs_solubility_limited(bcs_class)
    bcs1 = _is_bcs_one(bcs_class)

    if bcs1 or drug_solubility_mg_mL > 1.0:
        enhancement = 1.3
        sol_imp = 1.1
        abs_imp = 0.05
        recommended = False
        notes = "Limited benefit for soluble drug."
    elif sol_limited:
        # Scale with logP and MW
        logp_factor = _clamp(1.0 + drug_logP * 0.3, 1.5, 4.0)
        mw_factor = _clamp(1.0 + (drug_mw - 300) / 400, 1.0, 2.0)
        enhancement = _clamp(logp_factor * mw_factor, 2.0, 10.0)
        sol_imp = _clamp(1.5 + drug_logP * 0.4, 1.5, 6.0)
        abs_imp = _clamp(0.25 + drug_logP * 0.05, 0.20, 0.60)
        recommended = drug_logP >= 2.0
        notes = f"Nano particle size increases dissolution; logP={drug_logP:.1f} MW={drug_mw:.0f}."
    else:
        enhancement = 1.5
        sol_imp = 1.2
        abs_imp = 0.10
        recommended = False
        notes = "Moderate benefit for permeability-limited drug."
    return enhancement, sol_imp, abs_imp, recommended, notes


def _solid_dispersion(
    drug_logP: float, drug_mw: float, drug_solubility_mg_mL: float, bcs_class: str
) -> tuple[float, float, float, bool, str]:
    """Solid dispersion: best for high-logP BCS II drugs."""
    sol_limited = _bcs_solubility_limited(bcs_class)
    bcs1 = _is_bcs_one(bcs_class)

    if bcs1 or drug_solubility_mg_mL > 1.0:
        enhancement = 1.2
        sol_imp = 1.1
        abs_imp = 0.03
        recommended = False
        notes = "Not needed for already-soluble drug."
    elif sol_limited and drug_logP >= 3.0:
        # High logP benefits most from amorphous dispersion
        enhancement = _clamp(3.0 + (drug_logP - 3.0) * 1.5, 3.0, 15.0)
        sol_imp = _clamp(2.0 + drug_logP * 0.8, 3.0, 20.0)
        abs_imp = _clamp(0.40 + (drug_logP - 3.0) * 0.06, 0.35, 0.80)
        recommended = True
        notes = (
            f"Amorphous dispersion eliminates crystal lattice energy; "
            f"logP={drug_logP:.1f} favors strong improvement."
        )
    elif sol_limited:
        enhancement = _clamp(2.0 + drug_logP * 0.5, 1.5, 5.0)
        sol_imp = _clamp(1.5 + drug_logP * 0.4, 1.5, 5.0)
        abs_imp = 0.30
        recommended = True
        notes = "Good option for BCS II drug with moderate lipophilicity."
    else:
        enhancement = 1.4
        sol_imp = 1.2
        abs_imp = 0.08
        recommended = False
        notes = "Limited benefit for permeability-limited BCS class."
    return enhancement, sol_imp, abs_imp, recommended, notes


def _cyclodextrin_strategy(
    drug_logP: float, drug_mw: float, drug_solubility_mg_mL: float, bcs_class: str
) -> tuple[float, float, float, bool, str]:
    """Cyclodextrin complexation: effective for intermediate logP, MW < 800."""
    sol_limited = _bcs_solubility_limited(bcs_class)
    bcs1 = _is_bcs_one(bcs_class)

    # Cyclodextrin cavity fits MW 200-800
    mw_ok = 200 <= drug_mw <= 800

    if bcs1 or drug_solubility_mg_mL > 1.0:
        enhancement = 1.3
        sol_imp = 1.2
        abs_imp = 0.05
        recommended = False
        notes = "Drug already soluble; complexation offers minimal benefit."
    elif not mw_ok:
        enhancement = 1.2
        sol_imp = 1.1
        abs_imp = 0.03
        recommended = False
        notes = f"MW={drug_mw:.0f} outside optimal range (200–800) for CD cavity."
    elif sol_limited and 1.0 <= drug_logP <= 5.0:
        # Optimal logP range for CD
        cd_factor = _clamp(1.5 + drug_logP * 0.6, 2.0, 8.0)
        enhancement = cd_factor
        sol_imp = _clamp(2.0 + drug_logP * 0.5, 2.0, 10.0)
        abs_imp = _clamp(0.25 + drug_logP * 0.06, 0.20, 0.55)
        recommended = True
        notes = (
            f"Good logP ({drug_logP:.1f}) for HP-beta-CD complexation; "
            f"MW={drug_mw:.0f} fits cavity."
        )
    elif sol_limited and drug_logP > 5.0:
        # Very lipophilic — may precipitate on dilution
        enhancement = _clamp(2.5 + (6.0 - min(drug_logP, 8.0)) * 0.5, 1.5, 4.0)
        sol_imp = _clamp(1.5 + (6.0 - min(drug_logP, 7.0)) * 0.3, 1.2, 3.5)
        abs_imp = 0.20
        recommended = False
        notes = (
            f"High logP ({drug_logP:.1f}) may exceed CD solubilization capacity; consider SEDDS."
        )
    else:
        enhancement = 1.5
        sol_imp = 1.3
        abs_imp = 0.10
        recommended = False
        notes = "Moderate benefit expected."
    return enhancement, sol_imp, abs_imp, recommended, notes


def _self_emulsifying(
    drug_logP: float, drug_mw: float, drug_solubility_mg_mL: float, bcs_class: str
) -> tuple[float, float, float, bool, str]:
    """SEDDS/SMEDDS: best for highly lipophilic BCS II/IV drugs."""
    sol_limited = _bcs_solubility_limited(bcs_class)
    perm_limited = _bcs_permeability_limited(bcs_class)
    bcs1 = _is_bcs_one(bcs_class)

    if bcs1 or drug_solubility_mg_mL > 1.0:
        enhancement = 1.2
        sol_imp = 1.1
        abs_imp = 0.04
        recommended = False
        notes = "Not needed for already-soluble drug."
    elif sol_limited and drug_logP >= 4.0:
        # Lipophilic drug dissolves well in lipid vehicle
        enhancement = _clamp(4.0 + (drug_logP - 4.0) * 1.0, 3.0, 12.0)
        sol_imp = _clamp(3.0 + drug_logP * 0.5, 4.0, 15.0)
        abs_imp = _clamp(0.35 + (drug_logP - 4.0) * 0.05, 0.30, 0.70)
        recommended = True
        notes = (
            f"SEDDS excellent for high logP ({drug_logP:.1f}) BCS II drug; "
            "maintains solubilized state in GI."
        )
    elif sol_limited and drug_logP >= 2.0:
        enhancement = _clamp(2.5 + drug_logP * 0.3, 2.0, 5.0)
        sol_imp = _clamp(2.0 + drug_logP * 0.3, 2.0, 6.0)
        abs_imp = 0.25
        recommended = True
        notes = "Moderate lipophilicity; SEDDS provides good solubilization."
    elif perm_limited:
        enhancement = 1.5
        sol_imp = 1.3
        abs_imp = 0.08
        recommended = False
        notes = "Permeability-limited drug; SEDDS helps solubility but not absorption."
    else:
        enhancement = 1.3
        sol_imp = 1.1
        abs_imp = 0.05
        recommended = False
        notes = "Low expected benefit for this profile."
    return enhancement, sol_imp, abs_imp, recommended, notes


def _cocrystal(
    drug_logP: float, drug_mw: float, drug_solubility_mg_mL: float, bcs_class: str
) -> tuple[float, float, float, bool, str]:
    """Cocrystal: tunable crystal packing for moderate solubility improvement."""
    sol_limited = _bcs_solubility_limited(bcs_class)
    bcs1 = _is_bcs_one(bcs_class)

    if bcs1 or drug_solubility_mg_mL > 1.0:
        enhancement = 1.2
        sol_imp = 1.1
        abs_imp = 0.03
        recommended = False
        notes = "Cocrystallization not needed for already-soluble drug."
    elif sol_limited and drug_logP >= 1.0:
        # Cocrystal gives moderate and predictable improvement
        enhancement = _clamp(1.5 + drug_logP * 0.4, 1.5, 6.0)
        sol_imp = _clamp(1.3 + drug_logP * 0.35, 1.5, 5.0)
        abs_imp = _clamp(0.15 + drug_logP * 0.04, 0.10, 0.45)
        recommended = 1.5 <= drug_logP <= 5.0
        notes = (
            f"Cocrystal modifies crystal packing; retains crystallinity with "
            f"improved solubility. logP={drug_logP:.1f}."
        )
    else:
        enhancement = 1.3
        sol_imp = 1.1
        abs_imp = 0.05
        recommended = False
        notes = "Limited benefit for this drug profile."
    return enhancement, sol_imp, abs_imp, recommended, notes


_STRATEGY_FN = {
    "micronization": _micronization,
    "nanosuspension": _nanosuspension,
    "solid_dispersion": _solid_dispersion,
    "cyclodextrin": _cyclodextrin_strategy,
    "self_emulsifying": _self_emulsifying,
    "cocrystal": _cocrystal,
}


def predict_enhancement(
    drug_logP: float,
    drug_mw: float,
    drug_solubility_mg_mL: float,
    bcs_class: str,
    strategy: str,
) -> DissolutionEnhancementResult:
    """Predict dissolution enhancement for a given formulation strategy.

    Parameters
    ----------
    drug_logP:
        Octanol-water partition coefficient (log scale).
    drug_mw:
        Molecular weight in g/mol.
    drug_solubility_mg_mL:
        Aqueous solubility in mg/mL.
    bcs_class:
        BCS class string: 'I', 'II', 'III', or 'IV'.
    strategy:
        Formulation strategy. One of: 'micronization', 'nanosuspension',
        'solid_dispersion', 'cyclodextrin', 'self_emulsifying', 'cocrystal'.

    Returns
    -------
    DissolutionEnhancementResult

    Raises
    ------
    ValueError
        If strategy is not recognized or inputs are invalid.
    """
    strategy_lower = strategy.lower()
    if strategy_lower not in _STRATEGIES:
        valid = ", ".join(sorted(_STRATEGIES))
        raise ValueError(f"Unknown strategy '{strategy}'. Valid strategies: {valid}")
    if drug_mw <= 0:
        raise ValueError(f"drug_mw must be > 0, got {drug_mw}")
    if drug_solubility_mg_mL < 0:
        raise ValueError(f"drug_solubility_mg_mL must be >= 0, got {drug_solubility_mg_mL}")

    fn = _STRATEGY_FN[strategy_lower]
    enhancement, sol_imp, abs_imp, recommended, notes = fn(
        drug_logP, drug_mw, drug_solubility_mg_mL, bcs_class
    )

    return DissolutionEnhancementResult(
        strategy=strategy_lower,
        drug_logP=drug_logP,
        drug_mw=drug_mw,
        enhancement_factor=enhancement,
        solubility_improvement=sol_imp,
        absorption_improvement=_clamp(abs_imp, 0.0, 1.0),
        recommended=recommended,
        mechanism=_MECHANISMS[strategy_lower],
        notes=notes,
    )


def rank_strategies(
    drug_logP: float,
    drug_mw: float,
    drug_solubility_mg_mL: float,
    bcs_class: str,
) -> list[DissolutionEnhancementResult]:
    """Rank all dissolution enhancement strategies by enhancement_factor.

    Parameters
    ----------
    drug_logP:
        Octanol-water partition coefficient (log scale).
    drug_mw:
        Molecular weight in g/mol.
    drug_solubility_mg_mL:
        Aqueous solubility in mg/mL.
    bcs_class:
        BCS class string: 'I', 'II', 'III', or 'IV'.

    Returns
    -------
    list[DissolutionEnhancementResult]
        All 6 strategies sorted by enhancement_factor descending.
    """
    results = [
        predict_enhancement(drug_logP, drug_mw, drug_solubility_mg_mL, bcs_class, s)
        for s in sorted(_STRATEGIES)
    ]
    results.sort(key=lambda r: r.enhancement_factor, reverse=True)
    return results


def cyclodextrin_complexation(
    drug_logP: float,
    drug_mw: float,
    cd_type: str = "HP-beta-CD",
) -> dict[str, object]:
    """Predict cyclodextrin complexation properties.

    Higher logP drugs drive better hydrophobic interaction with the CD cavity.
    MW must be within the cavity size range (200–800 Da) for strong binding.

    Parameters
    ----------
    drug_logP:
        Drug lipophilicity (log P).
    drug_mw:
        Molecular weight in g/mol.
    cd_type:
        Type of cyclodextrin: 'HP-beta-CD', 'beta-CD', 'gamma-CD', 'alpha-CD'.

    Returns
    -------
    dict
        Keys: 'complexation_efficiency', 'solubility_fold_increase', 'notes'.

    Raises
    ------
    ValueError
        If cd_type is not recognized or drug_mw <= 0.
    """
    valid_cd = {"HP-beta-CD", "beta-CD", "gamma-CD", "alpha-CD"}
    if cd_type not in valid_cd:
        raise ValueError(f"Unknown cd_type '{cd_type}'. Valid types: {', '.join(sorted(valid_cd))}")
    if drug_mw <= 0:
        raise ValueError(f"drug_mw must be > 0, got {drug_mw}")

    # CD cavity size compatibility
    # alpha-CD: MW 500-600 → small molecules ~MW 100-250
    # beta-CD:  MW 700-800 → medium molecules ~MW 200-500
    # gamma-CD: MW 1100    → larger molecules ~MW 300-800
    cd_mw_ranges = {
        "alpha-CD": (80, 250),
        "beta-CD": (200, 500),
        "HP-beta-CD": (200, 600),
        "gamma-CD": (300, 800),
    }
    lo, hi = cd_mw_ranges[cd_type]
    mw_ok = lo <= drug_mw <= hi

    # Complexation efficiency based on logP (hydrophobic driving force)
    # Scale 0–1: low logP (<1) → poor; logP 2-5 → good; logP >6 → partial
    if drug_logP < 0:
        ce_base = 0.05
    elif drug_logP < 1.0:
        ce_base = 0.10 + drug_logP * 0.10
    elif drug_logP <= 5.0:
        ce_base = _clamp(0.20 + (drug_logP - 1.0) * 0.15, 0.20, 0.80)
    else:
        # Very lipophilic: may crystallize out after dilution
        ce_base = _clamp(0.80 - (drug_logP - 5.0) * 0.08, 0.30, 0.80)

    # MW penalty if outside optimal range
    if not mw_ok:
        ce = ce_base * 0.4
        mw_note = f"MW={drug_mw:.0f} outside {cd_type} optimal range ({lo}–{hi} Da)."
    else:
        ce = ce_base
        mw_note = f"MW={drug_mw:.0f} compatible with {cd_type} cavity."

    # Solubility fold increase correlates with complexation efficiency
    sol_fold = _clamp(1.0 + ce * 15.0, 1.0, 12.0)

    notes_parts = [mw_note]
    if drug_logP >= 2.0 and mw_ok:
        notes_parts.append(f"Good hydrophobic driving force (logP={drug_logP:.1f}) for {cd_type}.")
    elif drug_logP < 1.0:
        notes_parts.append("Low logP reduces hydrophobic driving force for CD complexation.")

    return {
        "complexation_efficiency": round(ce, 4),
        "solubility_fold_increase": round(sol_fold, 2),
        "notes": " ".join(notes_parts),
    }
