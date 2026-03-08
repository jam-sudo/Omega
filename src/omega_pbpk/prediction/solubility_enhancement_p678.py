"""Predict solubility enhancement by pharmaceutical excipients.

Covers cyclodextrins, surfactants (SLS, Cremophor), co-solvents (PEG 400),
and polymer-stabilised amorphous dispersions (HPMC).  Pure Python, no
external libraries.

References
----------
- Loftsson T & Brewster ME.  J Pharm Pharmacol. 2010 (cyclodextrins)
- Seedher N & Kanojia M.  AAPS PharmSciTech. 2009 (surfactants)
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_EXCIPIENTS = {"cyclodextrin", "SLS", "PEG400", "HPMC", "Cremophor", "none"}


@dataclass(frozen=True)
class SolubilityEnhancementResult:
    """Result of a solubility-enhancement prediction."""

    drug_name: str
    logP: float
    mw_Da: float
    intrinsic_solubility_mg_mL: float
    excipient_type: str
    excipient_concentration_pct: float
    enhanced_solubility_mg_mL: float
    solubility_ratio: float
    mechanism: str
    aqueous_solubility_class: str
    formulation_category: str
    bioavailability_impact_pct: float
    notes: str


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def predict_solubility_enhancement(
    drug_name: str,
    logP: float,
    mw_Da: float,
    intrinsic_solubility_mg_mL: float,
    excipient_type: str,
    excipient_concentration_pct: float = 10.0,
) -> SolubilityEnhancementResult:
    """Predict enhanced solubility for *drug_name* with *excipient_type*."""

    # --- validation ---
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError("intrinsic_solubility_mg_mL must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if excipient_concentration_pct <= 0:
        raise ValueError("excipient_concentration_pct must be > 0")
    if excipient_type not in VALID_EXCIPIENTS:
        raise ValueError(
            f"excipient_type must be one of {sorted(VALID_EXCIPIENTS)}, got '{excipient_type}'"
        )

    pct = excipient_concentration_pct

    # --- compute ratio and mechanism per excipient ---
    if excipient_type == "none":
        ratio = 1.0
        mechanism = "no_enhancement"
    elif excipient_type == "cyclodextrin":
        Ka = _clamp(1000.0 * (logP - 1.0), 100.0, 50000.0)
        cd_conc_M = pct / 100.0 * 1000.0 / 1135.0
        ratio = _clamp(1.0 + Ka * cd_conc_M, 1.0, 1000.0)
        mechanism = "inclusion_complex_formation"
    elif excipient_type == "SLS":
        if pct > 0.2:
            ratio = 1.0 + max(0.0, logP * 2.0) * (pct - 0.2) * 5.0
        else:
            ratio = 1.0
        ratio = _clamp(ratio, 1.0, 100.0)
        mechanism = "micellar_solubilization"
    elif excipient_type == "PEG400":
        ratio = 1.0 + pct * 0.1 * max(1.0, logP)
        ratio = _clamp(ratio, 1.0, 20.0)
        mechanism = "co_solvent_effect"
    elif excipient_type == "HPMC":
        ratio = 1.0 + pct * 0.05 * max(1.0, mw_Da / 300.0)
        ratio = _clamp(ratio, 1.0, 50.0)
        mechanism = "polymer_stabilized_amorphous_dispersion"
    elif excipient_type == "Cremophor":
        ratio = 1.0 + max(0.0, logP * 3.0) * pct * 0.1
        ratio = _clamp(ratio, 1.0, 200.0)
        mechanism = "non_ionic_micelle_solubilization"
    else:  # pragma: no cover
        ratio = 1.0
        mechanism = "unknown"

    enhanced_sol = intrinsic_solubility_mg_mL * ratio

    # --- solubility class ---
    if enhanced_sol >= 10.0:
        sol_class = "freely_soluble"
    elif enhanced_sol >= 1.0:
        sol_class = "soluble"
    elif enhanced_sol >= 0.1:
        sol_class = "sparingly_soluble"
    elif enhanced_sol >= 0.01:
        sol_class = "slightly_soluble"
    else:
        sol_class = "practically_insoluble"

    # --- formulation category ---
    if ratio > 100:
        form_cat = "high_enhancement_required"
    elif ratio > 10:
        form_cat = "significant_enhancement"
    elif ratio > 2:
        form_cat = "moderate_enhancement"
    else:
        form_cat = "minimal_enhancement"

    # --- bioavailability impact ---
    if logP > 2:
        impact = min(100.0, (ratio - 1.0) * 10.0)
    else:
        impact = min(20.0, (ratio - 1.0) * 5.0)
    impact = max(0.0, impact)

    return SolubilityEnhancementResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        excipient_type=excipient_type,
        excipient_concentration_pct=pct,
        enhanced_solubility_mg_mL=enhanced_sol,
        solubility_ratio=ratio,
        mechanism=mechanism,
        aqueous_solubility_class=sol_class,
        formulation_category=form_cat,
        bioavailability_impact_pct=impact,
        notes=f"excipient={excipient_type}, conc={pct}%",
    )


def compare_excipients(
    drug_name: str,
    logP: float,
    mw_Da: float,
    intrinsic_solubility_mg_mL: float,
    concentration_pct: float = 10.0,
) -> list[SolubilityEnhancementResult]:
    """Compare all excipients (excluding 'none') sorted by enhanced solubility descending."""

    results: list[SolubilityEnhancementResult] = []
    for exc in ("cyclodextrin", "SLS", "PEG400", "HPMC", "Cremophor"):
        r = predict_solubility_enhancement(
            drug_name=drug_name,
            logP=logP,
            mw_Da=mw_Da,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
            excipient_type=exc,
            excipient_concentration_pct=concentration_pct,
        )
        results.append(r)
    results.sort(key=lambda x: x.enhanced_solubility_mg_mL, reverse=True)
    return results
