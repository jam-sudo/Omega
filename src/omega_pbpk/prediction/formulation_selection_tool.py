"""Phase 225 — Formulation Selection Tool.

Select optimal formulation strategy based on drug properties (BCS class,
solubility, stability, dose, logP, molecular weight).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FormulationSelectionResult:
    """Result of formulation selection analysis."""

    drug_name: str
    bcs_class: int
    dose_mg: float
    solubility_mg_mL: float
    logP: float
    top_formulations: tuple[str, ...]
    formulation_scores: tuple[float, ...]
    primary_recommendation: str
    rationale: str
    development_risk: str  # "low", "moderate", "high"
    notes: str


def _validate_inputs(bcs_class: int, dose_mg: float, solubility_mg_mL: float) -> None:
    """Validate core inputs and raise ValueError on bad data."""
    if bcs_class not in (1, 2, 3, 4):
        raise ValueError(f"bcs_class must be 1, 2, 3, or 4; got {bcs_class}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0; got {dose_mg}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0; got {solubility_mg_mL}")


def _score_clip(score: float) -> float:
    """Clip score to [0, 100]."""
    return max(0.0, min(100.0, score))


def _evaluate_all_formulations(
    bcs_class: int,
    dose_mg: float,
    solubility_mg_mL: float,
    stability_ph_range: float,
    logP: float,
    mw_Da: float,
) -> list[tuple[str, float, str]]:
    """Return list of (formulation_name, score, rationale) for all candidates."""

    high_dose = dose_mg > 500.0
    narrow_ph_stability = stability_ph_range < 2.0
    very_low_solubility = solubility_mg_mL < 0.1
    moderate_logp = 1.0 <= logP <= 3.0
    high_logp = logP > 3.0
    high_mw = mw_Da > 500.0

    results: list[tuple[str, float, str]] = []

    # ------------------------------------------------------------------ #
    # 1. Simple tablet / capsule
    # ------------------------------------------------------------------ #
    tablet_score = 50.0
    if bcs_class == 1:
        tablet_score = 95.0  # highly soluble + highly permeable → ideal
    elif bcs_class == 3:
        tablet_score = 65.0  # solubility fine, permeability limited
    elif bcs_class == 2:
        tablet_score = 40.0  # solubility problem
    else:  # IV
        tablet_score = 25.0
    if high_dose:
        tablet_score += 10.0  # large tablets tolerated for high dose
    if narrow_ph_stability:
        tablet_score -= 15.0
    results.append(
        (
            "simple_tablet_capsule",
            _score_clip(tablet_score),
            "Low cost, high patient acceptance; best for BCS I drugs.",
        )
    )

    # ------------------------------------------------------------------ #
    # 2. Lipid-based (SMEDDS / SEDDS / NLC)
    # ------------------------------------------------------------------ #
    lipid_score = 30.0
    if bcs_class == 2 and high_logp:
        lipid_score = 88.0
    elif bcs_class == 2 and moderate_logp:
        lipid_score = 60.0
    elif bcs_class == 4 and high_logp:
        lipid_score = 70.0
    if high_dose:
        lipid_score -= 35.0  # SMEDDS low capacity; poor for > 500 mg
    if narrow_ph_stability:
        lipid_score += 5.0  # protected inside lipid matrix
    results.append(
        (
            "lipid_based_SMEDDS_SEDDS",
            _score_clip(lipid_score),
            "Solubilises lipophilic BCS II/IV drugs; poor for high dose.",
        )
    )

    # ------------------------------------------------------------------ #
    # 3. Amorphous solid dispersion (ASD / hot-melt extrusion)
    # ------------------------------------------------------------------ #
    asd_score = 25.0
    if bcs_class == 2 and high_logp:
        asd_score = 82.0
    elif bcs_class == 2 and moderate_logp:
        asd_score = 65.0
    elif bcs_class == 4:
        asd_score = 55.0
    if high_dose:
        asd_score -= 10.0  # manageable for ASD, not catastrophic
    if narrow_ph_stability:
        asd_score -= 10.0  # polymer degradation possible
    if high_mw:
        asd_score -= 8.0
    results.append(
        (
            "amorphous_solid_dispersion",
            _score_clip(asd_score),
            "Supersaturating ASD overcomes poor solubility for BCS II/IV.",
        )
    )

    # ------------------------------------------------------------------ #
    # 4. Salt form / crystal engineering
    # ------------------------------------------------------------------ #
    salt_score = 20.0
    if bcs_class == 2 and moderate_logp:
        salt_score = 80.0
    elif bcs_class == 2:
        salt_score = 55.0
    if very_low_solubility:
        salt_score -= 15.0  # salt disproportionation risk
    if narrow_ph_stability:
        salt_score -= 20.0  # acid-labile issues
    results.append(
        (
            "salt_form_crystal_engineering",
            _score_clip(salt_score),
            "Cost-effective for ionisable BCS II moderate logP drugs.",
        )
    )

    # ------------------------------------------------------------------ #
    # 5. Particle size reduction (micronisation / nanosuspension)
    # ------------------------------------------------------------------ #
    particle_score = 20.0
    if bcs_class == 2 and moderate_logp:
        particle_score = 72.0
    elif bcs_class == 2:
        particle_score = 50.0
    if high_dose:
        particle_score += 8.0  # particle reduction works at any dose
    if narrow_ph_stability:
        particle_score -= 5.0
    results.append(
        (
            "particle_size_reduction",
            _score_clip(particle_score),
            "Increases surface area; effective for low solubility drugs.",
        )
    )

    # ------------------------------------------------------------------ #
    # 6. Permeation enhancers / bioadhesive (BCS III)
    # ------------------------------------------------------------------ #
    enhance_score = 15.0
    if bcs_class == 3:
        enhance_score = 78.0
    elif bcs_class == 4:
        enhance_score = 55.0
    if high_dose:
        enhance_score -= 10.0
    results.append(
        (
            "permeation_enhancer_bioadhesive",
            _score_clip(enhance_score),
            "Overcomes poor permeability; recommended for BCS III.",
        )
    )

    # ------------------------------------------------------------------ #
    # 7. Pro-drug strategy
    # ------------------------------------------------------------------ #
    prodrug_score = 10.0
    if bcs_class == 4:
        prodrug_score = 65.0
    elif bcs_class == 3:
        prodrug_score = 40.0
    if high_mw:
        prodrug_score -= 10.0
    if narrow_ph_stability:
        prodrug_score += 5.0  # prodrug can be more stable
    results.append(
        (
            "prodrug_strategy",
            _score_clip(prodrug_score),
            "Combination solubility + permeability fix; complex synthesis.",
        )
    )

    # ------------------------------------------------------------------ #
    # 8. Enteric coating
    # ------------------------------------------------------------------ #
    enteric_score = 30.0
    if narrow_ph_stability:
        enteric_score = 85.0
    elif bcs_class == 1 and narrow_ph_stability:
        enteric_score = 90.0
    if high_dose:
        enteric_score -= 5.0
    results.append(
        (
            "enteric_coating",
            _score_clip(enteric_score),
            "Protects acid-labile drugs; strongly recommended for narrow pH stability.",
        )
    )

    # ------------------------------------------------------------------ #
    # 9. Modified-release / sustained-release
    # ------------------------------------------------------------------ #
    mr_score = 40.0
    if bcs_class == 1:
        mr_score = 70.0
    elif bcs_class == 3:
        mr_score = 55.0
    if high_dose:
        mr_score += 5.0
    if narrow_ph_stability:
        mr_score -= 20.0
    results.append(
        (
            "modified_release",
            _score_clip(mr_score),
            "Extended therapeutic window; suited for high permeability drugs.",
        )
    )

    return results


def select_formulation(
    drug_name: str,
    bcs_class: int,
    dose_mg: float,
    solubility_mg_mL: float,
    stability_ph_range: float = 4.0,
    logP: float = 2.0,
    mw_Da: float = 350.0,
) -> FormulationSelectionResult:
    """Select optimal formulation strategy for a drug.

    Parameters
    ----------
    drug_name:
        Identifier for the drug compound.
    bcs_class:
        BCS classification (1, 2, 3, or 4).
    dose_mg:
        Target dose in milligrams (must be > 0).
    solubility_mg_mL:
        Aqueous solubility in mg/mL (must be > 0).
    stability_ph_range:
        pH range over which the drug is stable (e.g., 2.0 = stable in pH 5-7).
        Values < 2.0 trigger enteric coating recommendation.
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons.

    Returns
    -------
    FormulationSelectionResult
    """
    _validate_inputs(bcs_class, dose_mg, solubility_mg_mL)

    all_forms = _evaluate_all_formulations(
        bcs_class, dose_mg, solubility_mg_mL, stability_ph_range, logP, mw_Da
    )
    # Sort by score descending
    all_forms_sorted = sorted(all_forms, key=lambda x: x[1], reverse=True)

    top3 = all_forms_sorted[:3]
    top_formulations = tuple(f[0] for f in top3)
    formulation_scores = tuple(f[1] for f in top3)
    primary_recommendation = top3[0][0]
    primary_rationale = top3[0][2]

    # Build human-readable rationale
    bcs_desc = {
        1: "high solubility / high permeability",
        2: "low solubility / high permeability",
        3: "high solubility / low permeability",
        4: "low solubility / low permeability",
    }[bcs_class]

    notes_parts: list[str] = []
    if stability_ph_range < 2.0:
        notes_parts.append("Narrow pH stability window; enteric coating strongly advised.")
    if dose_mg > 500.0:
        notes_parts.append(
            "High dose (>500 mg); SMEDDS/lipid-based formulations limited in capacity."
        )
    if logP > 3.0 and bcs_class in (2, 4):
        notes_parts.append("High logP: lipid-based or ASD preferred to enhance dissolution.")
    if bcs_class == 3:
        notes_parts.append("BCS III: permeation enhancers or bioadhesive systems recommended.")
    if bcs_class == 4:
        notes_parts.append("BCS IV: combination strategy or pro-drug approach warranted.")
    notes = " ".join(notes_parts) if notes_parts else "Standard formulation development path."

    # Development risk
    if bcs_class == 1 and not (stability_ph_range < 2.0):
        development_risk = "low"
    elif bcs_class == 4 or (bcs_class == 2 and logP > 5.0):
        development_risk = "high"
    else:
        development_risk = "moderate"

    rationale = (
        f"BCS {bcs_class} ({bcs_desc}), dose={dose_mg:.0f} mg, "
        f"logP={logP:.1f}, solubility={solubility_mg_mL:.3g} mg/mL. "
        f"{primary_rationale}"
    )

    return FormulationSelectionResult(
        drug_name=drug_name,
        bcs_class=bcs_class,
        dose_mg=dose_mg,
        solubility_mg_mL=solubility_mg_mL,
        logP=logP,
        top_formulations=top_formulations,
        formulation_scores=formulation_scores,
        primary_recommendation=primary_recommendation,
        rationale=rationale,
        development_risk=development_risk,
        notes=notes,
    )


def compare_formulation_strategies(
    drug_name: str,
    bcs_class: int,
    dose_mg: float,
    solubility_mg_mL: float,
    logP: float,
    stability_ph_range: float = 4.0,
    mw_Da: float = 350.0,
) -> list[tuple[str, float]]:
    """Return all evaluated formulation strategies sorted by score descending.

    Parameters
    ----------
    drug_name:
        Drug identifier (used for validation context only).
    bcs_class:
        BCS classification (1-4).
    dose_mg:
        Target dose in mg (> 0).
    solubility_mg_mL:
        Aqueous solubility in mg/mL (> 0).
    logP:
        Octanol-water partition coefficient.
    stability_ph_range:
        pH range of drug stability.
    mw_Da:
        Molecular weight in Daltons.

    Returns
    -------
    List of (formulation_name, score) sorted by score descending.
    """
    _validate_inputs(bcs_class, dose_mg, solubility_mg_mL)

    all_forms = _evaluate_all_formulations(
        bcs_class, dose_mg, solubility_mg_mL, stability_ph_range, logP, mw_Da
    )
    sorted_forms = sorted(all_forms, key=lambda x: x[1], reverse=True)
    return [(name, score) for name, score, _ in sorted_forms]
