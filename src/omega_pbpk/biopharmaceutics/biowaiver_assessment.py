"""Phase 764 — Drug Biowaiver Assessment.

Assesses biowaiver eligibility for solid oral dosage forms based on
FDA 2017 guidance and BCS classification.

BCS Class I (high sol + high perm) → biowaiver eligible if criteria met.
BCS Class III (high sol + low perm) → conditional biowaiver if fa >= 85%.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BiowaiverAssessmentResult:
    """Result of biowaiver eligibility assessment."""

    drug_name: str
    dose_number: float  # Do = dose_mg / (solubility_mg_mL * 250)
    bcs_class: str  # "I", "II", "III", "IV"
    high_solubility: bool
    high_permeability: bool
    rapid_dissolution: bool
    biowaiver_eligible: bool
    biowaiver_rationale: str
    risk_factors: str  # comma-joined, "" if none
    recommendation: str
    notes: str


def _determine_bcs_class(high_solubility: bool, high_permeability: bool) -> str:
    """Determine BCS class from solubility and permeability flags."""
    if high_solubility and high_permeability:
        return "I"
    elif not high_solubility and high_permeability:
        return "II"
    elif high_solubility and not high_permeability:
        return "III"
    else:
        return "IV"


def _build_rationale(
    bcs_class: str,
    biowaiver_eligible: bool,
    rapid_dissolution: bool,
    is_narrow_ti: bool,
    is_prodrug: bool,
    has_absorption_window: bool,
    has_pgp_inhibitor_excipient: bool,
    high_surfactant_load: bool,
    fa_pct: float,
) -> str:
    """Build human-readable rationale string."""
    if biowaiver_eligible:
        if bcs_class == "I":
            return (
                f"BCS Class {bcs_class} drug with rapid dissolution meets FDA 2017 biowaiver "
                "criteria. High solubility and high permeability with no disqualifying factors."
            )
        else:  # Class III
            return (
                f"BCS Class {bcs_class} drug qualifies for conditional biowaiver. "
                f"High solubility, rapid dissolution, fa={fa_pct:.1f}%>=85%, no excipient concerns."
            )

    reasons: list[str] = []
    if bcs_class not in ("I", "III"):
        reasons.append(f"BCS Class {bcs_class} does not qualify for biowaiver")
    if not rapid_dissolution:
        reasons.append("does not meet rapid dissolution criterion (>=85% in 30 min)")
    if is_narrow_ti:
        reasons.append("narrow therapeutic index drug")
    if is_prodrug:
        reasons.append("prodrug requiring active transport")
    if has_absorption_window:
        reasons.append("has absorption window (site-specific absorption)")
    if has_pgp_inhibitor_excipient:
        reasons.append("P-gp inhibitor excipient present")
    if high_surfactant_load:
        reasons.append("high surfactant load (>2%)")
    if bcs_class == "III" and fa_pct < 85.0:
        reasons.append(f"BCS Class III requires fa>=85% (actual fa={fa_pct:.1f}%)")

    return "Biowaiver not eligible: " + "; ".join(reasons) + "."


def _build_risk_factors(
    is_narrow_ti: bool,
    is_prodrug: bool,
    has_absorption_window: bool,
    has_pgp_inhibitor_excipient: bool,
    high_surfactant_load: bool,
) -> str:
    """Build comma-joined risk factors string."""
    risks: list[str] = []
    if is_narrow_ti:
        risks.append("narrow therapeutic index")
    if is_prodrug:
        risks.append("prodrug")
    if has_absorption_window:
        risks.append("absorption window")
    if has_pgp_inhibitor_excipient:
        risks.append("P-gp inhibitor excipient")
    if high_surfactant_load:
        risks.append("high surfactant load")
    return ", ".join(risks)


def _build_recommendation(biowaiver_eligible: bool, bcs_class: str) -> str:
    """Build recommendation string."""
    if biowaiver_eligible:
        if bcs_class == "I":
            return (
                "Biowaiver recommended. Proceed with in vitro dissolution testing "
                "at pH 1.2, 4.5, and 6.8 to confirm rapid dissolution."
            )
        else:
            return (
                "Conditional biowaiver recommended for BCS Class III. "
                "Confirm excipients do not affect permeability; document fa data."
            )
    else:
        return (
            "In vivo bioequivalence study required. Biowaiver criteria not met. "
            "Review risk factors and consider formulation changes if applicable."
        )


def _build_notes(
    dose_number: float,
    bcs_class: str,
    high_solubility: bool,
    high_permeability: bool,
    rapid_dissolution: bool,
    fa_pct: float,
) -> str:
    """Build notes string."""
    sol_str = "high" if high_solubility else "low"
    perm_str = "high" if high_permeability else "low"
    diss_str = "rapid" if rapid_dissolution else "not rapid"
    return (
        f"BCS Class {bcs_class}: dose number Do={dose_number:.3f} ({sol_str} solubility), "
        f"permeability {perm_str}, dissolution {diss_str}, fa={fa_pct:.1f}%. "
        "FDA 2017 biowaiver guidance applied."
    )


def assess_biowaiver(
    drug_name: str,
    dose_mg: float,
    solubility_pH4_5_mg_mL: float,
    peff_cm_s: float = -1.0,
    fa_pct: float = 80.0,
    dissolution_85_in_30min: bool = True,
    is_narrow_ti: bool = False,
    is_prodrug: bool = False,
    has_absorption_window: bool = False,
    has_pgp_inhibitor_excipient: bool = False,
    high_surfactant_load: bool = False,
) -> BiowaiverAssessmentResult:
    """Assess biowaiver eligibility for a solid oral dosage form.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Highest dose strength in mg.
    solubility_pH4_5_mg_mL:
        Equilibrium solubility at pH 4.5 in mg/mL.
    peff_cm_s:
        Effective intestinal permeability in cm/s (-1.0 if unknown).
    fa_pct:
        Fraction absorbed in percent (0-100).
    dissolution_85_in_30min:
        True if >=85% dissolved in 30 min at pH 1.2, 4.5, and 6.8.
    is_narrow_ti:
        True if narrow therapeutic index drug.
    is_prodrug:
        True if prodrug or peptide requiring active transport.
    has_absorption_window:
        True if site-specific absorption exists.
    has_pgp_inhibitor_excipient:
        True if formulation contains P-gp inhibitor excipients.
    high_surfactant_load:
        True if surfactant content exceeds 2%.

    Returns
    -------
    BiowaiverAssessmentResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if solubility_pH4_5_mg_mL <= 0:
        raise ValueError(f"solubility_pH4_5_mg_mL must be > 0, got {solubility_pH4_5_mg_mL}")

    # Dose number Do: dimensionless
    volume_mL = 250.0
    dose_number = dose_mg / (solubility_pH4_5_mg_mL * volume_mL)

    # BCS solubility criterion: Do < 1
    high_solubility = dose_number < 1.0

    # BCS permeability criterion: Peff > 2e-4 cm/s (primary) OR fa > 85% (secondary, Peff unknown)
    # When Peff is known, use it exclusively for BCS class; fa is used for biowaiver eligibility.
    peff_threshold = 2e-4  # cm/s
    if peff_cm_s > 0:
        high_permeability = peff_cm_s > peff_threshold
    else:
        # No Peff measurement: rely on fa for BCS permeability classification
        high_permeability = fa_pct > 85.0

    bcs_class = _determine_bcs_class(high_solubility, high_permeability)

    # Base eligibility criteria (apply to all classes)
    base_criteria_met = (
        dissolution_85_in_30min
        and not is_narrow_ti
        and not is_prodrug
        and not has_absorption_window
        and not has_pgp_inhibitor_excipient
        and not high_surfactant_load
    )

    # Class-specific eligibility
    if bcs_class == "I":
        biowaiver_eligible = base_criteria_met
    elif bcs_class == "III":
        # Additional condition: fa >= 85%
        biowaiver_eligible = base_criteria_met and (fa_pct >= 85.0)
    else:
        # Class II or IV: not eligible
        biowaiver_eligible = False

    risk_factors = _build_risk_factors(
        is_narrow_ti,
        is_prodrug,
        has_absorption_window,
        has_pgp_inhibitor_excipient,
        high_surfactant_load,
    )
    rationale = _build_rationale(
        bcs_class,
        biowaiver_eligible,
        dissolution_85_in_30min,
        is_narrow_ti,
        is_prodrug,
        has_absorption_window,
        has_pgp_inhibitor_excipient,
        high_surfactant_load,
        fa_pct,
    )
    recommendation = _build_recommendation(biowaiver_eligible, bcs_class)
    notes = _build_notes(
        dose_number,
        bcs_class,
        high_solubility,
        high_permeability,
        dissolution_85_in_30min,
        fa_pct,
    )

    return BiowaiverAssessmentResult(
        drug_name=drug_name,
        dose_number=dose_number,
        bcs_class=bcs_class,
        high_solubility=high_solubility,
        high_permeability=high_permeability,
        rapid_dissolution=dissolution_85_in_30min,
        biowaiver_eligible=biowaiver_eligible,
        biowaiver_rationale=rationale,
        risk_factors=risk_factors,
        recommendation=recommendation,
        notes=notes,
    )


def screen_biowaiver_candidates(
    drugs: list[dict],
) -> list[BiowaiverAssessmentResult]:
    """Screen multiple drug candidates for biowaiver eligibility.

    Parameters
    ----------
    drugs:
        List of dicts with keys matching assess_biowaiver parameters.

    Returns
    -------
    List of BiowaiverAssessmentResult sorted by biowaiver_eligible descending
    (eligible candidates first).
    """
    results: list[BiowaiverAssessmentResult] = []
    for d in drugs:
        result = assess_biowaiver(**d)
        results.append(result)
    # Sort: eligible (True=1) before not-eligible (False=0), descending
    results.sort(key=lambda r: r.biowaiver_eligible, reverse=True)
    return results
