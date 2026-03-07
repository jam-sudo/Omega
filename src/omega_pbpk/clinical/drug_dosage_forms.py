"""Phase 605 — Drug Dosage Forms Reference Guide.

Reference database for drug dosage forms with bioavailability, onset,
duration, and PK implications for formulation selection.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DosageFormProfile:
    """Profile for a single dosage form."""

    form_name: str
    category: str  # "oral","parenteral","topical","inhalation","transdermal"
    typical_bioavailability_pct: float  # 0-100
    onset_min: float  # typical onset in minutes
    duration_h: float  # typical duration in hours
    absorption_rate: str  # "immediate","slow","extended","variable"
    first_pass: bool  # subject to first-pass metabolism
    dose_number_relevant: bool  # whether BCS dose number matters
    key_pk_parameters: dict  # {"ka_range": (low, high), "tmax_range": (low, high)}
    advantages: str
    disadvantages: str
    notes: str


@dataclass(frozen=True)
class DosageFormSelectionResult:
    """Result of dosage form selection for a drug."""

    drug_name: str
    recommended_form: str
    alternative_forms: tuple
    selection_rationale: str
    bioavailability_estimate_pct: float
    onset_estimate_min: float
    pk_implications: str
    notes: str


# Database of dosage forms
DOSAGE_FORM_DATABASE: dict = {
    "immediate_release_tablet": DosageFormProfile(
        form_name="immediate_release_tablet",
        category="oral",
        typical_bioavailability_pct=70.0,
        onset_min=30.0,
        duration_h=6.0,
        absorption_rate="immediate",
        first_pass=True,
        dose_number_relevant=True,
        key_pk_parameters={"ka_range": (0.5, 2.0), "tmax_range": (0.5, 2.0)},
        advantages="Convenient, widely accepted, cost-effective",
        disadvantages="First-pass metabolism, multiple daily doses may be required",
        notes="Standard oral solid dosage form; most common formulation type",
    ),
    "extended_release_tablet": DosageFormProfile(
        form_name="extended_release_tablet",
        category="oral",
        typical_bioavailability_pct=65.0,
        onset_min=120.0,
        duration_h=24.0,
        absorption_rate="extended",
        first_pass=True,
        dose_number_relevant=True,
        key_pk_parameters={"ka_range": (0.05, 0.3), "tmax_range": (4.0, 12.0)},
        advantages="Once-daily dosing, reduced peak-trough fluctuation, improved adherence",
        disadvantages="Higher cost, dose-dumping risk, slower onset",
        notes="Matrix, osmotic, or coated ER systems; reduces dosing frequency",
    ),
    "oral_solution": DosageFormProfile(
        form_name="oral_solution",
        category="oral",
        typical_bioavailability_pct=85.0,
        onset_min=20.0,
        duration_h=5.0,
        absorption_rate="immediate",
        first_pass=True,
        dose_number_relevant=False,
        key_pk_parameters={"ka_range": (1.0, 4.0), "tmax_range": (0.25, 1.0)},
        advantages="Rapid absorption, good for patients with swallowing difficulties",
        disadvantages="Stability issues, taste masking required, portability",
        notes="Bypasses dissolution step; highest oral BA among solid/liquid comparisons",
    ),
    "sublingual_tablet": DosageFormProfile(
        form_name="sublingual_tablet",
        category="oral",
        typical_bioavailability_pct=75.0,
        onset_min=5.0,
        duration_h=4.0,
        absorption_rate="immediate",
        first_pass=False,
        dose_number_relevant=False,
        key_pk_parameters={"ka_range": (2.0, 8.0), "tmax_range": (0.05, 0.3)},
        advantages="Rapid onset, avoids first-pass metabolism, suitable for emergency use",
        disadvantages="Small dose volume, patient compliance, taste issues",
        notes="Buccal/sublingual route bypasses hepatic first-pass; used for nitroglycerin, buprenorphine",
    ),
    "iv_bolus": DosageFormProfile(
        form_name="iv_bolus",
        category="parenteral",
        typical_bioavailability_pct=100.0,
        onset_min=1.0,
        duration_h=2.0,
        absorption_rate="immediate",
        first_pass=False,
        dose_number_relevant=False,
        key_pk_parameters={"ka_range": (0.0, 0.0), "tmax_range": (0.0, 0.05)},
        advantages="100% bioavailability, immediate effect, precise dose delivery",
        disadvantages="Invasive, requires trained personnel, risk of adverse reactions",
        notes="Gold standard reference route; defines 100% BA for all other routes",
    ),
    "iv_infusion": DosageFormProfile(
        form_name="iv_infusion",
        category="parenteral",
        typical_bioavailability_pct=100.0,
        onset_min=5.0,
        duration_h=24.0,
        absorption_rate="extended",
        first_pass=False,
        dose_number_relevant=False,
        key_pk_parameters={"ka_range": (0.0, 0.0), "tmax_range": (1.0, 24.0)},
        advantages="Controlled delivery rate, suitable for drugs with narrow TI, continuous effect",
        disadvantages="Requires IV access, pump equipment, hospital setting",
        notes="Steady-state achieved at ~4-5 half-lives; rate-controlled PK profile",
    ),
    "im_injection": DosageFormProfile(
        form_name="im_injection",
        category="parenteral",
        typical_bioavailability_pct=95.0,
        onset_min=15.0,
        duration_h=8.0,
        absorption_rate="slow",
        first_pass=False,
        dose_number_relevant=False,
        key_pk_parameters={"ka_range": (0.2, 1.0), "tmax_range": (0.25, 2.0)},
        advantages="Near-complete BA, suitable for lipophilic drugs, depot formulations possible",
        disadvantages="Painful, requires trained personnel, injection site reactions",
        notes="Absorption depends on muscle blood flow; useful for long-acting depot formulations",
    ),
    "sc_injection": DosageFormProfile(
        form_name="sc_injection",
        category="parenteral",
        typical_bioavailability_pct=90.0,
        onset_min=30.0,
        duration_h=12.0,
        absorption_rate="slow",
        first_pass=False,
        dose_number_relevant=False,
        key_pk_parameters={"ka_range": (0.1, 0.5), "tmax_range": (1.0, 4.0)},
        advantages="Self-administration possible, suitable for biologics, sustained effect",
        disadvantages="Volume limitation (~1-2 mL), site reactions, variable absorption",
        notes="Common for insulin, mAbs; lymphatic absorption relevant for macromolecules",
    ),
    "transdermal_patch": DosageFormProfile(
        form_name="transdermal_patch",
        category="transdermal",
        typical_bioavailability_pct=50.0,
        onset_min=120.0,
        duration_h=72.0,
        absorption_rate="extended",
        first_pass=False,
        dose_number_relevant=False,
        key_pk_parameters={"ka_range": (0.01, 0.1), "tmax_range": (12.0, 36.0)},
        advantages="Avoids first-pass, continuous drug delivery, improved patient compliance",
        disadvantages="Limited to potent lipophilic drugs, skin irritation, slow onset",
        notes="Suitable for logP 1-3, MW<500 Da; fentanyl, nicotine, estradiol examples",
    ),
    "inhaled_dry_powder": DosageFormProfile(
        form_name="inhaled_dry_powder",
        category="inhalation",
        typical_bioavailability_pct=20.0,
        onset_min=5.0,
        duration_h=12.0,
        absorption_rate="immediate",
        first_pass=False,
        dose_number_relevant=False,
        key_pk_parameters={"ka_range": (0.5, 3.0), "tmax_range": (0.05, 0.5)},
        advantages="Local lung delivery, rapid effect for respiratory drugs, avoid systemic SE",
        disadvantages="Device dependent, technique-dependent absorption, particle size critical",
        notes="Lung deposition 10-40% of dose; remainder swallowed and absorbed orally",
    ),
    "topical_cream": DosageFormProfile(
        form_name="topical_cream",
        category="topical",
        typical_bioavailability_pct=5.0,
        onset_min=60.0,
        duration_h=8.0,
        absorption_rate="slow",
        first_pass=False,
        dose_number_relevant=False,
        key_pk_parameters={"ka_range": (0.001, 0.05), "tmax_range": (2.0, 8.0)},
        advantages="Local effect, minimal systemic exposure, patient self-administration",
        disadvantages="Very low systemic BA, formulation-dependent, skin condition dependent",
        notes="Designed for local effect; systemic exposure depends on skin integrity and occlusion",
    ),
    "rectal_suppository": DosageFormProfile(
        form_name="rectal_suppository",
        category="parenteral",
        typical_bioavailability_pct=50.0,
        onset_min=15.0,
        duration_h=6.0,
        absorption_rate="slow",
        first_pass=False,
        dose_number_relevant=False,
        key_pk_parameters={"ka_range": (0.2, 1.0), "tmax_range": (0.5, 2.0)},
        advantages="Useful when oral route unavailable, partial first-pass avoidance",
        disadvantages="Patient acceptability, variable absorption, position dependent",
        notes="Lower rectum drains to systemic circulation (partial first-pass bypass); upper to portal",
    ),
    "oral_disintegrating_tablet": DosageFormProfile(
        form_name="oral_disintegrating_tablet",
        category="oral",
        typical_bioavailability_pct=72.0,
        onset_min=15.0,
        duration_h=6.0,
        absorption_rate="immediate",
        first_pass=True,
        dose_number_relevant=True,
        key_pk_parameters={"ka_range": (0.8, 3.0), "tmax_range": (0.25, 1.5)},
        advantages="No water needed, rapid disintegration, suitable for dysphagia patients",
        disadvantages="Taste masking challenges, fragility, moisture sensitivity",
        notes="Disintegrates in mouth within seconds; absorption mainly from GI tract after swallowing",
    ),
}

_VALID_CATEGORIES = {"oral", "parenteral", "topical", "inhalation", "transdermal"}

# Injection-related form names (used for filtering)
_INJECTION_FORMS = {"iv_bolus", "iv_infusion", "im_injection", "sc_injection", "rectal_suppository"}

# Oral-related form names
_ORAL_FORMS = {
    "immediate_release_tablet",
    "extended_release_tablet",
    "oral_solution",
    "sublingual_tablet",
    "oral_disintegrating_tablet",
}


def get_dosage_form_profile(form_name: str) -> DosageFormProfile:
    """Retrieve profile for a named dosage form.

    Args:
        form_name: Name of the dosage form (must be in DOSAGE_FORM_DATABASE)

    Returns:
        DosageFormProfile for the named form

    Raises:
        ValueError: If form_name is not found in the database
    """
    if form_name not in DOSAGE_FORM_DATABASE:
        available = ", ".join(sorted(DOSAGE_FORM_DATABASE.keys()))
        raise ValueError(f"Unknown dosage form '{form_name}'. Available forms: {available}")
    return DOSAGE_FORM_DATABASE[form_name]


def list_forms_by_category(category: str) -> list:
    """Return list of form names in the given category.

    Args:
        category: One of "oral", "parenteral", "topical", "inhalation", "transdermal"

    Returns:
        List of form names belonging to the category

    Raises:
        ValueError: If category is not valid
    """
    if category not in _VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Must be one of: {sorted(_VALID_CATEGORIES)}"
        )
    return [name for name, profile in DOSAGE_FORM_DATABASE.items() if profile.category == category]


def _build_pk_implications(profile: DosageFormProfile) -> str:
    """Build PK implications text based on the dosage form profile."""
    parts = []

    ba = profile.typical_bioavailability_pct
    parts.append(f"Expected bioavailability ~{ba:.0f}%.")

    if profile.first_pass:
        parts.append("Subject to hepatic first-pass metabolism; individual variability expected.")
    else:
        parts.append("Bypasses hepatic first-pass metabolism; more predictable systemic exposure.")

    if profile.absorption_rate == "immediate":
        parts.append("Rapid absorption; high Cmax possible with short Tmax.")
    elif profile.absorption_rate == "slow":
        parts.append("Slow absorption; blunted Cmax with delayed Tmax.")
    elif profile.absorption_rate == "extended":
        parts.append(
            "Extended/controlled release; reduced peak-trough fluctuation, once-daily dosing possible."
        )

    if profile.dose_number_relevant:
        parts.append("Dose number (BCS) affects dissolution and absorption completeness.")

    onset_h = profile.onset_min / 60.0
    parts.append(
        f"Clinical onset approximately {profile.onset_min:.0f} min; duration ~{profile.duration_h:.0f} h."
    )

    if onset_h < 0.1:
        parts.append("Near-immediate onset suitable for acute/emergency indications.")

    return " ".join(parts)


def _build_selection_rationale(
    profile: DosageFormProfile,
    logP: float,
    mw: float,
    solubility_mg_mL: float,
    target_bioavailability_pct: float,
    required_onset_min: float,
    required_duration_h: float,
) -> str:
    """Build a brief selection rationale string."""
    reasons = []

    ba_diff = abs(profile.typical_bioavailability_pct - target_bioavailability_pct)
    reasons.append(
        f"Form selected with BA {profile.typical_bioavailability_pct:.0f}% "
        f"(target {target_bioavailability_pct:.0f}%, difference {ba_diff:.0f}%)."
    )

    if profile.onset_min <= required_onset_min:
        reasons.append(
            f"Onset {profile.onset_min:.0f} min meets required {required_onset_min:.0f} min."
        )

    if profile.duration_h >= required_duration_h:
        reasons.append(
            f"Duration {profile.duration_h:.0f} h meets required {required_duration_h:.0f} h."
        )

    if solubility_mg_mL < 0.1 and logP > 3:
        reasons.append(
            f"Low solubility ({solubility_mg_mL:.3f} mg/mL) and high logP ({logP:.1f}) "
            "indicate lipophilic drug; lipid-compatible route preferred."
        )

    if not profile.first_pass:
        reasons.append("Avoids first-pass metabolism, providing more consistent exposure.")

    return " ".join(reasons)


def select_dosage_form(
    drug_name: str,
    logP: float,
    mw: float,
    solubility_mg_mL: float,
    required_onset_min: float = 60.0,
    required_duration_h: float = 8.0,
    oral_acceptable: bool = True,
    injection_acceptable: bool = True,
    target_bioavailability_pct: float = 50.0,
) -> DosageFormSelectionResult:
    """Select the best dosage form for a drug based on its properties and requirements.

    Args:
        drug_name: Name of the drug
        logP: Octanol-water partition coefficient
        mw: Molecular weight (Da)
        solubility_mg_mL: Aqueous solubility (mg/mL)
        required_onset_min: Maximum acceptable time to onset (minutes)
        required_duration_h: Minimum required duration (hours)
        oral_acceptable: Whether oral administration is acceptable
        injection_acceptable: Whether injection routes are acceptable
        target_bioavailability_pct: Target bioavailability percentage (0-100)

    Returns:
        DosageFormSelectionResult with recommended form and alternatives

    Raises:
        ValueError: If parameters are out of valid range
    """
    if required_onset_min <= 0:
        raise ValueError(f"required_onset_min must be > 0, got {required_onset_min}")
    if required_duration_h <= 0:
        raise ValueError(f"required_duration_h must be > 0, got {required_duration_h}")

    # Identify preferred forms for low solubility / high logP drugs
    prefer_lipophilic = solubility_mg_mL < 0.1 and logP > 3

    # Filter candidate forms
    candidates = []
    for name, profile in DOSAGE_FORM_DATABASE.items():
        # Check onset and duration constraints
        if profile.onset_min > required_onset_min:
            continue
        if profile.duration_h < required_duration_h:
            continue

        # Check route acceptability
        if not oral_acceptable and name in _ORAL_FORMS:
            continue
        if not injection_acceptable and name in _INJECTION_FORMS:
            continue

        candidates.append(profile)

    # If prefer lipophilic forms and we have candidates, boost transdermal / iv_infusion
    if prefer_lipophilic and candidates:
        lipophilic_preferred = [
            p for p in candidates if p.category in ("transdermal",) or p.form_name == "iv_infusion"
        ]
        if lipophilic_preferred:
            # Move preferred to front
            others = [p for p in candidates if p not in lipophilic_preferred]
            candidates = lipophilic_preferred + others

    if not candidates:
        # Fall back: use all forms filtered only by route acceptability
        fallback = []
        for name, profile in DOSAGE_FORM_DATABASE.items():
            if not oral_acceptable and name in _ORAL_FORMS:
                continue
            if not injection_acceptable and name in _INJECTION_FORMS:
                continue
            fallback.append(profile)
        candidates = fallback if fallback else list(DOSAGE_FORM_DATABASE.values())

    # Rank by closeness to target BA (ascending delta)
    if prefer_lipophilic:
        # lipophilic preferred already boosted; sort within each group by BA delta
        preferred_set = {
            p for p in candidates if p.category in ("transdermal",) or p.form_name == "iv_infusion"
        }
        sorted_preferred = sorted(
            [p for p in candidates if p in preferred_set],
            key=lambda p: abs(p.typical_bioavailability_pct - target_bioavailability_pct),
        )
        sorted_others = sorted(
            [p for p in candidates if p not in preferred_set],
            key=lambda p: abs(p.typical_bioavailability_pct - target_bioavailability_pct),
        )
        candidates = sorted_preferred + sorted_others
    else:
        candidates = sorted(
            candidates,
            key=lambda p: abs(p.typical_bioavailability_pct - target_bioavailability_pct),
        )

    recommended = candidates[0]
    alternatives = tuple(p.form_name for p in candidates[1:3])

    rationale = _build_selection_rationale(
        recommended,
        logP,
        mw,
        solubility_mg_mL,
        target_bioavailability_pct,
        required_onset_min,
        required_duration_h,
    )

    pk_implications = _build_pk_implications(recommended)

    notes_parts = [f"Analysis for drug '{drug_name}'."]
    if mw > 500:
        notes_parts.append(f"High MW ({mw:.0f} Da) may limit oral absorption (Lipinski rule of 5).")
    if logP > 5:
        notes_parts.append(f"High logP ({logP:.1f}) may indicate poor aqueous solubility.")
    if solubility_mg_mL < 0.1:
        notes_parts.append(
            f"Very low solubility ({solubility_mg_mL:.4f} mg/mL); "
            "dissolution may be rate-limiting for oral forms."
        )
    if len(notes_parts) == 1:
        notes_parts.append("Drug properties within typical range for selected route.")

    return DosageFormSelectionResult(
        drug_name=drug_name,
        recommended_form=recommended.form_name,
        alternative_forms=alternatives,
        selection_rationale=rationale,
        bioavailability_estimate_pct=recommended.typical_bioavailability_pct,
        onset_estimate_min=recommended.onset_min,
        pk_implications=pk_implications,
        notes=" ".join(notes_parts),
    )
