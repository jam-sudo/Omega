"""Physiological fluid volumes, organ weights, blood flows, and pH values.

Comprehensive physiological constants database for PBPK modeling across
species and life stages. Reference sources: ICRP 2002, Davies & Morris 1993.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PhysiologyProfile:
    """Complete physiological profile for a given species/sex/age category."""

    species: str  # "human", "rat", "mouse", "dog", "monkey"
    sex: str  # "male", "female", "mixed"
    age_category: str  # "adult", "pediatric", "neonatal", "elderly"
    weight_kg: float

    # Fluid volumes (L)
    plasma_volume_L: float
    blood_volume_L: float
    interstitial_volume_L: float
    intracellular_volume_L: float
    total_body_water_L: float

    # Organ volumes (L)
    liver_volume_L: float
    kidney_volume_L: float
    lung_volume_L: float
    brain_volume_L: float
    muscle_volume_L: float
    fat_volume_L: float
    gut_volume_L: float
    heart_volume_L: float
    skin_volume_L: float
    bone_volume_L: float

    # Blood flows (L/h)
    cardiac_output_L_per_h: float
    hepatic_blood_flow_L_per_h: float
    renal_blood_flow_L_per_h: float
    pulmonary_blood_flow_L_per_h: float  # equals cardiac output

    # pH values
    plasma_ph: float
    urine_ph: float
    gastric_ph: float
    small_intestine_ph: float
    colon_ph: float

    # Organ weights (g)
    liver_weight_g: float
    kidney_weight_g: float

    notes: str


# ── Reference profiles ───────────────────────────────────────────────────────

_HUMAN_ADULT_MALE_70KG = PhysiologyProfile(
    species="human",
    sex="male",
    age_category="adult",
    weight_kg=70.0,
    # Fluid volumes (L) — ICRP 2002
    plasma_volume_L=3.0,
    blood_volume_L=5.6,
    interstitial_volume_L=12.0,
    intracellular_volume_L=25.0,
    total_body_water_L=42.0,
    # Organ volumes (L)
    liver_volume_L=1.5,
    kidney_volume_L=0.3,
    lung_volume_L=1.17,
    brain_volume_L=1.4,
    muscle_volume_L=22.0,
    fat_volume_L=14.4,
    gut_volume_L=1.0,
    heart_volume_L=0.3,
    skin_volume_L=3.0,
    bone_volume_L=8.0,
    # Blood flows (L/h) — 6.5 L/min cardiac output
    cardiac_output_L_per_h=390.0,
    hepatic_blood_flow_L_per_h=96.0,
    renal_blood_flow_L_per_h=72.0,
    pulmonary_blood_flow_L_per_h=390.0,
    # pH values
    plasma_ph=7.4,
    urine_ph=6.0,
    gastric_ph=1.5,
    small_intestine_ph=6.5,
    colon_ph=6.8,
    # Organ weights (g)
    liver_weight_g=1500.0,
    kidney_weight_g=310.0,
    notes="ICRP 2002 reference human adult male, 70 kg",
)

_HUMAN_ADULT_FEMALE_60KG = PhysiologyProfile(
    species="human",
    sex="female",
    age_category="adult",
    weight_kg=60.0,
    # Fluid volumes scaled ~0.857x from male (60/70)
    plasma_volume_L=2.5,
    blood_volume_L=4.7,
    interstitial_volume_L=10.3,
    intracellular_volume_L=21.4,
    total_body_water_L=35.0,
    # Organ volumes — slightly smaller
    liver_volume_L=1.3,
    kidney_volume_L=0.26,
    lung_volume_L=1.0,
    brain_volume_L=1.3,
    muscle_volume_L=17.0,
    fat_volume_L=16.0,
    gut_volume_L=0.86,
    heart_volume_L=0.25,
    skin_volume_L=2.6,
    bone_volume_L=6.9,
    # Blood flows — cardiac output ~5.5 L/min
    cardiac_output_L_per_h=330.0,
    hepatic_blood_flow_L_per_h=82.0,
    renal_blood_flow_L_per_h=62.0,
    pulmonary_blood_flow_L_per_h=330.0,
    # pH values same as male
    plasma_ph=7.4,
    urine_ph=6.0,
    gastric_ph=1.5,
    small_intestine_ph=6.5,
    colon_ph=6.8,
    liver_weight_g=1300.0,
    kidney_weight_g=265.0,
    notes="ICRP 2002 reference human adult female, 60 kg",
)

_RAT_ADULT_MALE_250G = PhysiologyProfile(
    species="rat",
    sex="male",
    age_category="adult",
    weight_kg=0.25,
    # Fluid volumes (L) — Davies & Morris 1993
    plasma_volume_L=0.0073,
    blood_volume_L=0.015,
    interstitial_volume_L=0.040,
    intracellular_volume_L=0.095,
    total_body_water_L=0.155,
    # Organ volumes (L)
    liver_volume_L=0.010,
    kidney_volume_L=0.002,
    lung_volume_L=0.0015,
    brain_volume_L=0.0018,
    muscle_volume_L=0.100,
    fat_volume_L=0.020,
    gut_volume_L=0.004,
    heart_volume_L=0.00085,
    skin_volume_L=0.018,
    bone_volume_L=0.030,
    # Blood flows (L/h) — 85 mL/min cardiac output
    cardiac_output_L_per_h=5.1,
    hepatic_blood_flow_L_per_h=1.25,
    renal_blood_flow_L_per_h=0.95,
    pulmonary_blood_flow_L_per_h=5.1,
    # pH values — rat gastric pH slightly higher than human
    plasma_ph=7.4,
    urine_ph=6.2,
    gastric_ph=3.5,
    small_intestine_ph=6.8,
    colon_ph=7.0,
    liver_weight_g=10.0,
    kidney_weight_g=2.0,
    notes="Davies & Morris 1993 reference rat adult male, 250 g",
)

_MOUSE_ADULT_MALE_20G = PhysiologyProfile(
    species="mouse",
    sex="male",
    age_category="adult",
    weight_kg=0.02,
    # Scale from rat using allometry
    plasma_volume_L=0.00077,
    blood_volume_L=0.0016,
    interstitial_volume_L=0.0042,
    intracellular_volume_L=0.0099,
    total_body_water_L=0.016,
    liver_volume_L=0.00105,
    kidney_volume_L=0.00021,
    lung_volume_L=0.00016,
    brain_volume_L=0.00042,
    muscle_volume_L=0.0105,
    fat_volume_L=0.0021,
    gut_volume_L=0.00042,
    heart_volume_L=0.000089,
    skin_volume_L=0.0019,
    bone_volume_L=0.0031,
    # Blood flows — ~1.8 L/h cardiac output (30 mL/min)
    cardiac_output_L_per_h=1.8,
    hepatic_blood_flow_L_per_h=0.44,
    renal_blood_flow_L_per_h=0.33,
    pulmonary_blood_flow_L_per_h=1.8,
    plasma_ph=7.4,
    urine_ph=6.2,
    gastric_ph=3.5,
    small_intestine_ph=6.8,
    colon_ph=7.0,
    liver_weight_g=1.05,
    kidney_weight_g=0.21,
    notes="Mouse adult male, 20 g; allometrically derived from rat",
)

_DOG_ADULT_MALE_10KG = PhysiologyProfile(
    species="dog",
    sex="male",
    age_category="adult",
    weight_kg=10.0,
    # Scale from human allometrically (10/70)^1.0 for volumes, (10/70)^0.75 for flows
    plasma_volume_L=0.43,
    blood_volume_L=0.80,
    interstitial_volume_L=1.71,
    intracellular_volume_L=3.57,
    total_body_water_L=6.0,
    liver_volume_L=0.21,
    kidney_volume_L=0.043,
    lung_volume_L=0.167,
    brain_volume_L=0.077,
    muscle_volume_L=3.14,
    fat_volume_L=2.06,
    gut_volume_L=0.143,
    heart_volume_L=0.043,
    skin_volume_L=0.43,
    bone_volume_L=1.14,
    # Cardiac output ~2.6 L/min (156 L/h) for 10 kg dog
    cardiac_output_L_per_h=110.0,
    hepatic_blood_flow_L_per_h=27.0,
    renal_blood_flow_L_per_h=20.0,
    pulmonary_blood_flow_L_per_h=110.0,
    plasma_ph=7.4,
    urine_ph=6.0,
    gastric_ph=2.0,
    small_intestine_ph=6.5,
    colon_ph=6.8,
    liver_weight_g=213.0,
    kidney_weight_g=44.0,
    notes="Dog adult male, 10 kg; scaled from human reference",
)

_MONKEY_ADULT_MALE_3KG = PhysiologyProfile(
    species="monkey",
    sex="male",
    age_category="adult",
    weight_kg=3.0,
    plasma_volume_L=0.13,
    blood_volume_L=0.24,
    interstitial_volume_L=0.51,
    intracellular_volume_L=1.07,
    total_body_water_L=1.8,
    liver_volume_L=0.064,
    kidney_volume_L=0.013,
    lung_volume_L=0.050,
    brain_volume_L=0.060,
    muscle_volume_L=0.94,
    fat_volume_L=0.62,
    gut_volume_L=0.043,
    heart_volume_L=0.013,
    skin_volume_L=0.13,
    bone_volume_L=0.34,
    # Cardiac output for 3 kg monkey
    cardiac_output_L_per_h=45.0,
    hepatic_blood_flow_L_per_h=11.0,
    renal_blood_flow_L_per_h=8.2,
    pulmonary_blood_flow_L_per_h=45.0,
    plasma_ph=7.4,
    urine_ph=6.0,
    gastric_ph=1.8,
    small_intestine_ph=6.5,
    colon_ph=6.8,
    liver_weight_g=64.0,
    kidney_weight_g=13.0,
    notes="Cynomolgus monkey adult male, 3 kg; scaled from human reference",
)

# Registry of built-in reference profiles
_REFERENCE_PROFILES: dict[tuple[str, str, str], PhysiologyProfile] = {
    ("human", "male", "adult"): _HUMAN_ADULT_MALE_70KG,
    ("human", "female", "adult"): _HUMAN_ADULT_FEMALE_60KG,
    ("human", "mixed", "adult"): _HUMAN_ADULT_MALE_70KG,  # use male as default mixed
    ("rat", "male", "adult"): _RAT_ADULT_MALE_250G,
    ("rat", "female", "adult"): _RAT_ADULT_MALE_250G,
    ("rat", "mixed", "adult"): _RAT_ADULT_MALE_250G,
    ("mouse", "male", "adult"): _MOUSE_ADULT_MALE_20G,
    ("mouse", "female", "adult"): _MOUSE_ADULT_MALE_20G,
    ("mouse", "mixed", "adult"): _MOUSE_ADULT_MALE_20G,
    ("dog", "male", "adult"): _DOG_ADULT_MALE_10KG,
    ("dog", "female", "adult"): _DOG_ADULT_MALE_10KG,
    ("dog", "mixed", "adult"): _DOG_ADULT_MALE_10KG,
    ("monkey", "male", "adult"): _MONKEY_ADULT_MALE_3KG,
    ("monkey", "female", "adult"): _MONKEY_ADULT_MALE_3KG,
    ("monkey", "mixed", "adult"): _MONKEY_ADULT_MALE_3KG,
}

_VALID_SPECIES = {"human", "rat", "mouse", "dog", "monkey"}
_VALID_SEXES = {"male", "female", "mixed"}
_VALID_AGE_CATEGORIES = {"adult", "pediatric", "neonatal", "elderly"}

# Age category volume scaling factors relative to adult
_AGE_VOLUME_SCALE = {
    "adult": 1.0,
    "elderly": 0.9,
    "pediatric": 0.7,
    "neonatal": 0.1,
}

# Age category blood flow scaling factors relative to adult
_AGE_FLOW_SCALE = {
    "adult": 1.0,
    "elderly": 0.85,
    "pediatric": 0.7,
    "neonatal": 0.15,
}


def _scale_profile(
    base: PhysiologyProfile,
    new_weight_kg: float,
    new_age_category: str | None = None,
) -> PhysiologyProfile:
    """Scale a profile to a new body weight and/or age category.

    Volumes scale linearly with weight (exponent 1.0).
    Blood flows scale with weight^0.75 (allometric).
    """
    wt_ratio = new_weight_kg / base.weight_kg
    flow_ratio = wt_ratio**0.75

    age = new_age_category if new_age_category is not None else base.age_category
    vol_age = _AGE_VOLUME_SCALE[age] / _AGE_VOLUME_SCALE[base.age_category]
    flow_age = _AGE_FLOW_SCALE[age] / _AGE_FLOW_SCALE[base.age_category]

    v = wt_ratio * vol_age
    f = flow_ratio * flow_age

    return replace(
        base,
        weight_kg=new_weight_kg,
        age_category=age,
        # Fluid volumes
        plasma_volume_L=base.plasma_volume_L * v,
        blood_volume_L=base.blood_volume_L * v,
        interstitial_volume_L=base.interstitial_volume_L * v,
        intracellular_volume_L=base.intracellular_volume_L * v,
        total_body_water_L=base.total_body_water_L * v,
        # Organ volumes
        liver_volume_L=base.liver_volume_L * v,
        kidney_volume_L=base.kidney_volume_L * v,
        lung_volume_L=base.lung_volume_L * v,
        brain_volume_L=base.brain_volume_L * v,
        muscle_volume_L=base.muscle_volume_L * v,
        fat_volume_L=base.fat_volume_L * v,
        gut_volume_L=base.gut_volume_L * v,
        heart_volume_L=base.heart_volume_L * v,
        skin_volume_L=base.skin_volume_L * v,
        bone_volume_L=base.bone_volume_L * v,
        # Blood flows
        cardiac_output_L_per_h=base.cardiac_output_L_per_h * f,
        hepatic_blood_flow_L_per_h=base.hepatic_blood_flow_L_per_h * f,
        renal_blood_flow_L_per_h=base.renal_blood_flow_L_per_h * f,
        pulmonary_blood_flow_L_per_h=base.pulmonary_blood_flow_L_per_h * f,
        # Organ weights
        liver_weight_g=base.liver_weight_g * v,
        kidney_weight_g=base.kidney_weight_g * v,
        notes=f"Scaled from {base.notes} to {new_weight_kg:.3f} kg ({age})",
    )


def get_physiology(
    species: str = "human",
    sex: str = "male",
    age_category: str = "adult",
    weight_kg: float | None = None,
) -> PhysiologyProfile:
    """Return a physiological profile for the given species/sex/age/weight.

    Parameters
    ----------
    species : str
        One of "human", "rat", "mouse", "dog", "monkey".
    sex : str
        One of "male", "female", "mixed".
    age_category : str
        One of "adult", "pediatric", "neonatal", "elderly".
    weight_kg : float, optional
        If provided, scale volumes and flows to this body weight.

    Returns
    -------
    PhysiologyProfile
        A frozen dataclass with fluid volumes, organ volumes, blood flows,
        pH values, and organ weights.

    Raises
    ------
    ValueError
        If species, sex, or age_category is invalid.
    """
    if species not in _VALID_SPECIES:
        raise ValueError(f"Unknown species '{species}'. Valid options: {sorted(_VALID_SPECIES)}")
    if sex not in _VALID_SEXES:
        raise ValueError(f"Unknown sex '{sex}'. Valid options: {sorted(_VALID_SEXES)}")
    if age_category not in _VALID_AGE_CATEGORIES:
        raise ValueError(
            f"Unknown age_category '{age_category}'. Valid options: {sorted(_VALID_AGE_CATEGORIES)}"
        )

    # Look up the adult reference (we store only adult references; age scaling applied below)
    key = (species, sex, "adult")
    if key not in _REFERENCE_PROFILES:
        # Fall back to male adult for that species
        key = (species, "male", "adult")

    base = _REFERENCE_PROFILES[key]

    # Determine effective weight for scaling
    if weight_kg is not None and weight_kg != base.weight_kg:
        profile = _scale_profile(base, weight_kg, age_category)
    elif age_category != "adult":
        # Scale by age but keep base weight
        profile = _scale_profile(base, base.weight_kg, age_category)
    else:
        profile = base

    return profile


def list_available_profiles() -> list[str]:
    """Return list of available (species, sex, age_category) profile descriptions.

    Returns
    -------
    list[str]
        Sorted list of strings describing available combinations.
    """
    base_combos = [
        f"{species}/{sex}/adult" for (species, sex, _) in sorted(_REFERENCE_PROFILES.keys())
    ]
    # Add age categories derived by scaling
    derived = []
    for species, sex, _ in sorted(_REFERENCE_PROFILES.keys()):
        for age in ("pediatric", "neonatal", "elderly"):
            derived.append(f"{species}/{sex}/{age}")

    return sorted(set(base_combos + derived))


def scale_physiology(base_profile: PhysiologyProfile, new_weight_kg: float) -> PhysiologyProfile:
    """Scale a physiological profile to a new body weight.

    Volumes scale linearly with weight (exponent 1.0).
    Blood flows scale with weight^0.75 (allometric).

    Parameters
    ----------
    base_profile : PhysiologyProfile
        The reference profile to scale from.
    new_weight_kg : float
        Target body weight in kg.

    Returns
    -------
    PhysiologyProfile
        New profile scaled to the target weight.
    """
    if new_weight_kg <= 0:
        raise ValueError(f"new_weight_kg must be positive, got {new_weight_kg}")
    return _scale_profile(base_profile, new_weight_kg)


__all__ = [
    "PhysiologyProfile",
    "get_physiology",
    "list_available_profiles",
    "scale_physiology",
]
