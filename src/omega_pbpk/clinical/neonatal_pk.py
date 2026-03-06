"""Neonatal PK: PK parameter adjustment for neonates (0-28 days) and infants (<1 year)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "NeonatalScalingFactors",
    "NeonatalDoseResult",
    "neonatal_scaling",
    "adjust_dose_neonatal",
]


@dataclass(frozen=True)
class NeonatalScalingFactors:
    age_days: float
    weight_kg: float
    gfr_mL_per_min_per_kg: float       # mL/min/1.73m²
    hepatic_enzyme_maturation: float   # 0-1 relative to adult
    bsa_m2: float                      # body surface area
    total_body_water_fraction: float   # TBW/BW (neonates: ~0.8, adults: ~0.6)
    plasma_protein_fraction: float     # albumin fraction relative to adult (neonates: ~0.5)
    cardiac_output_fraction: float     # relative to adult
    cyp3a4_maturation: float
    ugt_maturation: float
    renal_maturation: float


@dataclass(frozen=True)
class NeonatalDoseResult:
    drug_name: str
    age_days: float
    weight_kg: float
    adult_dose_mg_per_kg: float
    neonatal_dose_mg_per_kg: float
    neonatal_dose_mg: float           # absolute dose
    cl_adult_mL_per_min_per_kg: float
    cl_neonatal_mL_per_min_per_kg: float
    t_half_adult_h: float
    t_half_neonatal_h: float
    vd_adult_L_per_kg: float
    vd_neonatal_L_per_kg: float
    dose_interval_h: float
    rationale: str
    cautions: list[str]
    scaling: NeonatalScalingFactors


def neonatal_scaling(age_days: float, weight_kg: float) -> NeonatalScalingFactors:
    """Compute neonatal physiological scaling factors for a given age and weight."""
    if age_days < 0:
        raise ValueError("age_days must be >= 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")

    # GFR: sigmoidal maturation
    gfr = 1.0 + 29.0 * (age_days / (age_days + 14.0))  # mL/min per 1.73m²
    gfr_per_kg = gfr / max(weight_kg, 0.5)

    # CYP3A4: ~10% at birth, maturation toward 6 months
    cyp3a4 = 0.1 + 0.9 * (age_days / (age_days + 30.0))

    # UGT: 25% at birth
    ugt = 0.25 + 0.75 * (age_days / (age_days + 60.0))

    # Renal maturation: normalized to adult 120 mL/min
    renal = gfr / 120.0

    # Hepatic enzyme maturation (average CYP)
    hepatic_enz = 0.1 + 0.9 * (age_days / (age_days + 40.0))

    # BSA (Mosteller): BSA = sqrt(height_cm * weight_kg / 3600)
    height_cm = 50.0 + age_days * 0.1
    bsa = (height_cm * weight_kg / 3600.0) ** 0.5

    # Total body water: neonates ~80%, decreasing
    tbw = max(0.6, 0.80 - 0.002 * age_days)

    # Plasma protein: 50-60% of adult at birth
    ppb = min(1.0, 0.5 + 0.01 * age_days)

    # Cardiac output: per kg higher in neonates
    co = max(1.0, min(1.5, 1.5 - 0.01 * age_days))

    return NeonatalScalingFactors(
        age_days=age_days,
        weight_kg=weight_kg,
        gfr_mL_per_min_per_kg=gfr_per_kg,
        hepatic_enzyme_maturation=hepatic_enz,
        bsa_m2=bsa,
        total_body_water_fraction=tbw,
        plasma_protein_fraction=ppb,
        cardiac_output_fraction=co,
        cyp3a4_maturation=cyp3a4,
        ugt_maturation=ugt,
        renal_maturation=renal,
    )


def adjust_dose_neonatal(
    drug_name: str,
    adult_dose_mg_per_kg: float,
    cl_adult_mL_per_min_per_kg: float,
    vd_adult_L_per_kg: float,
    age_days: float,
    weight_kg: float,
    elimination: str = "hepatic",  # "hepatic", "renal", "mixed"
    t_half_adult_h: float = 12.0,
) -> NeonatalDoseResult:
    """Adjust adult dose to neonatal based on physiological maturation scaling."""
    # Validate inputs
    if adult_dose_mg_per_kg <= 0:
        raise ValueError("adult_dose_mg_per_kg must be > 0")
    if cl_adult_mL_per_min_per_kg <= 0:
        raise ValueError("cl_adult_mL_per_min_per_kg must be > 0")
    if vd_adult_L_per_kg <= 0:
        raise ValueError("vd_adult_L_per_kg must be > 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if age_days < 0:
        raise ValueError("age_days must be >= 0")

    valid_eliminations = {"hepatic", "renal", "mixed"}
    if elimination not in valid_eliminations:
        raise ValueError(
            f"elimination must be one of {valid_eliminations}, got '{elimination}'"
        )

    sf = neonatal_scaling(age_days, weight_kg)

    # CL scaling factor based on elimination pathway
    if elimination == "hepatic":
        cl_factor = sf.hepatic_enzyme_maturation
    elif elimination == "renal":
        cl_factor = sf.renal_maturation
    else:  # mixed
        cl_factor = 0.5 * sf.hepatic_enzyme_maturation + 0.5 * sf.renal_maturation

    # Neonatal CL
    cl_neo = cl_adult_mL_per_min_per_kg * cl_factor

    # Vd: higher in neonates due to higher TBW
    vd_factor = sf.total_body_water_fraction / 0.6  # relative to adult TBW
    vd_neo = vd_adult_L_per_kg * vd_factor

    # t_half neonatal: CL in L/min/kg
    cl_neo_L_per_min_per_kg = max(cl_neo / 1000.0, 1e-9)
    t_half_neo = 0.693 * vd_neo / cl_neo_L_per_min_per_kg / 60.0  # convert to hours

    # Neonatal dose per kg: proportional to CL factor
    neo_dose_per_kg = adult_dose_mg_per_kg * cl_factor
    neo_dose = neo_dose_per_kg * weight_kg

    # Dose interval based on neonatal t_half
    if t_half_neo > 24:
        interval = 24.0
    elif t_half_neo > 12:
        interval = 12.0
    else:
        interval = 8.0

    # Build rationale
    rationale = (
        f"Neonatal dose adjusted for age {age_days:.0f} days, weight {weight_kg:.2f} kg. "
        f"Elimination pathway: {elimination}. "
        f"CL maturation factor: {cl_factor:.2f}. "
        f"Neonatal t½: {t_half_neo:.1f} h (adult: {t_half_adult_h:.1f} h). "
        f"Vd neonatal: {vd_neo:.2f} L/kg (adult: {vd_adult_L_per_kg:.2f} L/kg). "
        f"Recommended interval: every {interval:.0f} h."
    )

    # Build cautions
    cautions: list[str] = []
    if sf.cyp3a4_maturation < 0.3:
        cautions.append(
            "CYP3A4 highly immature — avoid CYP3A4 substrates with narrow TI"
        )
    if sf.renal_maturation < 0.2:
        cautions.append(
            "Renal function very immature — renal dose adjustment critical"
        )
    if cl_factor < 0.2:
        cautions.append("Very low CL maturation — consider TDM")

    return NeonatalDoseResult(
        drug_name=drug_name,
        age_days=age_days,
        weight_kg=weight_kg,
        adult_dose_mg_per_kg=adult_dose_mg_per_kg,
        neonatal_dose_mg_per_kg=neo_dose_per_kg,
        neonatal_dose_mg=neo_dose,
        cl_adult_mL_per_min_per_kg=cl_adult_mL_per_min_per_kg,
        cl_neonatal_mL_per_min_per_kg=cl_neo,
        t_half_adult_h=t_half_adult_h,
        t_half_neonatal_h=t_half_neo,
        vd_adult_L_per_kg=vd_adult_L_per_kg,
        vd_neonatal_L_per_kg=vd_neo,
        dose_interval_h=interval,
        rationale=rationale,
        cautions=cautions,
        scaling=sf,
    )
