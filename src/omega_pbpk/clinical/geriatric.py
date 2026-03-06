"""Geriatric PK scaling for elderly patients (>=65 years).

Physiological changes with aging affect drug pharmacokinetics:
- Decreased hepatic blood flow and CYP activity
- Declining GFR (~1 mL/min/year after 40)
- Reduced cardiac output
- Increased body fat, decreased lean body mass
- Lower serum albumin
- Decreased intestinal P-glycoprotein

References: Turnheim 2003, Klotz 2009, McLachlan & Pont 2012.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeriatricScalingFactors:
    """Age-related physiological scaling factors."""

    age_years: float
    weight_kg: float
    sex: str
    height_cm: float
    gfr_mL_per_min: float
    hepatic_flow_factor: float
    cyp_activity_factor: float
    cardiac_output_factor: float
    lean_body_mass_kg: float
    body_fat_fraction: float
    albumin_factor: float
    pgp_intestinal_factor: float
    polypharmacy_ddi_risk: str


@dataclass(frozen=True)
class GeriatricDoseAdjustment:
    """Result of geriatric dose adjustment."""

    age_years: float
    sex: str
    base_dose_mg: float
    adjusted_dose_mg: float
    cl_scaling_factor: float
    vd_scaling_factor: float
    predicted_t_half_h: float
    dose_reduction_pct: float
    rationale: str
    cautions: list[str] = field(default_factory=list)
    scaling: GeriatricScalingFactors = field(default=None)  # type: ignore[assignment]


def _validate_inputs(age_years: float, weight_kg: float, sex: str) -> None:
    if age_years <= 0:
        raise ValueError(f"age_years must be > 0, got {age_years}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    sex_lower = sex.lower()
    if sex_lower not in ("male", "female", "m", "f"):
        raise ValueError(f"sex must be 'male' or 'female', got '{sex}'")


def _is_female(sex: str) -> bool:
    return sex.lower() in ("female", "f")


def age_normalized_gfr(age_years: float, weight_kg: float, sex: str) -> float:
    """Cockcroft-Gault GFR estimate (mL/min)."""
    _validate_inputs(age_years, weight_kg, sex)
    scr = 1.0  # default estimated serum creatinine (mg/dL)
    crcl = (140 - age_years) * weight_kg / (72 * scr)
    if _is_female(sex):
        crcl *= 0.85
    return float(max(5.0, min(120.0, crcl)))


def geriatric_scaling(
    age_years: float,
    weight_kg: float,
    sex: str,
    height_cm: float = 170.0,
    n_comedications: int = 0,
) -> GeriatricScalingFactors:
    """Compute geriatric physiological scaling factors."""
    _validate_inputs(age_years, weight_kg, sex)

    bmi = weight_kg / (height_cm / 100.0) ** 2
    female = _is_female(sex)

    # Lean body mass (Janmahasatian 2005)
    if female:
        lbm = 9.27e3 * weight_kg / (8.78e3 + 244 * bmi)
    else:
        lbm = 9.27e3 * weight_kg / (6.68e3 + 216 * bmi)

    body_fat_fraction = max(0.0, min(1.0, (weight_kg - lbm) / weight_kg))

    # GFR via CKD-EPI simplified (Cockcroft-Gault)
    gfr = age_normalized_gfr(age_years, weight_kg, sex)

    # Hepatic flow: 1.0 at age 25, linear decline 0.7%/yr after 40
    hepatic_flow = max(0.5, 1.0 - 0.007 * max(0.0, age_years - 40))

    # CYP activity: 1.0 at age 40, decline 1%/yr
    cyp = max(0.5, 1.0 - 0.010 * max(0.0, age_years - 40))

    # Cardiac output: 1.0 at age 30, decline 0.6%/yr
    co = max(0.6, 1.0 - 0.006 * max(0.0, age_years - 30))

    # Albumin: decline after 65
    alb = max(0.75, 1.0 - 0.0075 * max(0.0, age_years - 65))

    # P-gp intestinal: decline after 50
    pgp = max(0.6, 1.0 - 0.01 * max(0.0, age_years - 50))

    # Polypharmacy risk
    if n_comedications < 3:
        poly_risk = "low"
    elif n_comedications <= 5:
        poly_risk = "moderate"
    else:
        poly_risk = "high"

    return GeriatricScalingFactors(
        age_years=age_years,
        weight_kg=weight_kg,
        sex=sex.lower(),
        height_cm=height_cm,
        gfr_mL_per_min=gfr,
        hepatic_flow_factor=hepatic_flow,
        cyp_activity_factor=cyp,
        cardiac_output_factor=co,
        lean_body_mass_kg=lbm,
        body_fat_fraction=body_fat_fraction,
        albumin_factor=alb,
        pgp_intestinal_factor=pgp,
        polypharmacy_ddi_risk=poly_risk,
    )


def adjust_dose_geriatric(
    base_dose_mg: float,
    drug_cl_L_per_h: float,
    drug_vd_L: float,
    age_years: float,
    weight_kg: float,
    sex: str,
    height_cm: float = 170.0,
    elimination: str = "hepatic",
    protein_binding: float = 0.9,
    n_comedications: int = 0,
) -> GeriatricDoseAdjustment:
    """Adjust dose for geriatric patient based on age-related PK changes."""
    _validate_inputs(age_years, weight_kg, sex)

    sf = geriatric_scaling(age_years, weight_kg, sex, height_cm, n_comedications)

    # CL scaling
    if elimination == "hepatic":
        cl_factor = sf.cyp_activity_factor * sf.hepatic_flow_factor
    elif elimination == "renal":
        cl_factor = sf.gfr_mL_per_min / 100.0
    elif elimination == "mixed":
        cl_factor = 0.5 * (sf.cyp_activity_factor * sf.hepatic_flow_factor) + 0.5 * (
            sf.gfr_mL_per_min / 100.0
        )
    else:
        raise ValueError(f"elimination must be 'hepatic', 'renal', or 'mixed', got '{elimination}'")

    # Vd scaling
    if protein_binding > 0.9:
        # Lipophilic: Vd increases with fat
        vd_factor = 1.0 + 0.3 * sf.body_fat_fraction
    else:
        # Hydrophilic: Vd decreases with age
        if age_years > 40:
            vd_factor = max(0.7, 1.0 - 0.2 * (age_years - 40) / 40)
        else:
            vd_factor = 1.0

    adjusted_dose = base_dose_mg * cl_factor
    adjusted_cl = drug_cl_L_per_h * cl_factor
    adjusted_vd = drug_vd_L * vd_factor
    t_half = 0.693 * adjusted_vd / adjusted_cl if adjusted_cl > 0 else float("inf")
    dose_reduction_pct = max(0.0, min(100.0, (1.0 - cl_factor) * 100.0))

    # Rationale
    rationale = (
        f"Age {age_years:.0f} {sex}: CL scaled by {cl_factor:.2f} "
        f"({elimination} elimination), Vd scaled by {vd_factor:.2f}. "
        f"Dose reduced {dose_reduction_pct:.1f}%."
    )

    # Cautions
    cautions: list[str] = []
    if dose_reduction_pct > 40:
        cautions.append(
            "Significant dose reduction (>40%) — monitor for narrow therapeutic index drugs"
        )
    if n_comedications > 5:
        cautions.append("High polypharmacy risk — check for drug-drug interactions")
    if sf.gfr_mL_per_min < 60:
        cautions.append(
            f"Renal impairment (GFR {sf.gfr_mL_per_min:.0f} mL/min)"
            " — consider renal dose adjustment"
        )
    if sf.hepatic_flow_factor < 0.7:
        cautions.append("Reduced hepatic blood flow — monitor hepatically cleared drugs")

    return GeriatricDoseAdjustment(
        age_years=age_years,
        sex=sex.lower(),
        base_dose_mg=base_dose_mg,
        adjusted_dose_mg=adjusted_dose,
        cl_scaling_factor=cl_factor,
        vd_scaling_factor=vd_factor,
        predicted_t_half_h=t_half,
        dose_reduction_pct=dose_reduction_pct,
        rationale=rationale,
        cautions=cautions,
        scaling=sf,
    )
