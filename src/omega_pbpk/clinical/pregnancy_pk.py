"""Gestational age-dependent PK changes model for pregnancy."""

from __future__ import annotations

import math
from dataclasses import dataclass

_ALLOWED_ENZYMES = {"cyp3a4", "cyp2d6", "cyp1a2", "renal", "none"}

_TRIMESTER_REPRESENTATIVE_GA = {"first": 8.0, "second": 20.0, "third": 32.0}


@dataclass(frozen=True)
class PregnancyPKResult:
    drug_name: str
    gestational_age_weeks: float
    trimester: str  # "first", "second", "third"
    cl_scaling_factor: float  # CL relative to non-pregnant baseline
    vd_scaling_factor: float  # Vd relative to non-pregnant baseline
    adjusted_cl_L_per_h: float
    adjusted_vd_L: float
    adjusted_t_half_h: float
    plasma_volume_L: float  # expanded plasma volume
    gfr_mL_per_min: float  # increased GFR in pregnancy
    cyp3a4_induction_factor: float
    cyp2d6_induction_factor: float
    cyp1a2_suppression_factor: float
    p_glycoprotein_factor: float
    dose_adjustment_factor: float  # recommended dose / baseline dose
    notes: str


def _classify_trimester(gestational_age_weeks: float) -> str:
    if gestational_age_weeks <= 13:
        return "first"
    elif gestational_age_weeks <= 26:
        return "second"
    else:
        return "third"


def simulate_pregnancy_pk(
    drug_name: str,
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    gestational_age_weeks: float,  # 0-42
    primary_enzyme: str = "cyp3a4",
    baseline_gfr_mL_per_min: float = 110.0,
    protein_binding_fraction: float = 0.9,
) -> PregnancyPKResult:
    """Compute pregnancy-adjusted PK parameters at a given gestational age."""
    # --- Validation ---
    if gestational_age_weeks < 0 or gestational_age_weeks > 42:
        raise ValueError(f"gestational_age_weeks must be in [0, 42], got {gestational_age_weeks}")
    if baseline_cl_L_per_h <= 0:
        raise ValueError(f"baseline_cl_L_per_h must be > 0, got {baseline_cl_L_per_h}")
    if baseline_vd_L <= 0:
        raise ValueError(f"baseline_vd_L must be > 0, got {baseline_vd_L}")
    if not (0.0 <= protein_binding_fraction < 1.0):
        raise ValueError(
            f"protein_binding_fraction must be in [0, 1), got {protein_binding_fraction}"
        )
    if primary_enzyme not in _ALLOWED_ENZYMES:
        raise ValueError(
            f"primary_enzyme must be one of {_ALLOWED_ENZYMES}, got {primary_enzyme!r}"
        )

    ga = gestational_age_weeks
    ga_frac = ga / 40.0  # normalized 0–1 at term

    # Plasma volume expansion: baseline 2.75 L + 1.2*ga_frac L
    plasma_volume = 2.75 + 1.2 * ga_frac

    # GFR: increases up to 50% by mid-pregnancy (saturates at GA=20)
    gfr = baseline_gfr_mL_per_min * (1.0 + 0.5 * min(ga / 20.0, 1.0))

    # Enzyme induction/suppression factors
    cyp3a4_factor = 1.0 + 0.35 * ga_frac
    cyp2d6_factor = 1.0 + 0.45 * ga_frac
    cyp1a2_factor = max(0.7, 1.0 - 0.30 * ga_frac)
    pgp_factor = 1.0 + 0.15 * ga_frac

    # Albumin hemodilution effect on protein binding
    baseline_fu = 1.0 - protein_binding_fraction
    albumin_factor = 1.0 - 0.15 * ga_frac  # albumin decreases 15% at term
    # Avoid division by zero when albumin_factor is tiny
    if albumin_factor <= 0:
        albumin_factor = 1e-6
    fu_adjusted = min(1.0, baseline_fu / albumin_factor)
    # Binding CL factor: ratio of adjusted fu to baseline fu
    if baseline_fu <= 0:
        # Drug is completely unbound — binding changes have no effect
        binding_cl_factor = 1.0
    else:
        binding_cl_factor = fu_adjusted / baseline_fu

    # CL scaling based on primary elimination pathway
    if primary_enzyme == "renal":
        cl_factor = (gfr / baseline_gfr_mL_per_min) * binding_cl_factor
    elif primary_enzyme == "cyp3a4":
        cl_factor = cyp3a4_factor * binding_cl_factor
    elif primary_enzyme == "cyp2d6":
        cl_factor = cyp2d6_factor * binding_cl_factor
    elif primary_enzyme == "cyp1a2":
        cl_factor = cyp1a2_factor * binding_cl_factor
    else:  # "none"
        cl_factor = binding_cl_factor

    # Vd scaling: plasma expansion + tissue changes
    vd_factor = 1.0 + 0.4 * ga_frac

    # Adjusted PK
    adjusted_cl = baseline_cl_L_per_h * cl_factor
    adjusted_vd = baseline_vd_L * vd_factor
    adjusted_t_half = (math.log(2) * adjusted_vd) / adjusted_cl if adjusted_cl > 0 else float("nan")

    # Dose adjustment: scale proportionally to CL change (maintain same exposure)
    dose_adjustment_factor = cl_factor

    trimester = _classify_trimester(ga)

    notes = (
        f"Pregnancy PK at GA {ga:.1f} weeks ({trimester} trimester). "
        f"Primary enzyme: {primary_enzyme}. "
        f"CL factor: {cl_factor:.3f}, Vd factor: {vd_factor:.3f}."
    )

    return PregnancyPKResult(
        drug_name=drug_name,
        gestational_age_weeks=ga,
        trimester=trimester,
        cl_scaling_factor=cl_factor,
        vd_scaling_factor=vd_factor,
        adjusted_cl_L_per_h=adjusted_cl,
        adjusted_vd_L=adjusted_vd,
        adjusted_t_half_h=adjusted_t_half,
        plasma_volume_L=plasma_volume,
        gfr_mL_per_min=gfr,
        cyp3a4_induction_factor=cyp3a4_factor,
        cyp2d6_induction_factor=cyp2d6_factor,
        cyp1a2_suppression_factor=cyp1a2_factor,
        p_glycoprotein_factor=pgp_factor,
        dose_adjustment_factor=dose_adjustment_factor,
        notes=notes,
    )


def compare_trimesters(
    drug_name: str,
    baseline_cl_L_per_h: float,
    baseline_vd_L: float,
    primary_enzyme: str = "cyp3a4",
) -> list:
    """Return list of 3 PregnancyPKResult for representative GA weeks: 8, 20, 32."""
    results = []
    for ga in [8.0, 20.0, 32.0]:
        result = simulate_pregnancy_pk(
            drug_name=drug_name,
            baseline_cl_L_per_h=baseline_cl_L_per_h,
            baseline_vd_L=baseline_vd_L,
            gestational_age_weeks=ga,
            primary_enzyme=primary_enzyme,
        )
        results.append(result)
    return results
