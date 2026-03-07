"""
Phase 632 — Microbiome PK

Model drug-gut microbiome interactions: microbial metabolism, bioactivation,
and colonocyte effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MicrobiomePKResult:
    drug_name: str
    microbiome_metabolism_pct: float  # % of drug metabolized by microbiome
    bioactivation_pct: float  # % activated to toxic/active form by microbiome
    gut_exposure_mg_h: float  # AUC in gut lumen
    systemic_reduction_pct: (
        float  # % reduction in systemic bioavailability due to microbial metabolism
    )
    gut_half_life_h: float  # half-life of drug in gut lumen
    nitroreductase_risk: bool  # susceptible to nitroreductase (has NO2)
    azoreductase_risk: bool  # susceptible to azoreductase (has azo N=N)
    glucuronidase_risk: bool  # hydrolysis of glucuronide conjugates
    enterotype_sensitivity: str  # "low","moderate","high" sensitivity to microbiome variability
    colonocyte_cl_L_per_h: float  # colonocyte intrinsic CL
    notes: str


def estimate_microbial_clearance(
    logP: float,
    has_nitro: bool = False,
    has_azo: bool = False,
) -> float:
    """
    Estimate first-order microbial degradation rate (per hour).
    Higher for nitro/azo compounds (enzymatic reduction).
    """
    base = 0.1 + 0.2 * max(0.0, logP)
    if has_nitro:
        base += 1.0
    if has_azo:
        base += 0.8
    return max(0.0, base)


def gut_lumen_pk(
    dose_mg: float,
    transit_time_h: float,
    microbial_cl_per_h: float,
) -> dict:
    """
    Simple first-order gut lumen PK.
    C(t) = dose_mg / V_gut * exp(-microbial_cl_per_h * t)
    V_gut ≈ 1 L

    Returns: {"auc": float, "c_max": float, "t_half": float, "remaining_mg": float}
    """
    v_gut = 1.0  # L
    c_max = dose_mg / v_gut  # mg/L

    if microbial_cl_per_h > 0:
        # AUC = C0/k * (1 - exp(-k*t))
        auc = (
            (dose_mg / v_gut)
            / microbial_cl_per_h
            * (1.0 - math.exp(-microbial_cl_per_h * transit_time_h))
        )
        t_half = math.log(2.0) / microbial_cl_per_h
        remaining_mg = dose_mg * math.exp(-microbial_cl_per_h * transit_time_h)
    else:
        # No microbial clearance: AUC = C0 * t, remaining = dose
        auc = (dose_mg / v_gut) * transit_time_h
        t_half = float("inf")
        remaining_mg = dose_mg

    return {
        "auc": auc,
        "c_max": c_max,
        "t_half": t_half,
        "remaining_mg": remaining_mg,
    }


def simulate_microbiome_pk(
    drug_name: str,
    dose_mg: float,
    gut_transit_time_h: float = 24.0,
    microbial_cl_per_h: float = 0.5,
    f_bioactivated: float = 0.0,
    has_nitro: bool = False,
    has_azo: bool = False,
    has_glucuronide_conjugate: bool = False,
    absorption_fraction: float = 0.3,
    logP: float = 2.0,
    colonocyte_clint_mL_per_min: float = 0.0,
) -> MicrobiomePKResult:
    """
    Simulate drug-gut microbiome interaction PK.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if gut_transit_time_h <= 0:
        raise ValueError("gut_transit_time_h must be > 0")
    if microbial_cl_per_h < 0:
        raise ValueError("microbial_cl_per_h must be >= 0")
    if not (0.0 <= absorption_fraction <= 1.0):
        raise ValueError("absorption_fraction must be in [0, 1]")
    if not (0.0 <= f_bioactivated <= 1.0):
        raise ValueError("f_bioactivated must be in [0, 1]")

    # Compute gut lumen PK
    lumen = gut_lumen_pk(dose_mg, gut_transit_time_h, microbial_cl_per_h)

    # Fraction metabolized by microbiome
    if microbial_cl_per_h > 0:
        frac_metabolized = 1.0 - math.exp(-microbial_cl_per_h * gut_transit_time_h)
    else:
        frac_metabolized = 0.0

    microbiome_metabolism_pct = frac_metabolized * 100.0
    bioactivation_pct = f_bioactivated * frac_metabolized * 100.0

    # Systemic reduction: drug metabolized in gut that would otherwise be absorbed
    systemic_reduction_pct = frac_metabolized * (1.0 - absorption_fraction) * 100.0

    # Gut half-life
    if microbial_cl_per_h > 0:
        gut_half_life_h = math.log(2.0) / microbial_cl_per_h
    else:
        gut_half_life_h = float("inf")

    # Structural risk flags
    nitroreductase_risk = has_nitro
    azoreductase_risk = has_azo
    glucuronidase_risk = has_glucuronide_conjugate

    # Enterotype sensitivity
    if nitroreductase_risk or azoreductase_risk:
        enterotype_sensitivity = "high"
    elif glucuronidase_risk:
        enterotype_sensitivity = "moderate"
    else:
        enterotype_sensitivity = "low"

    # Colonocyte CL: mL/min → L/h
    colonocyte_cl_L_per_h = colonocyte_clint_mL_per_min * 60.0 / 1000.0

    notes_parts = [
        f"drug={drug_name}",
        f"microbial_cl={microbial_cl_per_h:.3f}/h",
        f"metabolism={microbiome_metabolism_pct:.1f}%",
        f"bioactivation={bioactivation_pct:.1f}%",
        f"enterotype_sensitivity={enterotype_sensitivity}",
    ]
    notes = "; ".join(notes_parts)

    return MicrobiomePKResult(
        drug_name=drug_name,
        microbiome_metabolism_pct=microbiome_metabolism_pct,
        bioactivation_pct=bioactivation_pct,
        gut_exposure_mg_h=lumen["auc"],
        systemic_reduction_pct=systemic_reduction_pct,
        gut_half_life_h=gut_half_life_h,
        nitroreductase_risk=nitroreductase_risk,
        azoreductase_risk=azoreductase_risk,
        glucuronidase_risk=glucuronidase_risk,
        enterotype_sensitivity=enterotype_sensitivity,
        colonocyte_cl_L_per_h=colonocyte_cl_L_per_h,
        notes=notes,
    )
