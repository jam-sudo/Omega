"""Neonatal PK: PK parameter adjustment for neonates (0-28 days) and infants (<1 year)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "NeonatalScalingFactors",
    "NeonatalDoseResult",
    "neonatal_scaling",
    "adjust_dose_neonatal",
    # Phase 287 additions
    "NeonatalScaling",
    "NeonatalDoseResult287",
    "NeonatalPKResult",
    "neonatal_scaling_factors",
    "scale_neonatal_dose",
    "simulate_neonatal_pk",
]


@dataclass(frozen=True)
class NeonatalScalingFactors:
    age_days: float
    weight_kg: float
    gfr_mL_per_min_per_kg: float  # mL/min/1.73m²
    hepatic_enzyme_maturation: float  # 0-1 relative to adult
    bsa_m2: float  # body surface area
    total_body_water_fraction: float  # TBW/BW (neonates: ~0.8, adults: ~0.6)
    plasma_protein_fraction: float  # albumin fraction relative to adult (neonates: ~0.5)
    cardiac_output_fraction: float  # relative to adult
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
    neonatal_dose_mg: float  # absolute dose
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
        raise ValueError(f"elimination must be one of {valid_eliminations}, got '{elimination}'")

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
        cautions.append("CYP3A4 highly immature — avoid CYP3A4 substrates with narrow TI")
    if sf.renal_maturation < 0.2:
        cautions.append("Renal function very immature — renal dose adjustment critical")
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


# ===========================================================================
# Phase 287 — Neonatal PK Scaling (new API)
# ===========================================================================


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


@dataclass(frozen=True)
class NeonatalScaling:
    """Physiological scaling factors for neonates/infants (Phase 287)."""

    gestational_age_weeks: float
    postnatal_age_days: float
    cyp3a4_factor: float
    cyp2d6_factor: float
    cyp2c9_factor: float
    cyp2c19_factor: float
    gfr_factor: float
    protein_binding_factor: float
    body_water_factor: float
    maturation_stage: str


@dataclass(frozen=True)
class NeonatalDoseResult287:
    """Result of neonatal dose scaling (Phase 287)."""

    drug_name: str
    weight_kg: float
    gestational_age_weeks: float
    postnatal_age_days: float
    scaling: NeonatalScaling
    adult_dose_mg_per_kg: float
    recommended_dose_mg: float
    cl_scale: float
    notes: str


@dataclass(frozen=True)
class NeonatalPKResult:
    """Result of neonatal PK simulation (Phase 287)."""

    drug_name: str
    times_h: list[float]
    c_plasma: list[float]
    cmax: float
    tmax_h: float
    auc: float
    t_half_h: float
    cl_neonatal: float
    vd_neonatal: float
    notes: str


def _maturation_stage(pna_days: float, ga_weeks: float) -> str:
    """Classify maturation stage based on GA and PNA."""
    pma_weeks = ga_weeks + pna_days / 7.0
    if pma_weeks < 28:
        return "extremely_preterm"
    elif pma_weeks < 32:
        return "very_preterm"
    elif pma_weeks < 37:
        return "preterm"
    elif pna_days <= 28:
        return "term_neonate"
    elif pna_days <= 182:
        return "young_infant"
    else:
        return "infant"


def neonatal_scaling_factors(
    gestational_age_weeks: float,
    postnatal_age_days: float,
) -> NeonatalScaling:
    """Compute physiological maturation scaling factors for neonates/infants.

    Parameters
    ----------
    gestational_age_weeks:
        Gestational age at birth (24–42 weeks).
    postnatal_age_days:
        Postnatal age in days (0–365).

    Returns
    -------
    NeonatalScaling
        Scaling factors relative to adult values (1.0 = adult).
    """
    if not (24.0 <= gestational_age_weeks <= 42.0):
        raise ValueError(f"gestational_age_weeks must be 24–42, got {gestational_age_weeks}")
    if not (0.0 <= postnatal_age_days <= 365.0):
        raise ValueError(f"postnatal_age_days must be 0–365, got {postnatal_age_days}")

    pna = postnatal_age_days

    cyp3a4 = 0.1 + 0.9 * _sigmoid((pna - 7.0) / 30.0)
    cyp2d6 = 0.05 + 0.95 * _sigmoid((pna - 14.0) / 45.0)
    cyp2c9 = 0.1 + 0.9 * _sigmoid((pna - 21.0) / 60.0)
    cyp2c19 = 0.05 + 0.95 * _sigmoid((pna - 30.0) / 90.0)
    gfr = 0.1 + 0.9 * _sigmoid((pna - 3.0) / 10.0)
    protein_binding = 0.7 + 0.3 * _sigmoid((pna - 7.0) / 14.0)
    body_water = 1.0 + 0.25 * math.exp(-pna / 30.0)

    stage = _maturation_stage(pna, gestational_age_weeks)

    return NeonatalScaling(
        gestational_age_weeks=gestational_age_weeks,
        postnatal_age_days=postnatal_age_days,
        cyp3a4_factor=cyp3a4,
        cyp2d6_factor=cyp2d6,
        cyp2c9_factor=cyp2c9,
        cyp2c19_factor=cyp2c19,
        gfr_factor=gfr,
        protein_binding_factor=protein_binding,
        body_water_factor=body_water,
        maturation_stage=stage,
    )


def scale_neonatal_dose(
    drug_name: str,
    adult_dose_mg_per_kg: float,
    weight_kg: float,
    gestational_age_weeks: float,
    postnatal_age_days: float,
    fm_cyp3a4: float = 0.5,
    fm_cyp2d6: float = 0.1,
    fm_cyp2c9: float = 0.2,
    fm_cyp2c19: float = 0.1,
    f_renal: float = 0.1,
) -> NeonatalDoseResult287:
    """Scale an adult mg/kg dose to a neonate/infant based on maturation factors.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    adult_dose_mg_per_kg:
        Adult dose in mg/kg.
    weight_kg:
        Neonatal/infant body weight in kg.
    gestational_age_weeks:
        GA at birth (24–42 weeks).
    postnatal_age_days:
        PNA in days (0–365).
    fm_cyp3a4, fm_cyp2d6, fm_cyp2c9, fm_cyp2c19:
        Fraction metabolised by each CYP isoform (sum with f_renal ≤ 1).
    f_renal:
        Fraction eliminated renally.
    """
    if adult_dose_mg_per_kg <= 0:
        raise ValueError("adult_dose_mg_per_kg must be > 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if any(f < 0 for f in (fm_cyp3a4, fm_cyp2d6, fm_cyp2c9, fm_cyp2c19, f_renal)):
        raise ValueError("All fm/f_renal fractions must be >= 0")
    total_fm = fm_cyp3a4 + fm_cyp2d6 + fm_cyp2c9 + fm_cyp2c19 + f_renal
    if total_fm > 1.0 + 1e-9:
        raise ValueError(f"Sum of fm fractions ({total_fm:.4f}) must not exceed 1.0")

    sc = neonatal_scaling_factors(gestational_age_weeks, postnatal_age_days)

    cl_scale = (
        fm_cyp3a4 * sc.cyp3a4_factor
        + fm_cyp2d6 * sc.cyp2d6_factor
        + fm_cyp2c9 * sc.cyp2c9_factor
        + fm_cyp2c19 * sc.cyp2c19_factor
        + f_renal * sc.gfr_factor
    )

    # Adjust for protein binding: lower protein_binding_factor → higher fu → higher unbound CL
    # If adult protein_binding_factor (reference) = 1.0, fu_ratio = 1/protein_binding_factor
    # But we cap the adjustment to avoid unrealistic amplification
    pb_adj = 1.0 / max(sc.protein_binding_factor, 0.5)
    cl_scale_adj = cl_scale * pb_adj

    recommended_dose_mg = adult_dose_mg_per_kg * weight_kg * cl_scale_adj

    notes = (
        f"CL scale: {cl_scale:.3f} (before PB adj), {cl_scale_adj:.3f} (after PB adj). "
        f"PB factor: {sc.protein_binding_factor:.3f}. "
        f"Stage: {sc.maturation_stage}."
    )

    return NeonatalDoseResult287(
        drug_name=drug_name,
        weight_kg=weight_kg,
        gestational_age_weeks=gestational_age_weeks,
        postnatal_age_days=postnatal_age_days,
        scaling=sc,
        adult_dose_mg_per_kg=adult_dose_mg_per_kg,
        recommended_dose_mg=recommended_dose_mg,
        cl_scale=cl_scale_adj,
        notes=notes,
    )


def simulate_neonatal_pk(
    drug_name: str,
    dose_mg: float,
    weight_kg: float,
    gestational_age_weeks: float,
    postnatal_age_days: float,
    cl_adult_per_kg_L_per_h: float,
    vd_adult_per_kg_L: float,
    fm_cyp3a4: float = 0.5,
    fm_cyp2d6: float = 0.1,
    fm_cyp2c9: float = 0.2,
    fm_cyp2c19: float = 0.1,
    f_renal: float = 0.1,
    route: str = "iv",
    ka_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> NeonatalPKResult:
    """Simulate neonatal/infant PK using a 1-compartment Forward Euler model.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Administered dose in mg.
    weight_kg:
        Neonatal body weight in kg.
    gestational_age_weeks:
        GA at birth (24–42 weeks).
    postnatal_age_days:
        PNA in days (0–365).
    cl_adult_per_kg_L_per_h:
        Adult clearance in L/h/kg.
    vd_adult_per_kg_L:
        Adult volume of distribution in L/kg.
    fm_cyp3a4, fm_cyp2d6, fm_cyp2c9, fm_cyp2c19, f_renal:
        Metabolic/renal fractions (sum ≤ 1).
    route:
        "iv" or "oral".
    ka_per_h:
        First-order absorption rate (oral only), h⁻¹.
    t_end_h:
        Simulation end time, hours.
    dt_h:
        Forward Euler time step, hours.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if cl_adult_per_kg_L_per_h <= 0:
        raise ValueError("cl_adult_per_kg_L_per_h must be > 0")
    if vd_adult_per_kg_L <= 0:
        raise ValueError("vd_adult_per_kg_L must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if any(f < 0 for f in (fm_cyp3a4, fm_cyp2d6, fm_cyp2c9, fm_cyp2c19, f_renal)):
        raise ValueError("All fm/f_renal fractions must be >= 0")
    total_fm = fm_cyp3a4 + fm_cyp2d6 + fm_cyp2c9 + fm_cyp2c19 + f_renal
    if total_fm > 1.0 + 1e-9:
        raise ValueError(f"Sum of fm fractions ({total_fm:.4f}) must not exceed 1.0")

    sc = neonatal_scaling_factors(gestational_age_weeks, postnatal_age_days)

    # CL scale factor (same logic as scale_neonatal_dose, without PB adjustment here
    # since we are working with total plasma concentration)
    cl_scale = (
        fm_cyp3a4 * sc.cyp3a4_factor
        + fm_cyp2d6 * sc.cyp2d6_factor
        + fm_cyp2c9 * sc.cyp2c9_factor
        + fm_cyp2c19 * sc.cyp2c19_factor
        + f_renal * sc.gfr_factor
    )

    cl_neo = cl_adult_per_kg_L_per_h * weight_kg * cl_scale  # L/h
    vd_neo = vd_adult_per_kg_L * weight_kg * sc.body_water_factor  # L

    ke = cl_neo / max(vd_neo, 1e-9)  # h⁻¹

    # Forward Euler simulation
    n_steps = max(1, int(t_end_h / dt_h))
    times: list[float] = []
    c_plasma: list[float] = []

    if route == "iv":
        c = dose_mg / max(vd_neo, 1e-9)
        gut = 0.0
    else:
        c = 0.0
        gut = dose_mg  # mg in gut

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        c_plasma.append(max(c, 0.0))

        if i < n_steps:
            if route == "oral":
                abs_rate = ka_per_h * gut
                dc_abs = abs_rate / max(vd_neo, 1e-9)
                dgut = -abs_rate * dt_h
                gut = max(gut + dgut, 0.0)
            else:
                dc_abs = 0.0
            dc = (dc_abs - ke * c) * dt_h
            c = max(c + dc, 0.0)

    arr_c = np.array(c_plasma)
    arr_t = np.array(times)

    cmax = float(arr_c.max())
    tmax_h = float(arr_t[int(arr_c.argmax())])

    # AUC via trapezoidal rule
    auc = float(np.trapz(arr_c, arr_t))

    t_half_h = 0.693 / max(ke, 1e-9)

    notes = (
        f"Route: {route}. CL scale: {cl_scale:.3f}. "
        f"Neo CL: {cl_neo:.3f} L/h, Neo Vd: {vd_neo:.3f} L. "
        f"Stage: {sc.maturation_stage}."
    )

    return NeonatalPKResult(
        drug_name=drug_name,
        times_h=times,
        c_plasma=c_plasma,
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        t_half_h=t_half_h,
        cl_neonatal=cl_neo,
        vd_neonatal=vd_neo,
        notes=notes,
    )
