"""
Phase 448 — Drug and metabolite excretion prediction into urine and bile
using mass balance approach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExcretionPredictionResult:
    """Predicted drug excretion profile via renal and biliary routes."""

    drug_name: str
    dose_mg: float
    fe_urine_unchanged_pct: float  # % of dose excreted unchanged in urine
    fe_urine_metabolites_pct: float  # % as metabolites in urine
    fe_bile_unchanged_pct: float  # % unchanged in bile
    fe_bile_metabolites_pct: float  # % as metabolites in bile
    fe_expired_air_pct: float  # % exhaled (volatile compounds)
    fe_total_accounted_pct: float  # should sum close to 100%
    primary_excretion_route: str  # "renal", "biliary", "mixed", "metabolic_then_renal"
    urinary_t50_h: float  # time to 50% of urinary excretion
    cumulative_urine_mg_at_24h: float  # absolute amount in urine at 24h
    renal_sensitivity_index: float  # fe_urine * (1 - fu_plasma)
    biliary_recycling_potential: str  # "high", "moderate", "low"
    notes: str


def predict_drug_excretion(
    drug_name: str,
    dose_mg: float,
    fu_plasma: float,
    logp: float,
    mw_Da: float,
    cl_hepatic_L_per_h: float,
    gfr_L_per_h: float = 7.5,
    t_half_h: float = 4.0,
) -> ExcretionPredictionResult:
    """
    Predict renal and biliary excretion fractions using a mass balance model.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        Administered dose (mg)
    fu_plasma : float
        Unbound fraction in plasma (0 < fu_plasma <= 1)
    logp : float
        Octanol-water partition coefficient
    mw_Da : float
        Molecular weight (Da)
    cl_hepatic_L_per_h : float
        Hepatic clearance (L/h), >= 0
    gfr_L_per_h : float
        Glomerular filtration rate (L/h), default 7.5 (~125 mL/min)
    t_half_h : float
        Systemic half-life (h), used for urinary timing estimates
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in range (0, 1]")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if cl_hepatic_L_per_h < 0:
        raise ValueError("cl_hepatic_L_per_h must be >= 0")
    if gfr_L_per_h <= 0:
        raise ValueError("gfr_L_per_h must be > 0")
    if t_half_h <= 0:
        raise ValueError("t_half_h must be > 0")

    # --- Renal filtration clearance ---
    cl_renal_filter = gfr_L_per_h * fu_plasma

    # --- Tubular reabsorption factor based on logP ---
    # logP > 1 → 70% reabsorbed; logP 0-1 → 30%; logP < 0 → 5%
    if logp > 1.0:
        reabsorption_factor = 0.70
    elif logp >= 0.0:
        reabsorption_factor = 0.30
    else:
        reabsorption_factor = 0.05

    cl_renal_net = cl_renal_filter * (1.0 - reabsorption_factor)

    # --- Total clearance ---
    cl_other = 0.1  # small other clearance
    cl_total = cl_hepatic_L_per_h + cl_renal_net + cl_other

    # --- Fraction excreted unchanged in urine ---
    fe_urine_unchanged = cl_renal_net / cl_total  # fraction (0–1)

    # --- Biliary fraction ---
    # High MW and logP → significant biliary excretion
    if mw_Da > 500 and logp > 2.0:
        biliary_frac = 0.15
    elif mw_Da > 500 or logp > 2.0:
        biliary_frac = 0.08
    else:
        biliary_frac = 0.02

    fe_bile_unchanged_frac = biliary_frac * (1.0 - fe_urine_unchanged)

    # --- Metabolic fraction excreted ---
    metabolic_remaining = 1.0 - fe_urine_unchanged - fe_bile_unchanged_frac
    metabolic_remaining = max(0.0, metabolic_remaining)

    fe_urine_metabolites_frac = metabolic_remaining * 0.70
    fe_bile_metabolites_frac = metabolic_remaining * 0.30

    # --- Expired air ---
    if logp > 4.0 or mw_Da < 100.0:
        fe_expired_air_frac = 0.02
    else:
        fe_expired_air_frac = 0.0

    # --- Convert all to percentages ---
    fe_urine_unchanged_pct = fe_urine_unchanged * 100.0
    fe_urine_metabolites_pct = fe_urine_metabolites_frac * 100.0
    fe_bile_unchanged_pct = fe_bile_unchanged_frac * 100.0
    fe_bile_metabolites_pct = fe_bile_metabolites_frac * 100.0
    fe_expired_air_pct = fe_expired_air_frac * 100.0

    fe_total_accounted_pct = (
        fe_urine_unchanged_pct
        + fe_urine_metabolites_pct
        + fe_bile_unchanged_pct
        + fe_bile_metabolites_pct
        + fe_expired_air_pct
    )

    # --- Primary excretion route ---
    fe_renal_total = fe_urine_unchanged_pct + fe_urine_metabolites_pct
    fe_bile_total = fe_bile_unchanged_pct + fe_bile_metabolites_pct

    if fe_urine_unchanged_pct >= 40.0:
        primary_route = "renal"
    elif fe_bile_total >= 20.0:
        primary_route = "biliary"
    elif fe_renal_total >= 30.0 and fe_bile_total >= 15.0:
        primary_route = "mixed"
    else:
        primary_route = "metabolic_then_renal"

    # --- Urinary t50 ---
    # Urinary rate follows systemic elimination: t50 ~ t_half * 0.8
    urinary_t50_h = t_half_h * 0.8

    # --- Cumulative urine at 24h ---
    # Amount = dose * fe_urine_total_fraction * (1 - exp(-0.693/t_half * 24))
    fe_urine_total_frac = (fe_urine_unchanged_pct + fe_urine_metabolites_pct) / 100.0
    frac_eliminated_24h = 1.0 - math.exp(-0.693 / t_half_h * 24.0)
    cumulative_urine_mg_at_24h = dose_mg * fe_urine_total_frac * frac_eliminated_24h

    # --- Renal sensitivity index ---
    renal_sensitivity_index = (fe_urine_unchanged_pct / 100.0) * (1.0 - fu_plasma)
    renal_sensitivity_index = max(0.0, min(1.0, renal_sensitivity_index))

    # --- Biliary recycling potential ---
    fe_bile_combined_pct = fe_bile_unchanged_pct + fe_bile_metabolites_pct
    if fe_bile_combined_pct > 15.0:
        biliary_recycling_potential = "high"
    elif fe_bile_combined_pct >= 5.0:
        biliary_recycling_potential = "moderate"
    else:
        biliary_recycling_potential = "low"

    notes = (
        f"Primary route: {primary_route}. "
        f"Renal unchanged: {fe_urine_unchanged_pct:.1f}%, "
        f"Biliary total: {fe_bile_combined_pct:.1f}%, "
        f"Metabolite-urine: {fe_urine_metabolites_pct:.1f}%. "
        f"logP={logp:.2f}, MW={mw_Da:.0f} Da, fu={fu_plasma:.3f}."
    )

    return ExcretionPredictionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fe_urine_unchanged_pct=fe_urine_unchanged_pct,
        fe_urine_metabolites_pct=fe_urine_metabolites_pct,
        fe_bile_unchanged_pct=fe_bile_unchanged_pct,
        fe_bile_metabolites_pct=fe_bile_metabolites_pct,
        fe_expired_air_pct=fe_expired_air_pct,
        fe_total_accounted_pct=fe_total_accounted_pct,
        primary_excretion_route=primary_route,
        urinary_t50_h=urinary_t50_h,
        cumulative_urine_mg_at_24h=cumulative_urine_mg_at_24h,
        renal_sensitivity_index=renal_sensitivity_index,
        biliary_recycling_potential=biliary_recycling_potential,
        notes=notes,
    )


def compare_excretion_profiles(
    compounds_list: list[dict[str, Any]],
) -> list[ExcretionPredictionResult]:
    """
    Compare excretion profiles for multiple compounds.

    Parameters
    ----------
    compounds_list : list of dict
        Each dict must contain the required keys for predict_drug_excretion:
        drug_name, dose_mg, fu_plasma, logp, mw_Da, cl_hepatic_L_per_h
        Optional: gfr_L_per_h, t_half_h

    Returns
    -------
    List of ExcretionPredictionResult sorted by fe_urine_unchanged_pct descending.
    """
    results = []
    for compound in compounds_list:
        res = predict_drug_excretion(
            drug_name=compound["drug_name"],
            dose_mg=compound["dose_mg"],
            fu_plasma=compound["fu_plasma"],
            logp=compound["logp"],
            mw_Da=compound["mw_Da"],
            cl_hepatic_L_per_h=compound["cl_hepatic_L_per_h"],
            gfr_L_per_h=compound.get("gfr_L_per_h", 7.5),
            t_half_h=compound.get("t_half_h", 4.0),
        )
        results.append(res)

    results.sort(key=lambda r: r.fe_urine_unchanged_pct, reverse=True)
    return results
