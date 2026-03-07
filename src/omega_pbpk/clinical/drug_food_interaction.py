"""Drug-food interaction modeling: quantitative PK impact of food on absorption.

This module provides quantitative PK food effect calculations distinct from
the existing food_drug_interaction.py qualitative assessments.
Follows FDA food effect guidance (2019).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FoodEffectResult",
    "assess_food_effect",
    "high_fat_meal_impact",
    # Phase 743 additions
    "FoodEffectPKResult",
    "assess_food_effect_pk",
    "compare_food_conditions",
]

_VALID_BCS = {"I", "II", "III", "IV"}
_VALID_FOOD_CONDITIONS = {"fasted", "fed_high_fat", "fed_low_fat"}
_VALID_FORMULATIONS = {"IR", "ER", "enteric_coated"}


@dataclass(frozen=True)
class FoodEffectResult:
    """Result of a drug-food interaction assessment."""

    drug_name: str
    bcs_class: str
    food_condition: str
    cmax_ratio_fed_fasted: float
    auc_ratio_fed_fasted: float
    tmax_delay_h: float
    food_effect_category: str
    clinical_relevance: str
    recommendation: str
    notes: str


def assess_food_effect(
    drug_name: str,
    bcs_class: str,
    logP: float,
    pka: float = 7.0,
    solubility_mg_mL: float = 1.0,
    dose_mg: float = 100.0,
    formulation: str = "IR",
    food_condition: str = "fed_high_fat",
) -> FoodEffectResult:
    """Assess quantitative food effect on drug PK based on BCS class and properties.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    bcs_class:
        BCS classification: "I", "II", "III", or "IV".
    logP:
        Octanol-water partition coefficient (log scale).
    pka:
        Drug pKa (used for solubility adjustments, reserved for extensions).
    solubility_mg_mL:
        Aqueous solubility (mg/mL).
    dose_mg:
        Dose in mg.
    formulation:
        Formulation type: "IR", "ER", or "enteric_coated".
    food_condition:
        Food condition being compared to fasted: "fasted", "fed_high_fat", "fed_low_fat".

    Returns
    -------
    FoodEffectResult
    """
    if bcs_class not in _VALID_BCS:
        raise ValueError(f"Invalid BCS class '{bcs_class}'. Must be one of: {sorted(_VALID_BCS)}")
    if food_condition not in _VALID_FOOD_CONDITIONS:
        valid = sorted(_VALID_FOOD_CONDITIONS)
        raise ValueError(f"Invalid food_condition '{food_condition}'. Must be one of: {valid}")
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(
            f"Invalid formulation '{formulation}'. Must be one of: {sorted(_VALID_FORMULATIONS)}"
        )

    # Fasted vs fasted: ratios are 1.0, no delay
    if food_condition == "fasted":
        auc_ratio = 1.0
        cmax_ratio = 1.0
        tmax_delay = 0.0
    else:
        # FDA-based AUC ratio model
        if bcs_class == "I":
            # Minimal food effect; slight positive if lipophilic
            auc_ratio = 1.0 + 0.05 * logP / 5.0
        elif bcs_class == "II":
            # Lipophilic drugs benefit from fat: bile-mediated solubilization
            auc_ratio = (
                1.0
                + 0.3 * min(logP, 5.0) / 5.0
                + (0.1 if food_condition == "fed_high_fat" else 0.0)
            )
        elif bcs_class == "III":
            # Reduced motility generally reduces absorption
            auc_ratio = 0.85
        else:  # BCS IV
            # Variable; slight benefit if somewhat lipophilic
            auc_ratio = 0.9 + 0.2 * min(logP, 3.0) / 3.0

        # Cmax: food delays absorption more than it affects AUC
        cmax_ratio = auc_ratio * (0.8 if food_condition == "fed_high_fat" else 0.9)

        # Tmax delay relative to fasted
        tmax_delay = 1.0 if food_condition == "fed_high_fat" else 0.5

    # Enteric-coated formulations have inherent delay; blunt food effect on tmax
    if formulation == "enteric_coated" and food_condition != "fasted":
        tmax_delay = max(tmax_delay, 2.0)

    # Determine food effect category based on AUC ratio
    food_effect = auc_ratio - 1.0
    if abs(food_effect) < 0.1:
        food_effect_category = "neutral"
    elif food_effect > 0:
        food_effect_category = "positive"
    else:
        food_effect_category = "negative"

    # Clinical relevance: FDA threshold is >20% change
    max_change = max(abs(auc_ratio - 1.0), abs(cmax_ratio - 1.0))
    if max_change < 0.1:
        clinical_relevance = "none"
    elif max_change < 0.2:
        clinical_relevance = "low"
    elif max_change < 0.5:
        clinical_relevance = "moderate"
    else:
        clinical_relevance = "high"

    # Recommendation
    if food_effect_category == "positive" and clinical_relevance in ("moderate", "high"):
        recommendation = "Take with food"
    elif food_effect_category == "negative" and clinical_relevance in ("moderate", "high"):
        recommendation = "Take on empty stomach"
    else:
        recommendation = "May be taken with or without food"

    notes = (
        f"BCS {bcs_class} drug (logP={logP:.1f}); "
        f"AUC ratio (fed/fasted)={auc_ratio:.2f}, "
        f"Cmax ratio={cmax_ratio:.2f}; "
        f"formulation={formulation}."
    )

    return FoodEffectResult(
        drug_name=drug_name,
        bcs_class=bcs_class,
        food_condition=food_condition,
        cmax_ratio_fed_fasted=round(cmax_ratio, 4),
        auc_ratio_fed_fasted=round(auc_ratio, 4),
        tmax_delay_h=round(tmax_delay, 2),
        food_effect_category=food_effect_category,
        clinical_relevance=clinical_relevance,
        recommendation=recommendation,
        notes=notes,
    )


def high_fat_meal_impact(
    drug_name: str,
    bcs_class: str,
    logP: float,
    **kwargs,
) -> list[FoodEffectResult]:
    """Return food effect results for all three food conditions.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    bcs_class:
        BCS classification: "I", "II", "III", or "IV".
    logP:
        Octanol-water partition coefficient (log scale).
    **kwargs:
        Additional keyword arguments forwarded to `assess_food_effect`
        (except `food_condition`).

    Returns
    -------
    list[FoodEffectResult]
        Results for "fasted", "fed_low_fat", and "fed_high_fat".
    """
    return [
        assess_food_effect(drug_name, bcs_class, logP, food_condition=cond, **kwargs)
        for cond in ("fasted", "fed_low_fat", "fed_high_fat")
    ]


# ===========================================================================
# Phase 743 — Drug-Food Interaction Predictor (simulation-based, 1-cpt ODE)
# ===========================================================================



@dataclass(frozen=True)
class FoodEffectPKResult:
    """Simulation-based drug-food interaction result (Phase 743)."""

    drug_name: str
    food_type: str
    cmax_fasted: float
    cmax_fed: float
    auc_fasted: float
    auc_fed: float
    tmax_fasted_h: float
    tmax_fed_h: float
    cmax_ratio: float
    auc_ratio: float
    food_effect_magnitude: str
    fda_recommendation: str
    notes: str


def _simulate_1cpt_food(
    dose_mg: float,
    kge: float,
    ka: float,
    ke: float,
    vd: float,
    f_eff: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[float, float, float]:
    """Simulate 1-cpt with stomach → gut → plasma ODE (Forward Euler).

    Returns (cmax, tmax_h, auc).
    f_eff is the effective bioavailability fraction applied at gut absorption.
    """
    n_steps = max(1, int(t_end_h / dt_h))
    a_stomach = dose_mg
    a_gut = 0.0
    c_plasma = 0.0
    cmax = 0.0
    tmax_h = 0.0
    prev_c = 0.0
    auc = 0.0

    for i in range(n_steps):
        # Record current state
        if c_plasma > cmax:
            cmax = c_plasma
            tmax_h = i * dt_h

        # Trapezoidal AUC accumulation
        if i > 0:
            auc += (prev_c + c_plasma) * dt_h / 2.0
        prev_c = c_plasma

        # ODEs (Forward Euler)
        d_stomach = -kge * a_stomach
        d_gut = kge * a_stomach - ka * a_gut
        # f_eff applied at the gut → plasma transfer
        dc_plasma = (ka * a_gut * f_eff / vd) - ke * c_plasma

        a_stomach = max(a_stomach + d_stomach * dt_h, 0.0)
        a_gut = max(a_gut + d_gut * dt_h, 0.0)
        c_plasma = max(c_plasma + dc_plasma * dt_h, 0.0)

    # Final step contribution
    if c_plasma > cmax:
        cmax = c_plasma
        tmax_h = n_steps * dt_h
    auc += (prev_c + c_plasma) * dt_h / 2.0

    return cmax, tmax_h, auc


def assess_food_effect_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 2.0,
    f_oral: float = 0.7,
    fm_cyp3a4: float = 0.0,
    ka_per_h: float = 1.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    food_type: str = "high_fat",
    grapefruit: bool = False,
    has_bile_enhancement: bool = False,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> FoodEffectPKResult:
    """Assess food effect on drug PK using 1-cpt forward Euler simulation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg. Must be > 0.
    logp:
        log octanol-water partition coefficient.
    f_oral:
        Fasted oral bioavailability (0 < f_oral <= 1).
    fm_cyp3a4:
        Fraction metabolized by CYP3A4 (0–1). Used for grapefruit DDI.
    ka_per_h:
        First-order absorption rate constant (h⁻¹).
    cl_L_per_h:
        Systemic clearance (L/h). Must be > 0.
    vd_L:
        Volume of distribution (L).
    food_type:
        "high_fat", "low_fat", or "fasted".
    grapefruit:
        If True, apply CYP3A4 inhibition by grapefruit juice.
    has_bile_enhancement:
        If True, bile salts increase dissolution under fat-fed conditions.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    FoodEffectPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0.0 < f_oral <= 1.0):
        raise ValueError("f_oral must be in (0, 1]")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")

    # Fasted PK parameters
    kge_fasted = 2.0  # h⁻¹
    ke = cl_L_per_h / max(vd_L, 1e-9)

    # Fed gastric emptying rate
    if food_type == "high_fat":
        kge_fed = 0.5
    elif food_type == "low_fat":
        kge_fed = 1.0
    else:
        # food_type == "fasted": same as fasted simulation
        kge_fed = kge_fasted

    # Fed bioavailability correction
    fed_f_correction = 1.0
    if food_type in ("high_fat", "low_fat"):
        if logp > 5.0:
            fed_f_correction += 0.3  # fat → lymphatic absorption
        if food_type == "high_fat" and has_bile_enhancement:
            fed_f_correction *= 1.2  # bile salt dissolution enhancement
        if grapefruit:
            fed_f_correction *= 1.0 + fm_cyp3a4 * 0.8  # CYP3A4 inhibition boosts F

    fed_f = min(1.0, f_oral * fed_f_correction)

    # Simulate fasted
    cmax_fasted, tmax_fasted, auc_fasted = _simulate_1cpt_food(
        dose_mg=dose_mg,
        kge=kge_fasted,
        ka=ka_per_h,
        ke=ke,
        vd=vd_L,
        f_eff=f_oral,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Simulate fed
    cmax_fed, tmax_fed, auc_fed = _simulate_1cpt_food(
        dose_mg=dose_mg,
        kge=kge_fed,
        ka=ka_per_h,
        ke=ke,
        vd=vd_L,
        f_eff=fed_f,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Ratios
    eps = 1e-12
    cmax_ratio = cmax_fed / max(cmax_fasted, eps)
    auc_ratio = auc_fed / max(auc_fasted, eps)

    # Food effect magnitude classification
    if 0.8 <= auc_ratio <= 1.25:
        food_effect_magnitude = "none"
    elif 0.5 <= auc_ratio < 0.8 or 1.25 < auc_ratio <= 2.0:
        food_effect_magnitude = "moderate"
    else:
        food_effect_magnitude = "significant"

    # FDA recommendation
    if auc_ratio > 1.5 and logp > 3.0:
        fda_recommendation = "take with food"
    elif auc_ratio < 0.7:
        fda_recommendation = "take fasted"
    else:
        fda_recommendation = "no restriction"

    notes_parts = [
        f"logP={logp:.1f}",
        f"food_type={food_type}",
        f"fed_f={fed_f:.3f}",
        f"fasted_f={f_oral:.3f}",
    ]
    if grapefruit:
        notes_parts.append(f"grapefruit CYP3A4 inhibition (fm={fm_cyp3a4:.2f})")
    if has_bile_enhancement:
        notes_parts.append("bile salt dissolution enhancement")
    notes = "; ".join(notes_parts)

    return FoodEffectPKResult(
        drug_name=drug_name,
        food_type=food_type,
        cmax_fasted=cmax_fasted,
        cmax_fed=cmax_fed,
        auc_fasted=auc_fasted,
        auc_fed=auc_fed,
        tmax_fasted_h=tmax_fasted,
        tmax_fed_h=tmax_fed,
        cmax_ratio=cmax_ratio,
        auc_ratio=auc_ratio,
        food_effect_magnitude=food_effect_magnitude,
        fda_recommendation=fda_recommendation,
        notes=notes,
    )


def compare_food_conditions(
    drug_name: str,
    dose_mg: float,
    logp: float = 2.0,
    f_oral: float = 0.7,
    fm_cyp3a4: float = 0.0,
    ka_per_h: float = 1.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    grapefruit: bool = False,
    has_bile_enhancement: bool = False,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list[FoodEffectPKResult]:
    """Compare drug PK across fasted, low_fat, and high_fat conditions.

    Returns a list of 3 FoodEffectPKResult objects.
    """
    return [
        assess_food_effect_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logp=logp,
            f_oral=f_oral,
            fm_cyp3a4=fm_cyp3a4,
            ka_per_h=ka_per_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            food_type=ft,
            grapefruit=grapefruit,
            has_bile_enhancement=has_bile_enhancement,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        for ft in ("fasted", "low_fat", "high_fat")
    ]
