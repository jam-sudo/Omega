"""Neonatal PK scaling for neonates (0-28 days postnatal age).

Scales adult PK parameters to neonates using postnatal age (PNA),
gestational age (GA), and body weight. Accounts for immature renal
function, enzyme expression, protein binding changes, and body composition.

Key references:
- Rhodin 2009: GFR maturation (sigmoidal Hill model, PMA-based)
- Anderson & Holford 2008: allometric scaling in pediatrics
- Kearns et al. 2003: developmental pharmacology review
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NeonatalPKResult:
    """Result of neonatal PK scaling.

    Attributes
    ----------
    drug_name : Drug identifier.
    ga_weeks : Gestational age at birth (weeks).
    pna_days : Postnatal age (days).
    neonatal_weight_kg : Neonatal body weight (kg).
    pma_weeks : Postmenstrual age (weeks) = GA + PNA/7.
    maturation_factor : Sigmoidal maturation factor for CL (0-1).
    maturation_stage : Descriptive maturation stage.
    cl_neonatal_L_per_h : Scaled neonatal clearance (L/h).
    vd_neonatal_L : Scaled neonatal volume of distribution (L).
    fu_neonatal : Unbound fraction in neonatal plasma.
    t_half_neonatal_h : Terminal half-life in neonate (h).
    dose_adjustment_factor : CL_neonatal / CL_allometric (ratio).
    notes : Summary note string.
    """

    drug_name: str
    ga_weeks: float
    pna_days: float
    neonatal_weight_kg: float
    pma_weeks: float
    maturation_factor: float
    maturation_stage: str
    cl_neonatal_L_per_h: float
    vd_neonatal_L: float
    fu_neonatal: float
    t_half_neonatal_h: float
    dose_adjustment_factor: float
    notes: str


def _validate_inputs(
    adult_cl_L_per_h: float,
    adult_vd_L: float,
    adult_weight_kg: float,
    neonatal_weight_kg: float,
    ga_weeks: float,
    pna_days: float,
    fu_adult: float,
    logp: float,
) -> None:
    if adult_cl_L_per_h <= 0:
        raise ValueError(f"adult_cl_L_per_h must be > 0, got {adult_cl_L_per_h}")
    if adult_vd_L <= 0:
        raise ValueError(f"adult_vd_L must be > 0, got {adult_vd_L}")
    if adult_weight_kg <= 0:
        raise ValueError(f"adult_weight_kg must be > 0, got {adult_weight_kg}")
    if neonatal_weight_kg <= 0:
        raise ValueError(f"neonatal_weight_kg must be > 0, got {neonatal_weight_kg}")
    if not (22.0 <= ga_weeks <= 44.0):
        raise ValueError(f"ga_weeks must be in [22, 44], got {ga_weeks}")
    if not (0.0 <= pna_days <= 28.0):
        raise ValueError(f"pna_days must be in [0, 28], got {pna_days}")
    if not (0.0 < fu_adult <= 1.0):
        raise ValueError(f"fu_adult must be in (0, 1], got {fu_adult}")
    if not (-5.0 <= logp <= 10.0):
        raise ValueError(f"logp must be in [-5, 10], got {logp}")


def _maturation_stage(ga_weeks: float) -> str:
    """Classify maturation stage from gestational age at birth."""
    if ga_weeks < 28.0:
        return "extremely_preterm"
    elif ga_weeks < 32.0:
        return "very_preterm"
    elif ga_weeks < 37.0:
        return "preterm"
    else:
        return "term"


def _maturation_factor(pma_weeks: float) -> float:
    """Sigmoidal Hill-type maturation factor based on postmenstrual age.

    Rhodin 2009 GFR maturation model:
        MF = PMA^3.4 / (PMA^3.4 + 47.7^3.4)
    """
    pma_34 = pma_weeks**3.4
    ref_34 = 47.7**3.4
    return pma_34 / (pma_34 + ref_34)


def scale_neonatal_pk(
    drug_name: str,
    adult_cl_L_per_h: float,
    adult_vd_L: float,
    adult_weight_kg: float,
    neonatal_weight_kg: float,
    ga_weeks: float,
    pna_days: float,
    fu_adult: float,
    logp: float,
) -> NeonatalPKResult:
    """Scale adult PK parameters to neonatal values.

    Parameters
    ----------
    drug_name : Drug identifier string.
    adult_cl_L_per_h : Adult total CL (L/h).
    adult_vd_L : Adult volume of distribution (L).
    adult_weight_kg : Adult body weight (kg).
    neonatal_weight_kg : Neonatal body weight (kg).
    ga_weeks : Gestational age at birth (weeks, 22-44).
    pna_days : Postnatal age (days, 0-28).
    fu_adult : Unbound fraction in adult plasma (0, 1].
    logp : Octanol-water partition coefficient [-5, 10].

    Returns
    -------
    NeonatalPKResult with scaled PK parameters.
    """
    _validate_inputs(
        adult_cl_L_per_h,
        adult_vd_L,
        adult_weight_kg,
        neonatal_weight_kg,
        ga_weeks,
        pna_days,
        fu_adult,
        logp,
    )

    # Postmenstrual age
    pma_weeks = ga_weeks + pna_days / 7.0

    # Allometric CL scaling (3/4 power)
    wt_ratio = neonatal_weight_kg / adult_weight_kg
    cl_allometric = adult_cl_L_per_h * (wt_ratio**0.75)

    # Maturation factor (Rhodin 2009)
    mf = _maturation_factor(pma_weeks)

    # Final neonatal CL
    cl_neonatal = cl_allometric * mf

    # Vd scaling — total body water component
    if ga_weeks < 32.0:
        tbw_fraction = 0.8
    elif ga_weeks < 37.0:
        tbw_fraction = 0.75
    else:
        tbw_fraction = 0.7

    vd_water_component = adult_vd_L * wt_ratio * (tbw_fraction / 0.6)

    # Vd scaling — fat component
    if ga_weeks < 32.0:
        fat_fraction = 0.01
    elif ga_weeks < 37.0:
        fat_fraction = 0.08
    else:
        fat_fraction = 0.12

    logp_factor = max(0.0, logp) / 5.0
    vd_fat_component = adult_vd_L * logp_factor * fat_fraction * neonatal_weight_kg

    vd_neonatal = vd_water_component + vd_fat_component

    # Protein binding — reduced albumin in neonates
    fu_neonatal = min(1.0, fu_adult * 1.5)

    # Terminal half-life
    t_half = 0.693 * vd_neonatal / cl_neonatal if cl_neonatal > 0 else float("inf")

    # Dose adjustment factor vs allometric prediction
    dose_adjustment_factor = cl_neonatal / cl_allometric if cl_allometric > 0 else 0.0

    stage = _maturation_stage(ga_weeks)

    notes = (
        f"{drug_name}: GA={ga_weeks:.1f}w PNA={pna_days:.0f}d "
        f"PMA={pma_weeks:.1f}w stage={stage} "
        f"MF={mf:.3f} CL={cl_neonatal:.4f}L/h Vd={vd_neonatal:.3f}L "
        f"t1/2={t_half:.1f}h"
    )

    return NeonatalPKResult(
        drug_name=drug_name,
        ga_weeks=ga_weeks,
        pna_days=pna_days,
        neonatal_weight_kg=neonatal_weight_kg,
        pma_weeks=pma_weeks,
        maturation_factor=mf,
        maturation_stage=stage,
        cl_neonatal_L_per_h=cl_neonatal,
        vd_neonatal_L=vd_neonatal,
        fu_neonatal=fu_neonatal,
        t_half_neonatal_h=t_half,
        dose_adjustment_factor=dose_adjustment_factor,
        notes=notes,
    )


def compare_gestational_ages(
    drug_name: str,
    adult_cl_L_per_h: float,
    adult_vd_L: float,
    adult_weight_kg: float,
    neonatal_weight_kg: float,
    pna_days: float,
    fu_adult: float,
    logp: float,
    ga_weeks_list: list[float],
) -> list[NeonatalPKResult]:
    """Compare neonatal PK across multiple gestational ages.

    Results are sorted ascending by cl_neonatal_L_per_h (most immature first).

    Parameters
    ----------
    drug_name : Drug identifier string.
    adult_cl_L_per_h : Adult total CL (L/h).
    adult_vd_L : Adult volume of distribution (L).
    adult_weight_kg : Adult body weight (kg).
    neonatal_weight_kg : Neonatal body weight (kg).
    pna_days : Postnatal age (days, 0-28).
    fu_adult : Unbound fraction in adult plasma (0, 1].
    logp : Octanol-water partition coefficient [-5, 10].
    ga_weeks_list : List of gestational ages to compare (weeks).

    Returns
    -------
    List of NeonatalPKResult sorted by cl_neonatal_L_per_h ascending.
    """
    results = [
        scale_neonatal_pk(
            drug_name=drug_name,
            adult_cl_L_per_h=adult_cl_L_per_h,
            adult_vd_L=adult_vd_L,
            adult_weight_kg=adult_weight_kg,
            neonatal_weight_kg=neonatal_weight_kg,
            ga_weeks=ga_weeks,
            pna_days=pna_days,
            fu_adult=fu_adult,
            logp=logp,
        )
        for ga_weeks in ga_weeks_list
    ]
    return sorted(results, key=lambda r: r.cl_neonatal_L_per_h)
