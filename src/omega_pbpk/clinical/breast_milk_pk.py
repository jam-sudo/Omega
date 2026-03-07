"""Neonatal drug exposure via breast milk — Phase 341.

Models drug transfer into breast milk and infant exposure using the
Rowe/Atkinson M/P ratio model with a 1-compartment maternal PK simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "BreastMilkResult",
    "calculate_milk_to_plasma_ratio",
    "simulate_breast_milk_pk",
    "screen_drugs_for_breastfeeding",
]

_MATERNAL_WEIGHT_KG = 60.0  # assumed maternal weight for RID calculation


@dataclass(frozen=True)
class BreastMilkResult:
    """Results of breast milk drug transfer simulation."""

    drug_name: str
    maternal_dose_mg: float
    milk_to_plasma_ratio: float  # M/P ratio

    # Maternal PK
    maternal_auc_mg_h_per_L: float
    maternal_cmax_mg_L: float

    # Infant exposure
    infant_daily_dose_mg_per_kg: float  # theoretical infant daily dose
    infant_dose_fraction_pct: float  # % of maternal weight-adjusted dose
    relative_infant_dose_pct: float  # RID = infant dose / maternal dose (weight-adjusted)

    # Risk classification
    risk_category: str  # "low" (<10% RID), "moderate" (10-25%), "high" (>25%)
    breastfeeding_recommendation: str  # "compatible", "use_with_caution", "avoid"

    times_h: list[float]
    c_milk_mg_L: list[float]  # drug concentration in breast milk

    notes: str


def calculate_milk_to_plasma_ratio(
    pka_acid: float | None,
    pka_base: float | None,
    logP: float,
    fu_plasma: float,
    milk_ph: float = 7.2,
    plasma_ph: float = 7.4,
) -> float:
    """Calculate milk-to-plasma (M/P) ratio using the Rowe/Atkinson model.

    Parameters
    ----------
    pka_acid : float or None
        pKa of the acidic group (None if not acidic drug).
    pka_base : float or None
        pKa of the basic group (None if not basic drug).
    logP : float
        Log octanol-water partition coefficient.
    fu_plasma : float
        Unbound fraction in plasma (0–1).
    milk_ph : float
        Milk pH (default 7.2).
    plasma_ph : float
        Plasma pH (default 7.4).

    Returns
    -------
    float
        M/P ratio, clamped to [0.05, 10.0].
    """
    if pka_acid is not None:
        mp = (
            (1.0 + 10.0 ** (milk_ph - pka_acid))
            / (1.0 + 10.0 ** (plasma_ph - pka_acid))
            * fu_plasma
        )
    elif pka_base is not None:
        mp = (
            (1.0 + 10.0 ** (pka_base - milk_ph))
            / (1.0 + 10.0 ** (pka_base - plasma_ph))
            * fu_plasma
        )
    else:
        # Neutral drug: simple logP-based estimate
        mp = logP * 0.1 + 0.5
        mp = max(0.1, min(5.0, mp))

    return float(max(0.05, min(10.0, mp)))


def simulate_breast_milk_pk(
    drug_name: str,
    maternal_dose_mg: float,
    cl_maternal_L_per_h: float,
    vd_maternal_L: float,
    milk_to_plasma_ratio: float,
    ka_per_h: float = 1.5,
    f_oral: float = 0.9,
    t_end_h: float = 24.0,
    infant_weight_kg: float = 5.0,
    milk_intake_mL_per_day: float = 150.0,
    infant_cl_L_per_h_per_kg: float = 0.3,
    dt_h: float = 0.1,
) -> BreastMilkResult:
    """Simulate maternal PK and infant drug exposure via breast milk.

    Uses a 1-compartment oral maternal PK model (forward Euler) and
    scales milk concentrations via the M/P ratio.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    maternal_dose_mg : float
        Single maternal dose (mg, > 0).
    cl_maternal_L_per_h : float
        Maternal clearance (L/h, > 0).
    vd_maternal_L : float
        Maternal volume of distribution (L, > 0).
    milk_to_plasma_ratio : float
        M/P ratio (> 0).
    ka_per_h : float
        Absorption rate constant (1/h).
    f_oral : float
        Oral bioavailability fraction (0, 1].
    t_end_h : float
        Simulation end time (h).
    infant_weight_kg : float
        Infant body weight (kg, > 0).
    milk_intake_mL_per_day : float
        Volume of breast milk consumed per day (mL/day, > 0).
    infant_cl_L_per_h_per_kg : float
        Infant weight-normalised clearance (L/h/kg).
    dt_h : float
        Integration time step (h).

    Returns
    -------
    BreastMilkResult
    """
    # Input validation
    if maternal_dose_mg <= 0:
        raise ValueError("maternal_dose_mg must be > 0")
    if cl_maternal_L_per_h <= 0:
        raise ValueError("cl_maternal_L_per_h must be > 0")
    if vd_maternal_L <= 0:
        raise ValueError("vd_maternal_L must be > 0")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")
    if infant_weight_kg <= 0:
        raise ValueError("infant_weight_kg must be > 0")
    if milk_intake_mL_per_day <= 0:
        raise ValueError("milk_intake_mL_per_day must be > 0")
    if milk_to_plasma_ratio <= 0:
        raise ValueError("milk_to_plasma_ratio must be > 0")

    ke = cl_maternal_L_per_h / vd_maternal_L
    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    # Forward Euler 1-compartment oral model
    # State: [A_gut (mg), C_plasma (mg/L)]
    a_gut = f_oral * maternal_dose_mg  # drug in gut at t=0
    c_plasma = 0.0

    c_plasma_arr = np.zeros(n_steps + 1)
    c_plasma_arr[0] = c_plasma

    for i in range(n_steps):
        # Absorption from gut
        absorbed_rate = ka_per_h * a_gut  # mg/h
        # Plasma concentration change
        dc = (absorbed_rate / vd_maternal_L - ke * c_plasma) * dt_h
        da_gut = -absorbed_rate * dt_h

        a_gut = max(0.0, a_gut + da_gut)
        c_plasma = max(0.0, c_plasma + dc)
        c_plasma_arr[i + 1] = c_plasma

    # Milk concentration
    c_milk_arr = c_plasma_arr * milk_to_plasma_ratio  # mg/L

    # Maternal AUC and Cmax
    maternal_auc = float(np_trapz(c_plasma_arr, times))
    maternal_cmax = float(np.max(c_plasma_arr))

    # Infant daily dose
    # Average milk concentration over 24 h = AUC_milk / t_end_h
    auc_milk = float(np_trapz(c_milk_arr, times))
    avg_c_milk_mg_L = auc_milk / t_end_h  # mg/L average

    # Volume consumed (L/day)
    milk_intake_L_per_day = milk_intake_mL_per_day / 1000.0
    infant_dose_mg_per_day = avg_c_milk_mg_L * milk_intake_L_per_day * 1000.0 / 1000.0
    # avg_c is mg/L, milk_intake_L = L/day → mg/day
    infant_dose_mg_per_day = avg_c_milk_mg_L * milk_intake_L_per_day

    infant_daily_dose_mg_per_kg = infant_dose_mg_per_day / infant_weight_kg

    # Relative Infant Dose (RID)
    maternal_dose_mg_per_kg_per_day = maternal_dose_mg / _MATERNAL_WEIGHT_KG
    rid_pct = (infant_daily_dose_mg_per_kg / max(maternal_dose_mg_per_kg_per_day, 1e-12)) * 100.0

    # Infant dose fraction (% of maternal dose, weight-adjusted)
    infant_dose_fraction_pct = rid_pct  # same calculation

    # Risk classification
    if rid_pct < 10.0:
        risk_category = "low"
        recommendation = "compatible"
    elif rid_pct <= 25.0:
        risk_category = "moderate"
        recommendation = "use_with_caution"
    else:
        risk_category = "high"
        recommendation = "avoid"

    notes = (
        f"1-cpt oral maternal PK (forward Euler, dt={dt_h} h). "
        f"M/P={milk_to_plasma_ratio:.3f}. "
        f"RID={rid_pct:.2f}%. "
        f"Maternal AUC={maternal_auc:.3f} mg·h/L. "
        f"Infant intake={milk_intake_mL_per_day} mL/day."
    )

    return BreastMilkResult(
        drug_name=drug_name,
        maternal_dose_mg=maternal_dose_mg,
        milk_to_plasma_ratio=milk_to_plasma_ratio,
        maternal_auc_mg_h_per_L=maternal_auc,
        maternal_cmax_mg_L=maternal_cmax,
        infant_daily_dose_mg_per_kg=infant_daily_dose_mg_per_kg,
        infant_dose_fraction_pct=infant_dose_fraction_pct,
        relative_infant_dose_pct=rid_pct,
        risk_category=risk_category,
        breastfeeding_recommendation=recommendation,
        times_h=times.tolist(),
        c_milk_mg_L=c_milk_arr.tolist(),
        notes=notes,
    )


def screen_drugs_for_breastfeeding(
    drugs_list: list[dict],
) -> list[BreastMilkResult]:
    """Screen multiple drugs for breastfeeding safety, sorted safest first.

    Parameters
    ----------
    drugs_list : list of dict
        Each dict: {"name": str, "maternal_dose_mg": float,
                    "cl_L_per_h": float, "vd_L": float, "m_p_ratio": float}

    Returns
    -------
    list[BreastMilkResult]
        Results sorted by relative_infant_dose_pct ascending (safest first).
    """
    results = []
    for drug in drugs_list:
        result = simulate_breast_milk_pk(
            drug_name=drug["name"],
            maternal_dose_mg=drug["maternal_dose_mg"],
            cl_maternal_L_per_h=drug["cl_L_per_h"],
            vd_maternal_L=drug["vd_L"],
            milk_to_plasma_ratio=drug["m_p_ratio"],
        )
        results.append(result)

    return sorted(results, key=lambda r: r.relative_infant_dose_pct)
