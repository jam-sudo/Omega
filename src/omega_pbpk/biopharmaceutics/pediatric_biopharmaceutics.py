"""Pediatric biopharmaceutics — age-dependent GI physiology and drug absorption.

Models how pediatric GI physiology affects oral drug absorption differently from adults:
- Higher gastric pH in neonates (pH 6-8 vs adult pH 1-3)
- Slower gastric emptying in neonates, faster by ~6 months
- Smaller intestinal surface area (scaled to body weight)
- Immature bile acid secretion at birth

References:
  Kearns GL et al. (2003) Developmental pharmacology — drug disposition,
    action, and therapy in infants and children. NEJM 349:1157-1167.
  Batchelor HK & Marriott JF (2015) Paediatric pharmacokinetics.
    Br J Clin Pharmacol 79:395-404.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

# ---------------------------------------------------------------------------
# Result dataclasses (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PediatricGIParams:
    """GI physiological parameters for a pediatric patient."""

    age_months: float
    weight_kg: float
    gastric_ph: float
    gastric_emptying_h: float  # gastric half-emptying time
    surface_fraction: float  # intestinal surface area relative to adult (0-1)
    bile_acid_factor: float  # bile acid maturation factor (0-1)
    transit_time_h: float  # total GI transit time (h)


@dataclass(frozen=True)
class PediatricAbsorptionResult:
    """Simulation result for pediatric oral drug absorption."""

    drug_name: str
    age_months: float
    weight_kg: float
    dose_mg: float
    times_h: list[float]
    c_plasma: list[float]  # plasma concentration (mg/L)
    cmax: float  # peak plasma concentration (mg/L)
    tmax_h: float  # time to peak (h)
    auc: float  # AUC_0-inf approximation (mg·h/L)
    ka_effective: float  # effective absorption rate constant (1/h)
    notes: str


# ---------------------------------------------------------------------------
# Core physiological functions
# ---------------------------------------------------------------------------


def pediatric_gastric_ph(age_months: float) -> float:
    """Predict gastric pH as a function of postnatal age.

    Neonates are born with near-neutral gastric pH (~6.5) due to alkaline
    amniotic fluid. Acid secretion matures and pH falls toward adult values
    (~2) within the first 3 months of life.

    Model: pH = 6.5 - 4.5 * (1 - exp(-age_months / 1.5))
    Clamped to [1.5, 7.0].

    Args:
        age_months: Postnatal age in months (must be >= 0).

    Returns:
        Predicted gastric pH.

    Raises:
        ValueError: If age_months < 0.
    """
    if age_months < 0:
        raise ValueError(f"age_months must be >= 0, got {age_months}")

    ph = 6.5 - 4.5 * (1.0 - math.exp(-age_months / 1.5))
    return max(1.5, min(7.0, ph))


def pediatric_gi_parameters(age_months: float, weight_kg: float) -> PediatricGIParams:
    """Compute age- and weight-dependent GI physiological parameters.

    Parameters:
    - gastric_ph: pH from pediatric_gastric_ph()
    - gastric_emptying_h: neonates have SLOWER emptying; matures by ~12 months
        t_gastric = 1.5 - 1.0 * min(1.0, age_months / 12)
    - surface_fraction: intestinal surface area relative to adult (70 kg reference)
        fraction = (weight_kg / 70) ** 0.75
    - bile_acid_factor: maturation of bile acid pool (0 at birth → 1 by 6 months)
        bile_factor = 0.3 + 0.7 * min(1.0, age_months / 6)
    - transit_time_h: full GI transit (~24 h adult, longer in neonates)
        transit = 18 + 6 * exp(-age_months / 12)

    Args:
        age_months: Postnatal age in months (must be >= 0).
        weight_kg: Body weight in kg (must be > 0).

    Returns:
        PediatricGIParams dataclass.

    Raises:
        ValueError: On invalid inputs.
    """
    if age_months < 0:
        raise ValueError(f"age_months must be >= 0, got {age_months}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")

    gastric_ph = pediatric_gastric_ph(age_months)
    gastric_emptying_h = 1.5 - 1.0 * min(1.0, age_months / 12.0)
    surface_fraction = (weight_kg / 70.0) ** 0.75
    bile_acid_factor = 0.3 + 0.7 * min(1.0, age_months / 6.0)
    transit_time_h = 18.0 + 6.0 * math.exp(-age_months / 12.0)

    return PediatricGIParams(
        age_months=age_months,
        weight_kg=weight_kg,
        gastric_ph=round(gastric_ph, 4),
        gastric_emptying_h=round(gastric_emptying_h, 4),
        surface_fraction=round(surface_fraction, 6),
        bile_acid_factor=round(bile_acid_factor, 4),
        transit_time_h=round(transit_time_h, 4),
    )


def _ph_solubility_factor(logP: float, pka: float, gi_ph: float) -> float:
    """Compute pH-adjusted solubility factor using Henderson-Hasselbalch.

    For a weak acid: fu = 1 / (1 + 10^(pH - pKa))
    For a weak base:  fu = 1 / (1 + 10^(pKa - pH))
    Assumes the input pka is for an acid if logP < 2, base otherwise.
    The factor scales solubility relative to intrinsic (unionised) solubility.

    Returns a factor >= 1.0 (higher pH-adjusted solubility relative to neutral).
    """
    if pka <= 0:
        return 1.0
    # Treat as weak base if logP >= 2 (lipophilic), weak acid otherwise
    if logP >= 2.0:
        # weak base: ionized at low pH
        ionization_ratio = 10.0 ** (pka - gi_ph)
    else:
        # weak acid: ionized at high pH
        ionization_ratio = 10.0 ** (gi_ph - pka)
    factor = 1.0 + ionization_ratio
    return max(1.0, min(factor, 1000.0))


def simulate_pediatric_absorption(
    drug_name: str,
    dose_mg_per_kg: float,
    weight_kg: float,
    age_months: float,
    logP: float,
    pka: float,
    solubility_mg_mL: float,
    cl_adult_L_per_h_per_kg: float,
    vd_adult_L_per_kg: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PediatricAbsorptionResult:
    """Simulate pediatric oral drug absorption using a 1-compartment PK model.

    Steps:
    1. Compute pediatric GI parameters for given age/weight.
    2. Adjust solubility for gastric pH (Henderson-Hasselbalch).
    3. Compute effective ka from intestinal surface fraction and bile acid maturity.
    4. Scale CL and Vd by weight (allometric 0.75 for CL, 1.0 for Vd).
    5. Integrate 1-cpt oral PK by forward Euler.

    Effective ka (1/h):
        Base ka_ref = 1.0 (typical well-absorbed drug)
        ka_effective = ka_ref * surface_fraction * bile_acid_factor

    CL scaling: CL = cl_adult_L_per_h_per_kg * weight_kg^0.75 / 70^0.75 * 70
    Simplified: CL (L/h) = cl_adult_L_per_h_per_kg * 70 * (weight_kg/70)^0.75
    Vd (L)     = vd_adult_L_per_kg * weight_kg

    Args:
        drug_name: Drug identifier string.
        dose_mg_per_kg: Dose in mg/kg.
        weight_kg: Body weight in kg (> 0).
        age_months: Postnatal age in months (>= 0).
        logP: Octanol-water partition coefficient.
        pka: Drug pKa (use 0 if neutral).
        solubility_mg_mL: Intrinsic (aqueous) solubility in mg/mL.
        cl_adult_L_per_h_per_kg: Adult total clearance per kg (L/h/kg, > 0).
        vd_adult_L_per_kg: Adult volume of distribution per kg (L/kg, > 0).
        t_end_h: Simulation end time (hours).
        dt_h: Forward Euler step size (hours).

    Returns:
        PediatricAbsorptionResult with full PK profile.

    Raises:
        ValueError: On invalid inputs.
    """
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if dose_mg_per_kg <= 0:
        raise ValueError(f"dose_mg_per_kg must be > 0, got {dose_mg_per_kg}")
    if age_months < 0:
        raise ValueError(f"age_months must be >= 0, got {age_months}")
    if cl_adult_L_per_h_per_kg <= 0:
        raise ValueError(f"cl_adult_L_per_h_per_kg must be > 0, got {cl_adult_L_per_h_per_kg}")
    if vd_adult_L_per_kg <= 0:
        raise ValueError(f"vd_adult_L_per_kg must be > 0, got {vd_adult_L_per_kg}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")

    gi_params = pediatric_gi_parameters(age_months, weight_kg)

    # Total dose in mg
    dose_mg = dose_mg_per_kg * weight_kg

    # pH-adjusted solubility factor (affects fraction absorbed, not ka directly)
    sol_factor = _ph_solubility_factor(logP, pka, gi_params.gastric_ph)
    sol_adjusted = solubility_mg_mL * sol_factor

    # Notes on solubility and GI conditions
    notes_parts = [
        f"Gastric pH={gi_params.gastric_ph:.2f}",
        f"Surface fraction={gi_params.surface_fraction:.3f}",
        f"Bile acid factor={gi_params.bile_acid_factor:.2f}",
        f"Adjusted solubility={sol_adjusted:.3f} mg/mL",
    ]

    # Effective absorption rate constant
    # Base ka = 1.0 h^-1 for a reference well-absorbed drug
    # Modulated by intestinal surface area and bile acid maturation
    ka_ref = 1.0
    ka_effective = ka_ref * gi_params.surface_fraction * gi_params.bile_acid_factor

    # Weight-scaled PK parameters (allometric)
    bw_fraction = weight_kg / 70.0
    cl_L_per_h = cl_adult_L_per_h_per_kg * 70.0 * (bw_fraction**0.75)
    vd_L = vd_adult_L_per_kg * weight_kg

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # Forward Euler 1-compartment oral PK
    # dX_gut/dt = -ka * X_gut
    # dX_central/dt = ka * X_gut - ke * X_central
    # C_plasma = X_central / Vd

    n_steps = max(int(t_end_h / dt_h), 1)
    times_h: list[float] = []
    c_plasma: list[float] = []

    x_gut = dose_mg  # drug in gut (mg)
    x_central = 0.0  # drug in central compartment (mg)

    cmax = 0.0
    tmax_h = 0.0

    for i in range(n_steps + 1):
        t = i * dt_h
        c = x_central / vd_L  # mg/L

        times_h.append(round(t, 4))
        c_plasma.append(round(c, 8))

        if c > cmax:
            cmax = c
            tmax_h = t

        if i < n_steps:
            dx_gut = -ka_effective * x_gut
            dx_central = ka_effective * x_gut - ke * x_central

            x_gut = max(x_gut + dx_gut * dt_h, 0.0)
            x_central = max(x_central + dx_central * dt_h, 0.0)

    times_arr = np.array(times_h)
    c_arr = np.array(c_plasma)
    auc = float(np_trapz(c_arr, times_arr))

    return PediatricAbsorptionResult(
        drug_name=drug_name,
        age_months=age_months,
        weight_kg=weight_kg,
        dose_mg=round(dose_mg, 4),
        times_h=times_h,
        c_plasma=c_plasma,
        cmax=round(cmax, 6),
        tmax_h=round(tmax_h, 4),
        auc=round(auc, 4),
        ka_effective=round(ka_effective, 6),
        notes="; ".join(notes_parts),
    )


def compare_age_groups(
    drug_name: str,
    dose_mg_per_kg: float,
    logP: float,
    pka: float,
    solubility_mg_mL: float,
    cl_adult_per_kg: float,
    vd_adult_per_kg: float,
) -> dict[str, PediatricAbsorptionResult]:
    """Compare pediatric drug absorption across standard age groups.

    Age groups simulated:
      - neonate:  0.5 months, 3.5 kg
      - infant:   6 months,   7 kg
      - toddler:  24 months,  12 kg
      - child:    96 months,  25 kg
      - adult:    216 months, 70 kg

    Args:
        drug_name: Drug identifier.
        dose_mg_per_kg: Dose in mg/kg (same across groups).
        logP: Octanol-water partition coefficient.
        pka: Drug pKa.
        solubility_mg_mL: Intrinsic aqueous solubility in mg/mL.
        cl_adult_per_kg: Adult clearance per kg body weight (L/h/kg).
        vd_adult_per_kg: Adult volume of distribution per kg (L/kg).

    Returns:
        Dictionary mapping age group name to PediatricAbsorptionResult.
    """
    age_groups = [
        ("neonate", 0.5, 3.5),
        ("infant", 6.0, 7.0),
        ("toddler", 24.0, 12.0),
        ("child", 96.0, 25.0),
        ("adult", 216.0, 70.0),
    ]

    results: dict[str, PediatricAbsorptionResult] = {}
    for label, age_mo, wt_kg in age_groups:
        results[label] = simulate_pediatric_absorption(
            drug_name=drug_name,
            dose_mg_per_kg=dose_mg_per_kg,
            weight_kg=wt_kg,
            age_months=age_mo,
            logP=logP,
            pka=pka,
            solubility_mg_mL=solubility_mg_mL,
            cl_adult_L_per_h_per_kg=cl_adult_per_kg,
            vd_adult_L_per_kg=vd_adult_per_kg,
        )

    return results


__all__ = [
    "PediatricGIParams",
    "PediatricAbsorptionResult",
    "pediatric_gastric_ph",
    "pediatric_gi_parameters",
    "simulate_pediatric_absorption",
    "compare_age_groups",
]
