"""Pediatric dosing optimization — Phase 312.

Optimize dosing for pediatric patients using allometric and maturation scaling.
Focuses on target AUC/Cmax optimization rather than simple table generation
(which is handled by pediatric_weight_dosing.py from Phase 272).

Functions:
- target_auc_dose: Scale CL allometrically + maturation, compute optimal dose
  for target AUC.
- mg_per_kg_dosing: Simple weight-based dosing with adult cap.
- optimize_pediatric_regimen: Find dose and interval achieving target
  trough/peak at steady state.

References
----------
- Holford N et al. (2013) The population approach: rationale, methods, and
  applications in clinical pharmacology and drug development. Handb Exp
  Pharmacol 216:137-175.
- Anderson BJ & Holford NH (2008) Mechanism-based concepts of size and
  maturity in pharmacokinetics. Annu Rev Pharmacol Toxicol 48:303-332.
- Johnson TN et al. (2006) Prediction of the clearance of eleven drugs and
  associated variability in neonates, infants and children. Clin Pharmacokinet
  45:931-956.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PediatricDoseOptResult",
    "PediatricRegimenResult",
    "target_auc_dose",
    "mg_per_kg_dosing",
    "optimize_pediatric_regimen",
]

# ---------------------------------------------------------------------------
# Practical dose steps (mg)
# ---------------------------------------------------------------------------

_PRACTICAL_DOSE_STEPS = [
    5.0,
    10.0,
    12.5,
    25.0,
    50.0,
    75.0,
    100.0,
    125.0,
    150.0,
    200.0,
    250.0,
]

# Maturation parameters (Hill equation for CL maturation)
# Ref: Anderson & Holford 2008; PNA50 = 47.7 weeks post-menstrual age (PMA)
# For simplicity we use postnatal age in months with TM50 = 3.8 months
_CL_TM50_MONTHS = 3.8  # half-maturation age (months)
_CL_HILL = 3.4  # Hill exponent

# ---------------------------------------------------------------------------
# Result dataclasses (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PediatricDoseOptResult:
    """Result of pediatric dose optimization."""

    drug_name: str
    weight_kg: float
    age_months: float
    target_auc: float
    cl_pediatric: float
    recommended_dose_mg: float
    dose_per_kg: float
    practical_dose_mg: float
    notes: str


@dataclass(frozen=True)
class PediatricRegimenResult:
    """Result of pediatric regimen optimization."""

    drug_name: str
    weight_kg: float
    age_months: float
    recommended_dose_mg: float
    interval_h: float
    predicted_trough: float
    predicted_peak: float
    in_therapeutic_range: bool
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _maturation_factor(age_months: float) -> float:
    """Compute CL maturation fraction (0–1) using Hill equation.

    For age >= 24 months, maturation is essentially complete (returns 1.0).

    Parameters
    ----------
    age_months : float
        Postnatal age in months (>= 0).

    Returns
    -------
    float
        Maturation fraction in [0, 1].
    """
    if age_months >= 24.0:
        return 1.0
    # Hill equation: mat = age^n / (TM50^n + age^n)
    # For age = 0 (neonate), mat -> 0; at TM50 = 3.8 months, mat = 0.5
    if age_months <= 0.0:
        return 0.01  # neonate: ~1% of adult CL per unit weight
    tm50 = _CL_TM50_MONTHS
    hill = _CL_HILL
    mat = (age_months**hill) / (tm50**hill + age_months**hill)
    return max(0.01, min(1.0, mat))


def _allometric_cl(cl_adult_L_per_h: float, weight_kg: float) -> float:
    """Allometric CL scaling from 70 kg adult.

    CL_child = CL_adult * (weight / 70)^0.75

    Parameters
    ----------
    cl_adult_L_per_h : float
        Adult clearance in L/h.
    weight_kg : float
        Child body weight in kg.

    Returns
    -------
    float
        Scaled clearance in L/h.
    """
    return cl_adult_L_per_h * (weight_kg / 70.0) ** 0.75


def _allometric_vd(vd_adult_L: float, weight_kg: float) -> float:
    """Allometric Vd scaling from 70 kg adult.

    Vd_child = Vd_adult * (weight / 70)^1.0  (linear with weight)

    Parameters
    ----------
    vd_adult_L : float
        Adult volume of distribution in L.
    weight_kg : float
        Child body weight in kg.

    Returns
    -------
    float
        Scaled volume in L.
    """
    return vd_adult_L * (weight_kg / 70.0)


def _round_to_practical(dose_mg: float) -> float:
    """Round dose to nearest practical step from _PRACTICAL_DOSE_STEPS."""
    if dose_mg <= 0:
        return _PRACTICAL_DOSE_STEPS[0]
    return min(_PRACTICAL_DOSE_STEPS, key=lambda x: abs(x - dose_mg))


def _steady_state_trough_peak(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    interval_h: float,
    route: str = "oral",
) -> tuple[float, float]:
    """Compute steady-state trough and peak concentrations.

    Analytical 1-compartment solution at steady state.

    For IV:
        C_peak_ss = D / Vd * 1 / (1 - exp(-ke * tau))
        C_trough_ss = C_peak_ss * exp(-ke * tau)

    For oral (ka >> ke approximation gives Cmax near absorption):
        C_ss_peak ≈ (D/Vd) * ke / (ka - ke) * accumulation factor
        We use a simplified analytical expression with ka = 1.0/h.

    Parameters
    ----------
    dose_mg : float
        Dose in mg.
    cl_L_per_h : float
        Clearance in L/h.
    vd_L : float
        Volume of distribution in L.
    interval_h : float
        Dosing interval in hours.
    route : str
        "oral" or "iv".

    Returns
    -------
    tuple[float, float]
        (trough_mg_L, peak_mg_L)
    """
    ke = cl_L_per_h / vd_L  # 1/h
    tau = interval_h

    if route == "iv":
        # Bolus IV: peak immediately after dose, trough just before next
        accum = 1.0 / (1.0 - math.exp(-ke * tau))
        peak_ss = (dose_mg / vd_L) * accum
        trough_ss = peak_ss * math.exp(-ke * tau)
    else:
        # Oral: 1-cpt analytical at steady state
        ka = 1.0  # typical oral absorption rate (1/h)
        if abs(ka - ke) < 1e-6:
            ka = ka + 1e-6

        accum_e = 1.0 / (1.0 - math.exp(-ke * tau))
        accum_a = 1.0 / (1.0 - math.exp(-ka * tau))

        # Concentration at time t after dose at steady state:
        # C_ss(t) = (D/Vd) * ka/(ka-ke) * [exp(-ke*t)*accum_e - exp(-ka*t)*accum_a]
        # Trough = C_ss(tau)
        c_trough = (
            (dose_mg / vd_L)
            * (ka / (ka - ke))
            * (
                math.exp(-ke * tau) * accum_e
                - math.exp(-ka * tau) * accum_a
            )
        )
        trough_ss = max(0.0, c_trough)

        # Find peak numerically: t_peak = ln(ka/ke)/(ka-ke)
        t_peak = math.log(ka / ke) / (ka - ke) if ka > ke else 1.0 / ke
        c_peak = (
            (dose_mg / vd_L)
            * (ka / (ka - ke))
            * (
                math.exp(-ke * t_peak) * accum_e
                - math.exp(-ka * t_peak) * accum_a
            )
        )
        peak_ss = max(0.0, c_peak)

    return trough_ss, peak_ss


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def target_auc_dose(
    drug_name: str,
    target_auc_mg_h_per_L: float,
    cl_adult_L_per_h: float,
    weight_kg: float,
    age_months: float,
    fm_cyp3a4: float = 0.5,
    fm_cyp2d6: float = 0.2,
    f_renal: float = 0.3,
) -> PediatricDoseOptResult:
    """Compute optimal pediatric dose to achieve a target AUC.

    Scales adult CL allometrically then applies maturation correction for
    children < 24 months. For oral dosing, bioavailability is assumed to
    be ~70% (F = 0.7).

    Dose = target_AUC * CL_pediatric / F

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    target_auc_mg_h_per_L : float
        Target AUC (mg·h/L). Must be > 0.
    cl_adult_L_per_h : float
        Adult total clearance (L/h). Must be > 0.
    weight_kg : float
        Child body weight (kg). Must be > 0.
    age_months : float
        Child age in months. Must be >= 0.
    fm_cyp3a4 : float
        Fraction of clearance via CYP3A4 (0–1). Used for maturation notes.
    fm_cyp2d6 : float
        Fraction of clearance via CYP2D6 (0–1). Used for maturation notes.
    f_renal : float
        Fraction of clearance via renal elimination (0–1).

    Returns
    -------
    PediatricDoseOptResult

    Raises
    ------
    ValueError
        If weight_kg <= 0, age_months < 0, target_auc <= 0, or cl_adult <= 0.
    """
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if age_months < 0:
        raise ValueError(f"age_months must be >= 0, got {age_months}")
    if target_auc_mg_h_per_L <= 0:
        raise ValueError(f"target_auc_mg_h_per_L must be > 0, got {target_auc_mg_h_per_L}")
    if cl_adult_L_per_h <= 0:
        raise ValueError(f"cl_adult_L_per_h must be > 0, got {cl_adult_L_per_h}")

    # Step 1: Allometric CL scaling
    cl_allometric = _allometric_cl(cl_adult_L_per_h, weight_kg)

    # Step 2: Maturation correction (only for age < 24 months)
    mat = _maturation_factor(age_months)
    cl_pediatric = cl_allometric * mat

    # Step 3: Optimal dose (oral, F = 0.7)
    f_bioavail = 0.7
    recommended_dose_mg = target_auc_mg_h_per_L * cl_pediatric / f_bioavail
    dose_per_kg = recommended_dose_mg / weight_kg

    # Step 4: Round to practical dose
    practical = _round_to_practical(recommended_dose_mg)

    notes_parts = [
        f"allometric_CL={cl_allometric:.3f} L/h",
        f"maturation_factor={mat:.3f}",
        f"cl_pediatric={cl_pediatric:.3f} L/h",
        f"optimal_dose(F=0.7)={recommended_dose_mg:.2f} mg",
        f"practical_dose={practical} mg",
    ]
    if age_months < 1:
        notes_parts.append("Neonate: high PK variability; TDM strongly recommended.")
    elif age_months < 6:
        notes_parts.append("Young infant: verify dose with neonatal specialist.")
    if fm_cyp3a4 > 0.3 and age_months < 12:
        notes_parts.append(
            f"CYP3A4 fm={fm_cyp3a4:.1f}: immature CYP3A4 may further reduce CL."
        )

    return PediatricDoseOptResult(
        drug_name=drug_name,
        weight_kg=weight_kg,
        age_months=age_months,
        target_auc=target_auc_mg_h_per_L,
        cl_pediatric=round(cl_pediatric, 6),
        recommended_dose_mg=round(recommended_dose_mg, 4),
        dose_per_kg=round(dose_per_kg, 4),
        practical_dose_mg=practical,
        notes="; ".join(notes_parts),
    )


def mg_per_kg_dosing(
    drug_name: str,
    reference_dose_mg_per_kg: float,
    weight_kg: float,
    age_months: float,
    max_dose_mg: float | None = None,
) -> PediatricDoseOptResult:
    """Simple weight-based dosing with adult dose cap.

    recommended_dose = min(ref_dose * weight, max_dose or inf)

    For CL estimation (used in PediatricDoseOptResult), assumes a typical
    adult CL of 1 L/h as placeholder and back-calculates target AUC.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    reference_dose_mg_per_kg : float
        Reference mg/kg dose. Must be > 0.
    weight_kg : float
        Child body weight (kg). Must be > 0.
    age_months : float
        Child age in months. Must be >= 0.
    max_dose_mg : float or None
        Adult maximum dose cap (mg). None = no cap.

    Returns
    -------
    PediatricDoseOptResult

    Raises
    ------
    ValueError
        If weight_kg <= 0, age_months < 0, or reference_dose_mg_per_kg <= 0.
    """
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if age_months < 0:
        raise ValueError(f"age_months must be >= 0, got {age_months}")
    if reference_dose_mg_per_kg <= 0:
        raise ValueError(
            f"reference_dose_mg_per_kg must be > 0, got {reference_dose_mg_per_kg}"
        )

    raw_dose = reference_dose_mg_per_kg * weight_kg
    was_capped = False
    if max_dose_mg is not None and raw_dose > max_dose_mg:
        raw_dose = max_dose_mg
        was_capped = True

    dose_per_kg = raw_dose / weight_kg
    practical = _round_to_practical(raw_dose)

    # Placeholder CL (back-calculate from dose, AUC = dose/CL)
    # Use a nominal CL based on allometric scaling from 1 L/h adult
    cl_placeholder = _allometric_cl(1.0, weight_kg) * _maturation_factor(age_months)
    # Back-calculated target AUC (nominal, F=0.7)
    target_auc = raw_dose * 0.7 / cl_placeholder if cl_placeholder > 0 else 0.0

    notes_parts = [
        f"ref_dose={reference_dose_mg_per_kg} mg/kg",
        f"raw_dose={reference_dose_mg_per_kg * weight_kg:.2f} mg",
    ]
    if was_capped:
        notes_parts.append(f"capped at max_dose={max_dose_mg} mg (adult limit)")
    if max_dose_mg is not None and not was_capped:
        notes_parts.append(f"below max_dose={max_dose_mg} mg; no cap applied")
    notes_parts.append(f"recommended={raw_dose:.2f} mg ({dose_per_kg:.2f} mg/kg)")

    return PediatricDoseOptResult(
        drug_name=drug_name,
        weight_kg=weight_kg,
        age_months=age_months,
        target_auc=round(target_auc, 4),
        cl_pediatric=round(cl_placeholder, 6),
        recommended_dose_mg=round(raw_dose, 4),
        dose_per_kg=round(dose_per_kg, 4),
        practical_dose_mg=practical,
        notes="; ".join(notes_parts),
    )


def optimize_pediatric_regimen(
    drug_name: str,
    target_trough_mg_L: float,
    target_peak_mg_L: float,
    cl_adult_L_per_h: float,
    vd_adult_L: float,
    weight_kg: float,
    age_months: float,
    route: str = "oral",
    interval_h: float = 12.0,
) -> PediatricRegimenResult:
    """Find the optimal pediatric dose and interval to achieve therapeutic targets.

    Scales CL and Vd allometrically with maturation correction. Evaluates
    candidate intervals (8, 12, 24 h) and finds the dose that achieves the
    target trough. Reports whether peak is also within therapeutic range.

    The best interval is selected as the one whose predicted steady-state
    trough is closest to the target trough while keeping peak < target_peak.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    target_trough_mg_L : float
        Desired minimum (trough) concentration at steady state (mg/L). Must be > 0.
    target_peak_mg_L : float
        Desired maximum (peak) concentration at steady state (mg/L). Must be > 0.
    cl_adult_L_per_h : float
        Adult clearance (L/h). Must be > 0.
    vd_adult_L : float
        Adult volume of distribution (L). Must be > 0.
    weight_kg : float
        Child body weight (kg). Must be > 0.
    age_months : float
        Child age in months. Must be >= 0.
    route : str
        Dosing route: "oral" or "iv".
    interval_h : float
        Preferred dosing interval in hours (8, 12, or 24 preferred).

    Returns
    -------
    PediatricRegimenResult

    Raises
    ------
    ValueError
        If any required parameter is invalid.
    """
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if age_months < 0:
        raise ValueError(f"age_months must be >= 0, got {age_months}")
    if target_trough_mg_L <= 0:
        raise ValueError(f"target_trough_mg_L must be > 0, got {target_trough_mg_L}")
    if target_peak_mg_L <= 0:
        raise ValueError(f"target_peak_mg_L must be > 0, got {target_peak_mg_L}")
    if cl_adult_L_per_h <= 0:
        raise ValueError(f"cl_adult_L_per_h must be > 0, got {cl_adult_L_per_h}")
    if vd_adult_L <= 0:
        raise ValueError(f"vd_adult_L must be > 0, got {vd_adult_L}")
    if route not in {"oral", "iv"}:
        raise ValueError(f"route must be 'oral' or 'iv', got {route!r}")

    # Pediatric PK parameter scaling
    cl_ped = _allometric_cl(cl_adult_L_per_h, weight_kg) * _maturation_factor(age_months)
    vd_ped = _allometric_vd(vd_adult_L, weight_kg)

    # Candidate intervals to evaluate
    candidate_intervals = [8.0, 12.0, 24.0]
    # Add the preferred interval if not already there
    if interval_h not in candidate_intervals:
        candidate_intervals.append(interval_h)
        candidate_intervals.sort()

    best_interval = interval_h
    best_dose = 0.0
    best_score = float("inf")

    for tau in candidate_intervals:
        # For target trough: solve for dose analytically
        # We compute trough and peak for a unit dose (1 mg), then scale
        trough_unit, peak_unit = _steady_state_trough_peak(1.0, cl_ped, vd_ped, tau, route)
        if trough_unit <= 0:
            continue

        # Dose needed to achieve target_trough
        dose_for_trough = target_trough_mg_L / trough_unit

        trough_pred, peak_pred = _steady_state_trough_peak(
            dose_for_trough, cl_ped, vd_ped, tau, route
        )

        # Score: minimize peak overshoot (peak > target_peak is penalized heavily)
        if peak_pred <= target_peak_mg_L:
            score = abs(trough_pred - target_trough_mg_L)
        else:
            # Peak too high: penalize
            score = abs(trough_pred - target_trough_mg_L) + 100.0 * (
                peak_pred - target_peak_mg_L
            )

        if score < best_score:
            best_score = score
            best_interval = tau
            best_dose = dose_for_trough

    practical_dose = _round_to_practical(best_dose)

    # Re-evaluate with practical dose
    trough_practical, peak_practical = _steady_state_trough_peak(
        practical_dose, cl_ped, vd_ped, best_interval, route
    )

    in_range = (trough_practical >= target_trough_mg_L * 0.8) and (
        peak_practical <= target_peak_mg_L * 1.2
    )

    notes_parts = [
        f"cl_ped={cl_ped:.3f} L/h; vd_ped={vd_ped:.2f} L",
        f"optimal_dose={best_dose:.2f} mg, interval={best_interval:.0f}h",
        f"practical_dose={practical_dose} mg",
        f"predicted_trough={trough_practical:.3f} mg/L (target>={target_trough_mg_L})",
        f"predicted_peak={peak_practical:.3f} mg/L (target<={target_peak_mg_L})",
    ]
    if not in_range:
        notes_parts.append("WARNING: practical dose outside ±20% of therapeutic targets.")

    return PediatricRegimenResult(
        drug_name=drug_name,
        weight_kg=weight_kg,
        age_months=age_months,
        recommended_dose_mg=practical_dose,
        interval_h=best_interval,
        predicted_trough=round(trough_practical, 6),
        predicted_peak=round(peak_practical, 6),
        in_therapeutic_range=in_range,
        notes="; ".join(notes_parts),
    )
