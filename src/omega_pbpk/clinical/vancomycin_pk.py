"""
Phase 963 — Vancomycin PK model with AUC-guided dosing.

ASHP/SIDP/IDSA 2020 guidelines:
  Target AUC24/MIC = 400-600 mg·h/L (MIC = 1 mg/L)
  2-compartment forward Euler, IV infusion, renal dosing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["VancomycinPKResult", "simulate_vancomycin_pk", "recommend_vancomycin_dose"]


@dataclass
class VancomycinPKResult:
    drug_name: str
    dose_mg: float
    weight_kg: float
    crcl_ml_min: float
    cl_L_per_h: float
    vd_central_L: float
    times_h: list[float] = field(default_factory=list)
    c_central_mg_L: list[float] = field(default_factory=list)
    cmax_mg_L: float = 0.0
    cmin_mg_L: float = 0.0
    auc24_mg_h_per_L: float = 0.0
    auc_mic_ratio: float = 0.0
    target_attained: bool = False
    recommended_dose_mg: float = 0.0
    notes: str = ""


def _cockcroft_gault(age_years: float, weight_kg: float, scr_mg_dL: float, sex: str) -> float:
    """CrCl in mL/min via Cockcroft-Gault."""
    crcl = (140.0 - age_years) * weight_kg / (72.0 * scr_mg_dL)
    if sex == "female":
        crcl *= 0.85
    return max(crcl, 1.0)  # floor at 1 mL/min


def _cl_from_crcl(crcl_ml_min: float, weight_kg: float) -> float:
    """Total CL in L/h from CrCl (Matzke 1984, per-kg then multiply by weight)."""
    cl_per_kg = 0.0028 * crcl_ml_min + 0.006  # L/h/kg
    return cl_per_kg * weight_kg


def _auc_trapezoidal(times: list, conc: list) -> float:
    """Trapezoidal AUC over the provided time/conc arrays."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (conc[i] + conc[i - 1]) * dt
    return auc


def _validate_params(dose_mg, weight_kg, age_years, scr_mg_dL, sex, n_doses):
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    if age_years <= 0:
        raise ValueError("age_years must be > 0")
    if scr_mg_dL <= 0:
        raise ValueError("scr_mg_dL must be > 0")
    if sex not in {"male", "female"}:
        raise ValueError("sex must be 'male' or 'female'")
    if n_doses < 1:
        raise ValueError("n_doses must be >= 1")


def simulate_vancomycin_pk(
    dose_mg: float,
    weight_kg: float = 70.0,
    age_years: float = 50.0,
    sex: str = "male",
    scr_mg_dL: float = 1.0,
    dosing_interval_h: float = 12.0,
    infusion_duration_h: float = 1.0,
    n_doses: int = 3,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
    mic_mg_L: float = 1.0,
) -> VancomycinPKResult:
    """Simulate vancomycin PK using 2-cpt forward Euler with IV infusion."""
    _validate_params(dose_mg, weight_kg, age_years, scr_mg_dL, sex, n_doses)

    crcl = _cockcroft_gault(age_years, weight_kg, scr_mg_dL, sex)
    cl = _cl_from_crcl(crcl, weight_kg)

    # 2-cpt population parameters
    vd_c = 0.4 * weight_kg  # L
    _vd_p = 0.4 * weight_kg  # L
    k12 = 0.4  # /h  (central → peripheral)
    k21 = 0.3  # /h  (peripheral → central)
    ke = cl / vd_c  # elimination rate /h

    infusion_rate = dose_mg / infusion_duration_h  # mg/h

    # Dose times
    dose_times = [i * dosing_interval_h for i in range(n_doses)]

    # Forward Euler simulation
    n_steps = int(math.ceil(t_end_h / dt_h)) + 1
    times = [i * dt_h for i in range(n_steps)]
    # Clamp last time to t_end_h
    if times[-1] > t_end_h:
        times[-1] = t_end_h

    A_c = 0.0  # mg in central compartment
    A_p = 0.0  # mg in peripheral compartment

    c_central = []

    for idx, t in enumerate(times):
        # Record concentration
        c_central.append(A_c / vd_c)

        # Determine infusion input at this moment
        R_in = 0.0
        for td in dose_times:
            if td <= t < td + infusion_duration_h:
                R_in += infusion_rate

        # Derivatives
        dAc_dt = R_in - (ke + k12) * A_c + k21 * A_p
        dAp_dt = k12 * A_c - k21 * A_p

        if idx < n_steps - 1:
            dt_actual = times[idx + 1] - t
            A_c = max(A_c + dAc_dt * dt_actual, 0.0)
            A_p = max(A_p + dAp_dt * dt_actual, 0.0)

    # Cmax over simulation
    cmax = max(c_central)

    # Cmin: trough before last dose (or last value)
    last_dose_time = dose_times[-1]
    # Find concentration just before last dose
    cmin = c_central[0]
    for idx, t in enumerate(times):
        if t < last_dose_time:
            cmin = c_central[idx]

    # AUC24: trapezoidal over first 24h
    times_24 = [t for t in times if t <= 24.0]
    conc_24 = c_central[: len(times_24)]
    if not times_24:
        times_24 = times[:2]
        conc_24 = c_central[:2]
    auc24 = _auc_trapezoidal(times_24, conc_24)

    auc_mic = auc24 / mic_mg_L
    target = 400.0 <= auc_mic <= 600.0

    # Analytical recommended dose: scale linearly to reach AUC/MIC=500
    target_auc = 500.0 * mic_mg_L
    if auc24 > 0:
        rec_dose = dose_mg * target_auc / auc24
    else:
        rec_dose = dose_mg

    notes = (
        f"CrCl={crcl:.1f} mL/min, CL={cl:.2f} L/h, Vd_central={vd_c:.1f} L. "
        f"AUC24/MIC={auc_mic:.1f}. "
        f"Target 400-600 {'ATTAINED' if target else 'NOT attained'}. "
        f"Recommended dose for AUC/MIC=500: {rec_dose:.0f} mg."
    )

    return VancomycinPKResult(
        drug_name="vancomycin",
        dose_mg=dose_mg,
        weight_kg=weight_kg,
        crcl_ml_min=crcl,
        cl_L_per_h=cl,
        vd_central_L=vd_c,
        times_h=times,
        c_central_mg_L=c_central,
        cmax_mg_L=cmax,
        cmin_mg_L=cmin,
        auc24_mg_h_per_L=auc24,
        auc_mic_ratio=auc_mic,
        target_attained=target,
        recommended_dose_mg=rec_dose,
        notes=notes,
    )


def recommend_vancomycin_dose(
    weight_kg: float,
    age_years: float,
    sex: str,
    scr_mg_dL: float,
    mic_mg_L: float = 1.0,
    target_auc: float = 500.0,
    dosing_interval_h: float = 12.0,
    infusion_duration_h: float = 1.0,
    n_doses: int = 3,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> dict:
    """Recommend vancomycin dose to achieve target AUC/MIC.

    Uses an initial dose of 15 mg/kg, simulates, then linearly scales.
    """
    _validate_params(15.0 * weight_kg, weight_kg, age_years, scr_mg_dL, sex, n_doses)

    initial_dose = 15.0 * weight_kg  # mg (15 mg/kg starting estimate)

    result = simulate_vancomycin_pk(
        dose_mg=initial_dose,
        weight_kg=weight_kg,
        age_years=age_years,
        sex=sex,
        scr_mg_dL=scr_mg_dL,
        dosing_interval_h=dosing_interval_h,
        infusion_duration_h=infusion_duration_h,
        n_doses=n_doses,
        t_end_h=t_end_h,
        dt_h=dt_h,
        mic_mg_L=mic_mg_L,
    )

    target_auc_total = target_auc * mic_mg_L
    if result.auc24_mg_h_per_L > 0:
        rec_dose = initial_dose * target_auc_total / result.auc24_mg_h_per_L
    else:
        rec_dose = initial_dose

    # Round to nearest 250 mg for practical use
    rec_dose_rounded = round(rec_dose / 250.0) * 250.0
    if rec_dose_rounded <= 0:
        rec_dose_rounded = 250.0

    # Predict AUC at rounded dose
    pred_result = simulate_vancomycin_pk(
        dose_mg=rec_dose_rounded,
        weight_kg=weight_kg,
        age_years=age_years,
        sex=sex,
        scr_mg_dL=scr_mg_dL,
        dosing_interval_h=dosing_interval_h,
        infusion_duration_h=infusion_duration_h,
        n_doses=n_doses,
        t_end_h=t_end_h,
        dt_h=dt_h,
        mic_mg_L=mic_mg_L,
    )

    predicted_auc24 = pred_result.auc24_mg_h_per_L
    auc_mic_pred = predicted_auc24 / mic_mg_L
    target_attained = 400.0 <= auc_mic_pred <= 600.0

    return {
        "recommended_dose_mg": rec_dose_rounded,
        "dosing_interval_h": dosing_interval_h,
        "predicted_auc24": predicted_auc24,
        "predicted_auc_mic_ratio": auc_mic_pred,
        "target_attained": target_attained,
        "crcl_ml_min": result.crcl_ml_min,
        "cl_L_per_h": result.cl_L_per_h,
    }
