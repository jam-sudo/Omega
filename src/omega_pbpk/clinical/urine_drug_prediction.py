"""Urine Drug Concentration Prediction for drug monitoring and compliance testing.

Phase 363: Predict urine drug concentrations for drug monitoring purposes
(urine drug testing for compliance/abuse monitoring).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UrinePredictionResult:
    """Result of urine drug concentration prediction."""

    drug_name: str
    dose_mg: float
    time_since_dose_h: float

    # Urine concentration
    c_urine_ng_mL: float
    c_urine_low_ng_mL: float
    c_urine_high_ng_mL: float

    # Detection windows
    detection_window_h: float
    cutoff_ng_mL: float
    above_cutoff: bool

    # PK basis
    plasma_c_mg_L: float
    urine_plasma_ratio: float
    cumulative_excreted_pct: float

    # Creatinine correction
    creatinine_corrected_ng_per_mg_creatinine: float

    notes: str


def _plasma_concentration(
    dose_mg: float,
    f_oral: float,
    ka_per_h: float,
    ke_per_h: float,
    vd_L: float,
    t_h: float,
) -> float:
    """1-compartment oral PK: plasma concentration at time t (mg/L)."""
    if t_h <= 0.0:
        return 0.0
    if abs(ka_per_h - ke_per_h) < 1e-9:
        # Avoid division by zero; perturb slightly
        ka_per_h = ka_per_h * 1.0001
    c = (
        (dose_mg * f_oral * ka_per_h)
        / (vd_L * (ka_per_h - ke_per_h))
        * (math.exp(-ke_per_h * t_h) - math.exp(-ka_per_h * t_h))
    )
    return max(0.0, c)


def _ionization_ratio_correction(
    urine_ph: float,
    pka_acid: float | None,
    pka_base: float | None,
) -> float:
    """pH-dependent ionization correction for urine vs plasma (pH 7.4)."""
    plasma_ph = 7.4
    if pka_acid is not None:
        # Acid: ratio_correction = (1 + 10^(urine_pH - pKa)) / (1 + 10^(7.4 - pKa))
        numerator = 1.0 + 10.0 ** (urine_ph - pka_acid)
        denominator = 1.0 + 10.0 ** (plasma_ph - pka_acid)
        return numerator / denominator
    if pka_base is not None:
        # Base: ratio_correction = (1 + 10^(pKa - urine_pH)) / (1 + 10^(pKa - 7.4))
        numerator = 1.0 + 10.0 ** (pka_base - urine_ph)
        denominator = 1.0 + 10.0 ** (pka_base - plasma_ph)
        return numerator / denominator
    return 1.0


def _cumulative_excreted_pct(
    dose_mg: float,
    f_oral: float,
    ka_per_h: float,
    ke_per_h: float,
    vd_L: float,
    cl_renal_L_per_h: float,
    t_end_h: float,
    dt_h: float = 0.1,
) -> float:
    """Cumulative % of dose excreted in urine from 0 to t_end_h."""
    if t_end_h <= 0.0:
        return 0.0
    times = np.arange(0.0, t_end_h + dt_h, dt_h)
    rates = np.array(
        [
            cl_renal_L_per_h * _plasma_concentration(dose_mg, f_oral, ka_per_h, ke_per_h, vd_L, t)
            for t in times
        ]
    )
    cumulative_mg = float(np.trapz(rates, times))
    pct = cumulative_mg / dose_mg * 100.0
    return min(100.0, max(0.0, pct))


def _estimate_detection_window(
    dose_mg: float,
    f_oral: float,
    ka_per_h: float,
    ke_per_h: float,
    vd_L: float,
    cl_renal_L_per_h: float,
    urine_flow_L_per_h: float,
    ratio_correction: float,
    cutoff_ng_mL: float,
    max_t_h: float = 200.0,
    dt_h: float = 0.5,
) -> float:
    """Estimate detection window: time until C_urine drops below cutoff."""
    cutoff_mg_L = cutoff_ng_mL / 1000.0
    up_ratio = cl_renal_L_per_h / urine_flow_L_per_h * ratio_correction

    # Scan forward to find last time above cutoff
    times = np.arange(dt_h, max_t_h + dt_h, dt_h)
    last_above = 0.0
    for t in times:
        c_plasma = _plasma_concentration(dose_mg, f_oral, ka_per_h, ke_per_h, vd_L, t)
        c_urine_mg_L = c_plasma * up_ratio
        if c_urine_mg_L >= cutoff_mg_L:
            last_above = t
        elif last_above > 0.0 and c_urine_mg_L < cutoff_mg_L * 0.5:
            # Clearly fallen below, stop scanning
            break
    return last_above


def predict_urine_concentration(
    drug_name: str,
    dose_mg: float,
    time_since_dose_h: float,
    cl_L_per_h: float,
    vd_L: float,
    fe_urine: float = 0.3,
    ka_per_h: float = 1.2,
    f_oral: float = 0.9,
    urine_ph: float = 6.0,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    cutoff_ng_mL: float = 300.0,
    urine_flow_mL_per_h: float = 60.0,
    creatinine_mg_dL: float = 100.0,
) -> UrinePredictionResult:
    """Predict urine drug concentration at a given time after oral dose.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    time_since_dose_h:
        Time since dose administration (hours).
    cl_L_per_h:
        Total clearance (L/h).
    vd_L:
        Volume of distribution (L).
    fe_urine:
        Fraction of clearance via renal route (0-1).
    ka_per_h:
        Absorption rate constant (per hour).
    f_oral:
        Oral bioavailability fraction.
    urine_ph:
        Urine pH (typical 4.5-8.0).
    pka_acid:
        pKa of acidic drug (None if not applicable).
    pka_base:
        pKa of basic drug (None if not applicable).
    cutoff_ng_mL:
        Screening cutoff concentration (ng/mL).
    urine_flow_mL_per_h:
        Urine flow rate (mL/h), default 60 mL/h (1 mL/min).
    creatinine_mg_dL:
        Urine creatinine concentration (mg/dL), default 100 mg/dL.

    Returns
    -------
    UrinePredictionResult
    """
    # Input validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")
    if not (0 < fe_urine <= 1):
        raise ValueError("fe_urine must be in (0, 1]")
    if time_since_dose_h < 0:
        raise ValueError("time_since_dose_h must be >= 0")
    if urine_flow_mL_per_h <= 0:
        raise ValueError("urine_flow_mL_per_h must be positive")

    ke = cl_L_per_h / vd_L
    cl_renal = fe_urine * cl_L_per_h
    urine_flow_L_per_h = urine_flow_mL_per_h / 1000.0

    # Plasma concentration at time t
    c_plasma_mg_L = _plasma_concentration(dose_mg, f_oral, ka_per_h, ke, vd_L, time_since_dose_h)

    # pH-dependent ionization correction
    ratio_correction = _ionization_ratio_correction(urine_ph, pka_acid, pka_base)

    # U/P ratio: CL_renal / urine_flow (both in L/h) * ionization correction
    up_ratio = cl_renal / urine_flow_L_per_h * ratio_correction

    # Urine concentration
    c_urine_mg_L = c_plasma_mg_L * up_ratio
    c_urine_ng_mL = c_urine_mg_L * 1000.0

    # Variability bounds: ±50%
    c_urine_low_ng_mL = c_urine_ng_mL * 0.5
    c_urine_high_ng_mL = c_urine_ng_mL * 1.5

    # Above cutoff?
    above_cutoff = c_urine_ng_mL > cutoff_ng_mL

    # Cumulative excretion
    cumulative_pct = _cumulative_excreted_pct(
        dose_mg, f_oral, ka_per_h, ke, vd_L, cl_renal, time_since_dose_h
    )

    # Detection window
    detection_window = _estimate_detection_window(
        dose_mg,
        f_oral,
        ka_per_h,
        ke,
        vd_L,
        cl_renal,
        urine_flow_L_per_h,
        ratio_correction,
        cutoff_ng_mL,
    )

    # Creatinine correction: ng/mg creatinine
    # Standard: ng/mg = (ng/mL) / (mg/dL / 100)  → multiply by 100/creatinine_mg_dL
    creatinine_corrected = c_urine_ng_mL / (creatinine_mg_dL / 100.0)

    notes_parts = []
    if pka_acid is not None:
        notes_parts.append(f"Acidic drug pKa={pka_acid}, pH correction={ratio_correction:.2f}")
    elif pka_base is not None:
        notes_parts.append(f"Basic drug pKa={pka_base}, pH correction={ratio_correction:.2f}")
    else:
        notes_parts.append("Neutral drug, no pH correction")
    if time_since_dose_h == 0.0:
        notes_parts.append("t=0: no drug absorbed yet")
    notes = "; ".join(notes_parts)

    return UrinePredictionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        time_since_dose_h=time_since_dose_h,
        c_urine_ng_mL=c_urine_ng_mL,
        c_urine_low_ng_mL=c_urine_low_ng_mL,
        c_urine_high_ng_mL=c_urine_high_ng_mL,
        detection_window_h=detection_window,
        cutoff_ng_mL=cutoff_ng_mL,
        above_cutoff=above_cutoff,
        plasma_c_mg_L=c_plasma_mg_L,
        urine_plasma_ratio=up_ratio,
        cumulative_excreted_pct=cumulative_pct,
        creatinine_corrected_ng_per_mg_creatinine=creatinine_corrected,
        notes=notes,
    )


def detection_timeline(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    fe_urine: float,
    cutoff_ng_mL: float = 300.0,
    n_timepoints: int = 48,
    ka_per_h: float = 1.2,
    f_oral: float = 0.9,
    urine_ph: float = 6.0,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    urine_flow_mL_per_h: float = 60.0,
) -> dict:
    """Evaluate urine concentration at hourly timepoints over n_timepoints hours.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose (mg).
    cl_L_per_h:
        Total clearance (L/h).
    vd_L:
        Volume of distribution (L).
    fe_urine:
        Fraction of clearance via renal route.
    cutoff_ng_mL:
        Screening cutoff concentration (ng/mL).
    n_timepoints:
        Number of hourly timepoints to evaluate (default 48).
    ka_per_h:
        Absorption rate constant (per hour).
    f_oral:
        Oral bioavailability fraction.
    urine_ph:
        Urine pH.
    pka_acid:
        pKa of acidic drug (None if not applicable).
    pka_base:
        pKa of basic drug (None if not applicable).
    urine_flow_mL_per_h:
        Urine flow rate (mL/h).

    Returns
    -------
    dict with keys: times_h, c_urine_ng_mL, above_cutoff, detection_window_h
    """
    times_h = list(range(n_timepoints))
    c_urine_list = []
    above_cutoff_list = []

    for t in times_h:
        result = predict_urine_concentration(
            drug_name=drug_name,
            dose_mg=dose_mg,
            time_since_dose_h=float(t),
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            fe_urine=fe_urine,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
            urine_ph=urine_ph,
            pka_acid=pka_acid,
            pka_base=pka_base,
            cutoff_ng_mL=cutoff_ng_mL,
            urine_flow_mL_per_h=urine_flow_mL_per_h,
        )
        c_urine_list.append(result.c_urine_ng_mL)
        above_cutoff_list.append(result.above_cutoff)

    # Detection window: last index above cutoff
    last_above = 0.0
    for i, above in enumerate(above_cutoff_list):
        if above:
            last_above = float(times_h[i])

    return {
        "times_h": times_h,
        "c_urine_ng_mL": c_urine_list,
        "above_cutoff": above_cutoff_list,
        "detection_window_h": last_above,
    }
