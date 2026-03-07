"""Microdose PK Extrapolation (Phase 553).

Extrapolate therapeutic PK from microdose (<=100 ug) human studies.
Assumes linear PK at low doses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MicrodosePKResult:
    drug_name: str
    microdose_ug: float
    therapeutic_dose_mg: float
    dose_ratio: float
    cl_L_per_h: float
    vd_L: float
    t_half_h: float
    f_bioavailable: float
    cmax_predicted_ng_mL: float
    auc_predicted_ng_h_mL: float
    utility_rating: str  # 'high', 'moderate', 'low'
    notes: str


def extrapolate_from_microdose(
    microdose_ug: float,
    therapeutic_dose_mg: float,
    microdose_cmax_ng_mL: float,
    microdose_auc_ng_h_mL: float,
    route: str = "oral",
    f_microdose: float | None = None,
    f_therapeutic: float | None = None,
) -> dict:
    """Extrapolate therapeutic PK from microdose data using linear scaling.

    Parameters
    ----------
    microdose_ug:
        Microdose amount in micrograms (must be <= 100 ug in practice).
    therapeutic_dose_mg:
        Therapeutic dose in milligrams.
    microdose_cmax_ng_mL:
        Observed Cmax from microdose study (ng/mL).
    microdose_auc_ng_h_mL:
        Observed AUC from microdose study (ng*h/mL).
    route:
        Administration route ('oral', 'iv').
    f_microdose:
        Bioavailability at microdose (fraction, 0-1). Optional.
    f_therapeutic:
        Bioavailability at therapeutic dose (fraction, 0-1). Optional.

    Returns
    -------
    dict with keys: dose_ratio, cmax_predicted_ng_mL, auc_predicted_ng_h_mL,
        cl_L_per_h, assumption_valid, notes
    """
    if microdose_ug <= 0:
        raise ValueError("microdose_ug must be > 0")
    if therapeutic_dose_mg <= 0:
        raise ValueError("therapeutic_dose_mg must be > 0")
    if microdose_cmax_ng_mL < 0:
        raise ValueError("microdose_cmax_ng_mL must be >= 0")
    if microdose_auc_ng_h_mL <= 0:
        raise ValueError("microdose_auc_ng_h_mL must be > 0")

    # Convert therapeutic dose to micrograms for ratio computation
    therapeutic_dose_ug = therapeutic_dose_mg * 1000.0
    dose_ratio = therapeutic_dose_ug / microdose_ug

    # Bioavailability correction factor
    if f_microdose is not None and f_therapeutic is not None:
        if f_microdose <= 0:
            raise ValueError("f_microdose must be > 0")
        f_correction = f_therapeutic / f_microdose
    else:
        f_correction = 1.0

    cmax_predicted = microdose_cmax_ng_mL * dose_ratio * f_correction
    auc_predicted = microdose_auc_ng_h_mL * dose_ratio * f_correction

    # CL calculation: CL = dose / AUC (with unit conversion)
    # dose in ng, AUC in ng*h/mL => CL in mL/h; convert to L/h
    microdose_ng = microdose_ug * 1000.0  # 1 ug = 1000 ng
    cl_mL_per_h = microdose_ng / microdose_auc_ng_h_mL
    cl_L_per_h = cl_mL_per_h / 1000.0

    # Assumption validity: dose ratio should not be too extreme
    assumption_valid = dose_ratio <= 10000.0

    notes_parts = [
        f"Route: {route}",
        f"Dose ratio: {dose_ratio:.1f}x",
    ]
    if f_correction != 1.0:
        notes_parts.append(f"F correction applied: {f_correction:.3f}")
    if not assumption_valid:
        notes_parts.append("WARNING: dose ratio > 10000; linear PK assumption may not hold")

    return {
        "dose_ratio": dose_ratio,
        "cmax_predicted_ng_mL": cmax_predicted,
        "auc_predicted_ng_h_mL": auc_predicted,
        "cl_L_per_h": cl_L_per_h,
        "assumption_valid": assumption_valid,
        "notes": "; ".join(notes_parts),
    }


def validate_linearity(
    dose1_mg: float,
    dose2_mg: float,
    auc1_ng_h_mL: float,
    auc2_ng_h_mL: float,
    tolerance_pct: float = 25.0,
) -> dict:
    """Check dose-proportionality between two doses.

    Parameters
    ----------
    dose1_mg:
        First dose in mg.
    dose2_mg:
        Second dose in mg.
    auc1_ng_h_mL:
        AUC at dose1 (ng*h/mL).
    auc2_ng_h_mL:
        Observed AUC at dose2 (ng*h/mL).
    tolerance_pct:
        Acceptable deviation from proportionality (%).

    Returns
    -------
    dict with keys: expected_auc2, observed_auc2, deviation_pct, linear, notes
    """
    if dose1_mg <= 0 or dose2_mg <= 0:
        raise ValueError("Doses must be > 0")
    if auc1_ng_h_mL <= 0:
        raise ValueError("auc1_ng_h_mL must be > 0")

    expected_auc2 = auc1_ng_h_mL * (dose2_mg / dose1_mg)
    deviation_pct = abs(auc2_ng_h_mL - expected_auc2) / expected_auc2 * 100.0
    linear = deviation_pct <= tolerance_pct

    if linear:
        notes = (
            f"AUC is dose-proportional within {tolerance_pct:.0f}% tolerance "
            f"(deviation: {deviation_pct:.1f}%)"
        )
    else:
        notes = (
            f"Non-linear PK detected: deviation {deviation_pct:.1f}% exceeds "
            f"{tolerance_pct:.0f}% tolerance; possible saturable processes"
        )

    return {
        "expected_auc2": expected_auc2,
        "observed_auc2": auc2_ng_h_mL,
        "deviation_pct": deviation_pct,
        "linear": linear,
        "notes": notes,
    }


def auc_from_sparse_sampling(
    times_h: list[float],
    concentrations_ng_mL: list[float],
    last_timepoint_h: float | None = None,
) -> dict:
    """Compute AUC from sparse (2-5 sample) microdose data.

    Uses trapezoidal rule for observed interval and log-linear extrapolation
    to infinity using the last two data points.

    Parameters
    ----------
    times_h:
        Sampling times in hours.
    concentrations_ng_mL:
        Corresponding concentrations in ng/mL.
    last_timepoint_h:
        Optional last timepoint for truncation (not used currently).

    Returns
    -------
    dict with keys: auc_0_last, auc_extrap, auc_total_ng_h_mL,
        ke_terminal_per_h, t_half_h, pct_extrap
    """
    if len(times_h) < 2:
        raise ValueError("At least 2 timepoints required")
    if len(times_h) != len(concentrations_ng_mL):
        raise ValueError("times_h and concentrations_ng_mL must have same length")
    if any(c < 0 for c in concentrations_ng_mL):
        raise ValueError("Concentrations must be >= 0")

    # Trapezoidal AUC from 0 to last point
    auc_0_last = 0.0
    for i in range(1, len(times_h)):
        dt = times_h[i] - times_h[i - 1]
        auc_0_last += 0.5 * (concentrations_ng_mL[i] + concentrations_ng_mL[i - 1]) * dt

    # Terminal elimination rate from last 2 points
    c_last = concentrations_ng_mL[-1]
    c_prev = concentrations_ng_mL[-2]
    t_last = times_h[-1]
    t_prev = times_h[-2]

    # Avoid log(0) or log of negative/zero concentration
    if c_last <= 0 or c_prev <= 0:
        ke_terminal = 0.1  # fallback
    else:
        dt = t_last - t_prev
        if dt <= 0:
            ke_terminal = 0.1
        else:
            ke_terminal = (math.log(c_prev) - math.log(c_last)) / dt
            if ke_terminal <= 0:
                ke_terminal = 0.01  # fallback for increasing concentrations

    t_half_h = math.log(2.0) / ke_terminal

    # Extrapolated AUC from last point to infinity
    auc_extrap = c_last / ke_terminal

    auc_total = auc_0_last + auc_extrap
    pct_extrap = (auc_extrap / auc_total * 100.0) if auc_total > 0 else 0.0

    return {
        "auc_0_last": auc_0_last,
        "auc_extrap": auc_extrap,
        "auc_total_ng_h_mL": auc_total,
        "ke_terminal_per_h": ke_terminal,
        "t_half_h": t_half_h,
        "pct_extrap": pct_extrap,
    }


def microdose_clinical_utility(
    drug_name: str,
    microdose_ug: float,
    therapeutic_dose_mg: float,
    f_microdose: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "oral",
) -> MicrodosePKResult:
    """Compute full PK parameters from microdose CL estimate.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    microdose_ug:
        Microdose in micrograms.
    therapeutic_dose_mg:
        Therapeutic dose in milligrams.
    f_microdose:
        Bioavailability at microdose (fraction, 0-1).
    cl_L_per_h:
        Clearance estimated from microdose (L/h).
    vd_L:
        Volume of distribution (L).
    route:
        Administration route.

    Returns
    -------
    MicrodosePKResult dataclass.
    """
    if microdose_ug <= 0:
        raise ValueError("microdose_ug must be > 0")
    if therapeutic_dose_mg <= 0:
        raise ValueError("therapeutic_dose_mg must be > 0")
    if f_microdose <= 0 or f_microdose > 1.0:
        raise ValueError("f_microdose must be in (0, 1]")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    therapeutic_dose_ug = therapeutic_dose_mg * 1000.0
    dose_ratio = therapeutic_dose_ug / microdose_ug

    t_half_h = math.log(2.0) * vd_L / cl_L_per_h

    # Predict AUC and Cmax at therapeutic dose
    # AUC = F * dose / CL
    therapeutic_dose_ng = therapeutic_dose_mg * 1e6  # mg -> ng
    auc_predicted_ng_h_mL = f_microdose * therapeutic_dose_ng / (cl_L_per_h * 1000.0)  # CL in mL/h

    # Approximate Cmax using 1-cpt oral model: Cmax ~ F*D/Vd (rough estimate)
    # More precisely, scale from microdose Cmax is not available here, so use:
    # Cmax = AUC * CL / (Vd * ka) but without ka, use AUC-based estimate
    # Simplified: Cmax ~ AUC * ke where ke = CL/Vd
    ke = cl_L_per_h / vd_L
    cmax_predicted_ng_mL = auc_predicted_ng_h_mL * ke

    # Utility rating based on dose ratio
    if dose_ratio <= 1000.0:
        utility_rating = "high"
    elif dose_ratio <= 5000.0:
        utility_rating = "moderate"
    else:
        utility_rating = "low"

    notes = (
        f"Dose ratio {dose_ratio:.1f}x; t1/2={t_half_h:.1f}h; "
        f"route={route}; utility={utility_rating}"
    )

    return MicrodosePKResult(
        drug_name=drug_name,
        microdose_ug=microdose_ug,
        therapeutic_dose_mg=therapeutic_dose_mg,
        dose_ratio=dose_ratio,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        t_half_h=t_half_h,
        f_bioavailable=f_microdose,
        cmax_predicted_ng_mL=cmax_predicted_ng_mL,
        auc_predicted_ng_h_mL=auc_predicted_ng_h_mL,
        utility_rating=utility_rating,
        notes=notes,
    )
