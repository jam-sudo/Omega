"""Phase 519 — Creatinine Clearance & Renal Function Estimation.

Provides Cockcroft-Gault, CKD-EPI, MDRD-4, and Schwartz equations for
estimating renal function, plus CKD staging and dose adjustment utilities.
"""

from __future__ import annotations


def cockcroft_gault(
    age_years: float,
    weight_kg: float,
    scr_mg_dL: float,
    sex: str,
) -> float:
    """Estimate creatinine clearance (mL/min) using the Cockcroft-Gault equation.

    Args:
        age_years: Patient age in years (>0).
        weight_kg: Patient weight in kg (>0).
        scr_mg_dL: Serum creatinine in mg/dL (>0).
        sex: 'male' or 'female'.

    Returns:
        CrCl in mL/min.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if age_years <= 0:
        raise ValueError("age_years must be positive")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be positive")
    if scr_mg_dL <= 0:
        raise ValueError("scr_mg_dL must be positive")
    sex = sex.lower()
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")

    crcl = (140.0 - age_years) * weight_kg / (72.0 * scr_mg_dL)
    if sex == "female":
        crcl *= 0.85
    return crcl


def ckd_epi_egfr(
    age_years: float,
    scr_mg_dL: float,
    sex: str,
    race: str = "non_black",
) -> float:
    """Estimate eGFR using the CKD-EPI 2009 equation (mL/min/1.73m²).

    Args:
        age_years: Patient age in years (>0).
        scr_mg_dL: Serum creatinine in mg/dL (>0).
        sex: 'male' or 'female'.
        race: 'black' or 'non_black' (default 'non_black').

    Returns:
        eGFR in mL/min/1.73m².

    Raises:
        ValueError: If any parameter is invalid.
    """
    if age_years <= 0:
        raise ValueError("age_years must be positive")
    if scr_mg_dL <= 0:
        raise ValueError("scr_mg_dL must be positive")
    sex = sex.lower()
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")
    race = race.lower()
    if race not in ("black", "non_black"):
        raise ValueError("race must be 'black' or 'non_black'")

    if sex == "female":
        kappa = 0.7
        alpha = -0.329
    else:
        kappa = 0.9
        alpha = -0.411

    ratio = scr_mg_dL / kappa
    min_ratio = min(ratio, 1.0)
    max_ratio = max(ratio, 1.0)

    egfr = 141.0 * (min_ratio**alpha) * (max_ratio**-1.209) * (0.993**age_years)
    if sex == "female":
        egfr *= 1.018
    if race == "black":
        egfr *= 1.159
    return egfr


def mdrd_egfr(
    age_years: float,
    scr_mg_dL: float,
    sex: str,
    race: str = "non_black",
) -> float:
    """Estimate eGFR using the MDRD-4 equation (mL/min/1.73m²).

    Args:
        age_years: Patient age in years (>0).
        scr_mg_dL: Serum creatinine in mg/dL (>0).
        sex: 'male' or 'female'.
        race: 'black' or 'non_black' (default 'non_black').

    Returns:
        eGFR in mL/min/1.73m².

    Raises:
        ValueError: If any parameter is invalid.
    """
    if age_years <= 0:
        raise ValueError("age_years must be positive")
    if scr_mg_dL <= 0:
        raise ValueError("scr_mg_dL must be positive")
    sex = sex.lower()
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")
    race = race.lower()
    if race not in ("black", "non_black"):
        raise ValueError("race must be 'black' or 'non_black'")

    egfr = 175.0 * (scr_mg_dL**-1.154) * (age_years**-0.203)
    if sex == "female":
        egfr *= 0.742
    if race == "black":
        egfr *= 1.212
    return egfr


def schwartz_egfr(height_cm: float, scr_mg_dL: float) -> float:
    """Estimate pediatric eGFR using the Schwartz equation (mL/min/1.73m²).

    Args:
        height_cm: Patient height in cm (>0).
        scr_mg_dL: Serum creatinine in mg/dL (>0).

    Returns:
        eGFR in mL/min/1.73m².

    Raises:
        ValueError: If any parameter is invalid.
    """
    if height_cm <= 0:
        raise ValueError("height_cm must be positive")
    if scr_mg_dL <= 0:
        raise ValueError("scr_mg_dL must be positive")

    return 0.413 * height_cm / scr_mg_dL


def classify_ckd_stage(egfr: float) -> dict:
    """Classify CKD stage based on eGFR value.

    Args:
        egfr: Estimated GFR in mL/min/1.73m².

    Returns:
        Dict with keys: stage, description, dose_adjustment_needed.

    Raises:
        ValueError: If egfr is negative.
    """
    if egfr < 0:
        raise ValueError("egfr must be non-negative")

    if egfr >= 90:
        stage = "G1"
        description = "Normal or high"
        adjustment = False
    elif egfr >= 60:
        stage = "G2"
        description = "Mildly decreased"
        adjustment = False
    elif egfr >= 45:
        stage = "G3a"
        description = "Mildly to moderately decreased"
        adjustment = True
    elif egfr >= 30:
        stage = "G3b"
        description = "Moderately to severely decreased"
        adjustment = True
    elif egfr >= 15:
        stage = "G4"
        description = "Severely decreased"
        adjustment = True
    else:
        stage = "G5"
        description = "Kidney failure"
        adjustment = True

    return {
        "stage": stage,
        "description": description,
        "dose_adjustment_needed": adjustment,
    }


def renal_dose_adjustment(
    cl_normal_L_per_h: float,
    egfr_patient: float,
    egfr_normal: float = 100.0,
) -> dict:
    """Calculate renal dose adjustment using linear GFR scaling.

    Args:
        cl_normal_L_per_h: Normal clearance in L/h (>0).
        egfr_patient: Patient eGFR in mL/min/1.73m² (>=0).
        egfr_normal: Reference normal eGFR (default 100 mL/min/1.73m²).

    Returns:
        Dict with dose_factor, recommendation, monitoring_interval.

    Raises:
        ValueError: If parameters are invalid.
    """
    if cl_normal_L_per_h <= 0:
        raise ValueError("cl_normal_L_per_h must be positive")
    if egfr_patient < 0:
        raise ValueError("egfr_patient must be non-negative")
    if egfr_normal <= 0:
        raise ValueError("egfr_normal must be positive")

    dose_factor = egfr_patient / egfr_normal

    stage_info = classify_ckd_stage(egfr_patient)

    if dose_factor >= 0.9:
        recommendation = "No dose adjustment required"
        monitoring_interval = "Routine monitoring"
    elif dose_factor >= 0.5:
        recommendation = f"Reduce dose to {dose_factor * 100:.0f}% of normal dose"
        monitoring_interval = "Monitor renal function every 3-6 months"
    elif dose_factor >= 0.25:
        recommendation = f"Significantly reduce dose to {dose_factor * 100:.0f}% of normal dose"
        monitoring_interval = "Monitor renal function monthly"
    else:
        recommendation = (
            f"Major dose reduction to {dose_factor * 100:.0f}% of normal; "
            "consider alternative therapy"
        )
        monitoring_interval = "Monitor renal function weekly"

    return {
        "dose_factor": dose_factor,
        "adjusted_cl_L_per_h": cl_normal_L_per_h * dose_factor,
        "recommendation": recommendation,
        "monitoring_interval": monitoring_interval,
        "ckd_stage": stage_info["stage"],
        "dose_adjustment_needed": stage_info["dose_adjustment_needed"],
    }
