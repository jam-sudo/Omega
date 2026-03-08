"""Phase 394 - Microdose PK Predictor.

Predict PK parameters at therapeutic doses from microdose (<=100 ug) human
studies, accounting for dose-proportionality and first-pass saturation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MicrodosePKResult:
    """Result from microdose PK prediction."""

    drug_name: str
    microdose_ug: float
    therapeutic_dose_mg: float
    dose_ratio: float
    cl_microdose_L_per_h: float
    cl_predicted_therapeutic_L_per_h: float
    vd_microdose_L: float
    vd_predicted_therapeutic_L: float
    f_microdose: float
    f_predicted_therapeutic: float
    auc_microdose_ng_h_per_mL: float
    auc_predicted_therapeutic_ng_h_per_mL: float
    dose_proportionality_flag: bool
    saturation_risk: str
    prediction_confidence: str
    notes: str


def predict_microdose_pk(
    drug_name: str,
    microdose_ug: float,
    therapeutic_dose_mg: float,
    cmax_microdose_ng_mL: float,
    tmax_microdose_h: float,
    auc_microdose_ng_h_mL: float,
    t_half_microdose_h: float,
    estimated_fa: float = 0.9,
    extraction_ratio: float = 0.3,
) -> MicrodosePKResult:
    """Predict PK at therapeutic doses from microdose human study data.

    Args:
        drug_name: Name of the drug.
        microdose_ug: Microdose administered in micrograms (must be <= 100 ug).
        therapeutic_dose_mg: Target therapeutic dose in mg.
        cmax_microdose_ng_mL: Observed Cmax at microdose in ng/mL.
        tmax_microdose_h: Observed Tmax at microdose in hours.
        auc_microdose_ng_h_mL: Observed AUC at microdose in ng*h/mL.
        t_half_microdose_h: Observed t1/2 at microdose in hours.
        estimated_fa: Estimated fraction absorbed (0-1]. Default 0.9.
        extraction_ratio: Hepatic extraction ratio [0-1]. Default 0.3.

    Returns:
        MicrodosePKResult with predicted PK at therapeutic dose.

    Raises:
        ValueError: If invalid parameters are provided.
    """
    if microdose_ug <= 0 or microdose_ug > 100:
        raise ValueError(f"microdose_ug must be > 0 and <= 100 ug, got {microdose_ug}")
    if therapeutic_dose_mg <= 0:
        raise ValueError(f"therapeutic_dose_mg must be > 0, got {therapeutic_dose_mg}")
    if cmax_microdose_ng_mL <= 0:
        raise ValueError(f"cmax_microdose_ng_mL must be > 0, got {cmax_microdose_ng_mL}")
    if auc_microdose_ng_h_mL <= 0:
        raise ValueError(f"auc_microdose_ng_h_mL must be > 0, got {auc_microdose_ng_h_mL}")
    if t_half_microdose_h <= 0:
        raise ValueError(f"t_half_microdose_h must be > 0, got {t_half_microdose_h}")
    if not (0.0 < estimated_fa <= 1.0):
        raise ValueError(f"estimated_fa must be in (0, 1], got {estimated_fa}")
    if not (0.0 <= extraction_ratio <= 1.0):
        raise ValueError(f"extraction_ratio must be in [0, 1], got {extraction_ratio}")

    # Dose ratio: therapeutic (mg) -> (ug) / microdose (ug)
    dose_ratio = (therapeutic_dose_mg * 1000.0) / microdose_ug

    # Bioavailability at microdose (linear, unsaturated)
    f_micro = estimated_fa * (1.0 - extraction_ratio)

    # Apparent CL/F at microdose
    # dose in mg, AUC in ng*h/mL
    # Convert: dose_mg / (auc_ng_h_mL * 1e-3 mg*h/L) = L/h
    dose_mg_micro = microdose_ug / 1000.0
    auc_mg_h_L = auc_microdose_ng_h_mL * 1e-3
    cl_apparent_micro = dose_mg_micro / auc_mg_h_L  # L/h (apparent, CL/F)

    # True CL at microdose
    cl_microdose = cl_apparent_micro * f_micro

    # Vd at microdose from CL and t1/2
    vd_microdose = cl_microdose * t_half_microdose_h / 0.693

    # Saturation risk assessment
    first_pass_saturation = extraction_ratio > 0.7 and dose_ratio > 100
    absorption_saturation = estimated_fa < 0.5 and dose_ratio > 10

    # Dose proportionality
    dose_proportionality_flag = not (first_pass_saturation or absorption_saturation)

    # Saturation risk string
    if first_pass_saturation and absorption_saturation:
        saturation_risk = "both"
    elif first_pass_saturation:
        saturation_risk = "first_pass_saturation"
    elif absorption_saturation:
        saturation_risk = "absorption_saturation"
    else:
        saturation_risk = "none"

    # F at therapeutic dose
    if dose_proportionality_flag:
        f_therapeutic = f_micro
    else:
        if first_pass_saturation:
            f_therapeutic = min(1.0, f_micro * (1.0 + 0.3 * math.log10(dose_ratio)))
        else:
            f_therapeutic = f_micro

    # CL at therapeutic dose
    cl_therapeutic = cl_microdose

    # Vd at therapeutic dose (typically unchanged with dose)
    vd_therapeutic = vd_microdose

    # AUC at therapeutic dose
    # Scale by dose ratio and adjust for any F change
    if f_micro > 0:
        auc_therapeutic = auc_microdose_ng_h_mL * dose_ratio * (f_therapeutic / f_micro)
    else:
        auc_therapeutic = auc_microdose_ng_h_mL * dose_ratio

    # Prediction confidence
    if dose_ratio < 100 and saturation_risk == "none":
        prediction_confidence = "high"
    elif dose_ratio < 1000 and saturation_risk != "both":
        prediction_confidence = "medium"
    else:
        prediction_confidence = "low"

    notes_parts = [
        f"Dose ratio: {dose_ratio:.1f}x.",
        f"Saturation risk: {saturation_risk}.",
        f"Dose proportional: {dose_proportionality_flag}.",
        f"Confidence: {prediction_confidence}.",
        f"F_micro={f_micro:.3f}, F_therapeutic={f_therapeutic:.3f}.",
    ]
    notes = " ".join(notes_parts)

    return MicrodosePKResult(
        drug_name=drug_name,
        microdose_ug=microdose_ug,
        therapeutic_dose_mg=therapeutic_dose_mg,
        dose_ratio=dose_ratio,
        cl_microdose_L_per_h=cl_microdose,
        cl_predicted_therapeutic_L_per_h=cl_therapeutic,
        vd_microdose_L=vd_microdose,
        vd_predicted_therapeutic_L=vd_therapeutic,
        f_microdose=f_micro,
        f_predicted_therapeutic=f_therapeutic,
        auc_microdose_ng_h_per_mL=auc_microdose_ng_h_mL,
        auc_predicted_therapeutic_ng_h_per_mL=auc_therapeutic,
        dose_proportionality_flag=dose_proportionality_flag,
        saturation_risk=saturation_risk,
        prediction_confidence=prediction_confidence,
        notes=notes,
    )


def compare_extraction_ratios(
    drug_name: str,
    microdose_ug: float,
    therapeutic_dose_mg: float,
    cmax_ng_mL: float,
    tmax_h: float,
    auc_ng_h_mL: float,
    t_half_h: float,
    ers: list = None,
) -> list:
    """Compare predicted therapeutic PK across different hepatic extraction ratios.

    Args:
        drug_name: Name of the drug.
        microdose_ug: Microdose in micrograms.
        therapeutic_dose_mg: Therapeutic dose in mg.
        cmax_ng_mL: Observed Cmax at microdose in ng/mL.
        tmax_h: Observed Tmax at microdose in hours.
        auc_ng_h_mL: Observed AUC at microdose in ng*h/mL.
        t_half_h: Observed t1/2 at microdose in hours.
        ers: List of extraction ratios to compare.
            Defaults to [0.1, 0.3, 0.5, 0.7, 0.9].

    Returns:
        List of MicrodosePKResult sorted by auc_predicted_therapeutic_ng_h_per_mL
        descending.
    """
    if ers is None:
        ers = [0.1, 0.3, 0.5, 0.7, 0.9]

    results = []
    for er in ers:
        result = predict_microdose_pk(
            drug_name=drug_name,
            microdose_ug=microdose_ug,
            therapeutic_dose_mg=therapeutic_dose_mg,
            cmax_microdose_ng_mL=cmax_ng_mL,
            tmax_microdose_h=tmax_h,
            auc_microdose_ng_h_mL=auc_ng_h_mL,
            t_half_microdose_h=t_half_h,
            extraction_ratio=er,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_predicted_therapeutic_ng_h_per_mL, reverse=True)
    return results
