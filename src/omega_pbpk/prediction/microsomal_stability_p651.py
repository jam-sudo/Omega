"""Hepatic microsomal stability prediction.

Predicts intrinsic clearance from in vitro microsomal incubation data and
scales to in vivo hepatic clearance using the well-stirred liver model.
Relevant for drug discovery metabolic screening.

Key concepts
------------
- In vitro CLint from substrate depletion half-life
- Scaling via microsomal protein content per gram liver
- Microsomal binding correction (fu_mic)
- Well-stirred hepatic clearance model

References
----------
- Obach RS, Drug Metab Dispos. 1999;27(11):1350-1359 (CLint scaling)
- Austin RP et al., Drug Metab Dispos. 2002;30(12):1497-1503 (fu_mic prediction)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MicrosomalStabilityResult:
    """Result container for microsomal stability prediction."""

    drug_name: str
    logP: float
    mw_Da: float
    t_half_in_vitro_min: float
    clint_mL_per_min_per_mg: float
    clint_scaled_L_per_h: float
    fu_mic: float
    hepatic_cl_L_per_h: float
    hepatic_extraction_ratio: float
    stability_class: str
    predicted_f_hepatic: float
    notes: str


def predict_microsomal_stability(
    drug_name: str,
    logP: float,
    mw_Da: float,
    t_half_in_vitro_min: float,
    microsomal_protein_mg_mL: float = 0.5,
    incubation_vol_mL: float = 0.1,
    liver_weight_g: float = 1500.0,
    microsomal_protein_per_g_liver: float = 45.0,
    hepatic_blood_flow_L_per_h: float = 90.0,
    fup: float = 1.0,
) -> MicrosomalStabilityResult:
    """Predict hepatic microsomal stability from in vitro half-life data.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    logP : float
        Logarithm of the octanol-water partition coefficient.
    mw_Da : float
        Molecular weight in Daltons.
    t_half_in_vitro_min : float
        In vitro microsomal half-life in minutes (must be > 0).
    microsomal_protein_mg_mL : float
        Microsomal protein concentration in mg/mL (default 0.5).
    incubation_vol_mL : float
        Incubation volume in mL (default 0.1).
    liver_weight_g : float
        Liver weight in grams (default 1500).
    microsomal_protein_per_g_liver : float
        Microsomal protein yield per gram liver (default 45).
    hepatic_blood_flow_L_per_h : float
        Hepatic blood flow in L/h (default 90).
    fup : float
        Fraction unbound in plasma (default 1.0).

    Returns
    -------
    MicrosomalStabilityResult
        Frozen dataclass with all computed parameters.
    """
    if t_half_in_vitro_min <= 0:
        raise ValueError("t_half_in_vitro_min must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if microsomal_protein_mg_mL <= 0:
        raise ValueError("microsomal_protein_mg_mL must be > 0")

    # In vitro CLint from substrate depletion half-life
    clint_intrinsic = 0.693 / t_half_in_vitro_min * incubation_vol_mL / microsomal_protein_mg_mL

    # Scale to in vivo (mL/min/mg → L/h whole liver)
    clint_scaled = clint_intrinsic * microsomal_protein_per_g_liver * liver_weight_g
    clint_scaled_L_per_h = clint_scaled * 60.0 / 1000.0

    # Microsomal binding correction
    if logP > 2.0:
        fu_mic = 1.0 / (1.0 + 0.072 * 10 ** (logP - 2.0))
    else:
        fu_mic = 1.0
    fu_mic = max(0.01, min(1.0, fu_mic))

    # Correct CLint for binding
    clint_corrected = clint_scaled_L_per_h * fup / fu_mic

    # Well-stirred hepatic clearance model
    hepatic_cl = (
        hepatic_blood_flow_L_per_h
        * clint_corrected
        / (hepatic_blood_flow_L_per_h + clint_corrected)
    )

    # Extraction ratio and bioavailability
    hepatic_extraction_ratio = max(0.0, min(1.0, hepatic_cl / hepatic_blood_flow_L_per_h))
    predicted_f_hepatic = 1.0 - hepatic_extraction_ratio

    # Stability classification
    if t_half_in_vitro_min >= 30.0:
        stability_class = "high"
    elif t_half_in_vitro_min >= 15.0:
        stability_class = "medium"
    else:
        stability_class = "low"

    notes = (
        f"Well-stirred model; fu_mic={'estimated' if logP > 2.0 else 'assumed 1.0'}; "
        f"stability={stability_class}"
    )

    return MicrosomalStabilityResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        t_half_in_vitro_min=t_half_in_vitro_min,
        clint_mL_per_min_per_mg=clint_intrinsic,
        clint_scaled_L_per_h=clint_scaled_L_per_h,
        fu_mic=fu_mic,
        hepatic_cl_L_per_h=hepatic_cl,
        hepatic_extraction_ratio=hepatic_extraction_ratio,
        stability_class=stability_class,
        predicted_f_hepatic=predicted_f_hepatic,
        notes=notes,
    )


def screen_microsomal_stability(
    compounds: list[dict],
) -> list[MicrosomalStabilityResult]:
    """Screen multiple compounds for microsomal stability.

    Parameters
    ----------
    compounds : list[dict]
        List of dicts, each containing keyword arguments for
        ``predict_microsomal_stability``.

    Returns
    -------
    list[MicrosomalStabilityResult]
        Results sorted by t_half_in_vitro_min descending (most stable first).
    """
    results = [predict_microsomal_stability(**c) for c in compounds]
    results.sort(key=lambda r: r.t_half_in_vitro_min, reverse=True)
    return results
