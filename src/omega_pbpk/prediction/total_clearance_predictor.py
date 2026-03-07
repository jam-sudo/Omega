"""Phase 654 — Total Body Clearance Predictor.

Predict total body clearance from physicochemical and in vitro data,
including hepatic (well-stirred), renal (GFR + secretion), and biliary routes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TotalClearancePrediction:
    compound_name: str
    cl_hepatic_L_per_h: float
    cl_renal_L_per_h: float
    cl_biliary_L_per_h: float
    cl_total_L_per_h: float
    cl_total_mL_per_min_per_kg: float
    hepatic_er: float
    renal_er: float
    fraction_hepatic: float
    fraction_renal: float
    fraction_biliary: float
    high_clearance_hepatic: bool
    t_half_predicted_h: float
    route_of_elimination: str
    notes: str


def _validate_predict_inputs(
    fup: float,
    clint_uL_per_min_per_mg: float,
    fe_urine: float,
    gfr_mL_per_min: float,
    liver_blood_flow_L_per_h: float,
    mw: float,
) -> None:
    if not (0 < fup <= 1):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if fup > 1.0:
        raise ValueError(f"fup must be <= 1, got {fup}")
    if clint_uL_per_min_per_mg < 0:
        raise ValueError(f"clint_uL_per_min_per_mg must be >= 0, got {clint_uL_per_min_per_mg}")
    if not (0 <= fe_urine <= 1):
        raise ValueError(f"fe_urine must be in [0, 1], got {fe_urine}")
    if gfr_mL_per_min <= 0:
        raise ValueError(f"gfr_mL_per_min must be > 0, got {gfr_mL_per_min}")
    if liver_blood_flow_L_per_h <= 0:
        raise ValueError(f"liver_blood_flow_L_per_h must be > 0, got {liver_blood_flow_L_per_h}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")


def predict_total_clearance(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    fup: float,
    clint_uL_per_min_per_mg: float,
    fe_urine: float = 0.0,
    active_secretion_cl_mL_per_min: float = 0.0,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    gfr_mL_per_min: float = 120.0,
    liver_blood_flow_L_per_h: float = 90.0,
    body_weight_kg: float = 70.0,
) -> TotalClearancePrediction:
    """Predict total body clearance from physicochemical and in vitro data."""
    _validate_predict_inputs(
        fup, clint_uL_per_min_per_mg, fe_urine, gfr_mL_per_min, liver_blood_flow_L_per_h, mw
    )

    QH = liver_blood_flow_L_per_h

    # --- Hepatic CL (well-stirred model) ---
    # Scale in vitro CLint to whole body:
    # clint_uL/min/mg_protein * (1 mg_protein/uL * 0.001) = mL/min/mg * 0.001
    # => L/h: * 0.001 * 60 * 40  (40 g microsomal protein per liver, rough IVIVE factor)
    CLint_body_L_per_h = clint_uL_per_min_per_mg * 0.001 * 60 * 40
    cl_hepatic = QH * fup * CLint_body_L_per_h / (QH + fup * CLint_body_L_per_h)

    # --- Renal CL ---
    GFR_L_per_h = gfr_mL_per_min * 0.06
    CL_filt = GFR_L_per_h * fup
    CL_sec = active_secretion_cl_mL_per_min * 0.06

    # Reabsorption fraction: high for lipophilic acids, lower for others
    if pka_acid is not None and pka_acid > 7:
        reabsorption_fraction = max(0.0, 0.9 - logP * 0.1)
    else:
        reabsorption_fraction = 0.3
    reabsorption_fraction = min(reabsorption_fraction, 0.95)

    cl_renal = (CL_filt + CL_sec) * (1.0 - reabsorption_fraction)

    # --- Biliary CL ---
    if mw > 400 and logP > 2:
        cl_biliary = 0.5 + (mw - 400) / 1000.0 + logP * 0.1
    else:
        cl_biliary = 0.0

    # --- Total CL ---
    cl_total = cl_hepatic + cl_renal + cl_biliary
    # Protect against zero total (edge case)
    if cl_total <= 0:
        cl_total = 1e-9

    # --- Normalized CL (mL/min/kg) ---
    cl_total_mL_per_min_per_kg = (cl_total * 1000.0 / 60.0) / body_weight_kg

    # --- Extraction ratios ---
    hepatic_er = min(1.0, cl_hepatic / QH)
    renal_er = min(1.0, cl_renal / GFR_L_per_h)

    # --- Fractions ---
    fraction_hepatic = cl_hepatic / cl_total
    fraction_renal = cl_renal / cl_total
    fraction_biliary = cl_biliary / cl_total

    # --- High clearance flag ---
    high_clearance_hepatic = hepatic_er > 0.7

    # --- Predicted t½ using rule-of-thumb Vd ---
    # Rule of thumb: Vd ~ 0.7 L/kg for moderate distribution
    vd_predicted_L = 0.7 * body_weight_kg
    t_half_predicted_h = math.log(2) * vd_predicted_L / cl_total

    # --- Route of elimination ---
    threshold = 0.4
    dominant_routes = []
    if fraction_hepatic >= threshold:
        dominant_routes.append("hepatic")
    if fraction_renal >= threshold:
        dominant_routes.append("renal")
    if fraction_biliary >= threshold:
        dominant_routes.append("biliary")

    if len(dominant_routes) == 0:
        route_of_elimination = "mixed"
    elif len(dominant_routes) == 1:
        route_of_elimination = dominant_routes[0]
    else:
        route_of_elimination = "mixed"

    notes = (
        f"Well-stirred hepatic model; CLint_body={CLint_body_L_per_h:.2f} L/h; "
        f"GFR={gfr_mL_per_min} mL/min; "
        f"reabsorption_frac={reabsorption_fraction:.2f}; "
        f"biliary={'yes' if cl_biliary > 0 else 'no'} (MW={mw}, logP={logP})"
    )

    return TotalClearancePrediction(
        compound_name=compound_name,
        cl_hepatic_L_per_h=cl_hepatic,
        cl_renal_L_per_h=cl_renal,
        cl_biliary_L_per_h=cl_biliary,
        cl_total_L_per_h=cl_total,
        cl_total_mL_per_min_per_kg=cl_total_mL_per_min_per_kg,
        hepatic_er=hepatic_er,
        renal_er=renal_er,
        fraction_hepatic=fraction_hepatic,
        fraction_renal=fraction_renal,
        fraction_biliary=fraction_biliary,
        high_clearance_hepatic=high_clearance_hepatic,
        t_half_predicted_h=t_half_predicted_h,
        route_of_elimination=route_of_elimination,
        notes=notes,
    )


def screen_clearance(compounds: list[dict]) -> list[TotalClearancePrediction]:
    """Screen multiple compounds, sorted by cl_total_L_per_h descending."""
    if not compounds:
        return []

    results = []
    for cmpd in compounds:
        pred = predict_total_clearance(
            compound_name=cmpd.get("compound_name", "unknown"),
            logP=cmpd.get("logP", 2.0),
            mw=cmpd.get("mw", 300.0),
            psa=cmpd.get("psa", 60.0),
            fup=cmpd.get("fup", 0.1),
            clint_uL_per_min_per_mg=cmpd.get("clint_uL_per_min_per_mg", 10.0),
            fe_urine=cmpd.get("fe_urine", 0.0),
            active_secretion_cl_mL_per_min=cmpd.get("active_secretion_cl_mL_per_min", 0.0),
            pka_acid=cmpd.get("pka_acid", None),
            pka_base=cmpd.get("pka_base", None),
            gfr_mL_per_min=cmpd.get("gfr_mL_per_min", 120.0),
            liver_blood_flow_L_per_h=cmpd.get("liver_blood_flow_L_per_h", 90.0),
            body_weight_kg=cmpd.get("body_weight_kg", 70.0),
        )
        results.append(pred)

    results.sort(key=lambda r: r.cl_total_L_per_h, reverse=True)
    return results
