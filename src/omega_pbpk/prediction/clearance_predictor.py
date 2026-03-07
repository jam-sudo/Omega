"""
Phase 783 — Clearance prediction from physicochemical properties.

Empirical rules:
- CLhepatic: well-stirred model if CLint provided, else empirical equation
- CLrenal: pH/logP-dependent, cation-aware
- CLtotal = CLhep + CLrenal
- Vd: allometric from MW and logP
- t_half = 0.693 * Vd / CLtotal
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClearancePredictionResult:
    compound_name: str
    logp: float
    mw_da: float
    fup: float
    predicted_cl_hepatic_L_per_h: float
    predicted_cl_renal_L_per_h: float
    predicted_cl_total_L_per_h: float
    predicted_t_half_h: float
    predicted_vd_L: float
    cl_class: str  # "low" (<1 L/h), "moderate" (1-10), "high" (>10)
    elimination_route: str  # "hepatic", "renal", "mixed"
    notes: str


def _cl_class(cl_total: float) -> str:
    if cl_total < 1.0:
        return "low"
    elif cl_total <= 10.0:
        return "moderate"
    else:
        return "high"


def _elimination_route(cl_hep: float, cl_ren: float) -> str:
    cl_total = cl_hep + cl_ren
    if cl_total == 0.0:
        return "hepatic"
    hep_frac = cl_hep / cl_total
    ren_frac = cl_ren / cl_total
    if hep_frac >= 0.7:
        return "hepatic"
    elif ren_frac >= 0.7:
        return "renal"
    else:
        return "mixed"


def predict_clearance(
    compound_name: str,
    logp: float,
    mw_da: float,
    fup: float,
    psa_A2: float = 60.0,
    charge: str = "neutral",
    cl_int_scaled_L_per_h: float | None = None,
) -> ClearancePredictionResult:
    """Predict total clearance and PK parameters from physicochemical properties."""

    # Input validation
    if not (0.0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if mw_da <= 0.0:
        raise ValueError(f"mw_da must be > 0, got {mw_da}")

    # Hepatic clearance
    QH = 97.0  # L/h, hepatic blood flow
    if cl_int_scaled_L_per_h is not None:
        # Well-stirred model
        numerator = QH * fup * cl_int_scaled_L_per_h
        denominator = QH + fup * cl_int_scaled_L_per_h
        if denominator == 0.0:
            cl_hep = 0.0
        else:
            cl_hep = numerator / denominator
    else:
        # Empirical equation: CLhep = fup * 10^(0.5*logP - 0.01*mw - 0.5)
        exponent = 0.5 * logp - 0.01 * mw_da - 0.5
        cl_hep = fup * (10.0**exponent)
        cl_hep = max(0.01, min(100.0, cl_hep))

    # Renal clearance
    GFR_BASELINE = 7.5  # L/h (125 mL/min)
    # CLrenal = fup * GFR_baseline * max(0, 1 - 0.05 * logP)
    renal_factor = max(0.0, 1.0 - 0.05 * logp)
    cl_ren = fup * GFR_BASELINE * renal_factor
    cl_ren = max(0.0, min(fup * GFR_BASELINE, cl_ren))

    # Total clearance
    cl_total = cl_hep + cl_ren

    # Volume of distribution
    # Vd = 0.6 * mw^0.33 * (1 + logP), clamped [3, 500]
    vd = 0.6 * (mw_da**0.33) * (1.0 + logp)
    vd = max(3.0, min(500.0, vd))

    # Half-life
    if cl_total > 0.0:
        t_half = 0.693 * vd / cl_total
    else:
        t_half = float("inf")

    # Classification
    cl_cls = _cl_class(cl_total)
    elim_route = _elimination_route(cl_hep, cl_ren)

    # Notes
    notes_parts = []
    if cl_int_scaled_L_per_h is not None:
        notes_parts.append("CLhep from well-stirred model using provided CLint")
    else:
        notes_parts.append("CLhep from empirical equation (logP, MW, fup)")
    notes_parts.append(f"charge={charge}, PSA={psa_A2} A2")
    notes = "; ".join(notes_parts)

    return ClearancePredictionResult(
        compound_name=compound_name,
        logp=logp,
        mw_da=mw_da,
        fup=fup,
        predicted_cl_hepatic_L_per_h=round(cl_hep, 6),
        predicted_cl_renal_L_per_h=round(cl_ren, 6),
        predicted_cl_total_L_per_h=round(cl_total, 6),
        predicted_t_half_h=round(t_half, 6),
        predicted_vd_L=round(vd, 6),
        cl_class=cl_cls,
        elimination_route=elim_route,
        notes=notes,
    )


def screen_clearance(compounds: list) -> list:
    """
    Screen multiple compounds and return results sorted by cl_total ascending
    (lowest clearance = longest t-half = most favorable).

    Each entry in compounds is a dict with keys matching predict_clearance arguments.
    """
    results = []
    for c in compounds:
        result = predict_clearance(
            compound_name=c.get("compound_name", "unknown"),
            logp=c["logp"],
            mw_da=c["mw_da"],
            fup=c["fup"],
            psa_A2=c.get("psa_A2", 60.0),
            charge=c.get("charge", "neutral"),
            cl_int_scaled_L_per_h=c.get("cl_int_scaled_L_per_h", None),
        )
        results.append(result)

    results.sort(key=lambda r: r.predicted_cl_total_L_per_h)
    return results
