"""Phase 1133 — Therapeutic Drug Level Calculator.

Calculate target therapeutic drug levels and individualized monitoring
parameters using 1-compartment analytical PK at steady state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Drug target database
# ---------------------------------------------------------------------------

_DRUG_TARGETS: dict[str, dict] = {
    "vancomycin": {
        "trough_target_mg_L": (10, 20),
        "peak_target_mg_L": (25, 40),
        "toxic_level_mg_L": 40,
    },
    "gentamicin": {
        "trough_target_mg_L": (0, 2),
        "peak_target_mg_L": (5, 10),
        "toxic_level_mg_L": 2,
    },
    "cyclosporin": {
        "trough_target_mg_L": (100, 400),
        "peak_target_mg_L": None,
        "toxic_level_mg_L": 400,
    },
    "digoxin": {
        "trough_target_mg_L": (0.8, 2.0),
        "peak_target_mg_L": None,
        "toxic_level_mg_L": 2.0,
    },
    "phenytoin": {
        "trough_target_mg_L": (10, 20),
        "peak_target_mg_L": None,
        "toxic_level_mg_L": 25,
    },
    "generic": {
        "trough_target_mg_L": (1, 10),
        "peak_target_mg_L": None,
        "toxic_level_mg_L": 20,
    },
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TherapeuticLevelResult:
    drug_name: str
    indication: str
    route: str
    dosing_interval_h: float
    cl_L_per_h: float
    vd_L: float
    ke_per_h: float
    t_half_h: float
    css_trough_mg_L: float
    css_peak_mg_L: float
    trough_in_range: bool
    peak_in_range: bool
    trough_target_low: float
    trough_target_high: float
    toxic_level_mg_L: float
    safety_margin: float
    dose_recommendation: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    cl_L_per_h: float,
    vd_L: float,
    dosing_interval_h: float,
    route: str,
    f_bioavail: float,
    ka_per_h: float,
) -> None:
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")
    if route not in {"oral", "iv"}:
        raise ValueError("route must be 'oral' or 'iv'")
    if not (0 < f_bioavail <= 1):
        raise ValueError("f_bioavail must be in (0, 1]")
    if route == "oral" and ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0 for oral route")


def _css_oral(
    dose_mg: float,
    f: float,
    ka: float,
    ke: float,
    vd: float,
    tau: float,
) -> tuple[float, float, float]:
    """Return (css_trough, css_peak, tmax) for 1-cpt oral at steady state."""
    if abs(ka - ke) < 1e-9:
        # Avoid division by zero; perturb ka slightly
        ka = ka * 1.0001
    denom = ka - ke
    # Trough (at t = tau)
    css_trough = (
        (f * dose_mg * ka)
        / (vd * denom)
        * (math.exp(-ke * tau) - math.exp(-ka * tau))
        / (1 - math.exp(-ke * tau))
    )
    # tmax at steady state (same formula as single dose for location)
    tmax = math.log(ka / ke) / denom
    # Css_peak at tmax
    css_peak = (
        (f * dose_mg * ka)
        / (vd * denom)
        * (math.exp(-ke * tmax) - math.exp(-ka * tmax))
        / (1 - math.exp(-ke * tau))
    )
    return css_trough, css_peak, tmax


def _css_iv(
    dose_mg: float,
    ke: float,
    vd: float,
    tau: float,
) -> tuple[float, float]:
    """Return (css_trough, css_peak) for 1-cpt IV bolus at steady state."""
    css_peak = (dose_mg / vd) / (1 - math.exp(-ke * tau))
    css_trough = css_peak * math.exp(-ke * tau)
    return css_trough, css_peak


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_target_level(
    drug_name: str,
    indication: str,
    route: str,
    dosing_interval_h: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_bioavail: float,
    dose_mg: float = 500.0,
) -> TherapeuticLevelResult:
    """Calculate steady-state therapeutic drug levels.

    Parameters
    ----------
    drug_name:
        Name of the drug (case-insensitive). Unknown names fall back to "generic".
    indication:
        Clinical indication string (informational only).
    route:
        "oral" or "iv".
    dosing_interval_h:
        Dosing interval in hours.
    cl_L_per_h:
        Total body clearance in L/h.
    vd_L:
        Volume of distribution in L.
    ka_per_h:
        Absorption rate constant (h⁻¹). Required for oral; ignored for IV.
    f_bioavail:
        Oral bioavailability fraction (0 < F ≤ 1). Ignored for IV (set to 1).
    dose_mg:
        Dose in mg (default 500 mg).

    Returns
    -------
    TherapeuticLevelResult
    """
    _validate_inputs(cl_L_per_h, vd_L, dosing_interval_h, route, f_bioavail, ka_per_h)

    key = drug_name.lower()
    targets = _DRUG_TARGETS.get(key, _DRUG_TARGETS["generic"])

    ke = cl_L_per_h / vd_L
    t_half = math.log(2) / ke
    tau = dosing_interval_h
    tmax_h: float

    if route == "oral":
        css_trough, css_peak, tmax_h = _css_oral(dose_mg, f_bioavail, ka_per_h, ke, vd_L, tau)
    else:
        css_trough, css_peak = _css_iv(dose_mg, ke, vd_L, tau)

    trough_low, trough_high = targets["trough_target_mg_L"]
    toxic = targets["toxic_level_mg_L"]
    peak_target = targets["peak_target_mg_L"]

    trough_in_range = trough_low <= css_trough <= trough_high
    if peak_target is None:
        peak_in_range = True
    else:
        pk_low, pk_high = peak_target
        peak_in_range = pk_low <= css_peak <= pk_high

    safety_margin = toxic / css_peak if css_peak > 0 else float("inf")

    # Dose recommendation
    if not trough_in_range:
        if css_trough < trough_low:
            dose_rec = "increase"
        else:
            dose_rec = "decrease"
    elif peak_target is not None and not peak_in_range:
        if css_peak < peak_target[0]:
            dose_rec = "increase"
        else:
            dose_rec = "decrease"
    else:
        dose_rec = "maintain"

    note_parts = []
    if css_peak >= toxic:
        note_parts.append("WARNING: Css_peak at or above toxic level")
    if trough_in_range and peak_in_range:
        note_parts.append("Levels within therapeutic range")
    notes = "; ".join(note_parts) if note_parts else "Monitoring recommended"

    return TherapeuticLevelResult(
        drug_name=drug_name,
        indication=indication,
        route=route,
        dosing_interval_h=dosing_interval_h,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ke_per_h=ke,
        t_half_h=t_half,
        css_trough_mg_L=css_trough,
        css_peak_mg_L=css_peak,
        trough_in_range=trough_in_range,
        peak_in_range=peak_in_range,
        trough_target_low=trough_low,
        trough_target_high=trough_high,
        toxic_level_mg_L=toxic,
        safety_margin=safety_margin,
        dose_recommendation=dose_rec,
        notes=notes,
    )


def recommend_sampling_time(result: TherapeuticLevelResult) -> dict:
    """Return optimal sampling times for trough and peak monitoring.

    Parameters
    ----------
    result:
        A TherapeuticLevelResult from calculate_target_level.

    Returns
    -------
    dict with keys: trough_time_h, peak_time_h, recommendation
    """
    trough_time = result.dosing_interval_h - 0.5  # 30 min before next dose

    if result.route == "oral":
        # We cannot access ka directly from the frozen result, so we use
        # a standard 1-2 h post-dose for oral as peak sampling time.
        peak_time = 1.0  # default for oral when exact tmax not stored
    else:
        peak_time = 0.5  # 30 min post-dose for IV bolus

    recommendation = (
        f"Draw trough sample {trough_time:.1f}h after last dose (30 min before next dose); "
        f"draw peak sample {peak_time:.1f}h post-dose."
    )

    return {
        "trough_time_h": trough_time,
        "peak_time_h": peak_time,
        "recommendation": recommendation,
    }
