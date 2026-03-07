"""Opioid pharmacodynamic model linking plasma concentration to pain relief and side effects.

Implements Hill/Emax model for analgesia, respiratory depression, and sedation
with 1-compartment PK for morphine-like opioids.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OpioidPDResult:
    """Result of opioid PD simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    effect_analgesia: list
    effect_respiratory: list
    effect_sedation: list
    nrs_pain_score: list
    peak_analgesia: float
    time_at_peak_analgesia_h: float
    duration_above_50pct_analgesia_h: float
    respiratory_depression_risk: str
    therapeutic_ratio: float
    notes: str


def _hill(c: float, ec50: float, n: float) -> float:
    """Hill/Emax equation: effect = C^n / (EC50^n + C^n)."""
    if c <= 0.0:
        return 0.0
    cn = c**n
    ec50n = ec50**n
    return cn / (ec50n + cn)


def simulate_opioid_pd(
    drug_name: str = "Morphine",
    dose_mg: float = 10.0,
    route: str = "oral",
    weight_kg: float = 70.0,
    ec50_analgesia_mg_L: float = 0.02,
    ec50_respiratory_mg_L: float = 0.05,
    hill_n: float = 2.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> OpioidPDResult:
    """Simulate opioid PK/PD.

    Args:
        drug_name: Name of the opioid drug.
        dose_mg: Dose in milligrams.
        route: Dosing route — "oral" or "iv".
        weight_kg: Patient body weight in kg.
        ec50_analgesia_mg_L: EC50 for analgesia (mg/L).
        ec50_respiratory_mg_L: EC50 for respiratory depression (mg/L).
        hill_n: Hill exponent.
        t_end_h: Simulation end time (hours).
        dt_h: Forward Euler time step (hours).

    Returns:
        OpioidPDResult with PK/PD time-course and summary metrics.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if route not in {"oral", "iv"}:
        raise ValueError(f"route must be 'oral' or 'iv', got {route!r}")
    if ec50_analgesia_mg_L <= 0:
        raise ValueError(f"ec50_analgesia_mg_L must be > 0, got {ec50_analgesia_mg_L}")

    # --- PK parameters (morphine-like) ---
    vd = 3.3 * weight_kg  # L
    cl = 0.015 * weight_kg  # L/h
    f = 0.35 if route == "oral" else 1.0
    ka = 1.2  # /h (oral absorption rate)
    ke = cl / vd  # /h

    ec50_sed = ec50_respiratory_mg_L * 1.5

    # --- Forward Euler ODE ---
    n_steps = max(1, int(round(t_end_h / dt_h)))

    times_h: list[float] = []
    c_plasma: list[float] = []
    effect_analgesia: list[float] = []
    effect_respiratory: list[float] = []
    effect_sedation: list[float] = []
    nrs_pain_score: list[float] = []

    if route == "oral":
        a_gut = f * dose_mg  # mg in gut compartment
        c = 0.0  # mg/L plasma concentration
        for step in range(n_steps + 1):
            t = step * dt_h
            times_h.append(t)
            c_plasma.append(c)

            # PD effects
            e_analg = _hill(c, ec50_analgesia_mg_L, hill_n)
            e_resp = _hill(c, ec50_respiratory_mg_L, hill_n)
            e_sed = _hill(c, ec50_sed, 1.5)
            effect_analgesia.append(e_analg)
            effect_respiratory.append(e_resp)
            effect_sedation.append(e_sed)
            nrs_pain_score.append(max(0.0, min(10.0, 10.0 - 9.0 * e_analg)))

            if step < n_steps:
                # Forward Euler update
                d_a_gut = -ka * a_gut
                d_c = (ka * a_gut / vd) - ke * c
                a_gut += d_a_gut * dt_h
                c += d_c * dt_h
                c = max(0.0, c)
                a_gut = max(0.0, a_gut)
    else:
        # IV bolus: C(0) = dose / Vd
        c = dose_mg / vd
        for step in range(n_steps + 1):
            t = step * dt_h
            times_h.append(t)
            c_plasma.append(c)

            e_analg = _hill(c, ec50_analgesia_mg_L, hill_n)
            e_resp = _hill(c, ec50_respiratory_mg_L, hill_n)
            e_sed = _hill(c, ec50_sed, 1.5)
            effect_analgesia.append(e_analg)
            effect_respiratory.append(e_resp)
            effect_sedation.append(e_sed)
            nrs_pain_score.append(max(0.0, min(10.0, 10.0 - 9.0 * e_analg)))

            if step < n_steps:
                dc = -ke * c
                c += dc * dt_h
                c = max(0.0, c)

    # --- Summary metrics ---
    peak_analgesia = max(effect_analgesia)

    # Time at peak analgesia
    peak_idx = effect_analgesia.index(peak_analgesia)
    time_at_peak_h = times_h[peak_idx]

    # Duration above 50% analgesia
    duration_50 = sum(dt_h for e in effect_analgesia if e > 0.5)

    # Respiratory depression risk
    peak_resp = max(effect_respiratory)
    if peak_resp > 0.8:
        resp_risk = "high"
    elif peak_resp > 0.5:
        resp_risk = "moderate"
    else:
        resp_risk = "low"

    therapeutic_ratio = peak_analgesia / max(peak_resp, 0.001)

    notes = (
        f"1-compartment PK: Vd={vd:.1f} L, CL={cl:.3f} L/h, ke={ke:.4f}/h. "
        f"Route: {route}, F={f}. "
        f"EC50 analgesia={ec50_analgesia_mg_L} mg/L, respiratory={ec50_respiratory_mg_L} mg/L."
    )

    return OpioidPDResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        effect_analgesia=effect_analgesia,
        effect_respiratory=effect_respiratory,
        effect_sedation=effect_sedation,
        nrs_pain_score=nrs_pain_score,
        peak_analgesia=peak_analgesia,
        time_at_peak_analgesia_h=time_at_peak_h,
        duration_above_50pct_analgesia_h=duration_50,
        respiratory_depression_risk=resp_risk,
        therapeutic_ratio=therapeutic_ratio,
        notes=notes,
    )
