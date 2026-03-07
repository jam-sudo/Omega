"""Drug concentration prediction from dose — Phase 758.

Analytical 1-compartment PK solutions for quick concentration estimates
without full PBPK simulation.

Supported routes: iv_bolus, iv_infusion, oral.
Supports multi-dose superposition and steady-state estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ConcentrationPredictionResult",
    "predict_concentration",
    "predict_concentration_profile",
]

_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcentrationPredictionResult:
    """Result of a single-point concentration prediction.

    Attributes
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in mg.
    t_h : float
        Time point (h).
    route : str
        Dosing route: 'iv_bolus', 'iv_infusion', or 'oral'.
    c_predicted_mg_L : float
        Predicted plasma concentration at t_h (mg/L).
    css_avg_mg_L : float
        Steady-state average concentration (mg/L).
    css_max_mg_L : float
        Steady-state maximum concentration at trough approach (oral, mg/L).
    css_min_mg_L : float
        Steady-state minimum concentration (mg/L).
    ke_per_h : float
        Elimination rate constant (1/h).
    t_half_h : float
        Half-life (h).
    notes : str
        Summary notes.
    """

    drug_name: str
    dose_mg: float
    t_h: float
    route: str
    c_predicted_mg_L: float
    css_avg_mg_L: float
    css_max_mg_L: float
    css_min_mg_L: float
    ke_per_h: float
    t_half_h: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _single_iv_bolus(dose_mg: float, vd_L: float, ke: float, t: float) -> float:
    """Concentration from a single IV bolus dose at time t."""
    if t < 0.0:
        return 0.0
    return (dose_mg / vd_L) * math.exp(-ke * t)


def _single_iv_infusion(
    dose_mg: float,
    cl_L_per_h: float,
    ke: float,
    t: float,
    duration_h: float,
) -> float:
    """Concentration from a constant-rate IV infusion at time t.

    Rate R = dose_mg / duration_h (mg/h).
    During infusion (0 <= t <= duration_h):
        C(t) = R/CL * (1 - exp(-ke*t))
    After infusion (t > duration_h):
        C(t) = C(duration_h) * exp(-ke * (t - duration_h))
    """
    if t < 0.0:
        return 0.0
    rate = dose_mg / duration_h
    c_at_end = (rate / cl_L_per_h) * (1.0 - math.exp(-ke * duration_h))
    if t <= duration_h:
        return (rate / cl_L_per_h) * (1.0 - math.exp(-ke * t))
    else:
        return c_at_end * math.exp(-ke * (t - duration_h))


def _single_oral(
    dose_mg: float,
    f_oral: float,
    vd_L: float,
    ka: float,
    ke: float,
    t: float,
) -> float:
    """Concentration from a single oral dose at time t (1-compartment, flip-flop aware).

    C(t) = F*D*ka / (Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))   [ka != ke]
    C(t) = F*D*ke / Vd * t * exp(-ke*t)                         [ka == ke, limit]
    """
    if t <= 0.0:
        return 0.0
    effective_dose = f_oral * dose_mg
    diff = ka - ke
    if abs(diff) < 1e-9:
        # Limiting case ka → ke
        return (effective_dose * ke / vd_L) * t * math.exp(-ke * t)
    return (effective_dose * ka / (vd_L * diff)) * (math.exp(-ke * t) - math.exp(-ka * t))


def _superpose_doses(
    single_fn,
    n_doses: int,
    tau: float,
    t_abs: float,
) -> float:
    """Multi-dose superposition: sum single_fn(t - n*tau) for n in 0..N-1."""
    total = 0.0
    for n in range(n_doses):
        t_since_dose = t_abs - n * tau
        if t_since_dose >= 0.0:
            total += single_fn(t_since_dose)
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_concentration(
    drug_name: str,
    dose_mg: float,
    t_h: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "oral",
    f_oral: float = 1.0,
    ka_per_h: float = 1.0,
    infusion_duration_h: float = 1.0,
    n_doses: int = 1,
    dosing_interval_h: float = 24.0,
) -> ConcentrationPredictionResult:
    """Predict plasma concentration at a specific time point.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose per administration (mg).
    t_h : float
        Time point of interest (h), measured from first dose.
    cl_L_per_h : float
        Total clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    route : str
        Dosing route: 'iv_bolus', 'iv_infusion', or 'oral'.
    f_oral : float
        Oral bioavailability fraction (used for oral route only).
    ka_per_h : float
        Absorption rate constant (1/h, oral route only).
    infusion_duration_h : float
        Infusion duration (h, iv_infusion route only).
    n_doses : int
        Number of doses administered (>= 1).
    dosing_interval_h : float
        Dosing interval tau (h).

    Returns
    -------
    ConcentrationPredictionResult

    Raises
    ------
    ValueError
        If any input parameter is out of valid range.
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if t_h < 0.0:
        raise ValueError(f"t_h must be >= 0, got {t_h}")
    if cl_L_per_h <= 0.0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0.0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if n_doses < 1:
        raise ValueError(f"n_doses must be >= 1, got {n_doses}")
    if route not in ("iv_bolus", "iv_infusion", "oral"):
        raise ValueError(f"route must be 'iv_bolus', 'iv_infusion', or 'oral', got '{route}'")

    ke = cl_L_per_h / vd_L
    t_half = _LN2 / ke
    tau = dosing_interval_h

    # Single-dose function depending on route
    if route == "iv_bolus":

        def single_fn(t: float) -> float:
            return _single_iv_bolus(dose_mg, vd_L, ke, t)
    elif route == "iv_infusion":

        def single_fn(t: float) -> float:
            return _single_iv_infusion(dose_mg, cl_L_per_h, ke, t, infusion_duration_h)
    else:  # oral

        def single_fn(t: float) -> float:
            return _single_oral(dose_mg, f_oral, vd_L, ka_per_h, ke, t)

    # Multi-dose superposition
    c_predicted = _superpose_doses(single_fn, n_doses, tau, t_h)

    # Steady-state calculations (assuming oral or bolus, infinite dosing)
    # css_avg is route-independent by mass balance
    if route == "oral":
        css_avg = f_oral * dose_mg / (cl_L_per_h * tau)
    else:
        # IV routes: F=1
        css_avg = dose_mg / (cl_L_per_h * tau)

    # css_max at steady state (oral, accumulation formula)
    # For iv_bolus/infusion, use bolus formula as approximation
    exp_ke_tau = math.exp(-ke * tau)
    if route == "oral":
        f_eff = f_oral
    else:
        f_eff = 1.0

    if abs(1.0 - exp_ke_tau) < 1e-12:
        # tau extremely large → no accumulation
        css_max = f_eff * dose_mg / vd_L
    else:
        css_max = f_eff * dose_mg / (vd_L * (1.0 - exp_ke_tau))

    css_min = css_max * exp_ke_tau

    # Notes
    note_parts: list[str] = [
        f"Route: {route}",
        f"ke={ke:.4f}/h, t½={t_half:.2f}h",
    ]
    if n_doses > 1:
        note_parts.append(f"Multi-dose superposition over {n_doses} doses")
    if route == "oral" and abs(ka_per_h - ke) < 1e-6:
        note_parts.append("Flip-flop kinetics (ka≈ke), limit formula applied")
    notes = "; ".join(note_parts)

    return ConcentrationPredictionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        t_h=t_h,
        route=route,
        c_predicted_mg_L=round(c_predicted, 8),
        css_avg_mg_L=round(css_avg, 8),
        css_max_mg_L=round(css_max, 8),
        css_min_mg_L=round(css_min, 8),
        ke_per_h=round(ke, 8),
        t_half_h=round(t_half, 8),
        notes=notes,
    )


def predict_concentration_profile(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    t_points: list[float],
    route: str = "oral",
    f_oral: float = 1.0,
    ka_per_h: float = 1.0,
    infusion_duration_h: float = 1.0,
    n_doses: int = 1,
    dosing_interval_h: float = 24.0,
) -> list[tuple[float, float]]:
    """Predict concentration at multiple time points.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose per administration (mg).
    cl_L_per_h : float
        Total clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    t_points : list[float]
        List of time points (h).
    route, f_oral, ka_per_h, infusion_duration_h, n_doses, dosing_interval_h :
        See predict_concentration.

    Returns
    -------
    list[tuple[float, float]]
        List of (time_h, concentration_mg_L) pairs.
    """
    profile: list[tuple[float, float]] = []
    for t in t_points:
        result = predict_concentration(
            drug_name=drug_name,
            dose_mg=dose_mg,
            t_h=t,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            route=route,
            f_oral=f_oral,
            ka_per_h=ka_per_h,
            infusion_duration_h=infusion_duration_h,
            n_doses=n_doses,
            dosing_interval_h=dosing_interval_h,
        )
        profile.append((t, result.c_predicted_mg_L))
    return profile
