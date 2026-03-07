"""Predict plasma drug concentration at a specific time after dose using 1-compartment model."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ConcentrationPredictionResult",
    "predict_concentration",
    "concentration_grid",
]

_VALID_ROUTES = ("iv", "oral")


@dataclass(frozen=True)
class ConcentrationPredictionResult:
    """Result from single time-point concentration prediction."""

    drug_name: str
    dose_mg: float
    route: str
    time_h: float
    n_prior_doses: int
    interval_h: float
    predicted_conc_mg_L: float
    ke_per_h: float
    t_half_h: float
    vd_L: float
    cl_L_per_h: float
    notes: str


def _single_dose_conc(
    dose_mg: float,
    vd_L: float,
    ke: float,
    ka: float,
    f_oral: float,
    route: str,
    t: float,
) -> float:
    """Compute 1-compartment concentration for a single dose at time t >= 0."""
    if t < 0:
        return 0.0
    if route == "iv":
        return (dose_mg / vd_L) * math.exp(-ke * t)
    else:
        # oral: absorption and elimination
        if abs(ka - ke) < 1e-9:
            # Avoid division by zero: use limiting form
            return (f_oral * dose_mg * ke / vd_L) * t * math.exp(-ke * t)
        return ((f_oral * dose_mg * ka) / (vd_L * (ka - ke))) * (
            math.exp(-ke * t) - math.exp(-ka * t)
        )


def predict_concentration(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "oral",
    time_h: float = 1.0,
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    n_prior_doses: int = 0,
    interval_h: float = 24.0,
) -> ConcentrationPredictionResult:
    """Predict plasma concentration at a given time using 1-compartment analytical model.

    For multiple prior doses, superposition is used: the concentration from each
    prior dose (dosed at -n*interval_h, n=1..n_prior_doses) is added to the
    contribution from the current dose.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Dose amount in mg.
    cl_L_per_h:
        Total clearance in L/h. Must be > 0.
    vd_L:
        Volume of distribution in L. Must be > 0.
    route:
        Administration route: ``"iv"`` or ``"oral"``.
    time_h:
        Time after the current dose at which to predict concentration [h]. Must be >= 0.
    ka_per_h:
        First-order oral absorption rate constant [1/h]. Ignored for IV route. Must be > 0.
    f_oral:
        Oral bioavailability fraction (0–1]. Ignored for IV route.
    n_prior_doses:
        Number of prior doses before the current one (for steady-state / multiple-dose
        superposition). Must be >= 0.
    interval_h:
        Dosing interval in hours. Must be > 0.

    Returns
    -------
    ConcentrationPredictionResult

    Raises
    ------
    ValueError
        On invalid parameter values.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got '{route}'")
    if time_h < 0:
        raise ValueError(f"time_h must be >= 0, got {time_h}")
    if route == "oral" and ka_per_h <= 0:
        raise ValueError(f"ka_per_h must be > 0 for oral route, got {ka_per_h}")
    if route == "oral" and not (0 < f_oral <= 1.0):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")
    if n_prior_doses < 0:
        raise ValueError(f"n_prior_doses must be >= 0, got {n_prior_doses}")
    if interval_h <= 0:
        raise ValueError(f"interval_h must be > 0, got {interval_h}")

    ke = cl_L_per_h / vd_L
    t_half_h = math.log(2.0) / ke

    ka = ka_per_h if route == "oral" else 1.0  # ka unused for iv

    # Current dose contribution
    conc = _single_dose_conc(dose_mg, vd_L, ke, ka, f_oral, route, time_h)

    # Prior doses via superposition (each dosed n*interval_h earlier)
    for n in range(1, n_prior_doses + 1):
        t_since_prior = time_h + n * interval_h
        conc += _single_dose_conc(dose_mg, vd_L, ke, ka, f_oral, route, t_since_prior)

    conc = max(conc, 0.0)

    # Estimate Cmax for % calculation
    if route == "iv":
        cmax = dose_mg / vd_L
    else:
        # Tmax analytical: tmax = ln(ka/ke)/(ka-ke) when ka != ke
        if abs(ka - ke) < 1e-9:
            tmax = 1.0 / ke
        else:
            tmax = math.log(ka / ke) / (ka - ke)
        tmax = max(tmax, 0.0)
        cmax = _single_dose_conc(dose_mg, vd_L, ke, ka, f_oral, route, tmax)
        cmax = max(cmax, 1e-12)

    percent_of_cmax = 100.0 * conc / cmax if cmax > 0 else 0.0

    notes = (
        f"ke={ke:.4f}/h; t_half={t_half_h:.2f}h; Cmax_est={cmax:.4f} mg/L; "
        f"pct_Cmax={percent_of_cmax:.1f}%; route={route}; "
        f"n_prior_doses={n_prior_doses}; tau={interval_h}h"
    )

    return ConcentrationPredictionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        time_h=time_h,
        n_prior_doses=n_prior_doses,
        interval_h=interval_h,
        predicted_conc_mg_L=conc,
        ke_per_h=ke,
        t_half_h=t_half_h,
        vd_L=vd_L,
        cl_L_per_h=cl_L_per_h,
        notes=notes,
    )


def concentration_grid(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "oral",
    time_points: list[float] | None = None,
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
) -> list[dict]:
    """Compute concentration at multiple time points (single-dose, no prior doses).

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Dose amount in mg.
    cl_L_per_h:
        Total clearance in L/h.
    vd_L:
        Volume of distribution in L.
    route:
        Administration route: ``"iv"`` or ``"oral"``.
    time_points:
        List of times in hours at which to evaluate concentration.
        Defaults to [0, 0.5, 1, 2, 4, 6, 8, 12, 24] if None.
    ka_per_h:
        Oral absorption rate constant [1/h].
    f_oral:
        Oral bioavailability fraction (0–1].

    Returns
    -------
    list[dict]
        Each element: ``{"time_h": float, "concentration_mg_L": float}``.

    Raises
    ------
    ValueError
        On invalid inputs.
    """
    if time_points is None:
        time_points = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]

    if not time_points:
        raise ValueError("time_points must not be empty")

    results = []
    for t in time_points:
        result = predict_concentration(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            route=route,
            time_h=t,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
            n_prior_doses=0,
            interval_h=24.0,
        )
        results.append({"time_h": t, "concentration_mg_L": result.predicted_conc_mg_L})

    return results
