"""Steady-state PK predictor using multi-dose superposition on PBPK simulations.

Computes: Css_max, Css_min, Css_avg, AUCtau, accumulation ratio, fluctuation %,
time_to_SS, n_doses_to_SS.

Method: Run a single-dose PBPK simulation over n_doses × tau hours. Apply
superposition to compute the concentration at steady state over the last
dosing interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz
from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug


@dataclass(frozen=True)
class SteadyStateResult:
    """Steady-state PK metrics from multi-dose superposition."""

    css_max_mg_L: float
    css_min_mg_L: float
    css_avg_mg_L: float
    auc_tau_mg_h_L: float
    accumulation_ratio: float
    fluctuation_pct: float
    time_to_ss_h: float
    n_doses_to_ss: int
    dose_mg: float
    interval_h: float
    route: str


def predict_steady_state(
    drug: Drug,
    dose_mg: float = 100.0,
    interval_h: float = 12.0,
    route: str = "oral",
    n_doses: int = 10,
    body_weight: float = 70.0,
) -> SteadyStateResult:
    """Predict steady-state PK via superposition of a single-dose PBPK simulation.

    Parameters
    ----------
    drug : Drug
        Drug object with physicochemical and PK parameters.
    dose_mg : float
        Dose per administration in mg.
    interval_h : float
        Dosing interval (tau) in hours.
    route : str
        "oral" or "iv".
    n_doses : int
        Number of doses to superimpose.
    body_weight : float
        Body weight in kg.

    Returns
    -------
    SteadyStateResult
        Frozen dataclass with steady-state PK metrics.
    """
    # 1. Run single-dose simulation over the full multi-dose window
    t_end = n_doses * interval_h
    model = WholeBodyPBPK(drug, body_weight=body_weight)
    if route == "iv":
        model.setup_iv(dose_mg)
    else:
        model.setup_oral(dose_mg)
    result = model.simulate(t_end_h=t_end, dt_h=0.1)
    t = result.time_h
    cp = result.plasma_concentration()

    # 2. Get single-dose PK
    pk = result.pk_summary()
    single_cmax = pk["Cmax_mg_L"]
    t_half = pk["half_life_h"]

    # 3. Superposition over last dosing interval [0, tau]
    n_points = max(int(interval_h / 0.1) + 1, 2)
    t_fine = np.linspace(0, interval_h, n_points)
    css = np.zeros_like(t_fine)
    for k in range(n_doses):
        # k-th dose was given at time k*tau from start.
        # In the last interval, time since dose k = t_fine + (n_doses - 1 - k) * tau
        t_since_dose_k = t_fine + (n_doses - 1 - k) * interval_h
        contrib = np.interp(t_since_dose_k, t, cp, left=0.0, right=0.0)
        css += contrib

    # 4. Compute SS metrics
    css_max = float(np.max(css))
    css_min = float(np.min(css))
    auc_tau = float(np_trapz(css, t_fine))
    css_avg = auc_tau / max(interval_h, 1e-12)
    accum = css_max / max(single_cmax, 1e-12)
    fluct = 100.0 * (css_max - css_min) / max(css_avg, 1e-12)
    if math.isinf(t_half) or t_half > 1e6:
        time_to_ss = float("inf")
        n_to_ss = n_doses
    else:
        time_to_ss = 3.32 * max(t_half, 0.01)
        n_to_ss = max(1, math.ceil(time_to_ss / interval_h))

    return SteadyStateResult(
        css_max_mg_L=css_max,
        css_min_mg_L=css_min,
        css_avg_mg_L=css_avg,
        auc_tau_mg_h_L=auc_tau,
        accumulation_ratio=accum,
        fluctuation_pct=fluct,
        time_to_ss_h=time_to_ss,
        n_doses_to_ss=n_to_ss,
        dose_mg=dose_mg,
        interval_h=interval_h,
        route=route,
    )
