"""Biomarker-linked PK/PD model — Phase 493.

Implements indirect response models linking drug PK to a turnover biomarker.
The biomarker has natural production (kin) and degradation (kout); drug alters
one of these rates via an Emax relationship.

Model types
-----------
inhibit_kin    : dR/dt = kin*(1 - Emax*C/(EC50+C)) - kout*R
stimulate_kin  : dR/dt = kin*(1 + Emax*C/(EC50+C)) - kout*R
inhibit_kout   : dR/dt = kin - kout*(1 - Emax*C/(EC50+C))*R
stimulate_kout : dR/dt = kin - kout*(1 + Emax*C/(EC50+C))*R

Baseline: R0 = kin / kout  (steady state without drug)

References
----------
- Dayneka NL et al., J Pharmacokinet Biopharm. 1993;21(4):457-78.
- Mager DE & Jusko WJ, J Pharmacokinet Pharmacodyn. 2001;28(6):507-32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_VALID_MODEL_TYPES = {"inhibit_kin", "stimulate_kin", "inhibit_kout", "stimulate_kout"}


@dataclass
class IndirectResponseResult:
    """Result of an indirect response PK/PD simulation.

    Attributes
    ----------
    drug_name : Drug identifier.
    model_type : One of inhibit_kin, stimulate_kin, inhibit_kout, stimulate_kout.
    times_h : Simulation time grid (hours).
    c_plasma_mg_L : Plasma concentration at each time point (mg/L).
    biomarker : Biomarker response at each time point (same units as kin/kout).
    baseline_biomarker : Equilibrium biomarker before drug administration = kin/kout.
    max_effect_pct_change : Maximum % change from baseline (signed).
    time_to_max_effect_h : Time of maximum absolute change from baseline (hours).
    time_to_return_to_50pct_baseline_h : Hours after peak to return to 50% of
        max deviation from baseline.  NaN if never reached within simulation.
    auc_effect_above_baseline : Trapezoidal integral of |biomarker - baseline|
        over the simulation (arbitrary units * hours).
    notes : Diagnostic string.
    """

    drug_name: str
    model_type: str
    times_h: list[float]
    c_plasma_mg_L: list[float]
    biomarker: list[float]
    baseline_biomarker: float
    max_effect_pct_change: float
    time_to_max_effect_h: float
    time_to_return_to_50pct_baseline_h: float
    auc_effect_above_baseline: float
    notes: str = field(default="")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def calculate_baseline_biomarker(kin: float, kout: float) -> float:
    """Return equilibrium biomarker value = kin / kout.

    Parameters
    ----------
    kin  : Zero-order production rate (> 0).
    kout : First-order elimination rate constant (> 0).

    Returns
    -------
    float : Baseline biomarker concentration.
    """
    if kin <= 0:
        raise ValueError(f"kin must be > 0, got {kin}")
    if kout <= 0:
        raise ValueError(f"kout must be > 0, got {kout}")
    return kin / kout


def predict_effect_onset(ec50_mg_L: float, cmax_mg_L: float, kout: float) -> float:
    """Estimate hours to peak effect after a single dose.

    Uses the approximation that peak effect lag ≈ 1 / kout when Cmax >> EC50,
    scaled by the saturation fraction:
        t_peak_effect ≈ ln(cmax / ec50) / kout

    When Cmax <= EC50 the effect is sub-maximal and the formula returns a
    smaller value (negative saturation log → clipped to 0 with a minimum of
    1 / kout).

    Parameters
    ----------
    ec50_mg_L  : EC50 in mg/L (> 0).
    cmax_mg_L  : Predicted peak plasma concentration in mg/L (> 0).
    kout       : First-order biomarker elimination rate (> 0).

    Returns
    -------
    float : Estimated time to peak biomarker effect (hours).
    """
    if ec50_mg_L <= 0:
        raise ValueError(f"ec50_mg_L must be > 0, got {ec50_mg_L}")
    if cmax_mg_L <= 0:
        raise ValueError(f"cmax_mg_L must be > 0, got {cmax_mg_L}")
    if kout <= 0:
        raise ValueError(f"kout must be > 0, got {kout}")
    ratio = cmax_mg_L / ec50_mg_L
    if ratio <= 1.0:
        return 1.0 / kout
    return math.log(ratio) / kout


def calculate_hysteresis_area(
    pk_times_h: list[float],
    conc_mg_L: list[float],
    pd_times_h: list[float],
    effect: list[float],
) -> float:
    """Calculate hysteresis loop area in the concentration–effect plane.

    Interpolates effect onto the PK time grid then computes the area enclosed
    by the ascending and descending limbs of the concentration–effect curve
    using the shoelace (surveyor's) formula.

    Parameters
    ----------
    pk_times_h  : PK time grid (hours, ascending).
    conc_mg_L   : Plasma concentration at each PK time (mg/L).
    pd_times_h  : PD time grid (hours, ascending, same or different from PK).
    effect      : Biomarker (or effect) values at each PD time.

    Returns
    -------
    float : Absolute area of the hysteresis loop (concentration * effect units).
    """
    if len(pk_times_h) < 2:
        raise ValueError("pk_times_h must have at least 2 points")
    if len(pk_times_h) != len(conc_mg_L):
        raise ValueError("pk_times_h and conc_mg_L must have the same length")
    if len(pd_times_h) != len(effect):
        raise ValueError("pd_times_h and effect must have the same length")
    if len(pd_times_h) < 2:
        raise ValueError("pd_times_h must have at least 2 points")

    # Interpolate effect onto PK time grid (linear interpolation)
    def _interp(t: float, xs: list[float], ys: list[float]) -> float:
        if t <= xs[0]:
            return ys[0]
        if t >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= t <= xs[i + 1]:
                frac = (t - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + frac * (ys[i + 1] - ys[i])
        return ys[-1]

    eff_at_pk = [_interp(t, pd_times_h, effect) for t in pk_times_h]

    # Shoelace formula for closed polygon (C, E) → (C_peak back to start)
    # We pair (conc, eff_at_pk) as the loop polygon
    n = len(pk_times_h)
    area = 0.0
    for i in range(n - 1):
        area += conc_mg_L[i] * eff_at_pk[i + 1] - conc_mg_L[i + 1] * eff_at_pk[i]
    # close the loop
    area += conc_mg_L[n - 1] * eff_at_pk[0] - conc_mg_L[0] * eff_at_pk[n - 1]
    return abs(area) / 2.0


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_indirect_response(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    kin: float,
    kout: float,
    ec50_mg_L: float,
    emax: float = 1.0,
    route: str = "iv",
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
    model_type: str = "inhibit_kin",
    ka_per_h: float = 1.0,
) -> IndirectResponseResult:
    """Simulate drug PK linked to an indirect-response biomarker model.

    Parameters
    ----------
    drug_name   : Identifier for the drug.
    dose_mg     : Administered dose (mg, > 0).
    cl_L_per_h  : Systemic clearance (L/h, > 0).
    vd_L        : Volume of distribution (L, > 0).
    kin         : Zero-order biomarker production rate (> 0).
    kout        : First-order biomarker elimination rate constant (> 0).
    ec50_mg_L   : Drug concentration producing 50% of Emax (mg/L, > 0).
    emax        : Maximum fractional effect (0 – 1 for inhibition models).
    route       : 'iv' (bolus) or 'oral' (first-order absorption via ka).
    t_end_h     : Simulation end time (hours).
    dt_h        : Integration step size (hours).
    model_type  : One of inhibit_kin, stimulate_kin, inhibit_kout, stimulate_kout.
    ka_per_h    : First-order oral absorption rate (h⁻¹); only used when
                  route='oral'.

    Returns
    -------
    IndirectResponseResult
    """
    # --- Input validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if kin <= 0:
        raise ValueError(f"kin must be > 0, got {kin}")
    if kout <= 0:
        raise ValueError(f"kout must be > 0, got {kout}")
    if ec50_mg_L <= 0:
        raise ValueError(f"ec50_mg_L must be > 0, got {ec50_mg_L}")
    if emax < 0:
        raise ValueError(f"emax must be >= 0, got {emax}")
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"model_type must be one of {sorted(_VALID_MODEL_TYPES)}, got '{model_type}'"
        )
    route = route.lower()
    if route not in {"iv", "oral"}:
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # --- Derived PK parameters ---
    ke = cl_L_per_h / vd_L  # elimination rate constant (h⁻¹)

    # --- Initial conditions ---
    # PK: C_plasma (mg/L), gut depot for oral
    c0 = dose_mg / vd_L if route == "iv" else 0.0
    gut0 = dose_mg if route == "oral" else 0.0  # mg in gut compartment
    r0 = kin / kout  # baseline biomarker

    # --- Forward Euler integration ---
    n_steps = max(1, round(t_end_h / dt_h))
    times_h: list[float] = []
    c_plasma: list[float] = []
    biomarker: list[float] = []

    c = c0
    gut = gut0
    r = r0
    t = 0.0

    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma.append(c)
        biomarker.append(r)

        # PD: Hill (Emax) effect
        effect_fraction = emax * c / (ec50_mg_L + c) if (ec50_mg_L + c) > 0 else 0.0

        # Biomarker ODE
        if model_type == "inhibit_kin":
            dr = kin * (1.0 - effect_fraction) - kout * r
        elif model_type == "stimulate_kin":
            dr = kin * (1.0 + effect_fraction) - kout * r
        elif model_type == "inhibit_kout":
            dr = kin - kout * (1.0 - effect_fraction) * r
        else:  # stimulate_kout
            dr = kin - kout * (1.0 + effect_fraction) * r

        # PK ODE
        if route == "oral":
            abs_rate = ka_per_h * gut / vd_L  # mg/h / L = mg/L/h
            d_gut = -ka_per_h * gut
            dc = abs_rate - ke * c
            gut += d_gut * dt_h
            gut = max(gut, 0.0)
        else:
            dc = -ke * c

        # Euler update
        c += dc * dt_h
        r += dr * dt_h
        c = max(c, 0.0)
        r = max(r, 0.0)
        t += dt_h

    # --- Post-processing ---
    baseline = r0

    # Max effect (signed % change from baseline)
    if baseline == 0.0:
        pct_changes = [0.0] * len(biomarker)
    else:
        pct_changes = [(b - baseline) / baseline * 100.0 for b in biomarker]

    abs_pct = [abs(p) for p in pct_changes]
    max_abs_idx = abs_pct.index(max(abs_pct))
    max_effect_pct = pct_changes[max_abs_idx]
    time_to_max_effect = times_h[max_abs_idx]

    # Time to return to 50% of max deviation (after peak)
    if baseline != 0.0:
        max_dev = abs(biomarker[max_abs_idx] - baseline)
        half_dev_threshold = 0.5 * max_dev
        return_time = float("nan")
        for i in range(max_abs_idx + 1, len(biomarker)):
            if abs(biomarker[i] - baseline) <= half_dev_threshold:
                return_time = times_h[i]
                break
    else:
        return_time = float("nan")

    # AUC of |biomarker - baseline| via trapezoidal rule
    auc = 0.0
    for i in range(len(times_h) - 1):
        y0 = abs(biomarker[i] - baseline)
        y1 = abs(biomarker[i + 1] - baseline)
        auc += 0.5 * (y0 + y1) * (times_h[i + 1] - times_h[i])

    notes = f"route={route}, ke={ke:.4f}/h, R0={baseline:.4f}, Emax={emax}, EC50={ec50_mg_L} mg/L"

    return IndirectResponseResult(
        drug_name=drug_name,
        model_type=model_type,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        biomarker=biomarker,
        baseline_biomarker=baseline,
        max_effect_pct_change=max_effect_pct,
        time_to_max_effect_h=time_to_max_effect,
        time_to_return_to_50pct_baseline_h=return_time,
        auc_effect_above_baseline=auc,
        notes=notes,
    )
