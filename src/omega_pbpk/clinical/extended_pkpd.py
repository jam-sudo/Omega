"""Extended PK/PD modeling tools — Phase 502.

Emax, sigmoidal Emax, indirect response, and effect-compartment PK/PD
models implemented in pure Python with forward Euler integration.

References
----------
- Gabrielsson J, Weiner D. Pharmacokinetic and Pharmacodynamic Data
  Analysis. Swedish Pharmaceutical Society, 2016.
- Sheiner LB et al., Clin Pharmacol Ther. 1979;25(3):358-71. (ke0)
- Jusko WJ, Ko HC, Clin Pharmacol Ther. 1994;56(4):406-19.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "PKPDResult",
    "emax_model",
    "sigmoidal_emax",
    "simulate_pkpd",
    "calculate_pdp_index",
    "time_above_ec50",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class PKPDResult:
    """Result of a PK/PD simulation.

    Attributes
    ----------
    drug_name : Compound identifier.
    model : PK/PD model used ('direct' or 'ce').
    times_h : Time points (h).
    c_plasma_mg_L : Plasma concentration profile (mg/L).
    effect : Effect profile (arbitrary units).
    baseline_effect : E0 value.
    max_effect : Peak effect value observed.
    time_to_max_effect_h : Time of peak effect (h).
    effect_auc : Trapezoidal AUC of effect vs time.
    time_above_ec50_h : Hours with plasma concentration > EC50.
    notes : Free-text summary.
    """

    drug_name: str
    model: str
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    effect: list[float] = field(default_factory=list)
    baseline_effect: float = 0.0
    max_effect: float = 0.0
    time_to_max_effect_h: float = 0.0
    effect_auc: float = 0.0
    time_above_ec50_h: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# PD Models
# ---------------------------------------------------------------------------


def emax_model(conc: float, emax: float, ec50: float, e0: float = 0.0) -> float:
    """Emax (hyperbolic) concentration-effect relationship.

    E = E0 + Emax * C / (EC50 + C)

    Parameters
    ----------
    conc : Drug concentration (mg/L or same units as ec50).
    emax : Maximum drug effect.
    ec50 : Concentration producing 50 % of Emax.
    e0 : Baseline effect (default 0).

    Returns
    -------
    float
        Predicted effect.

    Raises
    ------
    ValueError
        If ec50 ≤ 0 or conc < 0.
    """
    if ec50 <= 0:
        raise ValueError(f"ec50 must be positive, got {ec50}")
    if conc < 0:
        raise ValueError(f"conc must be ≥ 0, got {conc}")
    return e0 + emax * conc / (ec50 + conc)


def sigmoidal_emax(conc: float, emax: float, ec50: float, hill_n: float, e0: float = 0.0) -> float:
    """Sigmoidal (Hill) Emax model.

    E = E0 + Emax * C^n / (EC50^n + C^n)

    Parameters
    ----------
    conc : Drug concentration.
    emax : Maximum drug effect.
    ec50 : Concentration producing 50 % of Emax.
    hill_n : Hill exponent (steepness; n=1 reduces to simple Emax).
    e0 : Baseline effect (default 0).

    Returns
    -------
    float
        Predicted effect.

    Raises
    ------
    ValueError
        If ec50 ≤ 0, hill_n ≤ 0, or conc < 0.
    """
    if ec50 <= 0:
        raise ValueError(f"ec50 must be positive, got {ec50}")
    if hill_n <= 0:
        raise ValueError(f"hill_n must be positive, got {hill_n}")
    if conc < 0:
        raise ValueError(f"conc must be ≥ 0, got {conc}")
    if conc == 0:
        return e0
    cn = conc**hill_n
    ec50n = ec50**hill_n
    return e0 + emax * cn / (ec50n + cn)


# ---------------------------------------------------------------------------
# PK helpers (1-compartment, forward Euler)
# ---------------------------------------------------------------------------


def _simulate_pk_iv(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    times_h: list[float],
    dt_h: float,
) -> list[float]:
    """Simulate 1-cpt IV bolus PK analytically.

    C(t) = (Dose/Vd) * exp(-CL/Vd * t)
    """
    k_el = cl_L_per_h / vd_L
    c0 = dose_mg / vd_L
    return [c0 * _exp(-k_el * t) for t in times_h]


def _simulate_pk_oral(
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f: float,
    times_h: list[float],
) -> list[float]:
    """Simulate 1-cpt oral PK using 1-cpt analytical solution.

    C(t) = F*Dose*ka / (Vd*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
    """
    k_el = cl_L_per_h / vd_L
    if abs(ka_per_h - k_el) < 1e-6:
        # Avoid division by zero; perturb ka slightly
        ka_per_h = k_el * 1.001
    denom = vd_L * (ka_per_h - k_el)
    coeff = f * dose_mg * ka_per_h / denom
    result = []
    for t in times_h:
        c = coeff * (_exp(-k_el * t) - _exp(-ka_per_h * t))
        result.append(max(0.0, c))
    return result


def _exp(x: float) -> float:
    """Safe exponential — clamps large negative exponents."""
    import math

    if x < -700:
        return 0.0
    return math.exp(x)


def _linspace(start: float, stop: float, n: int) -> list[float]:
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_pkpd(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    emax: float,
    ec50_mg_L: float,
    hill_n: float = 1.0,
    e0: float = 0.0,
    route: str = "iv",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
    model: str = "direct",
) -> PKPDResult:
    """Simulate PK/PD using a 1-compartment PK model and selected PD model.

    Parameters
    ----------
    drug_name : Compound identifier.
    dose_mg : Administered dose (mg).
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).
    emax : Maximum drug effect.
    ec50_mg_L : EC50 (mg/L).
    hill_n : Hill coefficient (default 1.0 = simple Emax).
    e0 : Baseline effect.
    route : 'iv' or 'oral'.
    t_end_h : Simulation end time (h).
    dt_h : Time step for ODE integration (h); also controls output resolution.
    model : 'direct' or 'ce' (effect compartment with ke0=0.1/h).

    Returns
    -------
    PKPDResult

    Raises
    ------
    ValueError
        On invalid inputs.
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if ec50_mg_L <= 0:
        raise ValueError(f"ec50_mg_L must be positive, got {ec50_mg_L}")
    if hill_n <= 0:
        raise ValueError(f"hill_n must be positive, got {hill_n}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be positive, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be positive, got {dt_h}")
    if route not in ("iv", "oral"):
        raise ValueError(f"route must be 'iv' or 'oral', got '{route}'")
    if model not in ("direct", "ce"):
        raise ValueError(f"model must be 'direct' or 'ce', got '{model}'")

    # ------------------------------------------------------------------
    # Build time array
    # ------------------------------------------------------------------
    n_steps = max(2, int(t_end_h / dt_h) + 1)
    times_h = _linspace(0.0, t_end_h, n_steps)

    # ------------------------------------------------------------------
    # PK simulation
    # ------------------------------------------------------------------
    ka_per_h = 1.0  # default oral absorption rate constant
    f = 1.0  # bioavailability (assumed 100 %)

    if route == "iv":
        c_plasma = _simulate_pk_iv(dose_mg, cl_L_per_h, vd_L, times_h, dt_h)
    else:
        c_plasma = _simulate_pk_oral(dose_mg, cl_L_per_h, vd_L, ka_per_h, f, times_h)

    # ------------------------------------------------------------------
    # PD simulation
    # ------------------------------------------------------------------
    ke0 = 0.1  # effect compartment rate constant (h⁻¹)

    if model == "direct":
        effect = [sigmoidal_emax(c, emax, ec50_mg_L, hill_n, e0) for c in c_plasma]
    else:
        # Effect compartment: forward Euler
        # dCe/dt = ke0 * (Cp - Ce)
        # E(t) = sigmoidal_emax(Ce, ...)
        ce = 0.0
        effect = []
        for i, _t in enumerate(times_h):
            e_val = sigmoidal_emax(ce, emax, ec50_mg_L, hill_n, e0)
            effect.append(e_val)
            if i < len(times_h) - 1:
                cp = c_plasma[i]
                dce_dt = ke0 * (cp - ce)
                ce = max(0.0, ce + dce_dt * dt_h)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    max_eff = max(effect)
    idx_max = effect.index(max_eff)
    t_max_eff = times_h[idx_max]

    # Trapezoidal AUC of effect
    eff_auc = 0.0
    for i in range(1, len(times_h)):
        eff_auc += 0.5 * (effect[i] + effect[i - 1]) * (times_h[i] - times_h[i - 1])

    t_above = time_above_ec50(c_plasma, times_h, ec50_mg_L)

    notes = (
        f"Model: {model}. Route: {route}. Max effect: {max_eff:.3f} at "
        f"t={t_max_eff:.1f} h. Time above EC50: {t_above:.2f} h."
    )

    return PKPDResult(
        drug_name=drug_name,
        model=model,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        effect=effect,
        baseline_effect=e0,
        max_effect=max_eff,
        time_to_max_effect_h=t_max_eff,
        effect_auc=eff_auc,
        time_above_ec50_h=t_above,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# PK/PD indices
# ---------------------------------------------------------------------------


def calculate_pdp_index(
    effect_time_series: list[float],
    times_h: list[float],
    baseline_effect: float,
) -> float:
    """Calculate pharmacodynamic pressure (PDP) index.

    PDP is the AUC of the effect above baseline, normalised by the
    total simulation time.

    PDP = integral(max(E(t) - E0, 0) dt) / T_total

    A flat effect at baseline returns 0.

    Parameters
    ----------
    effect_time_series : Effect values at each time point.
    times_h : Corresponding time points (h).
    baseline_effect : Baseline effect E0.

    Returns
    -------
    float
        PDP index (same units as effect).

    Raises
    ------
    ValueError
        If lists have different lengths or fewer than 2 points.
    """
    if len(effect_time_series) != len(times_h):
        raise ValueError("effect_time_series and times_h must have the same length.")
    if len(times_h) < 2:
        raise ValueError("At least 2 time points are required.")

    t_total = times_h[-1] - times_h[0]
    if t_total <= 0:
        return 0.0

    auc_above = 0.0
    for i in range(1, len(times_h)):
        dt = times_h[i] - times_h[i - 1]
        e_prev = max(0.0, effect_time_series[i - 1] - baseline_effect)
        e_curr = max(0.0, effect_time_series[i] - baseline_effect)
        auc_above += 0.5 * (e_prev + e_curr) * dt

    return auc_above / t_total


def time_above_ec50(
    conc_time_series: list[float],
    times_h: list[float],
    ec50_mg_L: float,
) -> float:
    """Calculate time (hours) during which concentration exceeds EC50.

    Uses trapezoidal integration of the indicator function C(t) > EC50.

    Parameters
    ----------
    conc_time_series : Plasma concentration values (mg/L).
    times_h : Corresponding time points (h).
    ec50_mg_L : EC50 threshold (mg/L).

    Returns
    -------
    float
        Cumulative hours above EC50.

    Raises
    ------
    ValueError
        If lists differ in length or ec50_mg_L ≤ 0.
    """
    if ec50_mg_L <= 0:
        raise ValueError(f"ec50_mg_L must be positive, got {ec50_mg_L}")
    if len(conc_time_series) != len(times_h):
        raise ValueError("conc_time_series and times_h must have the same length.")
    if len(times_h) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(times_h)):
        c_prev = conc_time_series[i - 1]
        c_curr = conc_time_series[i]
        dt = times_h[i] - times_h[i - 1]
        above_prev = 1.0 if c_prev > ec50_mg_L else 0.0
        above_curr = 1.0 if c_curr > ec50_mg_L else 0.0
        total += 0.5 * (above_prev + above_curr) * dt

    return total
