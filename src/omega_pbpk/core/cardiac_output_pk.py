"""Phase 1079 — Cardiac Output Effect on PK.

Models how changes in cardiac output (CO) affect multi-organ drug distribution.
Relevant for heart failure, exercise, anesthesia, and critical illness.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Organ volume constants (litres)
# ---------------------------------------------------------------------------
_V_LIVER_L = 1.5
_V_KIDNEY_L = 0.3
_V_MUSCLE_L = 28.0
_VD_PLASMA_L = 3.5

# Fractional blood-flow allocation
_F_LIVER = 0.25
_F_KIDNEY = 0.20
_F_MUSCLE = 0.40
# _F_OTHER = 0.15  (remainder, lumped, not tracked as state)


def _kp_liver(logP: float) -> float:
    return max(0.5, min(20.0, 1.5 * logP))


def _kp_kidney(logP: float) -> float:
    return max(0.3, min(10.0, 0.8 * logP))


def _kp_muscle(logP: float) -> float:
    return max(0.1, min(5.0, 0.5 * logP))


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i - 1] + values[i]) * (times[i] - times[i - 1])
    return auc


def _find_thalf(times: list[float], values: list[float], cmax: float) -> float:
    """Return first time at which value falls to 50% of cmax."""
    target = 0.5 * cmax
    for i in range(1, len(values)):
        if values[i] <= target:
            t0, t1 = times[i - 1], times[i]
            v0, v1 = values[i - 1], values[i]
            if v1 < v0:
                frac = (v0 - target) / (v0 - v1)
                return t0 + frac * (t1 - t0)
    return times[-1]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class CardiacOutputPKResult:
    drug_name: str
    dose_mg: float
    cardiac_output_L_per_h: float
    logP: float
    times_h: list
    c_plasma_mg_L: list
    c_liver_mg_L: list
    c_kidney_mg_L: list
    c_muscle_mg_L: list
    cmax_plasma: float
    auc_plasma: float
    auc_liver: float
    auc_kidney: float
    auc_muscle: float
    t_half_plasma_h: float
    pk_risk_assessment: str
    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def simulate_cardiac_output_pk(
    drug_name: str,
    dose_mg: float,
    cardiac_output_L_per_h: float = 300.0,
    logP: float = 2.0,
    cl_intrinsic_L_per_h: float = 5.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> CardiacOutputPKResult:
    """Simulate 3-organ PBPK with IV bolus dose under given cardiac output."""
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cardiac_output_L_per_h <= 0:
        raise ValueError("cardiac_output_L_per_h must be > 0")
    if cl_intrinsic_L_per_h <= 0:
        raise ValueError("cl_intrinsic_L_per_h must be > 0")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError("logP must be in [-5, 10]")

    CO = cardiac_output_L_per_h
    Q_liver = _F_LIVER * CO
    Q_kidney = _F_KIDNEY * CO
    Q_muscle = _F_MUSCLE * CO

    Kp_liv = _kp_liver(logP)
    Kp_kid = _kp_kidney(logP)
    Kp_mus = _kp_muscle(logP)

    V_plasma = _VD_PLASMA_L
    V_liv = _V_LIVER_L
    V_kid = _V_KIDNEY_L
    V_mus = _V_MUSCLE_L

    # Initial conditions — IV bolus
    Cp = dose_mg / V_plasma
    Cl = 0.0
    Ck = 0.0
    Cm = 0.0

    times: list[float] = []
    c_plasma: list[float] = []
    c_liver: list[float] = []
    c_kidney: list[float] = []
    c_muscle: list[float] = []

    # Choose an internal sub-step to keep forward Euler stable.
    # Dominant rate constants (per hour):
    #   plasma: (CL_int + Q_liv + Q_kid + Q_mus) / V_plasma
    #   liver:  Q_liv / (V_liv * Kp_liv)   (return-flux term divided by V)
    #   kidney: Q_kid / (V_kid * Kp_kid)
    #   muscle: Q_mus / (V_mus * Kp_mus)
    # Stability requires dt * max_rate < 1.
    # We target dt * max_rate ≤ 0.5 for safety.
    _plasma_rate = (Q_liver + Q_kidney + Q_muscle) / V_plasma
    _liver_rate = Q_liver / (V_liv * Kp_liv) + cl_intrinsic_L_per_h / V_liv
    _kidney_rate = Q_kidney / (V_kid * Kp_kid)
    _muscle_rate = Q_muscle / (V_mus * Kp_mus)
    _max_rate = max(_plasma_rate, _liver_rate, _kidney_rate, _muscle_rate)
    _dt_stable = 0.5 / _max_rate if _max_rate > 0 else dt_h
    _dt_internal = min(dt_h, _dt_stable)

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        times.append(t)
        c_plasma.append(Cp)
        c_liver.append(Cl)
        c_kidney.append(Ck)
        c_muscle.append(Cm)

        if step == n_steps:
            break

        # Integrate from t to t+dt_h using internal sub-steps
        t_sub = 0.0
        interval = dt_h
        while t_sub < interval - 1e-12:
            h = min(_dt_internal, interval - t_sub)
            # ODEs — forward Euler
            # cl_intrinsic is hepatic metabolic clearance in the liver compartment.
            # Higher CO → more drug delivered to liver → faster elimination → lower plasma AUC.
            dCp = (
                -(Q_liver * (Cp - Cl / Kp_liv)) / V_plasma
                - (Q_kidney * (Cp - Ck / Kp_kid)) / V_plasma
                - (Q_muscle * (Cp - Cm / Kp_mus)) / V_plasma
            )
            dCl = Q_liver * (Cp - Cl / Kp_liv) / V_liv - cl_intrinsic_L_per_h * Cl / V_liv
            dCk = Q_kidney * (Cp - Ck / Kp_kid) / V_kid
            dCm = Q_muscle * (Cp - Cm / Kp_mus) / V_mus

            Cp = max(0.0, Cp + dCp * h)
            Cl = max(0.0, Cl + dCl * h)
            Ck = max(0.0, Ck + dCk * h)
            Cm = max(0.0, Cm + dCm * h)
            t_sub += h

        t += dt_h

    cmax_p = max(c_plasma)
    auc_p = _trapz(times, c_plasma)
    auc_l = _trapz(times, c_liver)
    auc_k = _trapz(times, c_kidney)
    auc_m = _trapz(times, c_muscle)
    t_half = _find_thalf(times, c_plasma, cmax_p)

    if CO < 180.0:
        risk = "elevated_exposure"
    elif CO <= 360.0:
        risk = "normal"
    else:
        risk = "reduced_exposure"

    notes = (
        f"CO={CO:.1f} L/h; Kp_liver={Kp_liv:.2f}, Kp_kidney={Kp_kid:.2f}, Kp_muscle={Kp_mus:.2f}"
    )

    return CardiacOutputPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cardiac_output_L_per_h=CO,
        logP=logP,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_liver_mg_L=c_liver,
        c_kidney_mg_L=c_kidney,
        c_muscle_mg_L=c_muscle,
        cmax_plasma=cmax_p,
        auc_plasma=auc_p,
        auc_liver=auc_l,
        auc_kidney=auc_k,
        auc_muscle=auc_m,
        t_half_plasma_h=t_half,
        pk_risk_assessment=risk,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------
def compare_cardiac_outputs(
    drug_name: str,
    dose_mg: float,
    co_values: list[float],
    logP: float = 2.0,
    cl_intrinsic_L_per_h: float = 5.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list[CardiacOutputPKResult]:
    """Simulate across a list of CO values; sorted by auc_plasma descending."""
    results = [
        simulate_cardiac_output_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cardiac_output_L_per_h=co,
            logP=logP,
            cl_intrinsic_L_per_h=cl_intrinsic_L_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        for co in co_values
    ]
    results.sort(key=lambda r: r.auc_plasma, reverse=True)
    return results
