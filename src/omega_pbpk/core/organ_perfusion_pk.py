"""
Phase 959 — Organ-Specific Perfusion-Limited PK Model

Blood flow-limited uptake into highly perfused organs.
Pure Python only (stdlib math, dataclasses). Forward Euler ODE solver.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["OrganPerfusionPKResult", "simulate_organ_perfusion", "screen_organ_exposure"]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class OrganPerfusionPKResult:
    drug_name: str
    dose_mg: float
    logp: float
    times_h: list[float]
    c_arterial_mg_L: list[float]
    c_liver_mg_L: list[float]
    c_kidney_mg_L: list[float]
    c_lung_mg_L: list[float]
    c_brain_mg_L: list[float]
    cmax_arterial: float
    auc_arterial: float
    auc_liver: float
    auc_brain: float
    liver_to_blood_auc_ratio: float
    brain_to_blood_auc_ratio: float
    notes: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kp_from_logp(logp: float):
    """Return (kp_liver, kp_kidney, kp_lung, kp_heart, kp_brain)."""
    kp_liver = max(0.5, 10 ** (0.35 * logp))
    kp_kidney = max(0.3, 10 ** (0.20 * logp))
    kp_lung = max(0.3, 10 ** (0.30 * logp))
    kp_heart = max(0.2, 10 ** (0.25 * logp))
    kp_brain = max(0.05, 10 ** (0.15 * logp - 0.5))
    return kp_liver, kp_kidney, kp_lung, kp_heart, kp_brain


def _trapz_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_organ_perfusion(
    drug_name: str,
    dose_mg: float,
    logp: float = 1.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    cl_hepatic_L_per_h: float = 10.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> OrganPerfusionPKResult:
    """Simulate perfusion-limited multi-organ PK using Forward Euler."""

    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_hepatic_L_per_h <= 0:
        raise ValueError("cl_hepatic_L_per_h must be > 0")
    if route not in {"iv", "oral"}:
        raise ValueError("route must be 'iv' or 'oral'")

    # --- Organ physiology ---
    Q_liver = 90.0  # L/h
    Q_kidney = 72.0  # L/h
    Q_lung = 300.0  # L/h  (all cardiac output)
    Q_heart = 15.0  # L/h
    Q_brain = 50.0  # L/h

    V_liver = 1.5  # L
    V_kidney = 0.3  # L
    V_lung = 0.5  # L
    V_heart = 0.3  # L
    V_brain = 1.5  # L
    V_blood = 5.0  # L

    kp_liver, kp_kidney, kp_lung, kp_heart, kp_brain = _kp_from_logp(logp)

    # State variables: [A_art, A_liver, A_kidney, A_lung, A_heart, A_brain, A_gut]
    # A_gut is the oral absorption depot (mg); ignored for IV
    # Concentrations = A_organ / V_organ (mg/L)

    # Initial conditions
    if route == "iv":
        A_art = dose_mg  # bolus into arterial blood
        A_gut = 0.0
    else:
        A_art = 0.0
        A_gut = dose_mg  # all dose in gut depot

    A_liver = 0.0
    A_kidney = 0.0
    A_lung = 0.0
    A_heart = 0.0
    A_brain = 0.0

    # Internal sub-step for numerical stability.
    # The stiffest eigenvalue is ~ Q_lung/V_lung/Kp_lung + Q_lung/V_blood ≈ 660 h⁻¹.
    # Euler stability requires dt < 2/660 ≈ 0.003 h → use dt_internal = 0.001 h.
    dt_internal = 0.001  # h — stable for all physiological flow/volume ratios
    output_dt = dt_h  # recording interval requested by caller

    times: list[float] = []
    c_art_list: list[float] = []
    c_liver_list: list[float] = []
    c_kidney_list: list[float] = []
    c_lung_list: list[float] = []
    c_brain_list: list[float] = []

    t = 0.0
    next_record = 0.0  # next time to record output
    max_t = t_end_h + 1e-9
    Q_total = Q_liver + Q_kidney + Q_lung + Q_heart + Q_brain

    while t <= max_t:
        # Record at requested output times
        if t >= next_record - 1e-12:
            C_art = A_art / V_blood
            C_liver = A_liver / V_liver
            C_kidney = A_kidney / V_kidney
            C_lung = A_lung / V_lung
            C_brain = A_brain / V_brain
            times.append(round(t, 6))
            c_art_list.append(C_art)
            c_liver_list.append(C_liver)
            c_kidney_list.append(C_kidney)
            c_lung_list.append(C_lung)
            c_brain_list.append(C_brain)
            next_record = round(next_record + output_dt, 10)

        if t >= max_t:
            break

        # Use the smaller of dt_internal and remaining time
        h = min(dt_internal, t_end_h - t, next_record - t)
        if h <= 0:
            h = dt_internal

        # Concentrations for derivatives
        C_art = A_art / V_blood
        C_liver = A_liver / V_liver
        C_kidney = A_kidney / V_kidney
        C_lung = A_lung / V_lung
        C_heart = A_heart / V_heart
        C_brain = A_brain / V_brain

        # Venous concentrations leaving each organ
        Cv_liver = C_liver / kp_liver
        Cv_kidney = C_kidney / kp_kidney
        Cv_lung = C_lung / kp_lung
        Cv_heart = C_heart / kp_heart
        Cv_brain = C_brain / kp_brain

        # Oral absorption input to arterial blood
        dose_input = ka_per_h * A_gut if route == "oral" else 0.0

        # Derivatives
        dA_liver = Q_liver * (C_art - Cv_liver) - cl_hepatic_L_per_h * Cv_liver
        dA_kidney = Q_kidney * (C_art - Cv_kidney)
        dA_lung = Q_lung * (C_art - Cv_lung)
        dA_heart = Q_heart * (C_art - Cv_heart)
        dA_brain = Q_brain * (C_art - Cv_brain)
        dA_gut = -ka_per_h * A_gut

        venous_return = (
            Q_liver * Cv_liver
            + Q_kidney * Cv_kidney
            + Q_lung * Cv_lung
            + Q_heart * Cv_heart
            + Q_brain * Cv_brain
        )
        co_outflow = Q_total * C_art
        dA_art = venous_return - co_outflow + dose_input

        # Forward Euler step
        A_art = max(0.0, A_art + h * dA_art)
        A_liver = max(0.0, A_liver + h * dA_liver)
        A_kidney = max(0.0, A_kidney + h * dA_kidney)
        A_lung = max(0.0, A_lung + h * dA_lung)
        A_heart = max(0.0, A_heart + h * dA_heart)
        A_brain = max(0.0, A_brain + h * dA_brain)
        A_gut = max(0.0, A_gut + h * dA_gut)

        t = round(t + h, 10)

    # Summary metrics
    cmax_art = max(c_art_list) if c_art_list else 0.0
    auc_art = _trapz_auc(times, c_art_list)
    auc_liv = _trapz_auc(times, c_liver_list)
    auc_brn = _trapz_auc(times, c_brain_list)

    liver_ratio = auc_liv / auc_art if auc_art > 0 else 0.0
    brain_ratio = auc_brn / auc_art if auc_art > 0 else 0.0

    notes = (
        f"Perfusion-limited 5-organ PBPK | route={route} | logP={logp:.2f} | "
        f"Kp_liver={kp_liver:.2f} Kp_brain={kp_brain:.3f} | "
        f"CL_hepatic={cl_hepatic_L_per_h:.1f} L/h"
    )

    return OrganPerfusionPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        times_h=times,
        c_arterial_mg_L=c_art_list,
        c_liver_mg_L=c_liver_list,
        c_kidney_mg_L=c_kidney_list,
        c_lung_mg_L=c_lung_list,
        c_brain_mg_L=c_brain_list,
        cmax_arterial=cmax_art,
        auc_arterial=auc_art,
        auc_liver=auc_liv,
        auc_brain=auc_brn,
        liver_to_blood_auc_ratio=liver_ratio,
        brain_to_blood_auc_ratio=brain_ratio,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Screening function
# ---------------------------------------------------------------------------


def screen_organ_exposure(
    drug_name: str,
    logp_values: list[float],
    dose_mg: float = 10.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    cl_hepatic_L_per_h: float = 10.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list[OrganPerfusionPKResult]:
    """Screen multiple logP values, sorted by brain_to_blood_auc_ratio descending."""
    results = []
    for logp in logp_values:
        r = simulate_organ_perfusion(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logp=logp,
            route=route,
            ka_per_h=ka_per_h,
            cl_hepatic_L_per_h=cl_hepatic_L_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(r)
    results.sort(key=lambda r: r.brain_to_blood_auc_ratio, reverse=True)
    return results
