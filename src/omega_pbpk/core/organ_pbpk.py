"""Simplified organ-level PBPK model — Phase 486."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OrganPBPKResult:
    """Result from organ-level PBPK simulation."""

    drug_name: str
    route: str
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_liver_mg_kg: list[float]
    c_kidney_mg_kg: list[float]
    c_lung_mg_kg: list[float]
    c_muscle_mg_kg: list[float]
    auc_plasma: float
    cmax_plasma: float
    kp_liver: float
    kp_kidney: float
    kp_lung: float
    kp_muscle: float
    notes: str


# -------------------------------------------------------------------------
# Physiological constants for a 70 kg human
# -------------------------------------------------------------------------
_Q_LIVER = 90.0  # L/h
_Q_KIDNEY = 72.0  # L/h
_Q_MUSCLE = 75.0  # L/h
_Q_LUNG = 390.0  # L/h  (cardiac output; lung treated as central extension)

_V_LIVER = 1.8  # L
_V_KIDNEY = 0.6  # L
_V_LUNG = 0.6  # L
_V_MUSCLE = 35.0  # L

# Central (plasma) volume — lung distributes into central via fast equilibrium
_V_CENTRAL = 5.0  # L

_KA_DEFAULT = 1.0  # oral absorption rate constant (h⁻¹)
_F_ABS = 1.0  # fraction absorbed

_VALID_ROUTES = ("iv", "oral")


def _validate_inputs(
    dose_mg: float,
    route: str,
    cl_int_L_per_h: float,
    fu_plasma: float,
    t_end_h: float,
    dt_h: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got '{route}'")
    if cl_int_L_per_h < 0:
        raise ValueError(f"cl_int_L_per_h must be >= 0, got {cl_int_L_per_h}")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")


def calculate_organ_kp(logP: float, fu_plasma: float, organ: str) -> float:
    """Calculate tissue-to-plasma partition coefficient (Kp) for an organ.

    Simplified Rodgers-Rowland approach:
    - Liver:  Kp = 1 + logP * 0.5
    - Kidney: Kp = 1 + logP * 0.3
    - Lung:   Kp = 1 + logP * 0.2
    - Muscle: Kp = 1 + logP * 0.1
    """
    valid_organs = ("liver", "kidney", "lung", "muscle")
    if organ not in valid_organs:
        raise ValueError(f"organ must be one of {valid_organs}, got '{organ}'")
    slopes = {"liver": 0.5, "kidney": 0.3, "lung": 0.2, "muscle": 0.1}
    kp = 1.0 + logP * slopes[organ]
    return max(0.1, kp)


def organ_steady_state_conc(c_plasma_ss: float, kp_organ: float) -> float:
    """Return steady-state organ concentration (mg/kg) given plasma Css and Kp."""
    if c_plasma_ss < 0:
        raise ValueError(f"c_plasma_ss must be >= 0, got {c_plasma_ss}")
    if kp_organ <= 0:
        raise ValueError(f"kp_organ must be > 0, got {kp_organ}")
    return c_plasma_ss * kp_organ


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def _organ_update_exact(
    c_organ: float,
    c_art: float,
    kp: float,
    q: float,
    v: float,
    dt: float,
    extra_cl: float = 0.0,
    fu: float = 1.0,
) -> tuple[float, float]:
    """Exact analytical solution for 1-cpt organ ODE over one dt.

    ODE: dC/dt = Q/V*(c_art - C/Kp) - extra_cl*fu*C/V

    The solution for constant c_art:
      Let alpha = Q/(V*Kp) + extra_cl*fu/V   (total clearance rate from organ)
      Let source = Q/V * c_art
      Css = source / alpha  = (Q*c_art) / (Q + extra_cl*fu*Kp)   (effective SS)
      C(t) = Css + (C0 - Css)*exp(-alpha*t)

    Returns (new_c_organ, mass_returned_to_central_mg_per_L_times_dt)
    The net drug leaving the organ to central = Q*(C_organ/Kp - c_art) * dt  [exact integral]
    """
    alpha = q / (v * kp) + extra_cl * fu / v
    source = q * c_art / v
    if alpha < 1e-15:
        # degenerate: no return
        new_c = c_organ + source * dt
        return max(0.0, new_c), 0.0

    css = source / alpha
    decay = math.exp(-alpha * dt)
    new_c = css + (c_organ - css) * decay
    new_c = max(0.0, new_c)

    # Net amount returned from organ to central over dt (mg)
    # = Q * integral(C_organ/Kp - c_art, 0, dt)
    # integral of C(t)/Kp dt = css/Kp * dt + (c_organ - css)/Kp * (1-decay)/alpha
    avg_return_flux = q / kp * (css * dt + (c_organ - css) * (1.0 - decay) / alpha) - q * c_art * dt
    return new_c, avg_return_flux


def simulate_organ_pbpk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_int_L_per_h: float = 0.5,
    fu_plasma: float = 0.1,
    logP: float = 2.0,
    mw: float = 400.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> OrganPBPKResult:
    """Simulate organ-level PBPK with stable semi-analytical integration.

    4-organ model: liver, kidney, lung, muscle.

    Architecture:
    - Lung is treated as a fast-equilibrating extension of the central volume
      (avoids stiffness from Q_lung = 390 L/h).
    - Liver, kidney, muscle: solved analytically each step (always stable).
    - Central compartment uses forward Euler with net exchange fluxes from organs.

    Elimination only from liver via CLint.
    """
    _validate_inputs(dose_mg, route, cl_int_L_per_h, fu_plasma, t_end_h, dt_h)

    kp_liver = calculate_organ_kp(logP, fu_plasma, "liver")
    kp_kidney = calculate_organ_kp(logP, fu_plasma, "kidney")
    kp_lung = calculate_organ_kp(logP, fu_plasma, "lung")
    kp_muscle = calculate_organ_kp(logP, fu_plasma, "muscle")

    # Effective central volume includes lung (fast equilibrium with plasma)
    v_eff = _V_CENTRAL + _V_LUNG * kp_lung

    # Initial conditions
    if route == "iv":
        a_central = dose_mg  # mg in central compartment
        a_gut = 0.0
    else:
        a_central = 0.0
        a_gut = dose_mg * _F_ABS

    c_liver = 0.0  # mg/L
    c_kidney = 0.0
    c_muscle = 0.0

    times: list[float] = []
    cp_trace: list[float] = []
    cl_trace: list[float] = []
    ck_trace: list[float] = []
    clu_trace: list[float] = []
    cm_trace: list[float] = []

    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h
        c_art = a_central / v_eff
        c_lung = c_art * kp_lung

        times.append(t)
        cp_trace.append(c_art)
        cl_trace.append(c_liver)
        ck_trace.append(c_kidney)
        clu_trace.append(c_lung)
        cm_trace.append(c_muscle)

        if step == n_steps:
            break

        # Oral absorption
        if route == "oral":
            absorbed = _KA_DEFAULT * a_gut * dt_h
            absorbed = min(absorbed, a_gut)
            a_gut -= absorbed
            a_gut = max(0.0, a_gut)
        else:
            absorbed = 0.0

        # Analytic organ updates (always stable)
        new_c_liver, flux_liver = _organ_update_exact(
            c_liver,
            c_art,
            kp_liver,
            _Q_LIVER,
            _V_LIVER,
            dt_h,
            extra_cl=cl_int_L_per_h,
            fu=fu_plasma,
        )
        new_c_kidney, flux_kidney = _organ_update_exact(
            c_kidney, c_art, kp_kidney, _Q_KIDNEY, _V_KIDNEY, dt_h
        )
        new_c_muscle, flux_muscle = _organ_update_exact(
            c_muscle, c_art, kp_muscle, _Q_MUSCLE, _V_MUSCLE, dt_h
        )

        # Net change in central amount (mg):
        # flux_* is the net amount returned from each organ (negative = drug left central)
        # absorbed adds drug to central
        da_central = absorbed + flux_liver + flux_kidney + flux_muscle

        # Update central amount
        a_central = max(0.0, a_central + da_central)

        # Update organ concentrations
        c_liver = new_c_liver
        c_kidney = new_c_kidney
        c_muscle = new_c_muscle

    auc = _trapezoidal_auc(times, cp_trace)
    cmax = max(cp_trace)

    notes = (
        f"4-organ PBPK ({route}), CLint={cl_int_L_per_h} L/h, "
        f"fu={fu_plasma}, logP={logP}, MW={mw} g/mol. "
        f"Lung modelled as fast-equilibrating central extension."
    )

    return OrganPBPKResult(
        drug_name=drug_name,
        route=route,
        times_h=times,
        c_plasma_mg_L=cp_trace,
        c_liver_mg_kg=cl_trace,
        c_kidney_mg_kg=ck_trace,
        c_lung_mg_kg=clu_trace,
        c_muscle_mg_kg=cm_trace,
        auc_plasma=auc,
        cmax_plasma=cmax,
        kp_liver=kp_liver,
        kp_kidney=kp_kidney,
        kp_lung=kp_lung,
        kp_muscle=kp_muscle,
        notes=notes,
    )
