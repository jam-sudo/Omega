"""Effect compartment model — ke0 hysteresis, PK-PD link."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class EffectCompartmentResult:
    drug_name: str
    times_h: list[float]
    cp_mg_L: list[float]      # plasma concentration
    ce_mg_L: list[float]      # effect compartment concentration
    effect: list[float]       # pharmacodynamic effect (0–1)
    ke0_per_h: float          # equilibration rate constant
    emax: float
    ec50_mg_L: float
    hill: float
    tmax_effect_h: float      # time of peak effect
    emax_observed: float      # peak effect achieved
    hysteresis_area: float    # area of CE-E loop (hysteresis index)


def _emax_effect(ce: float, ec50: float, emax: float, hill: float) -> float:
    """Hill equation: E = Emax * Ce^hill / (EC50^hill + Ce^hill)."""
    if ce <= 0:
        return 0.0
    ceh = ce ** hill
    ec50h = max(ec50, 1e-9) ** hill
    return float(emax * ceh / (ec50h + ceh))


def simulate_effect_compartment(
    drug_name: str,
    times_h: list[float],
    cp_mg_L: list[float],
    ke0_per_h: float,
    ec50_mg_L: float,
    emax: float = 1.0,
    hill: float = 1.0,
) -> EffectCompartmentResult:
    """Link plasma PK to effect compartment via ke0.

    dCe/dt = ke0 * (Cp - Ce)

    Parameters
    ----------
    times_h : list[float]
        Time array (h).
    cp_mg_L : list[float]
        Plasma concentration profile matching times_h.
    ke0_per_h : float
        Effect compartment equilibration rate (1/h).
    ec50_mg_L : float
        EC50 in effect compartment (mg/L).
    emax : float
        Maximum effect (0–1 or absolute units).
    hill : float
        Hill coefficient.
    """
    if ke0_per_h <= 0:
        raise ValueError("ke0_per_h must be > 0")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")

    t = np.asarray(times_h, dtype=float)
    cp = np.asarray(cp_mg_L, dtype=float)
    n = len(t)

    ce = np.zeros(n)
    effect_arr = np.zeros(n)
    effect_arr[0] = _emax_effect(ce[0], ec50_mg_L, emax, hill)

    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        dce_dt = ke0_per_h * (cp[i] - ce[i])
        ce[i + 1] = max(ce[i] + dce_dt * dt, 0.0)
        effect_arr[i + 1] = _emax_effect(ce[i + 1], ec50_mg_L, emax, hill)

    tmax_idx = int(np.argmax(effect_arr))
    tmax_effect = float(t[tmax_idx])
    emax_obs = float(np.max(effect_arr))

    # Hysteresis area: area enclosed by Ce-E loop (Cp vs Ce loop approximation)
    # Use shoelace formula on (cp, ce) trajectory
    if n >= 3:
        x = cp
        y = ce
        hysteresis = 0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))
    else:
        hysteresis = 0.0

    return EffectCompartmentResult(
        drug_name=drug_name,
        times_h=t.tolist(),
        cp_mg_L=cp.tolist(),
        ce_mg_L=ce.tolist(),
        effect=effect_arr.tolist(),
        ke0_per_h=ke0_per_h,
        emax=emax,
        ec50_mg_L=ec50_mg_L,
        hill=hill,
        tmax_effect_h=tmax_effect,
        emax_observed=emax_obs,
        hysteresis_area=hysteresis,
    )


def simulate_pkpd_iv(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ke0_per_h: float,
    ec50_mg_L: float,
    emax: float = 1.0,
    hill: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> EffectCompartmentResult:
    """Simulate IV bolus PK + effect compartment PD."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    ke = cl_L_per_h / vd_L
    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)
    cp = (dose_mg / vd_L) * np.exp(-ke * t)

    return simulate_effect_compartment(
        drug_name=drug_name,
        times_h=t.tolist(),
        cp_mg_L=cp.tolist(),
        ke0_per_h=ke0_per_h,
        ec50_mg_L=ec50_mg_L,
        emax=emax,
        hill=hill,
    )


def tmax_effect_analytical(ke: float, ke0: float) -> float:
    """Analytical tmax of effect compartment for 1-cpt IV bolus.

    tmax_ce = ln(ke0/ke) / (ke0 - ke)  (when ke0 != ke)
    """
    if abs(ke0 - ke) < 1e-9:
        return float(1.0 / ke)
    return float(math.log(ke0 / ke) / (ke0 - ke))


__all__ = [
    "EffectCompartmentResult",
    "simulate_effect_compartment",
    "simulate_pkpd_iv",
    "tmax_effect_analytical",
    "_emax_effect",
]
