"""Supersaturation and precipitation kinetics for amorphous solid dispersions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class SupersaturationResult:
    drug_name: str
    solubility_crystalline_mg_mL: float
    solubility_amorphous_mg_mL: float  # typically 10-1000x crystalline
    supersaturation_ratio: float  # C_aq / C_equilibrium
    times_h: list[float]
    conc_dissolved_mg_mL: list[float]  # dissolved concentration over time
    conc_precipitate_mg_mL: list[float]  # precipitated drug
    t_onset_precipitation_h: float  # time supersaturation begins to fall
    t50_precipitation_h: float  # time to 50% precipitation
    auc_dissolved: float  # AUC of dissolved drug (mg·h/mL)
    spring_and_parachute_auc: float  # AUC with polymer parachute (10% bonus)
    precipitation_complete: bool


def _amorphous_solubility(
    logP: float,
    crystalline_sol: float,
    temp_K: float = 310.0,
) -> float:
    """Estimate amorphous solubility from crystalline solubility.

    Amorphous ~ crystalline * exp(ΔGcryst/RT)
    Simplified: amorphous_sol = crystalline * fold_factor
    fold_factor = exp(0.5 * logP + 1.0) capped at 1000x.
    """
    fold = math.exp(0.5 * max(logP, 0.0) + 1.0)
    fold = min(fold, 1000.0)
    return crystalline_sol * fold


def simulate_supersaturation(
    drug_name: str,
    dose_mg: float,
    volume_gut_mL: float = 250.0,
    solubility_mg_mL: float = 0.01,
    logP: float = 3.0,
    precipitation_rate_per_h: float = 2.0,
    absorption_rate_per_h: float = 1.0,
    polymer_parachute: bool = False,
    t_end_h: float = 6.0,
    dt_h: float = 0.02,
) -> SupersaturationResult:
    """Simulate amorphous drug supersaturation and precipitation.

    Model:
    - Initial dissolved conc = min(dose/volume, C_amorphous)
    - Precipitation: dC/dt = -kp * (C - C_crystalline) when C > C_crystalline
    - Absorption: dC/dt -= ka * C (sink into enterocyte)

    Parameters
    ----------
    precipitation_rate_per_h : float
        First-order precipitation rate (1/h).
    polymer_parachute : bool
        If True, polymer slows precipitation by 50% (HPMC/PVP effect).
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if volume_gut_mL <= 0:
        raise ValueError("volume_gut_mL must be > 0")

    C_cryst = solubility_mg_mL
    C_amorph = _amorphous_solubility(logP, C_cryst)

    # Initial dissolved concentration
    C0 = min(dose_mg / volume_gut_mL, C_amorph)
    C_prec0 = dose_mg / volume_gut_mL - C0  # immediately precipitated

    kp = precipitation_rate_per_h * (0.5 if polymer_parachute else 1.0)
    ka = absorption_rate_per_h

    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)

    conc_diss = np.zeros(n + 1)
    conc_prec = np.zeros(n + 1)
    conc_diss[0] = C0
    conc_prec[0] = C_prec0

    t_onset = t_end_h  # default: no precipitation
    t50 = t_end_h

    for i in range(n):
        c = conc_diss[i]
        p = conc_prec[i]

        # Precipitation (if supersaturated)
        if c > C_cryst:
            precip_rate = kp * (c - C_cryst)
        else:
            precip_rate = 0.0

        # Absorption (proportional to dissolved conc)
        abs_rate = ka * c

        dC = -precip_rate - abs_rate
        dP = precip_rate

        conc_diss[i + 1] = max(c + dC * dt_h, 0.0)
        conc_prec[i + 1] = max(p + dP * dt_h, 0.0)

        # Record t_onset (first time supersaturation starts falling)
        if t_onset == t_end_h and precip_rate > 0 and i > 0:
            t_onset = float(t[i])

    # t50: time when 50% of supersaturation lost
    if C0 > C_cryst:
        supersaturation = C0 - C_cryst
        half_ss = supersaturation / 2.0
        t50 = t_end_h
        for i, c in enumerate(conc_diss):
            if c <= C_cryst + half_ss:
                t50 = float(t[i])
                break

    auc_diss = float(np_trapz(conc_diss, t))

    # Spring-and-parachute AUC (10% bonus from polymer stabilization)
    spring_auc = auc_diss * 1.10 if polymer_parachute else auc_diss

    prec_complete = bool(conc_diss[-1] <= C_cryst * 1.01)

    ss_ratio = C0 / C_cryst if C_cryst > 0 else 1.0

    return SupersaturationResult(
        drug_name=drug_name,
        solubility_crystalline_mg_mL=C_cryst,
        solubility_amorphous_mg_mL=C_amorph,
        supersaturation_ratio=ss_ratio,
        times_h=t.tolist(),
        conc_dissolved_mg_mL=conc_diss.tolist(),
        conc_precipitate_mg_mL=conc_prec.tolist(),
        t_onset_precipitation_h=t_onset,
        t50_precipitation_h=t50,
        auc_dissolved=auc_diss,
        spring_and_parachute_auc=spring_auc,
        precipitation_complete=prec_complete,
    )


def compare_formulations_supersaturation(
    drug_name: str,
    dose_mg: float,
    solubility_mg_mL: float,
    logP: float = 3.0,
) -> dict[str, SupersaturationResult]:
    """Compare crystalline vs amorphous (with/without polymer) formulations."""
    return {
        "crystalline": simulate_supersaturation(
            drug_name, dose_mg, solubility_mg_mL=solubility_mg_mL, logP=0.0
        ),
        "amorphous": simulate_supersaturation(
            drug_name, dose_mg, solubility_mg_mL=solubility_mg_mL, logP=logP
        ),
        "amorphous_polymer": simulate_supersaturation(
            drug_name, dose_mg, solubility_mg_mL=solubility_mg_mL, logP=logP, polymer_parachute=True
        ),
    }


__all__ = [
    "SupersaturationResult",
    "simulate_supersaturation",
    "compare_formulations_supersaturation",
]
