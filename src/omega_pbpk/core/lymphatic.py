"""Intestinal lymphatic absorption — alternative route for lipophilic drugs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class LymphaticAbsorptionResult:
    """Result of lymphatic absorption prediction."""

    drug_name: str
    logP: float
    dose_mg: float
    f_lymph: float  # fraction absorbed via lymph (0–1)
    f_portal: float  # fraction absorbed via portal vein (1 - f_lymph)
    auc_total_mg_h_L: float  # total plasma AUC
    auc_lymph_mg_h_L: float  # AUC contribution from lymphatic route
    auc_portal_mg_h_L: float  # AUC contribution from portal route
    fh_portal: float  # hepatic first-pass for portal fraction
    fh_lymph: float  # hepatic first-pass for lymphatic fraction (bypassed = 1.0)
    bioavailability_portal: float  # F via portal (with first-pass)
    bioavailability_lymph: float  # F via lymph (bypasses first-pass = f_lymph)
    bioavailability_total: float  # combined F
    times_h: list[float]
    plasma_conc: list[float]
    lymph_conc: list[float]  # drug in lymph (thoracic duct, mg/L)
    tmax_h: float
    cmax_mg_L: float


def _lymphatic_fraction(logP: float, solubility_mg_mL: float = 1.0) -> float:
    """Estimate fraction absorbed via intestinal lymphatics.

    Based on: Trevaskis et al. 2008 — logP and lipid solubility drive lymphatic uptake.
    Sigmoid function: f_lymph = f_max / (1 + exp(-k*(logP - logP50)))
    f_max = 0.50 (literature upper bound for most drugs)
    logP50 = 5.0 (half-maximal at logP 5)
    k = 1.5 (steepness)
    Attenuated by aqueous solubility (high solubility → less lymph).
    """
    f_max = 0.50
    logP50 = 5.0
    k = 1.5
    f_base = f_max / (1.0 + math.exp(-k * (logP - logP50)))
    # Solubility correction: high solubility reduces lymphatic fraction
    sol_factor = 1.0 / (1.0 + solubility_mg_mL / 0.1)
    return float(np.clip(f_base * sol_factor, 0.0, 0.50))


def _hepatic_first_pass(
    cl_L_per_h: float,
    q_liver_L_per_h: float = 90.0,
    fup: float = 1.0,
    cl_int_L_per_h: float | None = None,
) -> float:
    """Hepatic extraction fraction (well-stirred model). Returns fh = 1 - E."""
    if cl_int_L_per_h is not None:
        e = fup * cl_int_L_per_h / (q_liver_L_per_h + fup * cl_int_L_per_h)
    else:
        e = min(cl_L_per_h / q_liver_L_per_h, 0.999)
    return 1.0 - e


def simulate_lymphatic_absorption(
    drug_name: str,
    logP: float,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    fup: float = 1.0,
    solubility_mg_mL: float = 1.0,
    q_liver_L_per_h: float = 90.0,
    lymph_flow_L_per_h: float = 0.12,  # thoracic duct ~3 L/day
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> LymphaticAbsorptionResult:
    """Simulate oral absorption with intestinal lymphatic route.

    Lymphatic fraction bypasses hepatic first-pass (enters systemic via thoracic duct).
    Portal fraction undergoes normal hepatic extraction.

    Parameters
    ----------
    logP : float
        Lipophilicity — drives lymphatic uptake fraction.
    dose_mg : float
        Oral dose (mg).
    cl_L_per_h : float
        Total systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    ka_per_h : float
        Absorption rate constant (1/h).
    fup : float
        Fraction unbound in plasma.
    solubility_mg_mL : float
        Aqueous solubility (mg/mL) — high solubility reduces lymphatic fraction.
    lymph_flow_L_per_h : float
        Thoracic duct lymph flow (L/h), default ~3 L/day = 0.125 L/h.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    fup = float(np.clip(fup, 1e-6, 1.0))
    f_lymph = _lymphatic_fraction(logP, solubility_mg_mL)
    f_portal = 1.0 - f_lymph

    # Hepatic first-pass factor for portal route
    fh = _hepatic_first_pass(cl_L_per_h, q_liver_L_per_h, fup)

    # Bioavailabilities
    fa = 1.0  # assume complete gut absorption for simplicity
    F_portal = fa * f_portal * fh
    F_lymph = fa * f_lymph  # bypasses liver
    F_total = F_portal + F_lymph

    # Simulate plasma PK with two absorption inputs (forward Euler)
    n = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n + 1)
    ke = cl_L_per_h / vd_L

    # Gut compartment split between portal and lymph
    a_gut_portal = dose_mg * f_portal * fa
    a_gut_lymph = dose_mg * f_lymph * fa
    a_central = 0.0

    cp = np.zeros(n + 1)
    cl_lym = np.zeros(n + 1)  # lymph concentration (mg/L)

    for i in range(n):
        # Portal absorption → central (with first-pass applied at input)
        abs_portal = ka_per_h * a_gut_portal * dt_h * fh
        a_gut_portal -= ka_per_h * a_gut_portal * dt_h

        # Lymph absorption → central (no first-pass)
        abs_lymph = ka_per_h * a_gut_lymph * dt_h
        a_gut_lymph -= ka_per_h * a_gut_lymph * dt_h

        a_central += abs_portal + abs_lymph
        a_central -= ke * a_central * dt_h
        a_central = max(a_central, 0.0)
        a_gut_portal = max(a_gut_portal, 0.0)
        a_gut_lymph = max(a_gut_lymph, 0.0)

        cp[i + 1] = a_central / vd_L
        # Lymph drug concentration approximation
        cl_lym[i + 1] = (ka_per_h * a_gut_lymph * dt_h) / (lymph_flow_L_per_h * dt_h + 1e-9)

    # PK metrics
    auc_total = float(np_trapz(cp, t))
    cmax = float(np.max(cp))
    tmax_h = float(t[int(np.argmax(cp))])

    # AUC split (proportional to dose contribution)
    auc_portal = auc_total * F_portal / F_total if F_total > 0 else 0.0
    auc_lymph = auc_total * F_lymph / F_total if F_total > 0 else 0.0

    return LymphaticAbsorptionResult(
        drug_name=drug_name,
        logP=logP,
        dose_mg=dose_mg,
        f_lymph=f_lymph,
        f_portal=f_portal,
        auc_total_mg_h_L=auc_total,
        auc_lymph_mg_h_L=auc_lymph,
        auc_portal_mg_h_L=auc_portal,
        fh_portal=fh,
        fh_lymph=1.0,
        bioavailability_portal=F_portal,
        bioavailability_lymph=F_lymph,
        bioavailability_total=F_total,
        times_h=t.tolist(),
        plasma_conc=cp.tolist(),
        lymph_conc=cl_lym.tolist(),
        tmax_h=tmax_h,
        cmax_mg_L=cmax,
    )


def compare_food_effect(
    drug_name: str,
    logP: float,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float = 1.0,
    fup: float = 1.0,
    solubility_fasted_mg_mL: float = 1.0,
    solubility_fed_mg_mL: float = 0.05,  # fat meal reduces aqueous solubility
    lymph_flow_fasted_L_per_h: float = 0.05,
    lymph_flow_fed_L_per_h: float = 0.25,  # fat meal increases lymph flow ~5x
) -> dict[str, LymphaticAbsorptionResult]:
    """Compare fasted vs fed lymphatic absorption (food effect on lipophilic drugs)."""
    fasted = simulate_lymphatic_absorption(
        drug_name=drug_name,
        logP=logP,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ka_per_h=ka_per_h,
        fup=fup,
        solubility_mg_mL=solubility_fasted_mg_mL,
        lymph_flow_L_per_h=lymph_flow_fasted_L_per_h,
    )
    fed = simulate_lymphatic_absorption(
        drug_name=drug_name,
        logP=logP,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        ka_per_h=ka_per_h,
        fup=fup,
        solubility_mg_mL=solubility_fed_mg_mL,
        lymph_flow_L_per_h=lymph_flow_fed_L_per_h,
    )
    return {"fasted": fasted, "fed": fed}


__all__ = [
    "LymphaticAbsorptionResult",
    "simulate_lymphatic_absorption",
    "compare_food_effect",
]
