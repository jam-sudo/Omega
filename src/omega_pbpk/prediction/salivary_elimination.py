"""Salivary drug elimination model — Phase 1100.

Predicts drug concentration in saliva using pH-dependent Henderson-Hasselbalch
partitioning. Useful for non-invasive therapeutic drug monitoring and forensic
toxicology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_IONIZATION = frozenset({"acid", "base", "neutral"})


def _trapz(times: list[float], values: list[float]) -> float:
    """Trapezoidal integration."""
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        total += 0.5 * (values[i] + values[i - 1]) * dt
    return total


def _compute_sp_ratio(
    pka: float,
    ionization_type: str,
    fu_plasma: float,
    ph_saliva: float,
    ph_plasma: float,
) -> float:
    """Compute saliva/plasma concentration ratio using Henderson-Hasselbalch.

    For basic drugs: ion-trapping in saliva (lower pH → more ionized in saliva).
    For acid drugs: ion-trapping in plasma (higher pH → more ionized in plasma).
    For neutral: only unbound drug crosses → S/P = fu_plasma.
    """
    if ionization_type == "neutral":
        return fu_plasma
    elif ionization_type == "base":
        # Ratio of total (ionized + un-ionized) in saliva vs plasma
        # S/P = fu * (1 + 10^(pKa - pH_saliva)) / (1 + 10^(pKa - pH_plasma))
        numerator = 1.0 + math.pow(10.0, pka - ph_saliva)
        denominator = 1.0 + math.pow(10.0, pka - ph_plasma)
        return fu_plasma * numerator / denominator
    else:  # acid
        # S/P = fu * (1 + 10^(pH_saliva - pKa)) / (1 + 10^(pH_plasma - pKa))
        numerator = 1.0 + math.pow(10.0, ph_saliva - pka)
        denominator = 1.0 + math.pow(10.0, ph_plasma - pka)
        return fu_plasma * numerator / denominator


def _monitoring_utility(sp_ratio: float) -> str:
    """Classify monitoring utility based on S/P ratio."""
    if sp_ratio >= 0.8:
        return "good"
    elif sp_ratio >= 0.3:
        return "moderate"
    else:
        return "poor"


@dataclass(frozen=True)
class SalivaryEliminationResult:
    """Result from salivary concentration prediction — Phase 1100."""

    drug_name: str
    pka: float
    ionization_type: str
    ph_saliva: float
    ph_plasma: float
    fu_plasma: float
    sp_ratio: float
    c_plasma_mg_L: float
    c_saliva_mg_L: float
    sp_ratio_measured: float
    monitoring_utility: str
    forensic_window_h: float
    notes: str


def predict_salivary_concentration(
    drug_name: str,
    c_plasma_mg_L: float,
    pka: float,
    ionization_type: str,
    fu_plasma: float = 0.5,
    ph_saliva: float = 6.8,
    ph_plasma: float = 7.4,
) -> SalivaryEliminationResult:
    """Predict saliva concentration at a given plasma concentration.

    Args:
        drug_name: Name of the drug.
        c_plasma_mg_L: Plasma drug concentration (mg/L).
        pka: Drug pKa value.
        ionization_type: "acid", "base", or "neutral".
        fu_plasma: Unbound fraction in plasma (0-1].
        ph_saliva: Saliva pH (default 6.8).
        ph_plasma: Plasma pH (default 7.4).

    Returns:
        SalivaryEliminationResult with predicted saliva concentration.
    """
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION)}, got '{ionization_type}'"
        )
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if c_plasma_mg_L < 0:
        raise ValueError(f"c_plasma_mg_L must be >= 0, got {c_plasma_mg_L}")
    if not (5.0 <= ph_saliva <= 8.0):
        raise ValueError(f"ph_saliva must be in [5, 8], got {ph_saliva}")
    if not (7.0 <= ph_plasma <= 8.0):
        raise ValueError(f"ph_plasma must be in [7, 8], got {ph_plasma}")

    sp_ratio = _compute_sp_ratio(pka, ionization_type, fu_plasma, ph_saliva, ph_plasma)
    c_saliva = c_plasma_mg_L * sp_ratio

    # Measured S/P ratio from concentrations
    if c_plasma_mg_L > 0:
        sp_ratio_measured = c_saliva / c_plasma_mg_L
    else:
        sp_ratio_measured = sp_ratio  # theoretical when plasma is 0

    utility = _monitoring_utility(sp_ratio)

    # Forensic window: rough estimate based on monitoring utility
    # good S/P → reliable across entire elimination; moderate/poor → narrower window
    if utility == "good":
        forensic_window_h = 12.0
    elif utility == "moderate":
        forensic_window_h = 6.0
    else:
        forensic_window_h = 2.0

    notes_parts = [
        f"S/P={sp_ratio:.3f}",
        f"ionization={ionization_type}",
        f"pKa={pka}",
        f"pH_saliva={ph_saliva}",
        f"pH_plasma={ph_plasma}",
    ]
    notes = "; ".join(notes_parts)

    return SalivaryEliminationResult(
        drug_name=drug_name,
        pka=pka,
        ionization_type=ionization_type,
        ph_saliva=ph_saliva,
        ph_plasma=ph_plasma,
        fu_plasma=fu_plasma,
        sp_ratio=sp_ratio,
        c_plasma_mg_L=c_plasma_mg_L,
        c_saliva_mg_L=c_saliva,
        sp_ratio_measured=sp_ratio_measured,
        monitoring_utility=utility,
        forensic_window_h=forensic_window_h,
        notes=notes,
    )


def simulate_salivary_profile(
    drug_name: str,
    dose_mg: float,
    pka: float,
    ionization_type: str,
    fu_plasma: float = 0.5,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ka: float = 1.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
    ph_saliva: float = 6.8,
    ph_plasma: float = 7.4,
) -> dict:
    """Simulate plasma and saliva concentration-time profiles.

    One-compartment oral PK model with salivary lag of ~5 min (0.083 h).

    Args:
        drug_name: Name of the drug.
        dose_mg: Oral dose in mg.
        pka: Drug pKa.
        ionization_type: "acid", "base", or "neutral".
        fu_plasma: Unbound fraction in plasma (0-1].
        cl_L_per_h: Systemic clearance (L/h).
        vd_L: Volume of distribution (L).
        ka: Absorption rate constant (1/h).
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).
        ph_saliva: Saliva pH.
        ph_plasma: Plasma pH.

    Returns:
        Dict with keys: times_h, c_plasma, c_saliva, sp_ratio.
    """
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION)}, got '{ionization_type}'"
        )
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if not (5.0 <= ph_saliva <= 8.0):
        raise ValueError(f"ph_saliva must be in [5, 8], got {ph_saliva}")
    if not (7.0 <= ph_plasma <= 8.0):
        raise ValueError(f"ph_plasma must be in [7, 8], got {ph_plasma}")

    sp_ratio = _compute_sp_ratio(pka, ionization_type, fu_plasma, ph_saliva, ph_plasma)

    ke = cl_L_per_h / vd_L
    t_lag_saliva = 0.083  # ~5 minutes lag for salivary gland transit

    # Forward Euler simulation
    a_gut = dose_mg
    c_plasma_val = 0.0

    n_steps = int(t_end_h / dt_h) + 1
    times_h: list[float] = []
    c_plasma_list: list[float] = []

    t = 0.0
    for _ in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma_val)

        d_gut = -ka * a_gut
        d_plasma = (ka * a_gut / vd_L) - ke * c_plasma_val

        a_gut = max(0.0, a_gut + d_gut * dt_h)
        c_plasma_val = max(0.0, c_plasma_val + d_plasma * dt_h)
        t = round(t + dt_h, 10)

    # Compute saliva profile with t_lag offset
    # C_saliva(t) = sp_ratio * C_plasma(t - t_lag)
    c_saliva_list: list[float] = []
    lag_steps = int(t_lag_saliva / dt_h)

    for i in range(len(times_h)):
        plasma_idx = max(0, i - lag_steps)
        c_saliva_list.append(sp_ratio * c_plasma_list[plasma_idx])

    return {
        "times_h": times_h,
        "c_plasma": c_plasma_list,
        "c_saliva": c_saliva_list,
        "sp_ratio": sp_ratio,
    }
