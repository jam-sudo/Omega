"""Antibiotic PK/PD in biofilm infections.

Models drug penetration into biofilms and predicts efficacy against
planktonic vs biofilm bacterial populations.

References
----------
Stewart 2002 (Nat Rev Microbiol), Muller 2007 (Clin Pharmacokinet).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "BiofilmPKResult",
    "simulate_biofilm_pk",
    "compare_dosing_regimens",
]

_KE_BIOFILM = 0.05  # slow elimination from biofilm (h^-1)


@dataclass(frozen=True)
class BiofilmPKResult:
    """Results from biofilm PK/PD simulation."""

    drug_name: str
    dose_mg: float
    n_doses: int
    route: str
    mic_mg_L: float
    mbec_mg_L: float
    biofilm_penetration: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_biofilm_mg_L: list[float]
    cmax_plasma: float
    cmax_biofilm: float
    auc_24h_plasma: float
    auc_24h_biofilm: float
    ft_above_mic_pct: float
    auc_mic_ratio: float
    auc_mbec_ratio: float
    cmax_mic_ratio: float
    planktonic_kill_expected: bool
    biofilm_eradication_expected: bool
    notes: str


def simulate_biofilm_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    mic_mg_L: float = 1.0,
    mbec_mg_L: float = 100.0,
    biofilm_penetration: float = 0.05,
    biofilm_volume_L: float = 0.001,
    dosing_interval_h: float = 8.0,
    n_doses: int = 3,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> BiofilmPKResult:
    """Simulate antibiotic PK in plasma and biofilm compartments.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose per administration (mg).
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).
    route : 'iv' or 'oral'.
    ka_per_h : Absorption rate constant for oral (h^-1).
    f_oral : Oral bioavailability (0-1).
    mic_mg_L : Planktonic MIC (mg/L).
    mbec_mg_L : Minimum biofilm eradication concentration (mg/L).
    biofilm_penetration : Fraction of drug reaching biofilm (0-1).
    biofilm_volume_L : Biofilm compartment volume (L).
    dosing_interval_h : Hours between doses.
    n_doses : Number of doses.
    t_end_h : Simulation end time (h).
    dt_h : Time step (h).

    Returns
    -------
    BiofilmPKResult

    Raises
    ------
    ValueError
        If parameters are invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if mic_mg_L <= 0:
        raise ValueError("mic_mg_L must be > 0")
    if mbec_mg_L <= 0:
        raise ValueError("mbec_mg_L must be > 0")
    if biofilm_penetration < 0 or biofilm_penetration > 1:
        raise ValueError("biofilm_penetration must be in [0, 1]")

    ke = cl_L_per_h / vd_L
    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    c_plasma = np.zeros(n_steps + 1)
    c_biofilm = np.zeros(n_steps + 1)
    depot = np.zeros(n_steps + 1)

    # Dose times
    dose_times = [i * dosing_interval_h for i in range(n_doses)]

    # Apply initial IV dose or oral depot
    for dt_dose in dose_times:
        idx = int(round(dt_dose / dt_h))
        if idx <= n_steps:
            if route.lower() == "iv":
                c_plasma[idx] += dose_mg / vd_L
            else:
                depot[idx] += dose_mg * f_oral

    # ODE integration
    for i in range(n_steps):
        # Absorption from depot (oral)
        abs_rate = ka_per_h * depot[i] if route.lower() != "iv" else 0.0

        dc_plasma = (abs_rate / vd_L - ke * c_plasma[i]) * dt_h
        dc_biofilm = (
            biofilm_penetration * c_plasma[i] / biofilm_volume_L - _KE_BIOFILM * c_biofilm[i]
        ) * dt_h
        d_depot = -ka_per_h * depot[i] * dt_h if route.lower() != "iv" else 0.0

        c_plasma[i + 1] += max(0.0, c_plasma[i] + dc_plasma)
        c_biofilm[i + 1] += max(0.0, c_biofilm[i] + dc_biofilm)
        if route.lower() != "iv":
            depot[i + 1] += max(0.0, depot[i] + d_depot)

    # PK/PD indices
    cmax_plasma = float(np.max(c_plasma))
    cmax_biofilm = float(np.max(c_biofilm))

    auc_plasma = float(np_trapz(c_plasma, t))
    auc_biofilm = float(np_trapz(c_biofilm, t))

    # fT > MIC (fraction of time plasma exceeds MIC)
    above_mic = np.sum(c_plasma > mic_mg_L)
    ft_above_mic_pct = float(above_mic / len(c_plasma) * 100)

    auc_mic_ratio = auc_plasma / mic_mg_L if mic_mg_L > 0 else 0.0
    auc_mbec_ratio = auc_biofilm / mbec_mg_L if mbec_mg_L > 0 else 0.0
    cmax_mic_ratio = cmax_plasma / mic_mg_L if mic_mg_L > 0 else 0.0

    planktonic_kill = ft_above_mic_pct > 40 or auc_mic_ratio > 25
    biofilm_eradication = auc_mbec_ratio > 10

    notes = (
        f"Biofilm PK for {drug_name}: {n_doses}x{dose_mg}mg {route}. "
        f"Cmax_plasma={cmax_plasma:.2f}, AUC/MIC={auc_mic_ratio:.1f}, "
        f"AUC/MBEC={auc_mbec_ratio:.3f}."
    )

    return BiofilmPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        n_doses=n_doses,
        route=route,
        mic_mg_L=mic_mg_L,
        mbec_mg_L=mbec_mg_L,
        biofilm_penetration=biofilm_penetration,
        times_h=t.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        c_biofilm_mg_L=c_biofilm.tolist(),
        cmax_plasma=cmax_plasma,
        cmax_biofilm=cmax_biofilm,
        auc_24h_plasma=auc_plasma,
        auc_24h_biofilm=auc_biofilm,
        ft_above_mic_pct=ft_above_mic_pct,
        auc_mic_ratio=auc_mic_ratio,
        auc_mbec_ratio=auc_mbec_ratio,
        cmax_mic_ratio=cmax_mic_ratio,
        planktonic_kill_expected=planktonic_kill,
        biofilm_eradication_expected=biofilm_eradication,
        notes=notes,
    )


def compare_dosing_regimens(
    drug_name: str,
    cl_L_per_h: float,
    vd_L: float,
    regimens: list[dict],
    **kwargs,
) -> list[BiofilmPKResult]:
    """Compare multiple dosing regimens for biofilm PK.

    Parameters
    ----------
    drug_name : Drug name.
    cl_L_per_h : Clearance (L/h).
    vd_L : Volume of distribution (L).
    regimens : List of dicts with keys 'dose_mg', 'n_doses', 'dosing_interval_h'.
    **kwargs : Additional arguments passed to simulate_biofilm_pk.

    Returns
    -------
    list[BiofilmPKResult]
    """
    return [
        simulate_biofilm_pk(
            drug_name=drug_name,
            dose_mg=r["dose_mg"],
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            n_doses=r.get("n_doses", 3),
            dosing_interval_h=r.get("dosing_interval_h", 8.0),
            **kwargs,
        )
        for r in regimens
    ]
