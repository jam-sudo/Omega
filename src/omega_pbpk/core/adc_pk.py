"""Antibody-Drug Conjugate (ADC) pharmacokinetics model.

Models the complex PK of ADCs including:
- Linker stability and deconjugation kinetics
- DAR (drug-antibody ratio) distribution over time
- Payload release and independent small-molecule PK
- Total antibody (conjugated + unconjugated) tracking

ADC-specific considerations:
- Deconjugation rate depends on linker stability (k_deconj)
- Higher k_deconj → faster payload release
- Free payload has small-molecule PK (high CL, large Vd)
- ADC has antibody-like PK (low CL, small Vd, ~IgG)
- DAR decreases over time as payload deconjugates

References
----------
- Shah DK & Betts AM. J Pharmacokinet Pharmacodyn 2012;39:67-86
- Cilliers C et al. Clin Pharmacol Ther 2016;99:63-77
- Kamath AV. Drug Metab Dispos 2016;44:1958-64
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "ADCPKResult",
    "simulate_adc_pk",
    "compare_linker_stability",
]


@dataclass(frozen=True)
class ADCPKResult:
    """Results from ADC pharmacokinetic simulation.

    Attributes
    ----------
    drug_name : ADC drug name.
    total_antibody_dose_mg : Total antibody dose (mg).
    dar_initial : Drug-antibody ratio at time 0.
    times_h : Simulation time points (h).
    c_total_antibody_mg_L : Total antibody concentration (conjugated + unconjugated) (mg/L).
    c_conjugated_mg_L : Intact ADC (conjugated) concentration (mg/L).
    c_free_payload_mg_L : Free released payload concentration (mg/L).
    c_unconjugated_ab_mg_L : Naked antibody (unconjugated) concentration (mg/L).
    dar_mean_over_time : Mean DAR at each time point.
    auc_total_antibody : AUC of total antibody (mg*h/L).
    auc_conjugated : AUC of conjugated ADC (mg*h/L).
    auc_free_payload : AUC of free payload (mg*h/L).
    t_half_antibody_h : Antibody half-life (h).
    t_half_payload_h : Free payload half-life (h).
    payload_antibody_auc_ratio : Free payload AUC / total antibody AUC.
    fraction_deconjugated_by_24h : Fraction deconjugated at 24 h.
    fraction_deconjugated_by_168h : Fraction deconjugated at 168 h (1 week).
    notes : Simulation summary.
    """

    drug_name: str
    total_antibody_dose_mg: float
    dar_initial: float

    times_h: list[float]

    c_total_antibody_mg_L: list[float]
    c_conjugated_mg_L: list[float]
    c_free_payload_mg_L: list[float]
    c_unconjugated_ab_mg_L: list[float]

    dar_mean_over_time: list[float]

    auc_total_antibody: float
    auc_conjugated: float
    auc_free_payload: float

    t_half_antibody_h: float
    t_half_payload_h: float

    payload_antibody_auc_ratio: float

    fraction_deconjugated_by_24h: float
    fraction_deconjugated_by_168h: float

    notes: str


def simulate_adc_pk(
    drug_name: str,
    total_antibody_dose_mg: float,
    cl_antibody_L_per_day: float = 0.2,
    vd_antibody_L: float = 3.0,
    dar_initial: float = 4.0,
    k_deconj_per_day: float = 0.1,
    cl_payload_L_per_h: float = 20.0,
    vd_payload_L: float = 100.0,
    mw_antibody_kDa: float = 150.0,
    mw_payload_Da: float = 500.0,
    route: str = "iv",
    t_end_h: float = 672.0,
    dt_h: float = 1.0,
) -> ADCPKResult:
    """Simulate ADC pharmacokinetics following IV bolus administration.

    Uses a 3-compartment Forward Euler model tracking:
    1. Conjugated ADC in plasma
    2. Unconjugated (naked) antibody in plasma
    3. Free payload in plasma

    Parameters
    ----------
    drug_name : ADC drug name.
    total_antibody_dose_mg : Total antibody dose (mg).
    cl_antibody_L_per_day : Antibody clearance (L/day). Default 0.2.
    vd_antibody_L : Antibody volume of distribution (L). Default 3.0.
    dar_initial : Drug-antibody ratio at time 0. Default 4.0.
    k_deconj_per_day : Deconjugation rate constant (day^-1). Default 0.1.
    cl_payload_L_per_h : Payload clearance (L/h). Default 20.0.
    vd_payload_L : Payload volume of distribution (L). Default 100.0.
    mw_antibody_kDa : Antibody molecular weight (kDa). Default 150.0.
    mw_payload_Da : Payload molecular weight (Da). Default 500.0.
    route : Route of administration. Default "iv".
    t_end_h : Simulation end time (h). Default 672 (28 days).
    dt_h : Time step (h). Default 1.0.

    Returns
    -------
    ADCPKResult with full concentration-time profiles.

    Raises
    ------
    ValueError
        If input parameters violate constraints.
    """
    if total_antibody_dose_mg <= 0:
        raise ValueError(f"total_antibody_dose_mg must be > 0, got {total_antibody_dose_mg}")
    if cl_antibody_L_per_day <= 0:
        raise ValueError(f"cl_antibody_L_per_day must be > 0, got {cl_antibody_L_per_day}")
    if vd_antibody_L <= 0:
        raise ValueError(f"vd_antibody_L must be > 0, got {vd_antibody_L}")
    if dar_initial < 1.0:
        raise ValueError(f"dar_initial must be >= 1, got {dar_initial}")
    if k_deconj_per_day < 0:
        raise ValueError(f"k_deconj_per_day must be >= 0, got {k_deconj_per_day}")
    if cl_payload_L_per_h <= 0:
        raise ValueError(f"cl_payload_L_per_h must be > 0, got {cl_payload_L_per_h}")
    if vd_payload_L <= 0:
        raise ValueError(f"vd_payload_L must be > 0, got {vd_payload_L}")
    if mw_antibody_kDa <= 0:
        raise ValueError(f"mw_antibody_kDa must be > 0, got {mw_antibody_kDa}")
    if mw_payload_Da <= 0:
        raise ValueError(f"mw_payload_Da must be > 0, got {mw_payload_Da}")

    # Convert antibody PK to per-hour units
    cl_antibody_L_per_h = cl_antibody_L_per_day / 24.0
    k_deconj = k_deconj_per_day / 24.0  # h^-1

    # Rate constants
    ke_ab = cl_antibody_L_per_h / vd_antibody_L  # h^-1
    ke_payload = cl_payload_L_per_h / vd_payload_L  # h^-1

    # Payload mass release factor: payload released per unit mass of ADC deconjugated
    # moles_payload / moles_ADC = dar_initial
    # mass_payload = moles_payload * mw_payload_Da/1000 [kDa → g/mol cancel]
    # mass_ADC = moles_ADC * mw_antibody_kDa
    # ratio = dar_initial * mw_payload_Da / (mw_antibody_kDa * 1000)
    payload_mass_ratio = dar_initial * (mw_payload_Da / 1000.0) / mw_antibody_kDa

    # Initial conditions (IV bolus)
    c_adc0 = total_antibody_dose_mg / vd_antibody_L  # mg/L
    c_unconj0 = 0.0
    c_payload0 = 0.0

    n_steps = int(t_end_h / dt_h) + 1
    times = np.linspace(0.0, t_end_h, n_steps)

    c_adc_arr = np.zeros(n_steps)
    c_unconj_arr = np.zeros(n_steps)
    c_payload_arr = np.zeros(n_steps)

    c_adc = c_adc0
    c_unconj = c_unconj0
    c_payload = c_payload0

    for i in range(n_steps):
        c_adc_arr[i] = c_adc
        c_unconj_arr[i] = c_unconj
        c_payload_arr[i] = c_payload

        if i == n_steps - 1:
            break

        # ODEs (Forward Euler)
        # Conjugated ADC: lost by antibody elimination and deconjugation
        dc_adc = -(ke_ab + k_deconj) * c_adc

        # Unconjugated antibody: gained from deconjugation, lost by elimination
        dc_unconj = k_deconj * c_adc - ke_ab * c_unconj

        # Free payload: gained from ADC deconjugation, lost by payload CL
        # rate of payload formation (mg/L/h) = k_deconj * c_adc * (Vd_ab / Vd_payload) * mass_ratio
        payload_formation = k_deconj * c_adc * (vd_antibody_L / vd_payload_L) * payload_mass_ratio
        dc_payload = payload_formation - ke_payload * c_payload

        c_adc = max(0.0, c_adc + dc_adc * dt_h)
        c_unconj = max(0.0, c_unconj + dc_unconj * dt_h)
        c_payload = max(0.0, c_payload + dc_payload * dt_h)

    # Total antibody = conjugated + unconjugated
    c_total_ab_arr = c_adc_arr + c_unconj_arr

    # DAR over time: simplified exponential decay of DAR
    dar_arr = dar_initial * np.exp(-k_deconj * times)

    # AUC via trapezoidal
    auc_total_ab = float(np_trapz(c_total_ab_arr, times))
    auc_conjugated = float(np_trapz(c_adc_arr, times))
    auc_free_payload = float(np_trapz(c_payload_arr, times))

    # Half-lives
    t_half_antibody_h = math.log(2.0) / ke_ab if ke_ab > 0 else float("inf")
    t_half_payload_h = math.log(2.0) / ke_payload if ke_payload > 0 else float("inf")

    # Payload / antibody AUC ratio
    payload_antibody_auc_ratio = auc_free_payload / auc_total_ab if auc_total_ab > 0 else 0.0

    # Deconjugation fractions
    f_deconj_24h = 1.0 - math.exp(-k_deconj * 24.0)
    f_deconj_168h = 1.0 - math.exp(-k_deconj * 168.0)

    notes = (
        f"{drug_name} ADC PK: DAR={dar_initial:.1f}, dose={total_antibody_dose_mg:.1f} mg. "
        f"AUC_ADC={auc_conjugated:.1f} mg*h/L, AUC_payload={auc_free_payload:.3f} mg*h/L. "
        f"t½_antibody={t_half_antibody_h:.1f} h, t½_payload={t_half_payload_h:.2f} h. "
        f"Deconj@24h={f_deconj_24h:.1%}, Deconj@168h={f_deconj_168h:.1%}."
    )

    return ADCPKResult(
        drug_name=drug_name,
        total_antibody_dose_mg=total_antibody_dose_mg,
        dar_initial=dar_initial,
        times_h=times.tolist(),
        c_total_antibody_mg_L=c_total_ab_arr.tolist(),
        c_conjugated_mg_L=c_adc_arr.tolist(),
        c_free_payload_mg_L=c_payload_arr.tolist(),
        c_unconjugated_ab_mg_L=c_unconj_arr.tolist(),
        dar_mean_over_time=dar_arr.tolist(),
        auc_total_antibody=auc_total_ab,
        auc_conjugated=auc_conjugated,
        auc_free_payload=auc_free_payload,
        t_half_antibody_h=t_half_antibody_h,
        t_half_payload_h=t_half_payload_h,
        payload_antibody_auc_ratio=payload_antibody_auc_ratio,
        fraction_deconjugated_by_24h=f_deconj_24h,
        fraction_deconjugated_by_168h=f_deconj_168h,
        notes=notes,
    )


def compare_linker_stability(
    drug_name: str,
    total_antibody_dose_mg: float,
    k_deconj_values: list[float] | None = None,
    **kwargs,
) -> list[ADCPKResult]:
    """Compare ADC PK across different linker stabilities.

    Higher k_deconj → more payload release (less stable linker).
    Results are sorted by auc_conjugated descending (most stable linker first).

    Parameters
    ----------
    drug_name : ADC drug name.
    total_antibody_dose_mg : Total antibody dose (mg).
    k_deconj_values : List of deconjugation rate constants (day^-1).
        Default: [0.01, 0.05, 0.1, 0.3].
    **kwargs : Additional keyword arguments passed to simulate_adc_pk.

    Returns
    -------
    list[ADCPKResult]
        Results sorted by auc_conjugated descending (most stable first).

    Raises
    ------
    ValueError
        If k_deconj_values is empty.
    """
    if k_deconj_values is None:
        k_deconj_values = [0.01, 0.05, 0.1, 0.3]

    if not k_deconj_values:
        raise ValueError("k_deconj_values must not be empty")

    results: list[ADCPKResult] = []
    for k_val in k_deconj_values:
        result = simulate_adc_pk(
            drug_name=drug_name,
            total_antibody_dose_mg=total_antibody_dose_mg,
            k_deconj_per_day=k_val,
            **kwargs,
        )
        results.append(result)

    # Sort by auc_conjugated descending (most stable = highest conjugated AUC = first)
    results.sort(key=lambda r: r.auc_conjugated, reverse=True)
    return results
