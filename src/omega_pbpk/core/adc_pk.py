"""Antibody-Drug Conjugate (ADC) pharmacokinetics model.

Models the complex PK of ADCs, which consist of:
- Intact ADC (antibody + payload linked via linker)
- Released free payload (formed by linker cleavage/deconjugation)
- Total antibody (ADC + deconjugated antibody)

ADC-specific considerations:
- Deconjugation rate depends on DAR (drug-antibody ratio)
- Higher DAR → faster deconjugation (hydrophobic payload increases clearance)
- Free payload has small-molecule PK (high CL, large Vd)
- ADC has antibody-like PK (low CL, small Vd, ~IgG)

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


@dataclass(frozen=True)
class ADCPKResult:
    """Results from ADC pharmacokinetic simulation.

    Attributes
    ----------
    drug_name : ADC drug name.
    dose_mg : Total dose administered (mg = dose_mg_per_kg * bw_kg).
    dar : Drug-antibody ratio used.
    bw_kg : Body weight (kg).
    times_h : Simulation time points (h).
    c_adc_mg_L : Intact ADC plasma concentration (mg/L).
    c_payload_mg_L : Free payload plasma concentration (mg/L).
    cmax_adc : Peak intact ADC concentration (mg/L).
    tmax_adc_h : Time of peak ADC concentration (h). Should be ~0 for IV.
    auc_adc : AUC of intact ADC (mg*h/L).
    cmax_payload : Peak free payload concentration (mg/L).
    tmax_payload_h : Time of peak payload concentration (h).
    auc_payload : AUC of free payload (mg*h/L).
    t_half_adc_h : Estimated ADC terminal half-life (h).
    payload_exposure_ratio : Ratio auc_payload / auc_adc.
    notes : Summary of simulation.
    """

    drug_name: str
    dose_mg: float
    dar: int
    bw_kg: float
    times_h: list[float]
    c_adc_mg_L: list[float]
    c_payload_mg_L: list[float]
    cmax_adc: float
    tmax_adc_h: float
    auc_adc: float
    cmax_payload: float
    tmax_payload_h: float
    auc_payload: float
    t_half_adc_h: float
    payload_exposure_ratio: float
    notes: str


def _estimate_half_life(times: np.ndarray, conc: np.ndarray) -> float:
    """Estimate terminal half-life from last 30% of concentration profile."""
    n = len(times)
    start = int(n * 0.7)
    t_tail = times[start:]
    c_tail = conc[start:]

    mask = c_tail > 1e-15
    if np.sum(mask) < 2:
        return float("nan")

    t_fit = t_tail[mask]
    ln_c = np.log(c_tail[mask])

    n_pts = len(t_fit)
    sum_t = np.sum(t_fit)
    sum_lnc = np.sum(ln_c)
    sum_t2 = np.sum(t_fit**2)
    sum_t_lnc = np.sum(t_fit * ln_c)

    denom = n_pts * sum_t2 - sum_t**2
    if abs(denom) < 1e-30:
        return float("nan")

    slope = (n_pts * sum_t_lnc - sum_t * sum_lnc) / denom
    if slope >= 0:
        return float("nan")

    ke_app = -slope
    return float(math.log(2) / ke_app)


def simulate_adc_pk(
    drug_name: str,
    dose_mg_per_kg: float,
    bw_kg: float = 70.0,
    dar: int = 4,
    cl_adc_L_per_h: float = 0.01,
    vd_adc_L: float = 4.0,
    k_deconj_per_h: float = 0.005,
    cl_payload_L_per_h: float = 5.0,
    vd_payload_L: float = 50.0,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> ADCPKResult:
    """Simulate ADC pharmacokinetics following IV bolus administration.

    Uses a 2-compartment Forward Euler model tracking:
    1. Intact ADC in plasma (cleared by antibody CL + deconjugation)
    2. Free payload in plasma (formed from ADC deconjugation + own CL)

    Parameters
    ----------
    drug_name : ADC drug name.
    dose_mg_per_kg : Dose in mg/kg body weight.
    bw_kg : Body weight (kg). Default 70.0.
    dar : Drug-antibody ratio (number of payload molecules per antibody).
        Typically 2-8. Default 4.
    cl_adc_L_per_h : ADC total clearance (L/h). Default 0.01 (IgG-like).
    vd_adc_L : ADC volume of distribution (L). Default 4.0 (IgG-like).
    k_deconj_per_h : Deconjugation rate constant (h^-1). Default 0.005.
    cl_payload_L_per_h : Free payload clearance (L/h). Default 5.0.
    vd_payload_L : Free payload volume of distribution (L). Default 50.0.
    t_end_h : Simulation end time (h). Default 720 (30 days).
    dt_h : Time step (h). Default 1.0.

    Returns
    -------
    ADCPKResult with full concentration-time profiles.

    Raises
    ------
    ValueError
        If input parameters violate constraints.
    """
    if dose_mg_per_kg <= 0:
        raise ValueError(f"dose_mg_per_kg must be > 0, got {dose_mg_per_kg}")
    if bw_kg <= 0:
        raise ValueError(f"bw_kg must be > 0, got {bw_kg}")
    if not (1 <= dar <= 8):
        raise ValueError(f"dar must be in [1, 8], got {dar}")
    if k_deconj_per_h <= 0:
        raise ValueError(f"k_deconj_per_h must be > 0, got {k_deconj_per_h}")
    if cl_adc_L_per_h <= 0:
        raise ValueError(f"cl_adc_L_per_h must be > 0, got {cl_adc_L_per_h}")
    if vd_adc_L <= 0:
        raise ValueError(f"vd_adc_L must be > 0, got {vd_adc_L}")
    if cl_payload_L_per_h <= 0:
        raise ValueError(f"cl_payload_L_per_h must be > 0, got {cl_payload_L_per_h}")
    if vd_payload_L <= 0:
        raise ValueError(f"vd_payload_L must be > 0, got {vd_payload_L}")

    dose_mg = dose_mg_per_kg * bw_kg

    # Rate constants
    ke_adc = cl_adc_L_per_h / vd_adc_L  # ADC elimination (non-deconj)
    ke_payload = cl_payload_L_per_h / vd_payload_L  # payload elimination

    # ADC total loss rate = elimination + deconjugation
    k_adc_total = ke_adc + k_deconj_per_h

    # IV bolus: initial ADC concentration
    c_adc_0 = dose_mg / vd_adc_L  # mg/L
    c_payload_0 = 0.0  # no free payload at t=0

    n_steps = int(t_end_h / dt_h) + 1
    times = np.linspace(0.0, t_end_h, n_steps)

    c_adc_arr = np.zeros(n_steps)
    c_payload_arr = np.zeros(n_steps)

    c_adc = c_adc_0
    c_payload = c_payload_0

    for i in range(n_steps):
        c_adc_arr[i] = c_adc
        c_payload_arr[i] = c_payload

        if i == n_steps - 1:
            break

        # ADC: lost by elimination and deconjugation
        d_adc = -k_adc_total * c_adc

        # Payload: gained from ADC deconjugation, lost by its own CL
        # Deconjugation releases DAR payload molecules per ADC molecule
        # Mass balance: rate of payload formation = k_deconj * C_adc * Vd_adc * dar
        # (normalised to payload Vd)
        payload_formation_rate = k_deconj_per_h * c_adc * vd_adc_L * dar / vd_payload_L
        d_payload = payload_formation_rate - ke_payload * c_payload

        c_adc = max(0.0, c_adc + d_adc * dt_h)
        c_payload = max(0.0, c_payload + d_payload * dt_h)

    # Compute summary statistics
    cmax_adc = float(np.max(c_adc_arr))
    idx_tmax_adc = int(np.argmax(c_adc_arr))
    tmax_adc_h = float(times[idx_tmax_adc])

    cmax_payload = float(np.max(c_payload_arr))
    idx_tmax_payload = int(np.argmax(c_payload_arr))
    tmax_payload_h = float(times[idx_tmax_payload])

    auc_adc = float(np_trapz(c_adc_arr, times))
    auc_payload = float(np_trapz(c_payload_arr, times))

    t_half_adc_h = _estimate_half_life(times, c_adc_arr)

    payload_exposure_ratio = auc_payload / auc_adc if auc_adc > 0 else 0.0

    notes = (
        f"{drug_name} ADC PK: DAR={dar}, dose={dose_mg:.1f}mg ({dose_mg_per_kg}mg/kg). "
        f"Cmax_ADC={cmax_adc:.3f}mg/L, Cmax_payload={cmax_payload:.3f}mg/L. "
        f"t½_ADC={t_half_adc_h:.1f}h, payload_exposure_ratio={payload_exposure_ratio:.3f}."
    )

    return ADCPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        dar=dar,
        bw_kg=bw_kg,
        times_h=times.tolist(),
        c_adc_mg_L=c_adc_arr.tolist(),
        c_payload_mg_L=c_payload_arr.tolist(),
        cmax_adc=cmax_adc,
        tmax_adc_h=tmax_adc_h,
        auc_adc=auc_adc,
        cmax_payload=cmax_payload,
        tmax_payload_h=tmax_payload_h,
        auc_payload=auc_payload,
        t_half_adc_h=t_half_adc_h,
        payload_exposure_ratio=payload_exposure_ratio,
        notes=notes,
    )


def compare_dar(
    drug_name: str,
    dose_mg_per_kg: float,
    dar_values: list[int],
    **kwargs,
) -> list[ADCPKResult]:
    """Compare ADC PK across different drug-antibody ratios.

    Higher DAR leads to faster deconjugation and therefore more free payload
    exposure. The deconjugation rate scales with DAR as:
        k_deconj_effective = k_deconj_base * (dar / 4)^0.5

    Parameters
    ----------
    drug_name : ADC drug name.
    dose_mg_per_kg : Dose in mg/kg body weight.
    dar_values : List of DAR values to compare (each must be in [1, 8]).
    **kwargs : Additional keyword arguments passed to simulate_adc_pk.
        Note: k_deconj_per_h in kwargs serves as the base rate for DAR=4.

    Returns
    -------
    List of ADCPKResult objects sorted by auc_payload in descending order.

    Raises
    ------
    ValueError
        If dar_values is empty or any DAR is not in [1, 8].
    """
    if not dar_values:
        raise ValueError("dar_values must not be empty")

    base_k_deconj = kwargs.pop("k_deconj_per_h", 0.005)

    results: list[ADCPKResult] = []
    for dar in dar_values:
        if not (1 <= dar <= 8):
            raise ValueError(f"dar must be in [1, 8], got {dar}")
        # Scale deconjugation rate with DAR relative to DAR=4
        k_deconj_effective = base_k_deconj * ((dar / 4.0) ** 0.5)
        result = simulate_adc_pk(
            drug_name=drug_name,
            dose_mg_per_kg=dose_mg_per_kg,
            dar=dar,
            k_deconj_per_h=k_deconj_effective,
            **kwargs,
        )
        results.append(result)

    # Sort by auc_payload descending (higher DAR → more payload exposure)
    results.sort(key=lambda r: r.auc_payload, reverse=True)
    return results
