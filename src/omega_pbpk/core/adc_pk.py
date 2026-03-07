"""Antibody-Drug Conjugate (ADC) pharmacokinetics — Phase 497.

Three-compartment model:
1. Intact ADC (plasma)
2. Deconjugated antibody (free Ab after linker cleavage)
3. Released payload (free small molecule)

ODE system (IV bolus at t=0):
    dC_ADC/dt     = -(k_deconj + ke_adc) * C_ADC
    dC_Ab_free/dt = k_deconj * C_ADC - ke_ab * C_Ab_free
    dC_payload/dt = dar * k_deconj * C_ADC * (Vd_adc/Vd_payload)
                    - ke_payload * C_payload

References
----------
- Shah DK & Betts AM. J Pharmacokinet Pharmacodyn 2012;39:67-86
- Gibiansky & Gibiansky, AAPS J. 2014;16:303-14
- Deslandes, Br J Clin Pharmacol. 2021;87:9-17
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "ADCPKResult",
    "simulate_adc_pk",
    "calculate_dar_distribution",
    "therapeutic_index_adc",
    "linker_stability",
]


@dataclass
class ADCPKResult:
    """Results from ADC PK simulation.

    Attributes
    ----------
    drug_name : Name of the ADC drug.
    dar : Drug-to-antibody ratio.
    times_days : Simulation time points (days).
    c_adc_mg_L : Intact ADC concentration (mg/L).
    c_antibody_free_mg_L : Deconjugated antibody concentration (mg/L).
    c_payload_mg_L : Free payload concentration (mg/L).
    cmax_adc_mg_L : Peak ADC concentration (mg/L).
    cmax_payload_mg_L : Peak payload concentration (mg/L).
    auc_adc_mg_day_L : AUC for ADC (mg*day/L).
    auc_payload_mg_day_L : AUC for payload (mg*day/L).
    t_half_adc_days : ADC terminal half-life (days).
    notes : Summary text.
    """

    drug_name: str
    dar: int
    times_days: list[float] = field(default_factory=list)
    c_adc_mg_L: list[float] = field(default_factory=list)
    c_antibody_free_mg_L: list[float] = field(default_factory=list)
    c_payload_mg_L: list[float] = field(default_factory=list)
    cmax_adc_mg_L: float = 0.0
    cmax_payload_mg_L: float = 0.0
    auc_adc_mg_day_L: float = 0.0
    auc_payload_mg_day_L: float = 0.0
    t_half_adc_days: float = 0.0
    notes: str = ""


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


def simulate_adc_pk(
    drug_name: str,
    dose_mg_per_kg: float,
    weight_kg: float = 70.0,
    dar: int = 4,
    cl_adc_L_per_day: float = 0.3,
    vd_adc_L: float = 3.0,
    k_deconj_per_day: float = 0.1,
    cl_payload_L_per_day: float = 15.0,
    vd_payload_L: float = 50.0,
    ka_payload_per_day: float = 0.0,
    t_end_days: float = 21.0,
    dt_days: float = 0.05,
) -> ADCPKResult:
    """Simulate ADC pharmacokinetics using a 3-compartment model (IV bolus).

    Parameters
    ----------
    drug_name : Name of the ADC.
    dose_mg_per_kg : Dose in mg/kg.
    weight_kg : Patient weight (kg).
    dar : Drug-to-antibody ratio (average payload per antibody).
    cl_adc_L_per_day : ADC clearance (L/day).
    vd_adc_L : ADC volume of distribution (L).
    k_deconj_per_day : Linker deconjugation rate constant (1/day).
    cl_payload_L_per_day : Payload clearance (L/day).
    vd_payload_L : Payload volume of distribution (L).
    ka_payload_per_day : Payload absorption rate (kept for API compatibility, unused for IV).
    t_end_days : Simulation end time (days).
    dt_days : Integration step size (days).

    Returns
    -------
    ADCPKResult
    """
    if dose_mg_per_kg <= 0:
        raise ValueError(f"dose_mg_per_kg must be > 0, got {dose_mg_per_kg}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if dar < 1:
        raise ValueError(f"dar must be >= 1, got {dar}")
    if cl_adc_L_per_day <= 0:
        raise ValueError(f"cl_adc_L_per_day must be > 0, got {cl_adc_L_per_day}")
    if vd_adc_L <= 0:
        raise ValueError(f"vd_adc_L must be > 0, got {vd_adc_L}")
    if k_deconj_per_day < 0:
        raise ValueError(f"k_deconj_per_day must be >= 0, got {k_deconj_per_day}")
    if cl_payload_L_per_day <= 0:
        raise ValueError(f"cl_payload_L_per_day must be > 0, got {cl_payload_L_per_day}")
    if vd_payload_L <= 0:
        raise ValueError(f"vd_payload_L must be > 0, got {vd_payload_L}")
    if t_end_days <= 0:
        raise ValueError(f"t_end_days must be > 0, got {t_end_days}")
    if dt_days <= 0:
        raise ValueError(f"dt_days must be > 0, got {dt_days}")

    total_dose_mg = dose_mg_per_kg * weight_kg

    # Elimination rate constants (1/day)
    ke_adc = cl_adc_L_per_day / vd_adc_L
    ke_ab = ke_adc  # deconjugated Ab retains antibody-like clearance
    ke_payload = cl_payload_L_per_day / vd_payload_L

    # Volume ratio for mass-balance scaling of payload release
    vol_ratio = vd_adc_L / vd_payload_L

    # Initial conditions: IV bolus for ADC, zero for others
    c_adc = total_dose_mg / vd_adc_L
    c_ab_free = 0.0
    c_payload = 0.0

    times: list[float] = [0.0]
    c_adc_list: list[float] = [c_adc]
    c_ab_list: list[float] = [c_ab_free]
    c_payload_list: list[float] = [c_payload]

    t = 0.0
    n_steps = int(round(t_end_days / dt_days))

    for _ in range(n_steps):
        dc_adc = -(k_deconj_per_day + ke_adc) * c_adc
        dc_ab = k_deconj_per_day * c_adc - ke_ab * c_ab_free
        dc_payload = (
            dar * k_deconj_per_day * c_adc * vol_ratio - ke_payload * c_payload
        )

        c_adc = max(0.0, c_adc + dc_adc * dt_days)
        c_ab_free = max(0.0, c_ab_free + dc_ab * dt_days)
        c_payload = max(0.0, c_payload + dc_payload * dt_days)

        t += dt_days
        times.append(t)
        c_adc_list.append(c_adc)
        c_ab_list.append(c_ab_free)
        c_payload_list.append(c_payload)

    cmax_adc = max(c_adc_list)
    cmax_payload = max(c_payload_list)
    auc_adc = _trapz(times, c_adc_list)
    auc_payload = _trapz(times, c_payload_list)

    # Analytical t_half for ADC
    k_total_adc = k_deconj_per_day + ke_adc
    t_half_adc = math.log(2) / k_total_adc if k_total_adc > 0 else float("inf")

    notes = (
        f"ADC: {drug_name}, DAR={dar}, Dose={total_dose_mg:.1f} mg, "
        f"t½_ADC={t_half_adc:.2f} d, Cmax_payload={cmax_payload:.4f} mg/L"
    )

    return ADCPKResult(
        drug_name=drug_name,
        dar=dar,
        times_days=times,
        c_adc_mg_L=c_adc_list,
        c_antibody_free_mg_L=c_ab_list,
        c_payload_mg_L=c_payload_list,
        cmax_adc_mg_L=cmax_adc,
        cmax_payload_mg_L=cmax_payload,
        auc_adc_mg_day_L=auc_adc,
        auc_payload_mg_day_L=auc_payload,
        t_half_adc_days=t_half_adc,
        notes=notes,
    )


def calculate_dar_distribution(
    initial_dar: float,
    k_deconj: float,
    t_days: float,
) -> dict[str, float]:
    """Calculate drug-to-antibody ratio distribution over time.

    Models payload loss from the ADC as a first-order process.
    The remaining average DAR decreases exponentially.

    Parameters
    ----------
    initial_dar : Starting average DAR (e.g., 4 or 8).
    k_deconj : Deconjugation rate constant (1/day).
    t_days : Time point (days).

    Returns
    -------
    dict with keys: 'mean_dar', 'fraction_remaining', 'fraction_deconjugated',
                    'initial_dar', 't_days'
    """
    if initial_dar < 0:
        raise ValueError(f"initial_dar must be >= 0, got {initial_dar}")
    if k_deconj < 0:
        raise ValueError(f"k_deconj must be >= 0, got {k_deconj}")
    if t_days < 0:
        raise ValueError(f"t_days must be >= 0, got {t_days}")

    fraction_remaining = math.exp(-k_deconj * t_days)
    fraction_deconjugated = 1.0 - fraction_remaining
    mean_dar = initial_dar * fraction_remaining

    return {
        "mean_dar": mean_dar,
        "fraction_remaining": fraction_remaining,
        "fraction_deconjugated": fraction_deconjugated,
        "initial_dar": initial_dar,
        "t_days": t_days,
    }


def therapeutic_index_adc(
    cmax_payload_mg_L: float,
    target_conc_mg_L: float,
    toxic_conc_mg_L: float,
) -> float:
    """Calculate the therapeutic index for an ADC payload.

    TI = toxic_conc / target_conc.

    Parameters
    ----------
    cmax_payload_mg_L : Achieved peak payload concentration (mg/L).
    target_conc_mg_L : Minimum effective payload concentration (mg/L).
    toxic_conc_mg_L : Toxic threshold payload concentration (mg/L).

    Returns
    -------
    float
        Therapeutic index (toxic_conc / target_conc). Values > 1 indicate margin.
    """
    if target_conc_mg_L <= 0:
        raise ValueError(f"target_conc_mg_L must be > 0, got {target_conc_mg_L}")
    if toxic_conc_mg_L <= 0:
        raise ValueError(f"toxic_conc_mg_L must be > 0, got {toxic_conc_mg_L}")
    if cmax_payload_mg_L < 0:
        raise ValueError(f"cmax_payload_mg_L must be >= 0, got {cmax_payload_mg_L}")

    return toxic_conc_mg_L / target_conc_mg_L


def linker_stability(
    linker_type: str,
    temperature_C: float = 37.0,
    ph: float = 7.4,
) -> float:
    """Estimate linker stability as fraction intact per day.

    Models plasma stability for three linker classes:
    - 'cleavable': acid-labile or protease-sensitive; moderately stable at neutral pH
    - 'non_cleavable': thioether-based; very stable across pH range
    - 'disulfide': pH/redox-sensitive; less stable at acidic pH

    Parameters
    ----------
    linker_type : One of 'cleavable', 'non_cleavable', 'disulfide'.
    temperature_C : Temperature (°C). Higher temperature decreases stability.
    ph : pH of the environment. Lower pH destabilizes cleavable/disulfide linkers.

    Returns
    -------
    float
        Fraction of linker intact per day (0–1). Higher = more stable.
    """
    valid_linker_types = {"cleavable", "non_cleavable", "disulfide"}
    if linker_type not in valid_linker_types:
        raise ValueError(
            f"linker_type must be one of {valid_linker_types}, got {linker_type!r}"
        )
    if temperature_C <= 0:
        raise ValueError(f"temperature_C must be > 0, got {temperature_C}")

    # Base stability at 37°C, pH 7.4 (plasma conditions)
    base_stability = {
        "non_cleavable": 0.98,  # Very stable; minimal plasma deconjugation
        "cleavable": 0.92,       # Moderately stable in plasma, cleaved intracellularly
        "disulfide": 0.85,       # Reduced by plasma thiols; pH-sensitive
    }

    stability = base_stability[linker_type]

    # Temperature correction: Q10 rule approximation (10°C → 2x degradation rate)
    temp_factor = 2.0 ** ((temperature_C - 37.0) / 10.0)
    degradation_rate = (1.0 - stability) * temp_factor
    stability = max(0.0, 1.0 - degradation_rate)

    # pH correction
    if linker_type == "cleavable":
        # Acid-labile linkers less stable at low pH
        if ph < 7.0:
            stability = stability * (0.7 + 0.3 * (ph / 7.0))
        elif ph > 7.5:
            stability = min(1.0, stability * 1.02)
    elif linker_type == "disulfide":
        # Disulfide bonds sensitive to reducing environments and low pH
        if ph < 7.0:
            stability = stability * (0.5 + 0.4 * (ph / 7.0))
    # non_cleavable: negligible pH effect

    return max(0.0, min(1.0, stability))
