"""Phase 566 - Antibody Drug Conjugate (ADC) PK Model.

Models pharmacokinetics of an ADC: intact ADC, released payload,
and DAR (drug-antibody ratio) dynamics.
"""

from dataclasses import dataclass


@dataclass
class ADCPKResult:
    drug_name: str
    dose_mg_per_kg: float
    dar_initial: float
    times_h: list
    c_adc_mg_L: list
    c_payload_mg_L: list
    dar_profile: list
    cmax_adc_mg_L: float
    cmax_payload_mg_L: float
    auc_adc_mg_h_per_L: float
    auc_payload_mg_h_per_L: float
    t_half_adc_h: float
    t_half_payload_h: float
    therapeutic_index_estimate: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_adc_pk(
    drug_name: str,
    dose_mg_per_kg: float,
    weight_kg: float,
    dar_initial: float,
    cl_adc_L_per_h: float,
    vd_adc_L: float,
    k_deconjugation_per_h: float,
    cl_payload_L_per_h: float,
    vd_payload_L: float,
    t_end_h: float = 504.0,
    dt_h: float = 1.0,
) -> ADCPKResult:
    if dose_mg_per_kg <= 0:
        raise ValueError("dose_mg_per_kg must be > 0")
    if dar_initial <= 0:
        raise ValueError("dar_initial must be > 0")
    if cl_adc_L_per_h <= 0:
        raise ValueError("cl_adc_L_per_h must be > 0")
    if vd_adc_L <= 0:
        raise ValueError("vd_adc_L must be > 0")
    if k_deconjugation_per_h <= 0:
        raise ValueError("k_deconjugation_per_h must be > 0")
    if cl_payload_L_per_h <= 0:
        raise ValueError("cl_payload_L_per_h must be > 0")
    if vd_payload_L <= 0:
        raise ValueError("vd_payload_L must be > 0")

    # Initial conditions
    total_dose_mg = dose_mg_per_kg * weight_kg
    c_adc = total_dose_mg / vd_adc_L
    c_adc_0 = c_adc
    c_payload = 0.0

    # Rate constants
    k_el_adc = cl_adc_L_per_h / vd_adc_L
    k_el_payload = cl_payload_L_per_h / vd_payload_L

    # Half-lives
    t_half_adc = 0.693 * vd_adc_L / (cl_adc_L_per_h + k_deconjugation_per_h * vd_adc_L)
    t_half_payload = 0.693 * vd_payload_L / cl_payload_L_per_h

    times = []
    c_adc_list = []
    c_payload_list = []
    dar_list = []

    n_steps = int(t_end_h / dt_h) + 1
    t = 0.0

    cmax_adc = 0.0
    cmax_payload = 0.0

    for _ in range(n_steps):
        times.append(t)
        c_adc_list.append(c_adc)
        c_payload_list.append(c_payload)

        # DAR profile
        if c_adc_0 > 0:
            dar = dar_initial * c_adc / c_adc_0
        else:
            dar = 0.0
        dar_list.append(dar)

        if c_adc > cmax_adc:
            cmax_adc = c_adc
        if c_payload > cmax_payload:
            cmax_payload = c_payload

        # Derivatives
        d_adc = -(k_el_adc + k_deconjugation_per_h) * c_adc
        d_payload = (
            k_deconjugation_per_h * dar_initial * c_adc * vd_adc_L / vd_payload_L
            - k_el_payload * c_payload
        )

        # Forward Euler
        c_adc += d_adc * dt_h
        c_payload += d_payload * dt_h

        if c_adc < 0:
            c_adc = 0.0
        if c_payload < 0:
            c_payload = 0.0

        t += dt_h

    auc_adc = _trapezoidal_auc(times, c_adc_list)
    auc_payload = _trapezoidal_auc(times, c_payload_list)

    therapeutic_index = cmax_payload / cmax_adc if cmax_adc > 0 else 0.0

    return ADCPKResult(
        drug_name=drug_name,
        dose_mg_per_kg=dose_mg_per_kg,
        dar_initial=dar_initial,
        times_h=times,
        c_adc_mg_L=c_adc_list,
        c_payload_mg_L=c_payload_list,
        dar_profile=dar_list,
        cmax_adc_mg_L=cmax_adc,
        cmax_payload_mg_L=cmax_payload,
        auc_adc_mg_h_per_L=auc_adc,
        auc_payload_mg_h_per_L=auc_payload,
        t_half_adc_h=t_half_adc,
        t_half_payload_h=t_half_payload,
        therapeutic_index_estimate=therapeutic_index,
        notes=f"ADC PK model for {drug_name} (DAR={dar_initial})",
    )
