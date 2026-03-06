"""Signal transduction cascade PD model.

Models the chain: drug concentration -> receptor occupancy -> second messenger
-> downstream response, with receptor desensitization/resensitization dynamics.
"""

from __future__ import annotations

__all__ = [
    "SignalTransductionResult",
    "simulate_signal_cascade",
    "compare_signal_drugs",
]

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalTransductionResult:
    drug_name: str
    drug_conc_mg_L: float
    times_h: list[float]
    receptor_occupancy_pct: list[float]  # effective occupancy (%)
    second_messenger_norm: list[float]  # normalized second messenger
    response_norm: list[float]  # normalized downstream response
    peak_response_pct: float
    time_to_peak_h: float
    sustained_response_pct: float  # response at t_end / emax (%)
    receptor_desensitization_pct: float  # (1 - r_active_final) * 100
    notes: str


def simulate_signal_cascade(
    drug_name: str,
    drug_conc_mg_L: float,
    ec50_mg_L: float,
    emax: float = 1.0,
    hill_n: float = 1.0,
    k_desensitize_per_h: float = 0.05,
    k_resensitize_per_h: float = 0.02,
    k_second_messenger_per_h: float = 2.0,
    k_degrade_sm_per_h: float = 1.0,
    k_response_per_h: float = 1.0,
    k_decay_response_per_h: float = 0.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> SignalTransductionResult:
    """Simulate a 3-state signal transduction cascade.

    States:
    - r_active: fraction of receptors available (desensitization dynamics)
    - sm: second messenger concentration (normalized)
    - resp: downstream response (normalized)

    Parameters
    ----------
    drug_name:
        Name of the drug.
    drug_conc_mg_L:
        Drug concentration in mg/L (>= 0).
    ec50_mg_L:
        EC50 in mg/L (> 0).
    emax:
        Maximum effect (arbitrary units, default 1.0).
    hill_n:
        Hill coefficient.
    k_desensitize_per_h:
        Receptor desensitization rate constant (1/h).
    k_resensitize_per_h:
        Receptor resensitization rate constant (1/h).
    k_second_messenger_per_h:
        Second messenger activation rate (1/h).
    k_degrade_sm_per_h:
        Second messenger degradation rate (1/h).
    k_response_per_h:
        Response activation rate (1/h).
    k_decay_response_per_h:
        Response decay rate (1/h).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Time step in hours.

    Returns
    -------
    SignalTransductionResult
    """
    if drug_conc_mg_L < 0.0:
        raise ValueError(f"drug_conc_mg_L must be >= 0, got {drug_conc_mg_L}")
    if ec50_mg_L <= 0.0:
        raise ValueError(f"ec50_mg_L must be > 0, got {ec50_mg_L}")

    # Fractional receptor occupancy (Hill equation, constant concentration)
    if drug_conc_mg_L == 0.0:
        ro_0 = 0.0
    else:
        ro_0 = (drug_conc_mg_L**hill_n) / (ec50_mg_L**hill_n + drug_conc_mg_L**hill_n)

    # Initial conditions
    r_active = 1.0
    sm = 0.0
    resp = 0.0
    ro = ro_0  # constant drug concentration assumption

    n_steps = max(1, int(t_end_h / dt_h))

    times: list[float] = []
    ro_pct_list: list[float] = []
    sm_list: list[float] = []
    resp_list: list[float] = []

    for i in range(n_steps + 1):
        t = i * dt_h
        times.append(t)
        ro_pct_list.append(ro * r_active * 100.0)
        sm_list.append(sm)
        resp_list.append(resp)

        if i < n_steps:
            dr_active = (
                -k_desensitize_per_h * r_active * ro + k_resensitize_per_h * (1.0 - r_active)
            ) * dt_h
            dsm = (k_second_messenger_per_h * r_active * ro - k_degrade_sm_per_h * sm) * dt_h
            dresp = (k_response_per_h * sm - k_decay_response_per_h * resp) * dt_h

            r_active = max(0.0, min(1.0, r_active + dr_active))
            sm = max(0.0, sm + dsm)
            resp = max(0.0, resp + dresp)

    # Normalize response by emax
    resp_norm = [r / emax for r in resp_list]

    # Peak response
    if resp_norm:
        peak_idx = resp_norm.index(max(resp_norm))
        peak_response_pct = max(resp_norm) * 100.0
        time_to_peak_h = times[peak_idx]
    else:
        peak_response_pct = 0.0
        time_to_peak_h = 0.0

    sustained_response_pct = resp_norm[-1] * 100.0 if resp_norm else 0.0
    receptor_desensitization_pct = (1.0 - r_active) * 100.0

    notes = (
        f"Hill occupancy={ro_0:.4f}; final r_active={r_active:.4f}; "
        f"peak at t={time_to_peak_h:.2f} h"
    )

    return SignalTransductionResult(
        drug_name=drug_name,
        drug_conc_mg_L=drug_conc_mg_L,
        times_h=times,
        receptor_occupancy_pct=ro_pct_list,
        second_messenger_norm=sm_list,
        response_norm=resp_norm,
        peak_response_pct=peak_response_pct,
        time_to_peak_h=time_to_peak_h,
        sustained_response_pct=sustained_response_pct,
        receptor_desensitization_pct=receptor_desensitization_pct,
        notes=notes,
    )


def compare_signal_drugs(
    drugs: list[dict],
) -> list[SignalTransductionResult]:
    """Simulate signal cascade for multiple drugs and sort by peak_response_pct descending.

    Each entry in ``drugs`` must have keys:
    - ``name`` (str)
    - ``conc_mg_L`` (float)
    - ``ec50_mg_L`` (float)
    - Any additional keyword arguments accepted by :func:`simulate_signal_cascade`.

    Returns
    -------
    list[SignalTransductionResult]
        Sorted by ``peak_response_pct`` descending.
    """
    results: list[SignalTransductionResult] = []
    for d in drugs:
        d = dict(d)
        name = d.pop("name")
        conc = d.pop("conc_mg_L")
        ec50 = d.pop("ec50_mg_L")
        result = simulate_signal_cascade(
            drug_name=name,
            drug_conc_mg_L=conc,
            ec50_mg_L=ec50,
            **d,
        )
        results.append(result)

    results.sort(key=lambda r: r.peak_response_pct, reverse=True)
    return results
