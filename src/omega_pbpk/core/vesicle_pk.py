"""Phase 1046 — Drug Vesicle PK (liposome, exosome, niosome, transfersome)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_VESICLE_PARAMS: dict[str, tuple[float, float]] = {
    # type: (k_release /h, t_half_circ h)
    "liposome": (0.1, 24.0),
    "exosome": (0.05, 48.0),
    "niosome": (0.15, 18.0),
    "transfersome": (0.2, 12.0),
}


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal AUC."""
    if len(times) < 2:
        return 0.0
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i - 1] + values[i]) * (times[i] - times[i - 1])
    return auc


@dataclass
class VesiclePKResult:
    drug_name: str
    vesicle_type: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_encapsulated_mg_L: list = field(default_factory=list)
    c_free_mg_L: list = field(default_factory=list)
    encapsulation_efficiency: float = 0.0
    auc_free: float = 0.0
    auc_total: float = 0.0
    cmax_free: float = 0.0
    tmax_free_h: float = 0.0
    t_half_vesicle_h: float = 0.0
    release_rate_per_h: float = 0.0
    notes: str = ""


def simulate_vesicle_pk(
    drug_name: str,
    dose_mg: float,
    vesicle_type: str = "liposome",
    encapsulation_efficiency: float = 0.8,
    cl_free_L_per_h: float = 10.0,
    vd_free_L: float = 50.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> VesiclePKResult:
    """Simulate vesicle-encapsulated drug PK using forward Euler ODE."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if encapsulation_efficiency <= 0:
        raise ValueError("encapsulation_efficiency must be > 0")
    if encapsulation_efficiency > 1:
        raise ValueError("encapsulation_efficiency must be <= 1")
    if cl_free_L_per_h <= 0:
        raise ValueError("cl_free_L_per_h must be > 0")
    if vd_free_L <= 0:
        raise ValueError("vd_free_L must be > 0")
    if vesicle_type not in _VESICLE_PARAMS:
        raise ValueError(f"vesicle_type must be one of {sorted(_VESICLE_PARAMS.keys())}")

    k_release, t_half_circ = _VESICLE_PARAMS[vesicle_type]
    k_elim_vesicle = math.log(2) / t_half_circ

    # Initial conditions
    a_vesicle = dose_mg * encapsulation_efficiency  # mg in vesicle
    c_free = dose_mg * (1.0 - encapsulation_efficiency) / vd_free_L  # mg/L

    ke_free = cl_free_L_per_h / vd_free_L

    times: list[float] = []
    c_enc_list: list[float] = []
    c_free_list: list[float] = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _i in range(n_steps):
        # Record: convert a_vesicle to concentration equivalent (mg/L) via vd_free
        times.append(t)
        c_enc_list.append(a_vesicle / vd_free_L)
        c_free_list.append(c_free)

        if t >= t_end_h:
            break

        # Forward Euler
        da_dt = -(k_release + k_elim_vesicle) * a_vesicle
        dc_dt = k_release * a_vesicle / vd_free_L - ke_free * c_free

        a_vesicle += da_dt * dt_h
        c_free += dc_dt * dt_h

        # Clamp to zero to avoid negative values
        if a_vesicle < 0.0:
            a_vesicle = 0.0
        if c_free < 0.0:
            c_free = 0.0

        t += dt_h

    auc_free = _trapz(times, c_free_list)
    auc_total = _trapz(times, [e + f for e, f in zip(c_enc_list, c_free_list)])

    cmax_free = max(c_free_list)
    tmax_idx = c_free_list.index(cmax_free)
    tmax_free_h = times[tmax_idx]

    notes = (
        f"{vesicle_type.capitalize()} vesicle: k_release={k_release}/h,"
        f" t_half_circ={t_half_circ} h,"
        f" EE={encapsulation_efficiency:.0%}."
        f" AUC_free={auc_free:.2f} mg*h/L, Cmax_free={cmax_free:.4f} mg/L"
        f" at t={tmax_free_h:.1f} h."
    )

    return VesiclePKResult(
        drug_name=drug_name,
        vesicle_type=vesicle_type,
        dose_mg=dose_mg,
        times_h=times,
        c_encapsulated_mg_L=c_enc_list,
        c_free_mg_L=c_free_list,
        encapsulation_efficiency=encapsulation_efficiency,
        auc_free=auc_free,
        auc_total=auc_total,
        cmax_free=cmax_free,
        tmax_free_h=tmax_free_h,
        t_half_vesicle_h=t_half_circ,
        release_rate_per_h=k_release,
        notes=notes,
    )
