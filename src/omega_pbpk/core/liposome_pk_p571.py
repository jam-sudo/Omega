"""Phase 571 — Liposome Drug Encapsulation PK model."""

import math
from dataclasses import dataclass


@dataclass
class LiposomePKResult:
    """Result of liposomal drug PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_encapsulated_mg_L: list
    c_free_mg_L: list
    c_tissue_mg_L: list
    cmax_free_mg_L: float
    tmax_free_h: float
    auc_encapsulated_mg_h_per_L: float
    auc_free_mg_h_per_L: float
    t_half_liposome_h: float
    drug_release_time_h: float
    pegylated: bool
    notes: str


def simulate_liposome_pk(
    drug_name: str,
    dose_mg: float,
    pegylated: bool,
    cl_liposome_L_per_h: float,
    cl_free_L_per_h: float,
    vd_liposome_L: float,
    vd_free_L: float,
    k_release_per_h: float = 0.05,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> LiposomePKResult:
    """Simulate PK of liposomal drug formulations.

    Models encapsulated drug, released free drug, and tissue distribution
    using Forward Euler integration.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_liposome_L_per_h <= 0:
        raise ValueError("cl_liposome_L_per_h must be > 0")
    if cl_free_L_per_h <= 0:
        raise ValueError("cl_free_L_per_h must be > 0")
    if vd_liposome_L <= 0:
        raise ValueError("vd_liposome_L must be > 0")
    if vd_free_L <= 0:
        raise ValueError("vd_free_L must be > 0")
    if k_release_per_h <= 0:
        raise ValueError("k_release_per_h must be > 0")

    cl_liposome_adj = cl_liposome_L_per_h * 0.25 if pegylated else cl_liposome_L_per_h

    k_in = 0.1  # tissue uptake rate (1/h)
    k_out = 0.05  # tissue efflux rate (1/h)

    # State variables
    A_enc = dose_mg  # encapsulated drug amount (mg)
    C_free = 0.0  # free drug plasma concentration (mg/L)
    C_tiss = 0.0  # tissue concentration (mg/L)

    times_h = []
    c_encapsulated_mg_L = []
    c_free_mg_L = []
    c_tissue_mg_L = []

    cumulative_released = 0.0
    drug_release_time_h = math.inf

    n_steps = int(t_end_h / dt_h)

    for i in range(n_steps + 1):
        t = i * dt_h
        times_h.append(t)
        c_enc = A_enc / vd_liposome_L
        c_encapsulated_mg_L.append(c_enc)
        c_free_mg_L.append(C_free)
        c_tissue_mg_L.append(C_tiss)

        if i < n_steps:
            release_amount = k_release_per_h * A_enc * dt_h
            cumulative_released += release_amount

            if drug_release_time_h == math.inf and cumulative_released >= 0.5 * dose_mg:
                drug_release_time_h = t + dt_h

            dA_enc = -(cl_liposome_adj / vd_liposome_L + k_release_per_h) * A_enc
            dC_free = (
                k_release_per_h * A_enc / vd_free_L
                - (cl_free_L_per_h / vd_free_L) * C_free
                - k_in * C_free
                + k_out * C_tiss
            )
            dC_tiss = k_in * C_free - k_out * C_tiss

            A_enc += dA_enc * dt_h
            C_free += dC_free * dt_h
            C_tiss += dC_tiss * dt_h

            if A_enc < 0:
                A_enc = 0.0
            if C_free < 0:
                C_free = 0.0
            if C_tiss < 0:
                C_tiss = 0.0

    # Cmax and Tmax for free drug
    cmax_free = max(c_free_mg_L)
    tmax_idx = c_free_mg_L.index(cmax_free)
    tmax_free = times_h[tmax_idx]

    # AUC via trapezoidal rule
    auc_enc = sum(
        0.5 * (c_encapsulated_mg_L[i] + c_encapsulated_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_free = sum(
        0.5 * (c_free_mg_L[i] + c_free_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    t_half_liposome = 0.693 * vd_liposome_L / cl_liposome_adj

    notes_parts = []
    if pegylated:
        notes_parts.append("PEGylated liposome (4x longer circulation)")
    else:
        notes_parts.append("Non-PEGylated liposome")
    notes_parts.append(f"Forward Euler dt={dt_h}h")

    return LiposomePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_encapsulated_mg_L=c_encapsulated_mg_L,
        c_free_mg_L=c_free_mg_L,
        c_tissue_mg_L=c_tissue_mg_L,
        cmax_free_mg_L=cmax_free,
        tmax_free_h=tmax_free,
        auc_encapsulated_mg_h_per_L=auc_enc,
        auc_free_mg_h_per_L=auc_free,
        t_half_liposome_h=t_half_liposome,
        drug_release_time_h=drug_release_time_h,
        pegylated=pegylated,
        notes="; ".join(notes_parts),
    )
