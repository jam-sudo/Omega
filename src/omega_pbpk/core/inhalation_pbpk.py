"""Inhalation route PBPK model for inhaled drugs (anesthetics, bronchodilators).

3-compartment Forward Euler ODE:
- Lung (alveolar): receives inhaled dose, transfers to blood
- Central plasma
- Peripheral tissue
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class InhalationPKResult:
    """Result of an inhalation PK simulation."""

    drug_name: str
    dose_mg: float
    f_lung_abs: float
    times_h: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_h: float
    f_systemic: float  # total systemic fraction (lung + oral routes)
    lung_auc: float  # AUC in lung compartment
    notes: str


def _validate_inhalation_inputs(
    dose_mg: float,
    f_lung_abs: float,
    cl_L_per_h: float,
    vd_L: float,
) -> None:
    """Validate simulation inputs, raising ValueError on invalid values."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if f_lung_abs <= 0 or f_lung_abs > 1:
        raise ValueError(f"f_lung_abs must be in (0, 1], got {f_lung_abs}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")


def simulate_inhalation_pk(
    drug_name: str,
    dose_mg: float,
    f_lung_abs: float = 0.9,
    k_lung_abs_per_h: float = 10.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_swallowed: float = 0.1,
    f_oral_swallowed: float = 0.2,
    t_end_h: float = 8.0,
    dt_h: float = 0.01,
) -> InhalationPKResult:
    """Simulate inhalation PK using a 3-compartment Forward Euler ODE.

    Compartments:
    - A_lung: mass of drug remaining in lung (mg)
    - C_plasma: plasma concentration (mg/L)

    Oral swallowed fraction is delivered as zero-order input over the first 2 h.
    """
    _validate_inhalation_inputs(dose_mg, f_lung_abs, cl_L_per_h, vd_L)

    # Initial conditions
    a_lung_0 = dose_mg * f_lung_abs  # mg absorbed via lung
    a_oral_total = dose_mg * f_swallowed * f_oral_swallowed  # mg available from swallowed

    # Zero-order oral absorption over 2 h
    oral_duration_h = 2.0
    k_oral_mg_per_h = a_oral_total / oral_duration_h  # mg/h

    # Elimination rate constant
    ke = cl_L_per_h / vd_L  # per h

    # State variables
    a_lung = a_lung_0
    c_plasma = 0.0

    times_h = []
    c_plasma_list = []
    lung_conc_list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        lung_conc_list.append(a_lung)

        # Oral input (zero-order, active for first oral_duration_h)
        oral_input = k_oral_mg_per_h if t < oral_duration_h else 0.0

        # ODEs
        da_lung_dt = -k_lung_abs_per_h * a_lung
        dc_plasma_dt = k_lung_abs_per_h * a_lung / vd_L - ke * c_plasma + oral_input / vd_L

        # Forward Euler update
        a_lung = a_lung + da_lung_dt * dt_h
        if a_lung < 0.0:
            a_lung = 0.0
        c_plasma = c_plasma + dc_plasma_dt * dt_h
        if c_plasma < 0.0:
            c_plasma = 0.0

        t += dt_h

    # PK metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_h = times_h[c_plasma_list.index(cmax_mg_L)]

    # AUC via manual trapezoidal rule
    auc = 0.0
    for i in range(len(times_h) - 1):
        auc += (c_plasma_list[i] + c_plasma_list[i + 1]) / 2.0 * (times_h[i + 1] - times_h[i])

    # Lung AUC (mass-based, mg*h)
    lung_auc = 0.0
    for i in range(len(times_h) - 1):
        lung_auc += (
            (lung_conc_list[i] + lung_conc_list[i + 1]) / 2.0 * (times_h[i + 1] - times_h[i])
        )

    # Half-life from ke
    t_half_h = math.log(2.0) / ke if ke > 0 else float("inf")

    # Total systemic fraction
    f_systemic = f_lung_abs + f_swallowed * f_oral_swallowed
    f_systemic = min(f_systemic, 1.0)

    notes = (
        f"Inhalation PBPK simulation for {drug_name}. "
        f"Lung absorbed fraction={f_lung_abs:.2f}, "
        f"oral swallowed contribution={f_swallowed * f_oral_swallowed:.3f}. "
        f"ke={ke:.3f}/h, t_half={t_half_h:.2f}h."
    )

    return InhalationPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_lung_abs=f_lung_abs,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        t_half_h=t_half_h,
        f_systemic=f_systemic,
        lung_auc=lung_auc,
        notes=notes,
    )


def compare_inhalation_devices(
    drug_name: str,
    dose_mg: float,
    device_params: list[dict],
) -> list[InhalationPKResult]:
    """Compare multiple inhaler devices (DPI, MDI, nebulizer) with different f_lung_abs.

    Each dict in device_params should contain keyword arguments for simulate_inhalation_pk.
    The drug_name and dose_mg from the outer call are used as defaults if not overridden.
    """
    results = []
    for params in device_params:
        merged = {"drug_name": drug_name, "dose_mg": dose_mg}
        merged.update(params)
        result = simulate_inhalation_pk(**merged)
        results.append(result)
    return results
