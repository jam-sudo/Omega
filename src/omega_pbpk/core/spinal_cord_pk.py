"""Phase 919 — Drug Spinal Cord Distribution.

Models drug distribution in spinal cord with three compartments:
- CSF (lumbar)
- Spinal cord tissue
- Systemic plasma

Relevant for intrathecal analgesics and neuropathic pain drugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpinalCordPKResult:
    """Result of spinal cord PK simulation."""

    drug_name: str
    dose_mg: float
    route: str  # "intrathecal" or "systemic"
    times_h: list[float] = field(default_factory=list)
    c_csf_mg_L: list[float] = field(default_factory=list)
    c_spinal_cord_mg_L: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    cmax_csf: float = 0.0
    tmax_csf_h: float = 0.0
    auc_csf: float = 0.0
    cmax_spinal_cord: float = 0.0
    auc_spinal_cord: float = 0.0
    spinal_to_plasma_ratio: float = 0.0
    analgesic_duration_h: float = 0.0
    notes: str = ""


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return auc


def simulate_spinal_cord_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "intrathecal",
    k_csf_to_cord_per_h: float = 0.5,
    k_cord_to_csf_per_h: float = 0.1,
    k_csf_drain_per_h: float = 0.15,
    cl_sys_L_per_h: float = 10.0,
    vd_sys_L: float = 50.0,
    v_csf_L: float = 0.14,
    v_cord_L: float = 0.03,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> SpinalCordPKResult:
    """Simulate drug distribution in spinal cord.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    route:
        Administration route: "intrathecal" or "systemic".
    k_csf_to_cord_per_h:
        First-order transfer rate from CSF to spinal cord (1/h).
    k_cord_to_csf_per_h:
        First-order transfer rate from spinal cord to CSF (1/h).
    k_csf_drain_per_h:
        First-order drainage rate of CSF to blood (1/h).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution for systemic compartment (L).
    v_csf_L:
        Volume of lumbar CSF (L).
    v_cord_L:
        Volume of spinal cord tissue (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration step size (h).

    Returns
    -------
    SpinalCordPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if route not in {"intrathecal", "systemic"}:
        raise ValueError(f"route must be 'intrathecal' or 'systemic', got '{route}'")

    ke_sys = cl_sys_L_per_h / vd_sys_L  # systemic elimination rate (1/h)
    k_cord_elim = 0.05  # spinal cord elimination rate (1/h)

    # Initial amounts (mg)
    if route == "intrathecal":
        a_csf = dose_mg
        a_cord = 0.0
        a_plasma = 0.0
    else:
        a_csf = 0.0
        a_cord = 0.0
        a_plasma = dose_mg

    times: list[float] = []
    c_csf_list: list[float] = []
    c_cord_list: list[float] = []
    c_plasma_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        c_csf = a_csf / v_csf_L
        c_cord = a_cord / v_cord_L
        c_plasma = a_plasma / vd_sys_L

        times.append(t)
        c_csf_list.append(c_csf)
        c_cord_list.append(c_cord)
        c_plasma_list.append(c_plasma)

        if t >= t_end_h:
            break

        # ODEs
        if route == "intrathecal":
            bbb_in = 0.0
            da_csf = (
                -k_csf_drain_per_h * a_csf
                - k_csf_to_cord_per_h * a_csf
                + k_cord_to_csf_per_h * a_cord
                + bbb_in
            )
            da_cord = (
                k_csf_to_cord_per_h * a_csf - k_cord_to_csf_per_h * a_cord - k_cord_elim * a_cord
            )
            # CSF drains into systemic blood, minus systemic elimination
            da_plasma = k_csf_drain_per_h * a_csf - ke_sys * a_plasma
        else:
            # systemic route: 1% of plasma amount crosses into CSF
            bbb_in = 0.01 * a_plasma
            da_csf = (
                -k_csf_drain_per_h * a_csf
                - k_csf_to_cord_per_h * a_csf
                + k_cord_to_csf_per_h * a_cord
                + bbb_in
            )
            da_cord = (
                k_csf_to_cord_per_h * a_csf - k_cord_to_csf_per_h * a_cord - k_cord_elim * a_cord
            )
            # systemic: eliminate + BBB transfer out
            da_plasma = -ke_sys * a_plasma - 0.01 * a_plasma

        # Forward Euler
        a_csf = max(0.0, a_csf + da_csf * dt_h)
        a_cord = max(0.0, a_cord + da_cord * dt_h)
        a_plasma = max(0.0, a_plasma + da_plasma * dt_h)

        t = round(t + dt_h, 10)

    # Compute PK metrics
    cmax_csf = max(c_csf_list)
    tmax_csf_h = times[c_csf_list.index(cmax_csf)]
    auc_csf = _trapz(times, c_csf_list)

    cmax_spinal_cord = max(c_cord_list)
    auc_spinal_cord = _trapz(times, c_cord_list)

    auc_plasma = _trapz(times, c_plasma_list)

    spinal_to_plasma_ratio = auc_spinal_cord / auc_plasma if auc_plasma > 0.0 else 0.0

    # Analgesic duration: time above 10% of Cmax in spinal cord
    threshold = 0.1 * cmax_spinal_cord
    analgesic_steps = sum(1 for c in c_cord_list if c >= threshold)
    analgesic_duration_h = analgesic_steps * dt_h

    # Notes
    if route == "intrathecal" and spinal_to_plasma_ratio > 10.0:
        notes = (
            "Excellent spinal selectivity — suitable for spinally-targeted analgesia"
            " with minimal systemic effects"
        )
    elif spinal_to_plasma_ratio > 1.0:
        notes = "Good spinal cord penetration"
    else:
        notes = "Limited spinal cord distribution — consider intrathecal route"

    return SpinalCordPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_csf_mg_L=c_csf_list,
        c_spinal_cord_mg_L=c_cord_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_csf=cmax_csf,
        tmax_csf_h=tmax_csf_h,
        auc_csf=auc_csf,
        cmax_spinal_cord=cmax_spinal_cord,
        auc_spinal_cord=auc_spinal_cord,
        spinal_to_plasma_ratio=spinal_to_plasma_ratio,
        analgesic_duration_h=analgesic_duration_h,
        notes=notes,
    )
