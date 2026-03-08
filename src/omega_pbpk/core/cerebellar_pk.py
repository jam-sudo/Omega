"""Phase 1051 — Drug Cerebellar Distribution.

Models drug distribution into specific brain regions (cerebellum, cortex)
relevant for CNS drug targeting, with a 4-compartment ODE system:
plasma + cerebellum + cortex + CSF.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CerebellarPKResult:
    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_cerebellum_mg_L: list
    c_cortex_mg_L: list
    c_csf_mg_L: list
    auc_plasma: float
    auc_cerebellum: float
    auc_cortex: float
    cmax_cerebellum: float
    tmax_cerebellum_h: float
    cerebellum_to_plasma_ratio: float
    cortex_to_cerebellum_ratio: float
    bbb_penetration_class: str
    notes: str


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal AUC integration."""
    if len(times) < 2:
        return 0.0
    total = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        total += 0.5 * (values[i] + values[i + 1]) * dt
    return total


def simulate_cerebellar_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 2.0,
    mw_Da: float = 300.0,
    fu_plasma: float = 0.3,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 30.0,
    kp_bbb: float = 0.0,
    cerebellum_enrichment: float = 1.0,
    cortex_enrichment: float = 0.8,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> CerebellarPKResult:
    """Simulate drug distribution into cerebellum, cortex, and CSF.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        logp: Lipophilicity.
        mw_Da: Molecular weight in Da.
        fu_plasma: Unbound fraction in plasma (0, 1].
        cl_sys_L_per_h: Systemic clearance in L/h.
        vd_sys_L: Volume of distribution in L.
        kp_bbb: BBB partition coefficient (0 = compute from logP/MW/fu).
        cerebellum_enrichment: Cerebellum vs average brain enrichment (>1 = enriched).
        cortex_enrichment: Cortex vs average brain enrichment.
        t_end_h: Simulation end time in hours.
        dt_h: Time step in hours.

    Returns:
        CerebellarPKResult with full time-course and summary metrics.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if kp_bbb < 0:
        raise ValueError("kp_bbb must be >= 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")

    # Compute BBB Kp if not supplied
    if kp_bbb == 0.0:
        kp_bbb = max(0.01, min(2.0, 10 ** (0.5 * logp - 0.01 * mw_Da) * fu_plasma))

    # Regional Kp
    kp_cerebellum = kp_bbb * cerebellum_enrichment
    kp_cortex = kp_bbb * cortex_enrichment

    # Compartment volumes (L)
    v_cerebellum_L = 0.15
    v_cortex_L = 0.6
    v_csf_L = 0.14

    # Rate constants
    ke = cl_sys_L_per_h / vd_sys_L

    k_bbb = 0.1 * kp_bbb  # transfer rate coefficient

    k_p_cb = k_bbb * vd_sys_L / v_cerebellum_L
    k_cb_p = k_bbb / kp_cerebellum if kp_cerebellum > 0 else 0.0

    k_p_cx = k_bbb * vd_sys_L / v_cortex_L
    k_cx_p = k_bbb / kp_cortex if kp_cortex > 0 else 0.0

    k_p_csf = 0.01 * vd_sys_L / v_csf_L
    k_csf_p = 0.25

    # Initial conditions: IV bolus into plasma
    c_p = dose_mg / vd_sys_L
    c_cb = 0.0
    c_cx = 0.0
    c_csf = 0.0

    times = [0.0]
    cp_list = [c_p]
    ccb_list = [c_cb]
    ccx_list = [c_cx]
    ccsf_list = [c_csf]

    n_steps = int(round(t_end_h / dt_h))
    t = 0.0
    for _i in range(n_steps):
        # Forward Euler
        dcp = (
            -ke * c_p
            - k_p_cb * c_p
            + k_cb_p * c_cb
            - k_p_cx * c_p
            + k_cx_p * c_cx
            - k_p_csf * c_p
            + k_csf_p * c_csf
        )
        dcb = k_p_cb * c_p - k_cb_p * c_cb
        dcx = k_p_cx * c_p - k_cx_p * c_cx
        dcsf = k_p_csf * c_p - k_csf_p * c_csf

        c_p = max(0.0, c_p + dcp * dt_h)
        c_cb = max(0.0, c_cb + dcb * dt_h)
        c_cx = max(0.0, c_cx + dcx * dt_h)
        c_csf = max(0.0, c_csf + dcsf * dt_h)

        t += dt_h
        times.append(round(t, 10))
        cp_list.append(c_p)
        ccb_list.append(c_cb)
        ccx_list.append(c_cx)
        ccsf_list.append(c_csf)

    # AUCs
    auc_plasma = _trapz(times, cp_list)
    auc_cerebellum = _trapz(times, ccb_list)
    auc_cortex = _trapz(times, ccx_list)

    # Cmax and Tmax for cerebellum
    cmax_cerebellum = max(ccb_list)
    tmax_idx = ccb_list.index(cmax_cerebellum)
    tmax_cerebellum_h = times[tmax_idx]

    # Ratios
    cerebellum_to_plasma_ratio = auc_cerebellum / auc_plasma if auc_plasma > 0 else 0.0
    cortex_to_cerebellum_ratio = auc_cortex / auc_cerebellum if auc_cerebellum > 0 else 0.0

    # BBB penetration class
    if kp_bbb > 0.3:
        bbb_penetration_class = "high"
    elif kp_bbb >= 0.1:
        bbb_penetration_class = "moderate"
    else:
        bbb_penetration_class = "low"

    notes = (
        f"kp_bbb={kp_bbb:.4f}; "
        f"kp_cerebellum={kp_cerebellum:.4f}; "
        f"kp_cortex={kp_cortex:.4f}; "
        f"BBB class={bbb_penetration_class}; "
        f"cerebellum_to_plasma AUC ratio={cerebellum_to_plasma_ratio:.4f}"
    )

    return CerebellarPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_cerebellum_mg_L=ccb_list,
        c_cortex_mg_L=ccx_list,
        c_csf_mg_L=ccsf_list,
        auc_plasma=auc_plasma,
        auc_cerebellum=auc_cerebellum,
        auc_cortex=auc_cortex,
        cmax_cerebellum=cmax_cerebellum,
        tmax_cerebellum_h=tmax_cerebellum_h,
        cerebellum_to_plasma_ratio=cerebellum_to_plasma_ratio,
        cortex_to_cerebellum_ratio=cortex_to_cerebellum_ratio,
        bbb_penetration_class=bbb_penetration_class,
        notes=notes,
    )
