"""Phase 222 — Epidural Drug Delivery PK Model.

4-compartment forward Euler ODE model:
  - Epidural space
  - Spinal CSF
  - Spinal cord tissue
  - Systemic plasma
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpiduralPKResult:
    """Result of an epidural PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_epidural_mg_mL: list
    c_csf_mg_mL: list
    c_spinal_mg_g: list
    c_systemic_mg_L: list
    cmax_csf_mg_mL: float
    tmax_csf_h: float
    auc_csf_mg_h_per_mL: float
    cmax_systemic_mg_L: float
    auc_spinal_mg_h_per_g: float
    notes: str


# --- Physiological constants ---
_V_EPI_ML = 30.0  # epidural space volume (mL)
_V_CSF_ML = 75.0  # spinal CSF volume (mL)
_V_CORD_G = 30.0  # spinal cord volume (mL equivalent / g)

_K_ECS = 0.3  # /h  epidural → CSF transfer
_K_EV = 0.1  # /h  epidural → vascular absorption
_K_CSF_SYS = 0.05  # /h  CSF → systemic
_K_CSF_CORD = 0.2  # /h  CSF → spinal cord
_K_CORD_OUT = 0.02  # /h  spinal cord → systemic return


def _trapz(y: list, x: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total


def simulate_epidural_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> EpiduralPKResult:
    """Simulate epidural drug delivery using a 4-compartment ODE model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Epidural bolus dose in mg.
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_sys_L:
        Systemic volume of distribution in L.
    t_end_h:
        Simulation end time in hours (default 24 h).
    dt_h:
        Forward Euler time step in hours (default 0.01 h).

    Returns
    -------
    EpiduralPKResult with full time-course profiles and summary PK metrics.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}.")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}.")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}.")

    # Derived elimination rate constant
    ke = cl_L_per_h / vd_sys_L  # /h

    # Convert volumes for ODE unit consistency
    # V_epi in mL, V_csf in mL, V_cord in g, Vd_sys in L
    # Concentrations: C_epi (mg/mL), C_csf (mg/mL), C_spinal (mg/g), C_sys (mg/L)

    # Initial conditions — epidural bolus
    c_epi = dose_mg / _V_EPI_ML  # mg/mL
    c_csf = 0.0  # mg/mL
    c_spinal = 0.0  # mg/g
    c_sys = 0.0  # mg/L

    times: list = []
    epi_list: list = []
    csf_list: list = []
    spinal_list: list = []
    sys_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times.append(t)
        epi_list.append(c_epi)
        csf_list.append(c_csf)
        spinal_list.append(c_spinal)
        sys_list.append(c_sys)

        if _ == n_steps:
            break

        # ODEs — forward Euler
        dc_epi = -(_K_ECS + _K_EV) * c_epi

        dc_csf = _K_ECS * c_epi * _V_EPI_ML / _V_CSF_ML - _K_CSF_SYS * c_csf - _K_CSF_CORD * c_csf

        dc_spinal = _K_CSF_CORD * c_csf * _V_CSF_ML / _V_CORD_G - _K_CORD_OUT * c_spinal

        # Systemic: inputs from epidural vascular absorption + CSF bulk flow
        # k_ev * C_epi * V_epi gives rate in mg/h; divide by Vd_sys (L) for mg/L/h
        # k_csf_sys * C_csf * V_csf gives rate in mg/h; divide by Vd_sys (L)
        dc_sys = (
            _K_EV
            * c_epi
            * _V_EPI_ML
            / (vd_sys_L * 1000.0)  # mg/mL * mL/h / L → mg/L/h ... need unit fix
            + _K_CSF_SYS * c_csf * _V_CSF_ML / (vd_sys_L * 1000.0)
            - ke * c_sys
        )
        # Unit note: k_ev * C_epi (mg/mL) * V_epi (mL) = mg/h of drug flux
        # To get dC_sys (mg/L/h): divide mg/h by Vd_sys_L ... but Vd_sys is in L
        # So: mg/h / L = mg/(L*h) ✓
        # However we divided by vd_sys_L*1000 which converts L → mL, incorrect.
        # Fix: divide by vd_sys_L directly (already in L).
        # Recompute below (overwrite dc_sys):
        dc_sys = (
            _K_EV * c_epi * _V_EPI_ML / vd_sys_L
            + _K_CSF_SYS * c_csf * _V_CSF_ML / vd_sys_L
            - ke * c_sys
        )
        # Unit check:
        # k_ev (/h) * c_epi (mg/mL) * V_epi (mL) = mg/h
        # mg/h / vd_sys (L) = mg/(L*h) ✓ for dc_sys in mg/L/h

        c_epi = max(0.0, c_epi + dc_epi * dt_h)
        c_csf = max(0.0, c_csf + dc_csf * dt_h)
        c_spinal = max(0.0, c_spinal + dc_spinal * dt_h)
        c_sys = max(0.0, c_sys + dc_sys * dt_h)

        t += dt_h

    # Summary PK metrics
    cmax_csf = max(csf_list)
    tmax_csf = times[csf_list.index(cmax_csf)]
    auc_csf = _trapz(csf_list, times)
    cmax_sys = max(sys_list)
    auc_spinal = _trapz(spinal_list, times)

    notes = (
        f"Epidural bolus {dose_mg} mg, CL={cl_L_per_h} L/h, "
        f"Vd={vd_sys_L} L, ke={ke:.4f}/h. "
        f"Forward Euler dt={dt_h} h over {t_end_h} h."
    )

    return EpiduralPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_epidural_mg_mL=epi_list,
        c_csf_mg_mL=csf_list,
        c_spinal_mg_g=spinal_list,
        c_systemic_mg_L=sys_list,
        cmax_csf_mg_mL=cmax_csf,
        tmax_csf_h=tmax_csf,
        auc_csf_mg_h_per_mL=auc_csf,
        cmax_systemic_mg_L=cmax_sys,
        auc_spinal_mg_h_per_g=auc_spinal,
        notes=notes,
    )


def compare_epidural_doses(
    drug_name: str,
    doses_mg: list,
    cl_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> list:
    """Simulate epidural PK for multiple doses and sort by spinal cord AUC descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    doses_mg:
        List of epidural bolus doses in mg.
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_sys_L:
        Systemic volume of distribution in L.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    List of EpiduralPKResult sorted by auc_spinal_mg_h_per_g descending.
    """
    results = [
        simulate_epidural_pk(drug_name, dose, cl_L_per_h, vd_sys_L, t_end_h, dt_h)
        for dose in doses_mg
    ]
    return sorted(results, key=lambda r: r.auc_spinal_mg_h_per_g, reverse=True)
