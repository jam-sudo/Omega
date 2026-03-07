"""Phase 368 — Topical Drug Dermatokinetics Model.

Models drug concentration kinetics in skin layers after topical application
using a 4-compartment forward Euler ODE system:
  formulation → stratum_corneum → viable_epidermis → dermis → systemic
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DermatokineticsResult:
    """Result of a dermatokinetics simulation."""

    times_h: list
    c_formulation_mg_mL: list
    c_sc_mg_mL: list
    c_viable_epidermis_mg_mL: list
    c_dermis_mg_mL: list
    c_systemic_mg_L: list
    auc_dermis_mg_h_per_mL: float
    auc_systemic_mg_h_per_L: float
    cmax_dermis_mg_mL: float
    cmax_systemic_mg_L: float
    tmax_dermis_h: float
    ksc_factor: float
    systemic_bioavailability_pct: float
    notes: str


def _trapz(x: list[float], y: list[float]) -> float:
    """Trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_dermatokinetics(
    drug_name: str,
    applied_dose_mg: float,
    logP: float,
    mw_Da: float,
    k_formulation_per_h: float,
    k_sc_absorption_per_h: float,
    k_ve_transfer_per_h: float,
    k_dermis_to_sys_per_h: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float,
    dt_h: float = 0.01,
    area_cm2: float = 100.0,
    v_sc_mL: float = 0.5,
    v_ve_mL: float = 2.0,
    v_dermis_mL: float = 10.0,
) -> DermatokineticsResult:
    """Simulate topical drug dermatokinetics.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    applied_dose_mg:
        Total dose applied to the skin surface (mg).
    logP:
        Octanol-water partition coefficient.
    mw_Da:
        Molecular weight in Daltons.
    k_formulation_per_h:
        First-order release rate from formulation (h^-1).
    k_sc_absorption_per_h:
        Base stratum corneum permeation rate constant (h^-1).
    k_ve_transfer_per_h:
        Transfer rate from viable epidermis to dermis (h^-1).
    k_dermis_to_sys_per_h:
        Transfer rate from dermis to systemic circulation (h^-1).
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).
    area_cm2:
        Skin application area (cm^2).
    v_sc_mL:
        Volume of stratum corneum compartment (mL).
    v_ve_mL:
        Volume of viable epidermis compartment (mL).
    v_dermis_mL:
        Volume of dermis compartment (mL).

    Returns
    -------
    DermatokineticsResult
    """
    # --- Validation ---
    if applied_dose_mg <= 0:
        raise ValueError("applied_dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if k_formulation_per_h <= 0:
        raise ValueError("k_formulation_per_h must be > 0")

    # --- Stratum corneum permeability factor ---
    ksc_factor = max(0.01, min(10.0, 10.0 ** (0.5 * logP - 0.005 * mw_Da)))
    k_sc_eff = k_sc_absorption_per_h * ksc_factor

    # Systemic elimination rate constant
    ke_sys = cl_sys_L_per_h / vd_sys_L  # h^-1
    # Convert vd from L to mL for unit consistency within ODE, then back
    # Systemic compartment: C_sys in mg/L; vd_sys in L
    # Transfer from dermis adds: k_dermis_to_sys * C_dermis * v_dermis_mL / (vd_sys_L * 1000) * 1000
    # = k_dermis_to_sys * C_dermis * v_dermis_mL / vd_sys_L  [mg/L/h equivalent]
    # Actually handle units carefully:
    #   Flux from dermis = k_dermis_to_sys * C_dermis [mg/mL] * v_dermis_mL [mL] = mg/h
    #   Rate in systemic (mg/L): flux / vd_sys_L / ... need consistent mg/L
    #   dC_sys/dt [mg/L/h] = flux_mg_h / (vd_sys_L * 1000 mL/L) * 1000 = flux_mg_h / vd_sys_L  ... but vd_sys is in L
    #   dC_sys [mg/L/h] = k_dermis_to_sys * C_dermis [mg/mL] * v_dermis_mL / vd_sys_L / ...
    # Let's convert: C_sys in mg/L; dermis in mg/mL; 1 mg/mL = 1000 mg/L
    # Flux from dermis = k_dermis_to_sys * C_dermis_mg_mL * v_dermis_mL   [mg/h]
    # dC_sys/dt = flux_mg_h / (vd_sys_L * 1.0) ... but vd in L, C in mg/L
    # dC_sys/dt [mg/L/h] = flux [mg/h] / vd [L]
    # So: dC_sys = k_dermis_to_sys * C_dermis * v_dermis_mL / vd_sys_L - ke_sys * C_sys

    # Initial conditions
    # Formulation: amount in mg (not concentration)
    A_form = applied_dose_mg  # mg
    C_sc = 0.0  # mg/mL
    C_ve = 0.0  # mg/mL
    C_derm = 0.0  # mg/mL
    C_sys = 0.0  # mg/L

    times: list[float] = [0.0]
    c_form_list: list[float] = [A_form]
    c_sc_list: list[float] = [C_sc]
    c_ve_list: list[float] = [C_ve]
    c_derm_list: list[float] = [C_derm]
    c_sys_list: list[float] = [C_sys]

    t = 0.0
    n_steps = max(1, round(t_end_h / dt_h))
    actual_dt = t_end_h / n_steps

    for _ in range(n_steps):
        # Derivatives
        dA_form = -k_formulation_per_h * A_form

        # SC: input from formulation release (amount→concentration), output by permeation
        # release flux [mg/h] → dC_sc [mg/mL/h] = flux / v_sc
        dC_sc = (k_formulation_per_h * A_form / v_sc_mL) - k_sc_eff * C_sc

        # VE: input from SC permeation (C_sc * v_sc → rate into VE), output to dermis
        dC_ve = (k_sc_eff * C_sc * v_sc_mL / v_ve_mL) - k_ve_transfer_per_h * C_ve

        # Dermis: input from VE, output to systemic
        dC_derm = (
            k_ve_transfer_per_h * C_ve * v_ve_mL / v_dermis_mL
        ) - k_dermis_to_sys_per_h * C_derm

        # Systemic: input from dermis [mg/h] / vd_sys_L, elimination
        # Dermis flux = k_dermis_to_sys * C_derm [mg/mL] * v_dermis_mL [mL] = [mg/h]
        # dC_sys [mg/L/h] = flux [mg/h] / vd_sys [L] ... but C_sys is in mg/L
        # Convert: C_derm mg/mL * v_dermis_mL = mg; / vd_sys_L = mg/L ... consistent if vd in L
        dC_sys = (k_dermis_to_sys_per_h * C_derm * v_dermis_mL / vd_sys_L) - ke_sys * C_sys

        # Forward Euler update
        A_form += dA_form * actual_dt
        C_sc += dC_sc * actual_dt
        C_ve += dC_ve * actual_dt
        C_derm += dC_derm * actual_dt
        C_sys += dC_sys * actual_dt

        # Clamp to non-negative
        A_form = max(0.0, A_form)
        C_sc = max(0.0, C_sc)
        C_ve = max(0.0, C_ve)
        C_derm = max(0.0, C_derm)
        C_sys = max(0.0, C_sys)

        t += actual_dt
        times.append(t)
        c_form_list.append(A_form)
        c_sc_list.append(C_sc)
        c_ve_list.append(C_ve)
        c_derm_list.append(C_derm)
        c_sys_list.append(C_sys)

    # --- Summary metrics ---
    auc_dermis = _trapz(times, c_derm_list)
    auc_systemic = _trapz(times, c_sys_list)

    cmax_dermis = max(c_derm_list)
    cmax_systemic = max(c_sys_list)

    tmax_dermis_h = times[c_derm_list.index(cmax_dermis)]

    # Systemic bioavailability: compare AUC_sys to AUC expected from IV dose/CL
    # AUC_iv = dose_mg / cl_sys_L_per_h [mg/L * h] = mg·h/L
    # AUC_sys is in mg/L * h = mg·h/L (C_sys in mg/L)
    auc_iv_ref = applied_dose_mg / cl_sys_L_per_h
    systemic_ba_pct = auc_systemic / auc_iv_ref * 100.0

    notes = (
        f"ksc_factor={ksc_factor:.3f} (logP={logP}, MW={mw_Da} Da). "
        f"k_sc_eff={k_sc_eff:.4f}/h. "
        f"Systemic BA={systemic_ba_pct:.1f}%. "
        f"Tmax dermis={tmax_dermis_h:.2f}h."
    )

    return DermatokineticsResult(
        times_h=times,
        c_formulation_mg_mL=c_form_list,
        c_sc_mg_mL=c_sc_list,
        c_viable_epidermis_mg_mL=c_ve_list,
        c_dermis_mg_mL=c_derm_list,
        c_systemic_mg_L=c_sys_list,
        auc_dermis_mg_h_per_mL=auc_dermis,
        auc_systemic_mg_h_per_L=auc_systemic,
        cmax_dermis_mg_mL=cmax_dermis,
        cmax_systemic_mg_L=cmax_systemic,
        tmax_dermis_h=tmax_dermis_h,
        ksc_factor=ksc_factor,
        systemic_bioavailability_pct=systemic_ba_pct,
        notes=notes,
    )
