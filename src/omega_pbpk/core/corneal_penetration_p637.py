"""
Phase 637 — Drug Corneal Penetration
Models drug penetration through corneal layers (epithelium, stroma, endothelium)
for ophthalmic drops.
"""

import math
from dataclasses import dataclass


@dataclass
class CornealPenetrationResult:
    drug_name: str
    dose_mg: float
    times_h: list
    c_tear_mg_mL: list
    c_epithelium_mg_g: list
    c_stroma_mg_g: list
    c_aqueous_mg_mL: list
    cmax_aqueous_mg_mL: float
    tmax_aqueous_h: float
    auc_aqueous_mg_h_per_mL: float
    corneal_permeability_cm_s: float
    t_half_aqueous_h: float
    bioavailability_ocular_pct: float
    notes: str


def simulate_corneal_penetration(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    pka: float | None = None,
    drop_volume_uL: float = 30.0,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> CornealPenetrationResult:
    """
    Simulate corneal penetration of an ophthalmic drug drop.

    Parameters
    ----------
    drug_name : str
    dose_mg : float        total dose in drop (mg)
    logP : float           octanol-water partition coefficient
    mw_Da : float          molecular weight (Da)
    pka : float, optional  pKa (currently unused, reserved for ionization)
    drop_volume_uL : float drop volume (uL), default 30
    t_end_h : float        simulation end time (h)
    dt_h : float           time step (h)
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if drop_volume_uL <= 0:
        raise ValueError("drop_volume_uL must be > 0")
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")

    # --- Constants ---
    V_tear_uL = 7.0
    V_tear_eff_mL = (V_tear_uL + drop_volume_uL) / 1000.0  # mL

    k_drain = 20.0  # /h  (t1/2 ~2 min, ln2/0.033h ≈ 21 /h)

    epi_mass_g = 0.002
    stroma_mass_g = 0.06
    V_aq = 0.25  # mL
    k_turnover = 0.1  # /h (aqueous humor exchange)

    A_cornea = 1.04  # cm2

    # --- Permeability calculation ---
    # Epithelial permeability (lipophilic barrier)
    P_epi = max(1e-6, min(1e-4, 10 ** (-4.0 + 0.5 * logP - 0.001 * mw_Da)))  # cm/s

    # Stromal permeability (aqueous barrier, size-dependent)
    P_stroma = 1e-4 / (1.0 + mw_Da / 200.0)  # cm/s

    # Endothelial permeability (relatively permeable)
    P_endo = 1e-3  # cm/s

    # Harmonic mean (series barrier)
    P_cornea = 1.0 / (1.0 / P_epi + 1.0 / P_stroma + 1.0 / P_endo)  # cm/s

    # Absorption rate: convert permeability to first-order rate for tear compartment
    k_abs_vol = P_cornea * 3600.0 * A_cornea / V_tear_eff_mL  # /h
    k_abs_vol = max(0.5, min(20.0, k_abs_vol))

    # Fixed transfer rates between layers
    k_epi_to_str = 1.0  # /h
    k_str_to_aq = 2.0  # /h

    # --- Initial conditions (amounts in mg) ---
    A_tear = dose_mg
    A_epi = 0.0
    A_stroma = 0.0
    C_aq = 0.0  # mg/mL

    # --- Storage ---
    times_h = []
    c_tear_list = []
    c_epi_list = []
    c_stroma_list = []
    c_aq_list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for step in range(n_steps):
        # Record concentrations
        c_tear = A_tear / V_tear_eff_mL
        c_epi_g = A_epi / epi_mass_g
        c_str_g = A_stroma / stroma_mass_g

        times_h.append(t)
        c_tear_list.append(c_tear)
        c_epi_list.append(c_epi_g)
        c_stroma_list.append(c_str_g)
        c_aq_list.append(C_aq)

        if step == n_steps - 1:
            break

        # --- ODEs (Forward Euler) ---
        dA_tear = -(k_drain + k_abs_vol) * A_tear
        dA_epi = k_abs_vol * A_tear - k_epi_to_str * A_epi
        dA_stroma = k_epi_to_str * A_epi - k_str_to_aq * A_stroma
        dC_aq = k_str_to_aq * A_stroma / V_aq - k_turnover * C_aq

        A_tear = max(0.0, A_tear + dA_tear * dt_h)
        A_epi = max(0.0, A_epi + dA_epi * dt_h)
        A_stroma = max(0.0, A_stroma + dA_stroma * dt_h)
        C_aq = max(0.0, C_aq + dC_aq * dt_h)

        t += dt_h

    # --- Derived PK metrics ---
    cmax_aqueous = max(c_aq_list)
    tmax_idx = c_aq_list.index(cmax_aqueous)
    tmax_aqueous = times_h[tmax_idx]

    # Trapezoidal AUC
    auc_aqueous = sum(
        0.5 * (c_aq_list[i] + c_aq_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    t_half_aqueous = math.log(2.0) / k_turnover

    # Ocular bioavailability: fraction of dose that reached aqueous
    # AUC * k_turnover * V_aq = amount cleared through aqueous = proxy for absorbed amount
    bioavailability_ocular_pct = min(100.0, auc_aqueous * k_turnover * V_aq / dose_mg * 100.0)

    notes = (
        f"P_epi={P_epi:.3e} cm/s, P_stroma={P_stroma:.3e} cm/s, "
        f"P_endo={P_endo:.3e} cm/s, P_cornea={P_cornea:.3e} cm/s, "
        f"k_abs={k_abs_vol:.2f} /h"
    )

    return CornealPenetrationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_tear_mg_mL=c_tear_list,
        c_epithelium_mg_g=c_epi_list,
        c_stroma_mg_g=c_stroma_list,
        c_aqueous_mg_mL=c_aq_list,
        cmax_aqueous_mg_mL=cmax_aqueous,
        tmax_aqueous_h=tmax_aqueous,
        auc_aqueous_mg_h_per_mL=auc_aqueous,
        corneal_permeability_cm_s=P_cornea,
        t_half_aqueous_h=t_half_aqueous,
        bioavailability_ocular_pct=bioavailability_ocular_pct,
        notes=notes,
    )
