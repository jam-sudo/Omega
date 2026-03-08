"""
Phase 439: Ocular topical drug delivery — tear film kinetics, corneal penetration,
and aqueous humor PK.

3-compartment model: tear → cornea → aqueous humor
"""

import math
from dataclasses import dataclass


@dataclass
class OphthalmicTopicalResult:
    """Result of ocular topical PK simulation."""

    drug_name: str
    dose_mg: float
    drop_volume_uL: float
    times_min: list
    c_tear_mg_mL: list
    c_cornea_mg_g: list
    c_aqueous_mg_mL: list
    cmax_tear_mg_mL: float
    cmax_aqueous_mg_mL: float
    tmax_aqueous_min: float
    auc_tear_mg_h_per_mL: float
    auc_aqueous_mg_h_per_mL: float
    ocular_bioavailability_pct: float
    drainage_loss_pct: float
    t_half_tear_min: float
    notes: str


def _compute_peff_cornea_per_min(logp: float) -> float:
    """
    Compute corneal permeability coefficient in cm/min from logP.
    Peff_cornea (cm/s) = 1e-4 * 10^(logP * 0.2), clamped [1e-6, 5e-4] cm/s
    Returns in cm/min.
    """
    peff_cm_per_s = 1e-4 * 10 ** (logp * 0.2)
    # Clamp
    peff_cm_per_s = max(1e-6, min(5e-4, peff_cm_per_s))
    # Convert to per min
    peff_cm_per_min = peff_cm_per_s * 60.0
    return peff_cm_per_min


def simulate_ophthalmic_topical_pk(
    drug_name: str,
    dose_mg: float,
    logp: float,
    drop_volume_uL: float = 30.0,
    t_end_min: float = 120.0,
    dt_min: float = 0.1,
) -> OphthalmicTopicalResult:
    """
    Simulate ocular topical drug PK.

    Compartments:
    - Tear film (M_tear, mg)
    - Cornea (C_cornea, mg/g)
    - Aqueous humor (C_aq, mg/mL)

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        Applied dose in mg.
    logp : float
        Octanol-water partition coefficient (log scale).
    drop_volume_uL : float
        Drop volume in microliters (default 30 uL).
    t_end_min : float
        Simulation end time in minutes.
    dt_min : float
        ODE time step in minutes.

    Returns
    -------
    OphthalmicTopicalResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if drop_volume_uL <= 0:
        raise ValueError(f"drop_volume_uL must be > 0, got {drop_volume_uL}")
    if not (-5 <= logp <= 10):
        raise ValueError(f"logp must be between -5 and 10, got {logp}")
    if t_end_min <= 0:
        raise ValueError(f"t_end_min must be > 0, got {t_end_min}")
    if dt_min <= 0:
        raise ValueError(f"dt_min must be > 0, got {dt_min}")

    # --- PK Parameters ---
    # Tear film volume (mL): 7 uL normal + drop volume
    V_tear_mL = (7.0 + drop_volume_uL) / 1000.0

    # Rate constants (per min)
    k_drain = 0.7  # nasolacrimal drainage
    k_noncorneal = 0.1  # conjunctival/scleral absorption/loss

    # Corneal permeability
    A_cornea_cm2 = 1.04  # corneal surface area
    peff_cm_per_min = _compute_peff_cornea_per_min(logp)
    k_cornea = peff_cm_per_min * A_cornea_cm2 / V_tear_mL  # per min

    # Total tear elimination rate
    k_tear_total = k_drain + k_cornea + k_noncorneal

    # Cornea compartment
    V_cornea_g = 0.2  # g tissue
    k_cornea_to_aq = 0.5  # per min

    # Aqueous humor
    V_aq_mL = 0.25  # mL
    k_aq_elim = 0.02  # per min (t_half ~34.7 min)

    # --- Initial Conditions ---
    M_tear = dose_mg  # mg in tear film
    C_cornea = 0.0  # mg/g
    C_aq = 0.0  # mg/mL

    # --- ODE Integration (Forward Euler) ---
    times_min = [0.0]
    c_tear_mg_mL = [M_tear / V_tear_mL]
    c_cornea_mg_g = [C_cornea]
    c_aqueous_mg_mL = [C_aq]

    n_steps = int(math.ceil(t_end_min / dt_min))
    t = 0.0
    for _ in range(n_steps):
        # Rates
        dM_tear_dt = -k_tear_total * M_tear
        dC_cornea_dt = (k_cornea * M_tear / V_cornea_g) - k_cornea_to_aq * C_cornea
        dC_aq_dt = (k_cornea_to_aq * C_cornea * V_cornea_g / V_aq_mL) - k_aq_elim * C_aq

        # Update
        M_tear = max(0.0, M_tear + dM_tear_dt * dt_min)
        C_cornea = max(0.0, C_cornea + dC_cornea_dt * dt_min)
        C_aq = max(0.0, C_aq + dC_aq_dt * dt_min)

        t += dt_min
        times_min.append(t)
        c_tear_mg_mL.append(M_tear / V_tear_mL)
        c_cornea_mg_g.append(C_cornea)
        c_aqueous_mg_mL.append(C_aq)

    # --- Derived PK Metrics ---
    cmax_tear_mg_mL = max(c_tear_mg_mL)
    cmax_aqueous_mg_mL = max(c_aqueous_mg_mL)

    # tmax for aqueous
    idx_tmax_aq = c_aqueous_mg_mL.index(cmax_aqueous_mg_mL)
    tmax_aqueous_min = times_min[idx_tmax_aq]

    # AUC via trapezoidal (in mg*h/mL — convert min to hours)
    times_h = [t / 60.0 for t in times_min]

    auc_tear = sum(
        0.5 * (c_tear_mg_mL[i] + c_tear_mg_mL[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_aqueous = sum(
        0.5 * (c_aqueous_mg_mL[i] + c_aqueous_mg_mL[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Drainage loss percentage
    drainage_loss_pct = k_drain / k_tear_total * 100.0

    # Ocular bioavailability: fraction of dose reaching aqueous humor
    # Using analytical fraction approach
    f_corneal = k_cornea / k_tear_total
    f_cornea_to_aq = k_cornea_to_aq / (k_cornea_to_aq + k_aq_elim + 0.01)
    ocular_bioavailability_pct = f_corneal * f_cornea_to_aq * 100.0
    # Clamp to [0, 100]
    ocular_bioavailability_pct = max(0.0, min(100.0, ocular_bioavailability_pct))

    # Half-life in tear film
    t_half_tear_min = 0.693 / k_tear_total

    notes = (
        f"Tear drainage is the dominant loss pathway ({drainage_loss_pct:.1f}% of applied dose). "
        f"Ocular bioavailability (aqueous humor) is approximately {ocular_bioavailability_pct:.2f}%. "
        f"LogP={logp:.1f} determines corneal permeability; "
        f"t1/2 in tear film = {t_half_tear_min:.2f} min."
    )

    return OphthalmicTopicalResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        drop_volume_uL=drop_volume_uL,
        times_min=times_min,
        c_tear_mg_mL=c_tear_mg_mL,
        c_cornea_mg_g=c_cornea_mg_g,
        c_aqueous_mg_mL=c_aqueous_mg_mL,
        cmax_tear_mg_mL=cmax_tear_mg_mL,
        cmax_aqueous_mg_mL=cmax_aqueous_mg_mL,
        tmax_aqueous_min=tmax_aqueous_min,
        auc_tear_mg_h_per_mL=auc_tear,
        auc_aqueous_mg_h_per_mL=auc_aqueous,
        ocular_bioavailability_pct=ocular_bioavailability_pct,
        drainage_loss_pct=drainage_loss_pct,
        t_half_tear_min=t_half_tear_min,
        notes=notes,
    )


def compare_logp_ocular(
    drug_name: str,
    dose_mg: float,
    logp_values: list,
) -> list:
    """
    Compare ocular PK across a range of logP values.

    Returns list of OphthalmicTopicalResult sorted by auc_aqueous_mg_h_per_mL descending.
    """
    results = []
    for logp in logp_values:
        result = simulate_ophthalmic_topical_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logp=logp,
        )
        results.append(result)
    results.sort(key=lambda r: r.auc_aqueous_mg_h_per_mL, reverse=True)
    return results
