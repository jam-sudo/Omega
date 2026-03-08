"""Phase 618 — Drug Cardiac Tissue Distribution.

Models drug distribution into cardiac tissue and its implications for
cardiotoxicity assessment. Uses a 2-compartment model (plasma + cardiac)
with oral absorption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CardiacTissuePKResult:
    """Result of cardiac tissue PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_cardiac_mg_g: list  # drug concentration in cardiac tissue (mg/g)
    kp_cardiac: float  # cardiac tissue partition coefficient
    cmax_plasma_mg_L: float
    cmax_cardiac_mg_g: float
    cardiac_plasma_ratio_at_cmax: float  # c_cardiac / c_plasma at plasma Cmax
    auc_plasma_mg_h_per_L: float
    auc_cardiac_mg_h_per_g: float
    t_half_cardiac_h: float  # effective cardiac half-life
    cardiotoxicity_exposure_index: float  # AUC_cardiac * (1 + logP_factor)
    therapeutic_margin: float  # (5.0 - cmax_cardiac) / cmax_cardiac
    notes: str


def simulate_cardiac_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    cardiac_weight_g: float = 300.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> CardiacTissuePKResult:
    """Simulate drug distribution into cardiac tissue.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    logP:
        Lipophilicity (octanol-water partition coefficient).
    mw_Da:
        Molecular weight in Daltons.
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution of systemic (central) compartment (L).
    cardiac_weight_g:
        Weight of cardiac tissue in grams (default 300 g).
    t_end_h:
        Simulation end time (hours).
    dt_h:
        Time step for forward Euler (hours).

    Returns
    -------
    CardiacTissuePKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if cardiac_weight_g <= 0:
        raise ValueError("cardiac_weight_g must be > 0")

    # --- Cardiac Kp from logP and MW ---
    kp_cardiac_raw = (logP * 1.5 + 0.5) / max(1.0, mw_Da / 200.0)
    kp_cardiac = max(0.1, min(20.0, kp_cardiac_raw))
    # Ensure minimum of 0.5 per spec (max(0.5, ...) before MW correction)
    kp_cardiac = max(0.5 / max(1.0, mw_Da / 200.0), kp_cardiac)
    kp_cardiac = max(0.1, min(20.0, kp_cardiac))

    # --- Transfer rate parameters ---
    Qc = 18.0  # L/h, coronary blood flow (~5% of CO)
    Vd_cardiac_L = cardiac_weight_g * kp_cardiac / 1000.0  # L

    k12 = Qc / vd_sys_L  # /h, plasma → cardiac
    k21 = Qc / Vd_cardiac_L  # /h, cardiac → plasma

    # --- Oral absorption ---
    ka = 1.2  # /h
    f_oral = 0.85
    ke = cl_sys_L_per_h / vd_sys_L  # /h, elimination rate constant

    # --- Initial conditions ---
    A_gut = dose_mg * f_oral  # mg in gut (available for absorption)
    C_plasma = 0.0  # mg/L in plasma
    C_cardiac = 0.0  # mg/L in cardiac compartment (concentration basis)

    # --- Storage ---
    n_steps = int(round(t_end_h / dt_h)) + 1
    times_h: list = []
    c_plasma_mg_L: list = []
    c_cardiac_mg_g: list = []

    def _cardiac_mg_g(c_card: float) -> float:
        """Convert cardiac compartment concentration (mg/L) to mg/g tissue."""
        # Amount in cardiac = C_cardiac * Vd_cardiac_L (mg)
        # Concentration per gram = amount / cardiac_weight_g
        return c_card * Vd_cardiac_L * 1000.0 / cardiac_weight_g

    def _record(t: float) -> None:
        times_h.append(t)
        c_plasma_mg_L.append(C_plasma)
        c_cardiac_mg_g.append(_cardiac_mg_g(C_cardiac))

    _record(0.0)

    # --- Forward Euler ---
    for step in range(1, n_steps):
        t = step * dt_h

        dA_gut = -ka * A_gut
        dC_plasma = (
            ka * A_gut / vd_sys_L
            - ke * C_plasma
            - k12 * C_plasma
            + k21 * C_cardiac * (Vd_cardiac_L / vd_sys_L)
        )
        dC_cardiac = k12 * C_plasma - k21 * C_cardiac

        A_gut = max(0.0, A_gut + dA_gut * dt_h)
        C_plasma = max(0.0, C_plasma + dC_plasma * dt_h)
        C_cardiac = max(0.0, C_cardiac + dC_cardiac * dt_h)

        _record(t)

    # --- Derived metrics ---
    cmax_plasma_mg_L = max(c_plasma_mg_L)
    cmax_cardiac_mg_g = max(c_cardiac_mg_g)

    # cardiac_plasma_ratio at time of plasma Cmax
    idx_cmax = c_plasma_mg_L.index(cmax_plasma_mg_L)
    c_plasma_at_cmax = c_plasma_mg_L[idx_cmax]
    c_cardiac_at_cmax = c_cardiac_mg_g[idx_cmax]
    if c_plasma_at_cmax > 0.0:
        cardiac_plasma_ratio_at_cmax = c_cardiac_at_cmax / c_plasma_at_cmax
    else:
        cardiac_plasma_ratio_at_cmax = 0.0

    # Manual trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_cardiac = sum(
        0.5 * (c_cardiac_mg_g[i] + c_cardiac_mg_g[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Effective cardiac half-life dominated by washout
    t_half_cardiac_h = math.log(2.0) / k21

    # Cardiotoxicity exposure index
    logP_factor = max(1.0, logP)
    cardiotoxicity_exposure_index = auc_cardiac * logP_factor

    # Therapeutic margin: (threshold - cmax_cardiac) / cmax_cardiac
    cardiac_toxicity_threshold = 5.0  # mg/g
    therapeutic_margin = (cardiac_toxicity_threshold - cmax_cardiac_mg_g) / max(
        0.001, cmax_cardiac_mg_g
    )

    notes = (
        f"Cardiac tissue PK simulation for {drug_name}. "
        f"kp_cardiac={kp_cardiac:.3f}, k12={k12:.3f}/h, k21={k21:.3f}/h, "
        f"Vd_cardiac={Vd_cardiac_L:.4f} L."
    )

    return CardiacTissuePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        c_cardiac_mg_g=c_cardiac_mg_g,
        kp_cardiac=kp_cardiac,
        cmax_plasma_mg_L=cmax_plasma_mg_L,
        cmax_cardiac_mg_g=cmax_cardiac_mg_g,
        cardiac_plasma_ratio_at_cmax=cardiac_plasma_ratio_at_cmax,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_cardiac_mg_h_per_g=auc_cardiac,
        t_half_cardiac_h=t_half_cardiac_h,
        cardiotoxicity_exposure_index=cardiotoxicity_exposure_index,
        therapeutic_margin=therapeutic_margin,
        notes=notes,
    )
