"""Phase 617 — Drug Enterocyte Metabolism.

Models drug metabolism in intestinal enterocytes during absorption (gut-wall
first-pass effect). Tracks drug through lumen → enterocyte → portal vein,
accounting for CYP3A4-mediated gut-wall extraction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnterocyteMetabolismResult:
    """Result of enterocyte metabolism simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_lumen_mg_mL: list  # drug in intestinal lumen (mg/mL)
    c_enterocyte_mg_g: list  # drug in enterocyte (mg per g tissue)
    c_portal_mg_L: list  # drug entering portal vein (mg/L)
    c_metabolite_mg_L: list  # metabolite in portal vein (mg/L)
    f_gut: float  # fraction surviving gut wall (1 - extraction_gut_wall)
    f_absorbed: float  # fraction absorbed from lumen
    auc_portal_mg_h_per_L: float
    auc_metabolite_mg_h_per_L: float
    extraction_gut_wall: float  # gut wall extraction ratio (E_gut = 1 - f_gut)
    cyp3a4_contribution: float  # fraction of gut metabolism via CYP3A4 (0-1)
    notes: str


def simulate_enterocyte_metabolism(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_int_gut_L_per_h: float,
    fm_cyp3a4: float,
    vmax_gut_umol_h: float = None,
    km_gut_uM: float = None,
    t_end_h: float = 6.0,
    dt_h: float = 0.02,
) -> EnterocyteMetabolismResult:
    """Simulate drug metabolism in intestinal enterocytes.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    logP:
        Lipophilicity (octanol-water partition coefficient).
    cl_int_gut_L_per_h:
        Intrinsic clearance in gut wall (L/h).
    fm_cyp3a4:
        Fraction of gut-wall metabolism via CYP3A4 (0–1).
    vmax_gut_umol_h:
        Optional Michaelis-Menten Vmax (unused directly; cl_int_gut used).
    km_gut_uM:
        Optional Michaelis-Menten Km (unused directly; cl_int_gut used).
    t_end_h:
        Simulation end time (hours).
    dt_h:
        Time step for forward Euler (hours).

    Returns
    -------
    EnterocyteMetabolismResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_int_gut_L_per_h < 0:
        raise ValueError("cl_int_gut_L_per_h must be >= 0")
    if not (0.0 <= fm_cyp3a4 <= 1.0):
        raise ValueError("fm_cyp3a4 must be in [0, 1]")

    # --- Parameters ---
    # Absorption rate constant from logP, clamped to [0.05, 2.0] /h
    ka = 0.3 * max(0.1, logP)
    ka = max(0.05, min(2.0, ka))

    V_lumen = 0.4  # L, intestinal lumen volume
    V_ent = 0.05  # L, enterocyte compartment (concentration basis)
    V_portal = 3.0  # L, portal vein volume
    ent_mass_g = 50.0  # g, total enterocyte mass
    transit_rate = 1.5  # /h, passive transit rate out of enterocyte
    portal_drain = 0.5  # /h, portal drainage rate

    # First-order gut-wall metabolism rate
    k_met_gut = cl_int_gut_L_per_h / V_ent  # /h

    # --- Initial conditions ---
    A_lumen = dose_mg  # mg in lumen
    C_ent = 0.0  # mg/g in enterocyte tissue
    C_portal = 0.0  # mg/L in portal vein
    C_met = 0.0  # mg/L metabolite in portal

    # --- Storage ---
    n_steps = int(round(t_end_h / dt_h)) + 1
    times_h: list = []
    c_lumen_mg_mL: list = []
    c_enterocyte_mg_g: list = []
    c_portal_mg_L: list = []
    c_metabolite_mg_L: list = []

    def _record(t: float) -> None:
        times_h.append(t)
        # Lumen concentration: A_lumen (mg) / V_lumen (L) → mg/L → /1000 = mg/mL
        c_lumen_mg_mL.append(A_lumen / V_lumen / 1000.0)
        c_enterocyte_mg_g.append(C_ent)
        c_portal_mg_L.append(C_portal)
        c_metabolite_mg_L.append(C_met)

    _record(0.0)

    # --- Forward Euler ---
    for step in range(1, n_steps):
        t = step * dt_h

        flux_abs = ka * A_lumen  # mg/h absorbed into enterocyte
        rate_transit = transit_rate * C_ent * ent_mass_g  # mg/h transit from enterocyte
        rate_metabolized = k_met_gut * C_ent * ent_mass_g  # mg/h metabolised in enterocyte

        dA_lumen = -ka * A_lumen
        dC_ent = flux_abs / ent_mass_g - k_met_gut * C_ent - transit_rate * C_ent
        dC_portal = rate_transit * (1.0 - fm_cyp3a4) / V_portal - portal_drain * C_portal
        dC_met = (
            rate_transit * fm_cyp3a4 / V_portal + rate_metabolized / V_portal - portal_drain * C_met
        )

        A_lumen = max(0.0, A_lumen + dA_lumen * dt_h)
        C_ent = max(0.0, C_ent + dC_ent * dt_h)
        C_portal = max(0.0, C_portal + dC_portal * dt_h)
        C_met = max(0.0, C_met + dC_met * dt_h)

        _record(t)

    # --- Derived metrics ---
    f_absorbed = (dose_mg - A_lumen) / dose_mg

    # Manual trapezoidal AUC
    auc_portal = sum(
        0.5 * (c_portal_mg_L[i] + c_portal_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_metabolite = sum(
        0.5 * (c_metabolite_mg_L[i] + c_metabolite_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    auc_sum = auc_portal + auc_metabolite
    extraction_gut_wall = auc_metabolite / auc_sum if auc_sum > 0.0 else 0.0
    f_gut = 1.0 - extraction_gut_wall

    notes = (
        f"Enterocyte gut-wall first-pass simulation for {drug_name}. "
        f"ka={ka:.3f}/h, k_met_gut={k_met_gut:.3f}/h, fm_cyp3a4={fm_cyp3a4:.2f}."
    )

    return EnterocyteMetabolismResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_lumen_mg_mL=c_lumen_mg_mL,
        c_enterocyte_mg_g=c_enterocyte_mg_g,
        c_portal_mg_L=c_portal_mg_L,
        c_metabolite_mg_L=c_metabolite_mg_L,
        f_gut=f_gut,
        f_absorbed=f_absorbed,
        auc_portal_mg_h_per_L=auc_portal,
        auc_metabolite_mg_h_per_L=auc_metabolite,
        extraction_gut_wall=extraction_gut_wall,
        cyp3a4_contribution=fm_cyp3a4,
        notes=notes,
    )
