"""Phase 329 — Extended Renal Drug Transport Model.

Mechanistic model of renal tubular transport with OAT, OCT/MATE transporters,
passive reabsorption, and 3-compartment (plasma/tubular fluid/urine) simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

__all__ = [
    "RenalTransportResult",
    "oat_secretion_rate",
    "oct_secretion_rate",
    "reabsorption_rate",
    "simulate_renal_transport",
    "drug_drug_renal_interaction",
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenalTransportResult:
    """Result of extended renal transport simulation."""

    drug_name: str
    times_h: list[float]
    c_plasma: list[float]
    c_tubular: list[float]
    cumulative_urine_mg: list[float]
    cl_renal_eff: float
    cl_filtration: float
    cl_secretion_net: float
    fe_urine: float
    auc_plasma: float
    cmax: float
    tmax_h: float
    notes: str


# ---------------------------------------------------------------------------
# Core transport rate functions
# ---------------------------------------------------------------------------


def oat_secretion_rate(
    c_plasma_mg_L: float,
    vmax_OAT_mg_per_h: float,
    km_OAT_mg_L: float,
    fu_plasma: float = 1.0,
) -> float:
    """OAT1/OAT3 tubular secretion rate for anions (mg/h).

    Uses Michaelis-Menten kinetics on unbound plasma concentration.

    Parameters
    ----------
    c_plasma_mg_L : float
        Total plasma concentration (mg/L).
    vmax_OAT_mg_per_h : float
        Maximum secretion rate (mg/h).
    km_OAT_mg_L : float
        Michaelis constant (mg/L).
    fu_plasma : float
        Fraction unbound in plasma (0, 1].

    Returns
    -------
    float
        Secretion rate (mg/h).
    """
    if vmax_OAT_mg_per_h <= 0 or km_OAT_mg_L <= 0:
        raise ValueError("vmax_OAT and km_OAT must be > 0")
    fu = float(np.clip(fu_plasma, 1e-6, 1.0))
    c_unbound = fu * max(c_plasma_mg_L, 0.0)
    return vmax_OAT_mg_per_h * c_unbound / (km_OAT_mg_L + c_unbound)


def oct_secretion_rate(
    c_plasma_mg_L: float,
    vmax_OCT_mg_per_h: float,
    km_OCT_mg_L: float,
    fu_plasma: float = 1.0,
) -> float:
    """OCT2/MATE tubular secretion rate for cations (mg/h).

    Uses Michaelis-Menten kinetics on unbound plasma concentration.

    Parameters
    ----------
    c_plasma_mg_L : float
        Total plasma concentration (mg/L).
    vmax_OCT_mg_per_h : float
        Maximum secretion rate (mg/h).
    km_OCT_mg_L : float
        Michaelis constant (mg/L).
    fu_plasma : float
        Fraction unbound in plasma (0, 1].

    Returns
    -------
    float
        Secretion rate (mg/h).
    """
    if vmax_OCT_mg_per_h <= 0 or km_OCT_mg_L <= 0:
        raise ValueError("vmax_OCT and km_OCT must be > 0")
    fu = float(np.clip(fu_plasma, 1e-6, 1.0))
    c_unbound = fu * max(c_plasma_mg_L, 0.0)
    return vmax_OCT_mg_per_h * c_unbound / (km_OCT_mg_L + c_unbound)


def reabsorption_rate(
    c_tubule_mg_L: float,
    logP: float,
    urine_flow_mL_per_h: float = 60.0,
) -> float:
    """Passive reabsorption rate from tubular fluid (mg/h).

    Reabsorption fraction follows a sigmoid function of logP:
      k_reabs = sigmoid(logP) * 0.8

    Parameters
    ----------
    c_tubule_mg_L : float
        Tubular fluid concentration (mg/L).
    logP : float
        Lipophilicity. Higher logP → more reabsorption.
    urine_flow_mL_per_h : float
        Urine flow rate (mL/h).

    Returns
    -------
    float
        Reabsorption rate (mg/h).
    """
    # sigmoid centred at logP = 0
    k_reabs = 0.8 / (1.0 + math.exp(-logP))
    # tubular fluid volume flow in L/h
    flow_L_h = urine_flow_mL_per_h / 1000.0
    return k_reabs * max(c_tubule_mg_L, 0.0) * flow_L_h


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_simulate_inputs(
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    gfr_mL_per_min: float,
    fu_plasma: float,
    vmax_OAT: float,
    km_OAT: float,
    vmax_OCT: float,
    km_OCT: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_hepatic_L_per_h < 0:
        raise ValueError("cl_hepatic_L_per_h must be >= 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if gfr_mL_per_min <= 0:
        raise ValueError("gfr_mL_per_min must be > 0")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if vmax_OAT <= 0:
        raise ValueError("vmax_OAT must be > 0")
    if km_OAT <= 0:
        raise ValueError("km_OAT must be > 0")
    if vmax_OCT <= 0:
        raise ValueError("vmax_OCT must be > 0")
    if km_OCT <= 0:
        raise ValueError("km_OCT must be > 0")


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate_renal_transport(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vd_L: float,
    vmax_OAT: float,
    km_OAT: float,
    vmax_OCT: float,
    km_OCT: float,
    gfr_mL_per_min: float = 120.0,
    fu_plasma: float = 0.1,
    logP: float = 0.0,
    route: str = "oral",
    ka_per_h: float = 1.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> RenalTransportResult:
    """Simulate 3-compartment renal transport (plasma, tubule, urine).

    Compartments
    ------------
    1. Plasma: absorption, hepatic elimination, renal loss (filtration + secretion)
    2. Tubular fluid: drug filtered + secreted - reabsorbed
    3. Urine (sink): cumulative excretion

    ODE system (Forward Euler):
      dA_gut/dt     = -ka * A_gut                    (oral only)
      dA_plasma/dt  = abs_rate - (CL_hep/Vd)*A_plasma
                      - (CL_filt/Vd)*A_plasma
                      - (R_OAT + R_OCT)/Vd * Vd  ... (secretion terms)
      dA_tubule/dt  = CL_filt * C_plasma
                      + R_OAT + R_OCT
                      - R_reabs
                      - CL_tubule_out * C_tubule
      A_urine (cumulative) += dt * CL_tubule_out * C_tubule

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose (mg).
    cl_hepatic_L_per_h : float
        Hepatic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    vmax_OAT : float
        OAT transporter Vmax (mg/h).
    km_OAT : float
        OAT transporter Km (mg/L).
    vmax_OCT : float
        OCT/MATE transporter Vmax (mg/h).
    km_OCT : float
        OCT/MATE transporter Km (mg/L).
    gfr_mL_per_min : float
        Glomerular filtration rate (mL/min).
    fu_plasma : float
        Fraction unbound in plasma.
    logP : float
        Drug lipophilicity (for reabsorption).
    route : str
        'oral' or 'iv'.
    ka_per_h : float
        First-order oral absorption rate constant (h^-1).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration time step (h).

    Returns
    -------
    RenalTransportResult
    """
    _validate_simulate_inputs(
        dose_mg,
        cl_hepatic_L_per_h,
        vd_L,
        gfr_mL_per_min,
        fu_plasma,
        vmax_OAT,
        km_OAT,
        vmax_OCT,
        km_OCT,
    )

    # GFR in L/h
    gfr_L_h = gfr_mL_per_min * 0.06
    cl_filtration = gfr_L_h * fu_plasma  # L/h

    # Urine flow (mL/h): typical ~60 mL/h
    urine_flow_mL_h = 60.0

    # Tubule volume (mL) — conceptual, ~120 mL
    v_tubule_L = 0.12  # L

    # Time array
    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]

    # State: A_plasma (mg), A_gut (mg), A_tubule (mg), A_urine (mg cumul)
    a_gut = dose_mg if route == "oral" else 0.0
    a_plasma = 0.0 if route == "oral" else dose_mg
    a_tubule = 0.0
    a_urine = 0.0

    c_plasma_list: list[float] = []
    c_tubular_list: list[float] = []
    cumul_urine_list: list[float] = []

    # Track secretion rates for summary
    total_secreted = 0.0
    total_reabsorbed = 0.0

    for _step in range(n_steps):
        c_plasma = max(a_plasma / vd_L, 0.0)
        c_tubule = max(a_tubule / v_tubule_L, 0.0)

        c_plasma_list.append(c_plasma)
        c_tubular_list.append(c_tubule)
        cumul_urine_list.append(a_urine)

        # Absorption rate (mg/h)
        abs_rate = ka_per_h * a_gut if route == "oral" else 0.0

        # Filtration rate (mg/h): filtered amount enters tubule
        r_filt = cl_filtration * c_plasma  # mg/h

        # Active secretion (mg/h)
        r_oat = oat_secretion_rate(c_plasma, vmax_OAT, km_OAT, fu_plasma)
        r_oct = oct_secretion_rate(c_plasma, vmax_OCT, km_OCT, fu_plasma)
        r_secretion = r_oat + r_oct

        # Reabsorption from tubule (mg/h)
        r_reabs = reabsorption_rate(c_tubule, logP, urine_flow_mL_h)

        # Urinary excretion (tubule → urine): urine_flow clears tubule
        urine_flow_L_h = urine_flow_mL_h / 1000.0
        r_excretion = urine_flow_L_h * c_tubule  # mg/h

        # Hepatic elimination (mg/h)
        r_hep = (cl_hepatic_L_per_h / vd_L) * a_plasma

        # Renal removal from plasma: filtration + secretion (net from plasma)
        r_renal_plasma = r_filt + r_secretion

        # Forward Euler updates
        d_gut = -ka_per_h * a_gut if route == "oral" else 0.0
        d_plasma = abs_rate - r_hep - r_renal_plasma
        d_tubule = r_filt + r_secretion - r_reabs - r_excretion
        d_urine = max(r_excretion - r_reabs, 0.0) + r_reabs * 0.0  # net to urine
        # corrected: net urine accumulation = excretion (after reabsorption already removed)
        d_urine = r_excretion

        a_gut = max(a_gut + dt_h * d_gut, 0.0)
        a_plasma = max(a_plasma + dt_h * d_plasma, 0.0)
        a_tubule = max(a_tubule + dt_h * d_tubule, 0.0)
        a_urine = a_urine + dt_h * d_urine

        total_secreted += dt_h * r_secretion
        total_reabsorbed += dt_h * r_reabs

    times_arr = np.array(times)
    cp_arr = np.array(c_plasma_list)

    # AUC via trapezoidal rule
    auc_plasma = float(np_trapz(cp_arr, times_arr))
    cmax = float(np.max(cp_arr))
    tmax_h = float(times_arr[int(np.argmax(cp_arr))])

    # Effective renal clearances
    fe_urine = cumul_urine_list[-1] / dose_mg if dose_mg > 0 else math.nan
    cl_renal_eff = auc_plasma * 0.0  # will compute from AUC
    if auc_plasma > 0:
        cl_renal_eff = cumul_urine_list[-1] / auc_plasma
    cl_secretion_net = total_secreted / max(auc_plasma, 1e-12)

    notes = (
        f"Route: {route}. "
        f"Total secreted: {total_secreted:.3f} mg, "
        f"total reabsorbed: {total_reabsorbed:.3f} mg, "
        f"fe_urine: {fe_urine:.3f}."
    )

    return RenalTransportResult(
        drug_name=drug_name,
        times_h=times,
        c_plasma=c_plasma_list,
        c_tubular=c_tubular_list,
        cumulative_urine_mg=cumul_urine_list,
        cl_renal_eff=cl_renal_eff,
        cl_filtration=cl_filtration,
        cl_secretion_net=cl_secretion_net,
        fe_urine=float(fe_urine),
        auc_plasma=auc_plasma,
        cmax=cmax,
        tmax_h=tmax_h,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# DDI via renal transporter inhibition
# ---------------------------------------------------------------------------


def drug_drug_renal_interaction(
    substrate: dict,
    inhibitor: dict,
) -> dict:
    """Estimate AUCR from renal transporter inhibition.

    Uses static model: R = 1 + I_kidney / Ki_OAT

    Parameters
    ----------
    substrate : dict
        Keys: 'drug_name', 'vmax_OAT' (mg/h), 'km_OAT' (mg/L),
              'vmax_OCT' (mg/h), 'km_OCT' (mg/L), 'fu_plasma',
              'cl_hepatic_L_per_h', 'vd_L', 'dose_mg'.
    inhibitor : dict
        Keys: 'drug_name', 'dose_mg', 'ki_OAT' (mg/L), 'ki_OCT' (mg/L),
              'vd_L' (volume of distribution L), 'fu_plasma'.

    Returns
    -------
    dict
        'aucr_OAT', 'aucr_OCT', 'aucr_combined', 'classification',
        'inhibitor', 'substrate', 'notes'.
    """
    # Estimate inhibitor kidney concentration ≈ portal/systemic concentration
    inh_dose = float(inhibitor.get("dose_mg", 0.0))
    inh_vd = float(inhibitor.get("vd_L", 50.0))
    inh_fu = float(inhibitor.get("fu_plasma", 0.1))
    # Peak unbound plasma concentration as surrogate for kidney exposure
    i_kidney = (inh_dose / inh_vd) * inh_fu

    ki_oat = float(inhibitor.get("ki_OAT", float("inf")))
    ki_oct = float(inhibitor.get("ki_OCT", float("inf")))

    # Inhibition of OAT-mediated secretion
    r_oat = 1.0 + i_kidney / ki_oat if ki_oat > 0 else 1.0
    r_oct = 1.0 + i_kidney / ki_oct if ki_oct > 0 else 1.0

    # AUCR: combined inhibition of secretion reduces renal CL,
    # increasing plasma AUC proportionally (simplified static model)
    # Fraction of renal CL via secretion (assumed 50% each from OAT/OCT)
    f_oat = 0.5
    f_oct = 0.5
    # Fold-increase in substrate AUC = 1 / (1 - fi_renal * (1 - 1/R))
    fi_oat = f_oat * (1.0 - 1.0 / r_oat)
    fi_oct = f_oct * (1.0 - 1.0 / r_oct)
    fi_combined = fi_oat + fi_oct
    # Prevent division by zero
    denom = max(1.0 - fi_combined, 0.01)
    aucr_combined = 1.0 / denom

    # Classification per FDA guidance
    if aucr_combined >= 2.0:
        classification = "strong inhibition"
    elif aucr_combined >= 1.5:
        classification = "moderate inhibition"
    elif aucr_combined >= 1.25:
        classification = "weak inhibition"
    else:
        classification = "negligible"

    return {
        "aucr_OAT": round(r_oat, 4),
        "aucr_OCT": round(r_oct, 4),
        "aucr_combined": round(aucr_combined, 4),
        "classification": classification,
        "inhibitor": inhibitor.get("drug_name", "unknown"),
        "substrate": substrate.get("drug_name", "unknown"),
        "i_kidney_mg_L": round(i_kidney, 6),
        "notes": (f"I_kidney={i_kidney:.4f} mg/L; Ki_OAT={ki_oat} mg/L; Ki_OCT={ki_oct} mg/L."),
    }
