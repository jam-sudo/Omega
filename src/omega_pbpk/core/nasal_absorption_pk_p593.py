"""Nasal drug absorption PK model (Phase 593).

Models nasal drug delivery PK with mucociliary clearance, absorption through
nasal epithelium, and systemic distribution using a two-compartment approach:
nasal cavity and systemic circulation.

Model structure
---------------
Two compartments:

  1. Nasal cavity — drug enters here, absorbs to systemic or clears via MCC
  2. Systemic — absorbed drug distributes and eliminates

ODE system
----------
  dA_nasal/dt = -(ka_nasal + k_mcc) * A_nasal
  dC_sys/dt   = ka_nasal * A_nasal / vd_sys - (cl_sys / vd_sys) * C_sys

References
----------
- Illum L, J Pharm Pharmacol. 2003;55(1):3-12 (nasal drug delivery)
- Washington N et al., Eur J Pharm Biopharm. 2000;50:149-158 (mucociliary clearance)
- Merkus FW et al., Pharm Res. 1998;15(4):550-556 (nasal bioavailability)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_K_MCC = 0.5  # mucociliary clearance rate (/h)
_KA_MIN = 0.05
_KA_MAX = 3.0
_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class NasalAbsorptionPKResult:
    """Results from nasal absorption PK simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    dose_mg : Administered dose (mg).
    route : Administration route ('nasal').
    logP : Lipophilicity (log partition coefficient).
    times_h : Simulation time points (h).
    c_nasal_mg_mL : Drug concentration in nasal cavity (mg/mL).
    c_systemic_mg_L : Systemic plasma concentration (mg/L).
    cmax_mg_L : Peak systemic concentration (mg/L).
    tmax_h : Time of peak systemic concentration (h).
    auc_mg_h_per_L : AUC (mg*h/L), trapezoidal rule.
    t_half_h : Systemic elimination half-life (h).
    bioavailability_pct : Bioavailability percentage relative to theoretical.
    f_absorbed_nasal : Fraction of nasal dose absorbed systemically (vs cleared).
    notes : Summary text.
    """

    drug_name: str
    dose_mg: float
    route: str
    logP: float
    times_h: list
    c_nasal_mg_mL: list
    c_systemic_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_h: float
    bioavailability_pct: float
    f_absorbed_nasal: float
    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _trapezoidal_auc(times: list, concs: list) -> float:
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_nasal_absorption(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    mw_Da: float = 300.0,
    nasal_volume_mL: float = 0.5,
    t_end_h: float = 8.0,
    dt_h: float = 0.02,
) -> NasalAbsorptionPKResult:
    """Simulate nasal drug absorption PK.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Administered nasal dose (mg).
    logP : Lipophilicity (octanol-water log partition coefficient).
    cl_sys_L_per_h : Systemic clearance (L/h).
    vd_sys_L : Systemic volume of distribution (L).
    mw_Da : Molecular weight (Da). Larger MW reduces absorption.
    nasal_volume_mL : Volume of nasal cavity fluid (mL).
    t_end_h : Simulation end time (h).
    dt_h : Forward Euler time step (h).

    Returns
    -------
    NasalAbsorptionPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be positive, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be positive, got {vd_sys_L}")
    if nasal_volume_mL <= 0:
        raise ValueError(f"nasal_volume_mL must be positive, got {nasal_volume_mL}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be positive, got {mw_Da}")

    # --- Compute nasal absorption rate constant ---
    ka_nasal_raw = 0.5 * logP + 0.3
    ka_nasal_raw = _clamp(ka_nasal_raw, _KA_MIN, _KA_MAX)
    # MW effect: larger MW reduces absorption
    ka_nasal = ka_nasal_raw * (500.0 / max(300.0, mw_Da))
    ka_nasal = _clamp(ka_nasal, _KA_MIN, _KA_MAX)

    k_mcc = _K_MCC
    k_total = ka_nasal + k_mcc

    # --- Fraction absorbed nasally (vs mucociliary clearance) ---
    f_absorbed_nasal = ka_nasal / k_total

    # --- Forward Euler ODE integration ---
    n_steps = max(1, int(round(t_end_h / dt_h)))
    times_h: list = []
    c_nasal_mg_mL: list = []
    c_systemic_mg_L: list = []

    a_nasal = dose_mg  # amount in nasal cavity (mg)
    c_sys = 0.0  # systemic concentration (mg/L)

    t = 0.0
    times_h.append(t)
    c_nasal_mg_mL.append(a_nasal / nasal_volume_mL)
    c_systemic_mg_L.append(c_sys)

    for _ in range(n_steps):
        da_nasal = -k_total * a_nasal
        dc_sys = (ka_nasal * a_nasal / vd_sys_L) - (cl_sys_L_per_h / vd_sys_L) * c_sys

        a_nasal = max(0.0, a_nasal + da_nasal * dt_h)
        c_sys = max(0.0, c_sys + dc_sys * dt_h)
        t += dt_h

        times_h.append(t)
        c_nasal_mg_mL.append(a_nasal / nasal_volume_mL)
        c_systemic_mg_L.append(c_sys)

    # --- PK metrics ---
    cmax_mg_L = max(c_systemic_mg_L)
    tmax_h = times_h[c_systemic_mg_L.index(cmax_mg_L)]
    auc = _trapezoidal_auc(times_h, c_systemic_mg_L)

    # Elimination half-life from terminal phase
    k_el = cl_sys_L_per_h / vd_sys_L
    t_half_h = _LN2 / k_el if k_el > 0 else float("inf")

    # Bioavailability vs theoretical
    auc_theoretical = dose_mg * f_absorbed_nasal / cl_sys_L_per_h
    if auc_theoretical > 0:
        bioavailability_pct = min(100.0, auc / auc_theoretical * 100.0)
    else:
        bioavailability_pct = 0.0

    notes = (
        f"Nasal absorption: ka={ka_nasal:.3f}/h, k_mcc={k_mcc:.2f}/h, "
        f"f_absorbed={f_absorbed_nasal:.3f}, "
        f"MW_effect={500.0 / max(300.0, mw_Da):.3f}, "
        f"t_half={t_half_h:.2f}h"
    )

    return NasalAbsorptionPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="nasal",
        logP=logP,
        times_h=times_h,
        c_nasal_mg_mL=c_nasal_mg_mL,
        c_systemic_mg_L=c_systemic_mg_L,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        t_half_h=t_half_h,
        bioavailability_pct=bioavailability_pct,
        f_absorbed_nasal=f_absorbed_nasal,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Route comparison
# ---------------------------------------------------------------------------


def compare_nasal_routes(
    drug_name: str,
    dose_mg: float,
    logp_values: list,
    cl_sys: float,
    vd_sys: float,
) -> list:
    """Compare nasal absorption across different logP values.

    Simulates nasal absorption for each logP value and returns results
    sorted by bioavailability descending.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Administered dose (mg).
    logp_values : List of logP values to evaluate.
    cl_sys : Systemic clearance (L/h).
    vd_sys : Systemic volume of distribution (L).

    Returns
    -------
    list of NasalAbsorptionPKResult sorted by bioavailability_pct descending.
    """
    results = []
    for logp in logp_values:
        result = simulate_nasal_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logP=logp,
            cl_sys_L_per_h=cl_sys,
            vd_sys_L=vd_sys,
        )
        results.append(result)

    results.sort(key=lambda r: r.bioavailability_pct, reverse=True)
    return results
