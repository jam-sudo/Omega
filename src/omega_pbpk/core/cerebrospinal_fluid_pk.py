"""Phase 1145 — Cerebrospinal fluid (CSF) drug PK model.

Models drug distribution between plasma and CSF via the blood-CSF barrier
(BCSFB) for CNS infections (meningitis) and intrathecal drug delivery.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CSFPKResult:
    drug_name: str
    dose_mg: float
    logP: float
    route: str
    fu_plasma: float
    times_h: list
    c_plasma_mg_L: list
    c_csf_mg_L: list
    cmax_csf_mg_L: float
    tmax_csf_h: float
    auc_csf_mg_h_per_L: float
    auc_plasma_mg_h_per_L: float
    csf_to_plasma_auc_ratio: float
    csf_penetration_class: str
    meningitis_utility: bool
    notes: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_V_CSF_L = 0.15  # 150 mL total CSF volume
_KCP = 0.005  # /h  back-transfer CSF -> plasma
_K_CSF_DRAIN = (0.4 * 60) / 150  # = 0.16/h  (mL/min * min/h) / mL
_KPC_BASE = 0.01  # /h  base plasma -> CSF transfer coefficient


# ---------------------------------------------------------------------------
# Helper: trapz AUC
# ---------------------------------------------------------------------------


def _trapz_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt
    return auc


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_csf_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    route: str = "iv",
    fu_plasma: float = 0.1,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> CSFPKResult:
    """Simulate CSF drug PK using a 2-compartment plasma-CSF model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    logP:
        Octanol-water partition coefficient (log scale).
    route:
        ``"iv"`` (bolus into plasma) or ``"intrathecal"`` (bolus into CSF).
    fu_plasma:
        Unbound fraction in plasma (0 < fu_plasma <= 1).
    cl_L_per_h:
        Total plasma clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler time step (h).

    Returns
    -------
    CSFPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (-5 <= logP <= 10):
        raise ValueError("logP must be in [-5, 10]")
    if route not in {"iv", "intrathecal"}:
        raise ValueError("route must be 'iv' or 'intrathecal'")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")

    # --- Transfer rate constants ---
    logP_factor = max(0.1, min(3.0, logP / 3.0))
    fu_factor = min(2.0, fu_plasma * 10)
    kpc = _KPC_BASE * logP_factor * fu_factor  # plasma -> CSF (/h)
    kcp = _KCP  # CSF -> plasma (/h)
    k_csf_drain = _K_CSF_DRAIN  # CSF bulk drainage (/h)

    # Elimination rate from plasma (first-order approximation)
    ke = cl_L_per_h / vd_L  # /h

    # --- Initial conditions ---
    if route == "iv":
        c_plasma = dose_mg / vd_L  # mg/L
        c_csf = 0.0
    else:  # intrathecal
        c_plasma = 0.0
        c_csf = dose_mg / _V_CSF_L  # mg/L

    # --- Forward Euler simulation ---
    times: list = []
    c_plasma_list: list = []
    c_csf_list: list = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _ in range(n_steps):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_csf_list.append(c_csf)

        # ODEs
        # dC_plasma/dt = -ke*C_plasma - kpc*C_plasma + kcp*C_csf*(V_csf/Vd)
        dc_plasma_dt = -ke * c_plasma - kpc * c_plasma + kcp * c_csf * (_V_CSF_L / vd_L)
        # dC_csf/dt = kpc*(Vd/V_csf)*C_plasma - kcp*C_csf - k_csf_drain*C_csf
        dc_csf_dt = kpc * (vd_L / _V_CSF_L) * c_plasma - kcp * c_csf - k_csf_drain * c_csf

        c_plasma = max(0.0, c_plasma + dc_plasma_dt * dt_h)
        c_csf = max(0.0, c_csf + dc_csf_dt * dt_h)

        t = round(t + dt_h, 10)

    # --- Derived metrics ---
    cmax_csf = max(c_csf_list)
    tmax_csf = times[c_csf_list.index(cmax_csf)]
    auc_csf = _trapz_auc(times, c_csf_list)
    auc_plasma = _trapz_auc(times, c_plasma_list)

    if auc_plasma > 0:
        csf_to_plasma_ratio = auc_csf / auc_plasma
    else:
        csf_to_plasma_ratio = 0.0

    # Penetration class
    if csf_to_plasma_ratio < 0.05:
        penetration_class = "poor"
    elif csf_to_plasma_ratio < 0.2:
        penetration_class = "moderate"
    else:
        penetration_class = "good"

    meningitis_utility = csf_to_plasma_ratio > 0.1

    notes = (
        f"kpc={kpc:.4f}/h (logP_factor={logP_factor:.2f}, "
        f"fu_factor={fu_factor:.2f}), kcp={kcp}/h, "
        f"k_csf_drain={k_csf_drain:.4f}/h"
    )

    return CSFPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        route=route,
        fu_plasma=fu_plasma,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_csf_mg_L=c_csf_list,
        cmax_csf_mg_L=cmax_csf,
        tmax_csf_h=tmax_csf,
        auc_csf_mg_h_per_L=auc_csf,
        auc_plasma_mg_h_per_L=auc_plasma,
        csf_to_plasma_auc_ratio=csf_to_plasma_ratio,
        csf_penetration_class=penetration_class,
        meningitis_utility=meningitis_utility,
        notes=notes,
    )
