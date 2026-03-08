"""Fed vs Fasted State Drug Dissolution Model (Phase 611).

Models dissolution and absorption differences between fed and fasted GI states
using the Noyes-Whitney simplified dissolution model with state-dependent
physiological parameters.

Fed state changes:
- Increased GI fluid volume (500 vs 250 mL)
- Delayed gastric emptying (t1/2 = 2 h vs 0.5 h)
- Solubility modulation by bile acids (BCS-class dependent)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (plain — contains list fields)
# ---------------------------------------------------------------------------


@dataclass
class FedFastedDissolutionResult:
    """Results from a single fed or fasted dissolution simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    dose_mg : Administered dose (mg).
    fed_state : True if fed, False if fasted.
    times_h : Time vector (h).
    c_dissolved_mg_mL : Dissolved drug concentration in GI fluid (mg/mL).
    c_absorbed_mg : Cumulative absorbed amount (mg).
    c_plasma_mg_L : Systemic plasma concentration (mg/L).
    cmax_mg_L : Peak plasma concentration (mg/L).
    tmax_h : Time of peak plasma concentration (h).
    auc_mg_h_per_L : AUC0-t (mg*h/L) using trapezoidal rule.
    fraction_absorbed : Total absorbed / dose (clamped to 1.0).
    dissolution_rate_mg_per_h : Average dissolution rate (mg/h).
    food_effect_pct : % change in AUC vs fasted baseline (0 for single run).
    notes : Human-readable summary.
    """

    drug_name: str
    dose_mg: float
    fed_state: bool
    times_h: list
    c_dissolved_mg_mL: list
    c_absorbed_mg: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    fraction_absorbed: float
    dissolution_rate_mg_per_h: float
    food_effect_pct: float
    notes: str


# ---------------------------------------------------------------------------
# BCS solubility multipliers for fed state
# ---------------------------------------------------------------------------

_FED_SOLUBILITY_MULTIPLIER = {
    1: 1.0,  # high sol / high perm — no change
    2: 3.0,  # low sol / high perm  — bile acids boost
    3: 0.8,  # high sol / low perm  — slight decrease
    4: 2.0,  # low sol / low perm   — partial bile acid benefit
}

_HIGH_PERM_BCS = {1, 2}  # ka_abs = 0.5 /h
_LOW_PERM_BCS = {3, 4}  # ka_abs = 0.1 /h


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    dose_mg: float,
    bcs_class: int,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    solubility_fasted_mg_mL: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if bcs_class not in (1, 2, 3, 4):
        raise ValueError(f"bcs_class must be 1, 2, 3, or 4; got {bcs_class}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if solubility_fasted_mg_mL <= 0:
        raise ValueError(f"solubility_fasted_mg_mL must be > 0, got {solubility_fasted_mg_mL}")


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_fed_fasted_dissolution(
    drug_name: str,
    dose_mg: float,
    solubility_fasted_mg_mL: float,
    bcs_class: int,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    fed_state: bool = False,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> FedFastedDissolutionResult:
    """Simulate drug dissolution and absorption in fed or fasted state.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Oral dose (mg).
    solubility_fasted_mg_mL : Aqueous solubility in fasted state (mg/mL).
    bcs_class : BCS classification (1, 2, 3, or 4).
    logP : Octanol-water partition coefficient.
    mw_Da : Molecular weight (Da).
    cl_sys_L_per_h : Systemic clearance (L/h).
    vd_sys_L : Volume of distribution (L).
    fed_state : If True, apply fed-state GI parameters.
    t_end_h : Simulation duration (h).
    dt_h : Integration time step (h).

    Returns
    -------
    FedFastedDissolutionResult
    """
    _validate_inputs(dose_mg, bcs_class, cl_sys_L_per_h, vd_sys_L, solubility_fasted_mg_mL)

    # ------------------------------------------------------------------
    # State-dependent physiological parameters
    # ------------------------------------------------------------------
    V_gi = 500.0 if fed_state else 250.0  # mL
    k_empty = math.log(2) / 2.0 if fed_state else math.log(2) / 0.5  # /h

    sol_multiplier = _FED_SOLUBILITY_MULTIPLIER[bcs_class] if fed_state else 1.0
    effective_solubility = solubility_fasted_mg_mL * sol_multiplier  # mg/mL

    # Absorption rate constant — permeability-based
    ka_abs = 0.5 if bcs_class in _HIGH_PERM_BCS else 0.1  # /h

    # Dissolution rate constant (Noyes-Whitney simplified, MW-dependent)
    k_diss = 0.5 * (300.0 / max(200.0, mw_Da))  # /h

    # Elimination rate constant
    k_el = cl_sys_L_per_h / vd_sys_L  # /h

    C_sat = effective_solubility  # mg/mL

    # ------------------------------------------------------------------
    # State variables
    # ------------------------------------------------------------------
    A_undiss = dose_mg  # undissolved drug (mg)
    C_dissolved = 0.0  # dissolved drug in GI fluid (mg/mL)
    C_plasma = 0.0  # plasma concentration (mg/L)
    absorbed = 0.0  # cumulative absorbed amount (mg)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    n_steps = int(round(t_end_h / dt_h)) + 1
    times_h: list = []
    c_diss_list: list = []
    c_abs_list: list = []
    c_plasma_list: list = []

    t = 0.0
    for _ in range(n_steps):
        times_h.append(t)
        c_diss_list.append(C_dissolved)
        c_abs_list.append(absorbed)
        c_plasma_list.append(C_plasma)

        # Dissolution driver (Noyes-Whitney, undersaturated only)
        sat_factor = max(0.0, 1.0 - C_dissolved / C_sat) if C_sat > 0 else 0.0
        diss_flux = k_diss * A_undiss * sat_factor  # mg/h

        # Rate of change for dissolved concentration (mg/mL/h)
        abs_flux = ka_abs * C_dissolved  # /h * mg/mL → mg/mL/h (absorbed)
        empty_flux = k_empty * C_dissolved  # drainage /h
        dC_dissolved_dt = diss_flux / V_gi - abs_flux - empty_flux

        # Cumulative absorbed (mg/h × dt)
        d_absorbed = ka_abs * C_dissolved * V_gi  # mg/h
        # (C_dissolved in mg/mL × V_gi in mL → mg/h flux)

        # Plasma concentration (mg/L/h)
        # Input: ka_abs * C_dissolved [mg/mL] * V_gi [mL] = mg/h  → / Vd [L] = mg/L/h
        dC_plasma_dt = (ka_abs * C_dissolved * V_gi / vd_sys_L) - k_el * C_plasma

        # Forward Euler update
        A_undiss = max(0.0, A_undiss - diss_flux * dt_h)
        C_dissolved = max(0.0, C_dissolved + dC_dissolved_dt * dt_h)
        absorbed = absorbed + d_absorbed * dt_h
        C_plasma = max(0.0, C_plasma + dC_plasma_dt * dt_h)

        t += dt_h

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    cmax_mg_L = max(c_plasma_list)
    tmax_h = times_h[c_plasma_list.index(cmax_mg_L)]

    # Trapezoidal AUC
    x = times_h
    y = c_plasma_list
    auc = sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))

    fraction_absorbed = min(1.0, absorbed / dose_mg) if dose_mg > 0 else 0.0
    dissolution_rate_mg_per_h = absorbed / t_end_h if t_end_h > 0 else 0.0

    state_label = "fed" if fed_state else "fasted"
    notes = (
        f"BCS class {bcs_class}, {state_label} state. "
        f"Effective solubility {effective_solubility:.3f} mg/mL "
        f"(multiplier {sol_multiplier}x). "
        f"V_GI={V_gi:.0f} mL, k_empty={k_empty:.3f}/h, "
        f"k_diss={k_diss:.3f}/h, ka_abs={ka_abs}/h."
    )

    return FedFastedDissolutionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fed_state=fed_state,
        times_h=times_h,
        c_dissolved_mg_mL=c_diss_list,
        c_absorbed_mg=c_abs_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        fraction_absorbed=fraction_absorbed,
        dissolution_rate_mg_per_h=dissolution_rate_mg_per_h,
        food_effect_pct=0.0,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison wrapper
# ---------------------------------------------------------------------------


def compare_fed_fasted(
    drug_name: str,
    dose_mg: float,
    solubility_fasted_mg_mL: float,
    bcs_class: int,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
) -> dict:
    """Compare fed vs fasted dissolution for the same drug.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Oral dose (mg).
    solubility_fasted_mg_mL : Fasted-state aqueous solubility (mg/mL).
    bcs_class : BCS classification (1–4).
    logP : Octanol-water partition coefficient.
    mw_Da : Molecular weight (Da).
    cl_sys_L_per_h : Systemic clearance (L/h).
    vd_sys_L : Volume of distribution (L).

    Returns
    -------
    dict with keys "fed", "fasted", and "food_effect_pct".
    """
    fasted_result = simulate_fed_fasted_dissolution(
        drug_name=drug_name,
        dose_mg=dose_mg,
        solubility_fasted_mg_mL=solubility_fasted_mg_mL,
        bcs_class=bcs_class,
        logP=logP,
        mw_Da=mw_Da,
        cl_sys_L_per_h=cl_sys_L_per_h,
        vd_sys_L=vd_sys_L,
        fed_state=False,
    )

    fed_result = simulate_fed_fasted_dissolution(
        drug_name=drug_name,
        dose_mg=dose_mg,
        solubility_fasted_mg_mL=solubility_fasted_mg_mL,
        bcs_class=bcs_class,
        logP=logP,
        mw_Da=mw_Da,
        cl_sys_L_per_h=cl_sys_L_per_h,
        vd_sys_L=vd_sys_L,
        fed_state=True,
    )

    if fasted_result.auc_mg_h_per_L > 0:
        food_effect_pct = (
            (fed_result.auc_mg_h_per_L - fasted_result.auc_mg_h_per_L)
            / fasted_result.auc_mg_h_per_L
            * 100.0
        )
    else:
        food_effect_pct = 0.0

    return {
        "fasted": fasted_result,
        "fed": fed_result,
        "food_effect_pct": food_effect_pct,
    }
