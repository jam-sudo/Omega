"""Phase 276 — Mechanistic renal tubular secretion PK model.

Different from core/renal_tubular_secretion_pk.py (Phase 969) in that this
module focuses specifically on the active secretion contribution, providing
a simpler 1-compartment model with configurable Vmax/Km, IV/oral routes,
urine concentration tracking, and a compare_secretion_capacity helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "RenalSecretionPKResult",
    "simulate_renal_secretion_pk",
    "compare_secretion_capacity",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GFR_L_PER_H = 7.5  # 120 mL/min
_V_PLASMA_L = 3.0
_CL_REAB_L_PER_H = 0.5  # passive tubular reabsorption
_KA_PER_H = 1.0  # oral absorption rate constant
_F_ORAL = 0.8  # oral bioavailability
_V_URINE_PER_H_L = 0.12  # urine flow ~2 mL/min = 0.12 L/h


# ---------------------------------------------------------------------------
# Result dataclass (plain — has list fields)
# ---------------------------------------------------------------------------


@dataclass
class RenalSecretionPKResult:
    drug_name: str
    dose_mg: float
    route: str
    times_h: list
    c_plasma_mg_L: list
    c_urine_mg_mL: list
    cmax_plasma_mg_L: float
    auc_plasma_mg_h_per_L: float
    cumulative_urine_mg: float
    renal_clearance_L_per_h: float
    fraction_excreted_unchanged: float
    secretion_contribution_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(times: list, values: list) -> float:
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def _cl_secretion(c_plasma: float, vmax_sec: float, km_sec: float) -> float:
    """Michaelis-Menten apparent secretion clearance (L/h)."""
    if c_plasma <= 0.0:
        return vmax_sec / km_sec  # intrinsic at zero concentration
    return vmax_sec / (km_sec + c_plasma)


def _cl_renal(c_plasma: float, fu: float, vmax_sec: float, km_sec: float) -> float:
    """Net renal clearance at a given plasma concentration (L/h)."""
    cl_filt = _GFR_L_PER_H * fu
    cl_sec = vmax_sec / (km_sec + max(0.0, c_plasma))
    cl_r = cl_filt + cl_sec - _CL_REAB_L_PER_H
    return max(0.0, cl_r)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_renal_secretion_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_hepatic_L_per_h: float,
    fu: float = 0.3,
    vmax_sec: float = 10.0,
    km_sec: float = 2.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> RenalSecretionPKResult:
    """Simulate 1-compartment PK with mechanistic renal tubular secretion.

    Parameters
    ----------
    drug_name : str
    dose_mg : float  Dose (mg); must be > 0
    route : str  "iv" or "oral"
    cl_hepatic_L_per_h : float  Hepatic clearance (L/h); must be >= 0
    fu : float  Unbound fraction in plasma (default 0.3)
    vmax_sec : float  Maximal active secretion rate (mg/h, default 10.0)
    km_sec : float  Michaelis constant for secretion (mg/L, default 2.0)
    t_end_h : float  Simulation end time (h, default 24)
    dt_h : float  Forward Euler step (h, default 0.05)

    Returns
    -------
    RenalSecretionPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_hepatic_L_per_h < 0:
        raise ValueError("cl_hepatic_L_per_h must be >= 0")
    if route not in {"iv", "oral"}:
        raise ValueError("route must be 'iv' or 'oral'")

    n_steps = int(math.ceil(t_end_h / dt_h)) + 1

    times: list = []
    c_plasma_list: list = []
    c_urine_list: list = []
    cumulative_urine_amount = 0.0
    urine_conc_prev = 0.0

    # Initial conditions
    if route == "iv":
        C = dose_mg / _V_PLASMA_L
        A_gut = 0.0
    else:
        C = 0.0
        A_gut = dose_mg  # full dose in depot (F applied via absorption flux)

    for i in range(n_steps):
        t = i * dt_h
        c_curr = max(0.0, C)

        # Net renal clearance at current concentration
        cl_r = _cl_renal(c_curr, fu, vmax_sec, km_sec)

        # Total elimination rate
        ke_total = (cl_hepatic_L_per_h + cl_r) / _V_PLASMA_L

        # Urine concentration: dC_urine/dt = CLr * C_plasma / V_urine_flow
        # We track cumulative amount and then divide by cumulative volume
        dU_dt = cl_r * c_curr  # mg/h going into urine

        # Accumulate urine
        if i > 0:
            cumulative_urine_amount += dU_dt * dt_h
            cumul_vol_L = t * _V_URINE_PER_H_L
            if cumul_vol_L > 0:
                urine_conc_prev = (cumulative_urine_amount / cumul_vol_L) * 1000.0 / 1000.0
                # mg/L → mg/mL: divide by 1000; mg/L already, convert: mg/L * (1L/1000mL) = mg/mL
                urine_conc_prev = cumulative_urine_amount / (cumul_vol_L * 1000.0)

        times.append(t)
        c_plasma_list.append(c_curr)
        c_urine_list.append(urine_conc_prev)

        # ODE step (forward Euler)
        if route == "oral":
            oral_input = _KA_PER_H * A_gut * _F_ORAL / _V_PLASMA_L
            dC = oral_input - ke_total * c_curr
            A_gut = max(0.0, A_gut - _KA_PER_H * A_gut * dt_h)
        else:
            dC = -ke_total * c_curr

        C = max(0.0, C + dC * dt_h)

    # Summary statistics
    cmax = max(c_plasma_list)
    auc = _trapz(times, c_plasma_list)

    # Average renal clearance (at Cmax for representativeness)
    cl_r_avg = _cl_renal(cmax, fu, vmax_sec, km_sec)
    cl_filt = _GFR_L_PER_H * fu

    # Fraction excreted unchanged = cumulative_urine / dose
    fe = cumulative_urine_amount / dose_mg if dose_mg > 0 else 0.0
    fe = max(0.0, min(1.0, fe))

    # Secretion contribution: what fraction of renal clearance is secretion?
    cl_sec_at_cmax = vmax_sec / (km_sec + max(0.0, cmax))
    if cl_r_avg > 0:
        sec_pct = 100.0 * cl_sec_at_cmax / (cl_r_avg + cl_sec_at_cmax)
    else:
        sec_pct = 0.0

    notes = (
        f"route={route}; CLfilt={cl_filt:.2f} L/h; "
        f"CLsec(Cmax)={cl_sec_at_cmax:.2f} L/h; "
        f"CLreab={_CL_REAB_L_PER_H:.2f} L/h; "
        f"CLr_net={cl_r_avg:.2f} L/h; fe={fe:.3f}"
    )

    return RenalSecretionPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_urine_mg_mL=c_urine_list,
        cmax_plasma_mg_L=cmax,
        auc_plasma_mg_h_per_L=auc,
        cumulative_urine_mg=cumulative_urine_amount,
        renal_clearance_L_per_h=cl_r_avg,
        fraction_excreted_unchanged=fe,
        secretion_contribution_pct=sec_pct,
        notes=notes,
    )


def compare_secretion_capacity(
    drug_name: str,
    dose_mg: float,
    cl_hepatic_L_per_h: float,
    vmax_values: list = None,
) -> list[RenalSecretionPKResult]:
    """Compare secretion capacity across multiple Vmax values.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    cl_hepatic_L_per_h : float
    vmax_values : list[float], optional  defaults to [1.0, 5.0, 10.0, 20.0]

    Returns
    -------
    list[RenalSecretionPKResult] sorted by fraction_excreted_unchanged descending
    """
    if vmax_values is None:
        vmax_values = [1.0, 5.0, 10.0, 20.0]

    results = []
    for vmax in vmax_values:
        r = simulate_renal_secretion_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route="iv",
            cl_hepatic_L_per_h=cl_hepatic_L_per_h,
            vmax_sec=vmax,
        )
        results.append(r)

    results.sort(key=lambda r: r.fraction_excreted_unchanged, reverse=True)
    return results
