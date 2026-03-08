"""Phase 214 — Dynamic drug-protein binding kinetics in plasma (3-compartment ODE).

Simulates reversible binding of a drug to albumin and alpha-1-acid glycoprotein (AAG)
using a forward-Euler ODE system following an IV bolus dose.

Compartments
------------
C_free       : unbound drug (mg/L)
C_bound_alb  : albumin-bound drug (mg/L)
C_bound_aag  : AAG-bound drug (mg/L)

ODEs
----
dC_free/dt     = -kon_alb * C_free * (Alb_cap - C_bound_alb) + koff_alb * C_bound_alb
                 - kon_aag * C_free * (AAG_cap - C_bound_aag) + koff_aag * C_bound_aag
                 - ke * C_free
dC_bound_alb/dt =  kon_alb * C_free * (Alb_cap - C_bound_alb) - koff_alb * C_bound_alb
dC_bound_aag/dt =  kon_aag * C_free * (AAG_cap - C_bound_aag) - koff_aag * C_bound_aag
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Default binding parameters
# ---------------------------------------------------------------------------
_ALB_CAPACITY_DEFAULT = 40.0  # mg/L  (simplified albumin binding capacity)
_AAG_CAPACITY_DEFAULT = 5.0  # mg/L  (alpha-1-acid glycoprotein capacity)
_KON_ALB = 0.5  # /h/(mg/L)  — albumin association
_KOFF_ALB = 5.0  # /h         — albumin dissociation  → Kd = 10 mg/L
_KON_AAG = 1.0  # /h/(mg/L)  — AAG association
_KOFF_AAG = 2.0  # /h         — AAG dissociation  → Kd = 2 mg/L

_DT_H = 0.01  # forward-Euler time step (h)


# ---------------------------------------------------------------------------
# Result dataclass (plain — contains list fields)
# ---------------------------------------------------------------------------


@dataclass
class ProteinBindingODEResult:
    """Simulation result for plasma drug-protein binding ODE."""

    drug_name: str
    dose_mg: float
    times_h: list[float] = field(default_factory=list)
    c_free_mg_L: list[float] = field(default_factory=list)
    c_bound_alb_mg_L: list[float] = field(default_factory=list)
    c_bound_aag_mg_L: list[float] = field(default_factory=list)
    c_total_mg_L: list[float] = field(default_factory=list)
    fu_at_cmax: float = 0.0
    fu_at_trough: float = 0.0
    auc_free_mg_h_per_L: float = 0.0
    auc_total_mg_h_per_L: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_simulate_inputs(dose_mg: float, vd_L: float, cl_L_per_h: float) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")


def _trapz(xs: list[float], ys: list[float]) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(xs)):
        total += 0.5 * (ys[i - 1] + ys[i]) * (xs[i] - xs[i - 1])
    return total


def _safe_fu(c_free: float, c_total: float) -> float:
    """fu = C_free / C_total; returns 1.0 when both are essentially zero."""
    if c_total <= 1e-15:
        return 1.0
    return max(0.0, min(1.0, c_free / c_total))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_protein_binding(
    drug_name: str,
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    alb_capacity: float = _ALB_CAPACITY_DEFAULT,
    aag_capacity: float = _AAG_CAPACITY_DEFAULT,
    kon_alb: float = _KON_ALB,
    koff_alb: float = _KOFF_ALB,
    kon_aag: float = _KON_AAG,
    koff_aag: float = _KOFF_AAG,
    t_end_h: float = 24.0,
    dt_h: float = _DT_H,
    notes: str = "",
) -> ProteinBindingODEResult:
    """Simulate plasma drug-protein binding kinetics via forward-Euler ODE.

    Assumes an instantaneous IV bolus at t=0 distributing into volume *vd_L*.
    Elimination acts only on the free (unbound) fraction.

    Parameters
    ----------
    drug_name:
        Compound identifier.
    dose_mg:
        IV bolus dose in milligrams.
    vd_L:
        Volume of distribution (litres).
    cl_L_per_h:
        Total clearance (L/h); applied as ke = CL/Vd on C_free.
    alb_capacity:
        Maximum albumin binding capacity (mg/L).
    aag_capacity:
        Maximum AAG binding capacity (mg/L).
    kon_alb, koff_alb:
        Association/dissociation rate constants for albumin.
    kon_aag, koff_aag:
        Association/dissociation rate constants for AAG.
    t_end_h:
        Simulation end time (hours).
    dt_h:
        Forward-Euler time step (hours).
    notes:
        Optional annotation.

    Returns
    -------
    ProteinBindingODEResult
    """
    _validate_simulate_inputs(dose_mg, vd_L, cl_L_per_h)

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # Initial conditions: full dose in free compartment
    c_free = dose_mg / vd_L
    c_bound_alb = 0.0
    c_bound_aag = 0.0

    times: list[float] = []
    cf_list: list[float] = []
    ca_list: list[float] = []
    cb_list: list[float] = []
    ct_list: list[float] = []

    t = 0.0
    n_steps = max(1, int(round(t_end_h / dt_h)))
    for _ in range(n_steps + 1):
        c_total = c_free + c_bound_alb + c_bound_aag

        times.append(t)
        cf_list.append(c_free)
        ca_list.append(c_bound_alb)
        cb_list.append(c_bound_aag)
        ct_list.append(c_total)

        if t >= t_end_h:
            break

        # Free albumin / AAG capacity (bounded below at 0)
        alb_free_cap = max(0.0, alb_capacity - c_bound_alb)
        aag_free_cap = max(0.0, aag_capacity - c_bound_aag)

        dcf = (
            -kon_alb * c_free * alb_free_cap
            + koff_alb * c_bound_alb
            - kon_aag * c_free * aag_free_cap
            + koff_aag * c_bound_aag
            - ke * c_free
        )
        dca = kon_alb * c_free * alb_free_cap - koff_alb * c_bound_alb
        dcb = kon_aag * c_free * aag_free_cap - koff_aag * c_bound_aag

        c_free = max(0.0, c_free + dcf * dt_h)
        c_bound_alb = max(0.0, c_bound_alb + dca * dt_h)
        c_bound_aag = max(0.0, c_bound_aag + dcb * dt_h)

        t = round(t + dt_h, 10)

    # Cmax occurs at t=0 (IV bolus)
    fu_at_cmax = _safe_fu(cf_list[0], ct_list[0])

    # Trough = last time point with non-zero total
    fu_at_trough = _safe_fu(cf_list[-1], ct_list[-1])

    auc_free = _trapz(times, cf_list)
    auc_total = _trapz(times, ct_list)

    return ProteinBindingODEResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_free_mg_L=cf_list,
        c_bound_alb_mg_L=ca_list,
        c_bound_aag_mg_L=cb_list,
        c_total_mg_L=ct_list,
        fu_at_cmax=fu_at_cmax,
        fu_at_trough=fu_at_trough,
        auc_free_mg_h_per_L=auc_free,
        auc_total_mg_h_per_L=auc_total,
        notes=notes,
    )


def assess_binding_saturation(result: ProteinBindingODEResult) -> dict[str, object]:
    """Assess whether albumin or AAG binding is saturated at peak concentration.

    Parameters
    ----------
    result:
        Output of :func:`simulate_protein_binding`.

    Returns
    -------
    dict with keys:
    - ``"alb_saturated"`` (bool): True if C_bound_alb ≥ 90 % of Alb_capacity at Cmax.
    - ``"aag_saturated"`` (bool): True if C_bound_aag ≥ 90 % of AAG_capacity at Cmax.
    - ``"fu_at_cmax"`` (float): Unbound fraction at Cmax.
    - ``"fu_at_trough"`` (float): Unbound fraction at trough.
    """
    # Saturation thresholds — 90 % of default capacities
    alb_sat_threshold = 0.9 * _ALB_CAPACITY_DEFAULT
    aag_sat_threshold = 0.9 * _AAG_CAPACITY_DEFAULT

    # Peak bound concentrations (at t=0 everything is free; peak bound develops quickly)
    peak_alb = max(result.c_bound_alb_mg_L) if result.c_bound_alb_mg_L else 0.0
    peak_aag = max(result.c_bound_aag_mg_L) if result.c_bound_aag_mg_L else 0.0

    return {
        "alb_saturated": peak_alb >= alb_sat_threshold,
        "aag_saturated": peak_aag >= aag_sat_threshold,
        "fu_at_cmax": result.fu_at_cmax,
        "fu_at_trough": result.fu_at_trough,
    }
