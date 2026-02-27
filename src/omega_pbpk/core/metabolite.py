"""Parent-metabolite PBPK extension.

Models the fate of one or more metabolites formed from a parent drug.
Each metabolite is assigned:

- Formation fraction (fm_met): fraction of total CLint producing this metabolite
- Its own PBPK parameters: fup, rbp, CLr, CLint (further metabolism), Kp, mw
- Route of elimination: hepatic metabolism, renal excretion, or both

The metabolite concentration profile is solved using a simplified 3-compartment
PBPK model (blood, liver, rest of body) rather than the full 35-state parent
ODE to keep computational cost manageable.  This is appropriate when:

  * The metabolite has different distribution properties from the parent
  * Metabolite PK is needed for efficacy/safety predictions
  * The metabolite itself undergoes further metabolism (e.g. active metabolite)

Metabolite ODEs (3-state: blood, liver, rest):
  dA_met_blood/dt  = fm_met × CLh_parent × C_parent_venous
                     - Q_total × (A_met_blood/V_blood - A_met_liver/V_liver/Kp_liver)
                     - CLr_met × A_met_blood/V_blood
  dA_met_liver/dt  = Q_liver × A_met_blood/V_blood
                     - Q_liver × A_met_liver/(V_liver × Kp_liver_met)
                     - CLint_met × fup_met × A_met_liver/(V_liver × Kp_liver_met)
  dA_met_rest/dt   = Q_rest × A_met_blood/V_blood
                     - Q_rest × A_met_rest/(V_rest × Kp_rest_met)

References:
  Leahy & Rowland (1984) Drug Metab Dispos 12:292
  Pang & Gillette (1979) J Pharmacokinet Biopharm 7:275
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from numpy.typing import NDArray
from typing import Any


__all__ = [
    "MetaboliteSpec",
    "MetaboliteResult",
    "simulate_metabolite",
]


# ---------------------------------------------------------------------------
# Metabolite specification
# ---------------------------------------------------------------------------

@dataclass
class MetaboliteSpec:
    """Parameters for a single metabolite.

    Args:
        name: Metabolite name (e.g. "M1", "nordiazepam").
        fm_parent: Fraction of parent drug that forms this metabolite
                   (0–1; sum over all metabolites ≤ 1.0).
        mw: Metabolite molecular weight (g/mol). Used for molar PK if needed.
        fup: Fraction unbound in plasma (0–1).
        rbp: Blood-to-plasma concentration ratio.
        clint_met_L_per_h: Intrinsic hepatic clearance for further metabolism of
                            this metabolite (L/h, 70 kg). 0 = not further metabolised.
        clr_met_L_per_h: Renal clearance of the metabolite (L/h, 70 kg).
        kp_liver: Liver:blood partition coefficient for the metabolite.
        kp_rest: Rest-of-body:blood partition coefficient (average peripheral).
        vd_L_per_kg: Volume of distribution of the metabolite (L/kg body weight).
                     Used to estimate 'rest' compartment volume.
        is_active: Whether this metabolite has pharmacological activity.
        is_toxic: Whether this metabolite has toxic potential.
    """

    name: str
    fm_parent: float = 1.0          # fraction of parent CLint forming this met
    mw: float = 300.0
    fup: float = 0.5
    rbp: float = 1.0
    clint_met_L_per_h: float = 0.0  # further metabolism of metabolite
    clr_met_L_per_h: float = 0.0    # renal clearance of metabolite
    kp_liver: float = 1.0
    kp_rest: float = 1.0
    vd_L_per_kg: float = 1.0        # apparent Vd in L/kg
    is_active: bool = False
    is_toxic: bool = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MetaboliteResult:
    """Simulation result for a single metabolite.

    Attributes:
        name: Metabolite name.
        time_h: Simulation time array (h).
        amounts_mg: Array of metabolite amounts [blood, liver, rest] (mg).
        plasma_conc_mg_L: Metabolite plasma concentration (mg/L).
        pk: Summary PK parameters dict.
        spec: Original MetaboliteSpec.
    """

    name: str
    time_h: NDArray[np.floating[Any]]
    amounts_mg: NDArray[np.floating[Any]]   # shape (n_time, 3) — blood, liver, rest
    plasma_conc_mg_L: NDArray[np.floating[Any]]
    spec: MetaboliteSpec

    @property
    def pk(self) -> dict[str, Any]:
        """Basic PK parameters: Cmax, Tmax, AUC, t½."""
        cp = self.plasma_conc_mg_L
        t = self.time_h
        cmax = float(np.max(cp))
        tmax = float(t[np.argmax(cp)])
        auc = float(np.trapezoid(cp, t))

        # Terminal half-life
        n = len(t)
        start = max(n // 2, 1)
        cp_term = cp[start:]
        t_term = t[start:]
        mask = cp_term > cmax * 0.01
        if np.sum(mask) >= 3:
            log_cp = np.log(np.maximum(cp_term[mask], 1e-30))
            coeffs = np.polyfit(t_term[mask], log_cp, 1)
            ke = -coeffs[0]
            half_life = 0.693 / max(ke, 1e-12) if ke > 0 else float("inf")
        else:
            half_life = float("inf")

        return {
            "Cmax_mg_L": round(cmax, 6),
            "Tmax_h": round(tmax, 4),
            "AUC_mg_h_L": round(auc, 4),
            "half_life_h": round(half_life, 4) if half_life < 1e6 else float("inf"),
        }


# ---------------------------------------------------------------------------
# Simulation function
# ---------------------------------------------------------------------------

def simulate_metabolite(
    metabolite: MetaboliteSpec,
    parent_time_h: NDArray[np.floating[Any]],
    parent_clh_rate_mg_h: NDArray[np.floating[Any]],
    body_weight: float = 70.0,
) -> MetaboliteResult:
    """Simulate metabolite PK given parent hepatic clearance rate profile.

    Uses Euler integration with the 3-compartment metabolite model:
      Blood → Liver (active formation site) → Rest of body
    Metabolite is formed at a rate = fm_parent × CLh_formation_rate(t).

    Args:
        metabolite: MetaboliteSpec with metabolite PK parameters.
        parent_time_h: Time array from parent simulation (h).
        parent_clh_rate_mg_h: Hepatic clearance rate of parent drug at each
                               time point (mg/h). This is the source term.
                               Typically: met_hepatic[i] / dt or dA_met_hepatic/dt
        body_weight: Body weight (kg) for organ volume scaling.

    Returns:
        MetaboliteResult with time course and PK summary.
    """
    from scipy.integrate import solve_ivp

    met = metabolite
    scale = body_weight / 70.0

    # Organ volumes (mL → L)
    v_blood = 5.2 * scale       # total blood volume (L)
    v_liver = 1.80 * scale      # liver volume (L)
    v_rest = max(met.vd_L_per_kg * body_weight - v_blood - v_liver, 0.5 * scale)

    # Organ blood flows (L/h)
    co = 390.0 * scale          # cardiac output
    q_liver = 96.6 * scale      # hepatic blood flow (HA + PV)
    q_rest = co - q_liver

    fup = max(met.fup, 1e-9)
    rbp = max(met.rbp, 1e-9)
    kp_liver = max(met.kp_liver, 1e-9)
    kp_rest = max(met.kp_rest, 1e-9)

    # Hepatic clearance of metabolite (well-stirred model)
    clint_met = met.clint_met_L_per_h * scale
    clh_met = (q_liver * fup * clint_met) / max(q_liver + fup * clint_met, 1e-12)

    # Interpolation function for parent-driven formation rate
    def formation_rate_at_t(t_val: float) -> float:
        """Metabolite formation rate (mg/h) at time t via linear interpolation."""
        return float(np.interp(t_val, parent_time_h, parent_clh_rate_mg_h))

    # ODE: [blood_amount, liver_amount, rest_amount]
    def _rhs_met(t: float, y: list[float]) -> list[float]:
        A_blood, A_liver, A_rest = y

        # Concentrations
        c_blood = max(A_blood, 0.0) / max(v_blood, 1e-12)
        c_liver = max(A_liver, 0.0) / max(v_liver * kp_liver, 1e-12)
        c_rest  = max(A_rest,  0.0) / max(v_rest  * kp_rest,  1e-12)

        # Formation from parent (metabolite enters liver)
        formation = met.fm_parent * formation_rate_at_t(t)

        # Hepatic elimination (well-stirred: acts on liver content)
        liver_plasma_conc = c_liver / rbp
        hepatic_elim = clh_met * liver_plasma_conc

        # Renal clearance (from blood)
        blood_plasma_conc = c_blood / rbp
        renal_elim = met.clr_met_L_per_h * scale * blood_plasma_conc

        # Mass transfer between compartments
        # Blood ↔ Liver
        flux_liver = q_liver * (c_blood - c_liver)
        # Blood ↔ Rest
        flux_rest  = q_rest  * (c_blood - c_rest)

        dA_blood = -flux_liver - flux_rest - renal_elim
        dA_liver = flux_liver  + formation  - hepatic_elim
        dA_rest  = flux_rest

        return [dA_blood, dA_liver, dA_rest]

    t_span = (float(parent_time_h[0]), float(parent_time_h[-1]))
    y0 = [0.0, 0.0, 0.0]

    sol = solve_ivp(
        _rhs_met,
        t_span=t_span,
        y0=y0,
        method="LSODA",
        t_eval=parent_time_h,
        rtol=1e-7,
        atol=1e-9,
        max_step=0.1,
    )

    amounts = np.maximum(sol.y.T, 0.0)  # shape (n_time, 3)

    # Plasma concentration from blood compartment
    v_plasma = v_blood / max(rbp, 1e-9)
    plasma_conc = amounts[:, 0] / max(v_plasma, 1e-12)

    return MetaboliteResult(
        name=met.name,
        time_h=sol.t,
        amounts_mg=amounts,
        plasma_conc_mg_L=plasma_conc,
        spec=met,
    )


def simulate_all_metabolites(
    metabolites: list[MetaboliteSpec],
    parent_time_h: NDArray[np.floating[Any]],
    parent_hepatic_amounts_mg: NDArray[np.floating[Any]],
    body_weight: float = 70.0,
    dt_h: float = 0.1,
) -> list[MetaboliteResult]:
    """Simulate all metabolites from a parent simulation.

    Computes the hepatic clearance rate from the cumulative `IDX_MET_HEPATIC`
    state of the parent simulation (derivative of metabolised drug amount).

    Args:
        metabolites: List of MetaboliteSpec objects.
        parent_time_h: Time array from parent simulation (h).
        parent_hepatic_amounts_mg: Cumulative metabolised parent drug (mg)
            at each time point — corresponds to amounts[:, IDX_MET_HEPATIC].
        body_weight: Body weight (kg).
        dt_h: Time step for numerical differentiation (h).

    Returns:
        List of MetaboliteResult, one per metabolite.
    """
    # Differentiate cumulative metabolised amount → instantaneous formation rate (mg/h)
    met_hepatic = np.asarray(parent_hepatic_amounts_mg, dtype=float)
    t = np.asarray(parent_time_h, dtype=float)
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]) if len(t) > 1 else 1.0)
    dt[dt <= 0] = dt_h
    clh_rate = np.diff(met_hepatic, prepend=0.0) / np.maximum(dt, 1e-12)
    clh_rate = np.maximum(clh_rate, 0.0)

    results = []
    for met in metabolites:
        result = simulate_metabolite(met, t, clh_rate, body_weight)
        results.append(result)
    return results
