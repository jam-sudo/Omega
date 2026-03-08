"""Prodrug Activation Kinetics — Phase 601.

Models the conversion of a prodrug to its active metabolite in plasma/liver,
and the subsequent pharmacokinetics of the active drug.

Model structure
---------------
Three-compartment oral model:

  A_gut  : drug mass in gut lumen (mg), absorbs at rate ka
  C_prod : prodrug plasma concentration (mg/L)
  C_act  : active drug plasma concentration (mg/L)

ODEs (Forward Euler):
  dA_gut/dt      = -ka * A_gut
  dC_prod/dt     = ka * A_gut / Vd_prod - (k_act + k_el_prod) * C_prod
  dC_act/dt      = k_act * fm * Vd_prod / Vd_act * C_prod - k_el_act * C_act

References
----------
- Huttunen KM et al., Pharmacol Rev. 2011;63(3):750-71 (prodrug design)
- Rautio J et al., Nat Rev Drug Discov. 2018;17(8):559-87 (prodrug strategies)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ProdrugActivationResult",
    "simulate_prodrug_activation",
    "compare_activation_rates",
]


@dataclass
class ProdrugActivationResult:
    """Results from a prodrug activation simulation.

    Attributes
    ----------
    drug_name : Name of the active drug.
    prodrug_name : Name of the prodrug.
    dose_mg : Administered prodrug dose (mg).
    times_h : Simulation time points (h).
    c_prodrug_mg_L : Prodrug plasma concentration over time (mg/L).
    c_active_mg_L : Active drug plasma concentration over time (mg/L).
    cmax_prodrug_mg_L : Peak prodrug concentration (mg/L).
    cmax_active_mg_L : Peak active drug concentration (mg/L).
    tmax_prodrug_h : Time of peak prodrug concentration (h).
    tmax_active_h : Time of peak active drug concentration (h).
    auc_prodrug_mg_h_per_L : AUC of prodrug (mg·h/L).
    auc_active_mg_h_per_L : AUC of active drug (mg·h/L).
    activation_efficiency : Fraction of absorbed dose converted to active drug (0–1).
    t_half_prodrug_h : Prodrug elimination half-life (h).
    t_half_active_h : Active drug elimination half-life (h).
    notes : Summary text.
    """

    drug_name: str
    prodrug_name: str
    dose_mg: float
    times_h: list
    c_prodrug_mg_L: list
    c_active_mg_L: list
    cmax_prodrug_mg_L: float
    cmax_active_mg_L: float
    tmax_prodrug_h: float
    tmax_active_h: float
    auc_prodrug_mg_h_per_L: float
    auc_active_mg_h_per_L: float
    activation_efficiency: float
    t_half_prodrug_h: float
    t_half_active_h: float
    notes: str


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Compute AUC using manual trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1])
    return auc


def simulate_prodrug_activation(
    drug_name: str,
    prodrug_name: str,
    dose_mg: float,
    ka_absorption_per_h: float,
    f_oral: float,
    k_activation_per_h: float,
    k_el_prodrug_per_h: float,
    k_el_active_per_h: float,
    vd_prodrug_L: float,
    vd_active_L: float,
    fm_activation: float = 0.9,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> ProdrugActivationResult:
    """Simulate prodrug-to-active-drug conversion kinetics.

    Parameters
    ----------
    drug_name : Name of the active drug.
    prodrug_name : Name of the prodrug.
    dose_mg : Prodrug dose administered orally (mg).
    ka_absorption_per_h : First-order GI absorption rate constant (h⁻¹).
    f_oral : Oral bioavailability fraction of the prodrug (0–1].
    k_activation_per_h : First-order activation rate constant (prodrug→active, h⁻¹).
    k_el_prodrug_per_h : Prodrug elimination rate constant (non-activation pathway, h⁻¹).
    k_el_active_per_h : Active drug elimination rate constant (h⁻¹).
    vd_prodrug_L : Apparent volume of distribution of prodrug (L).
    vd_active_L : Apparent volume of distribution of active drug (L).
    fm_activation : Fraction of absorbed prodrug converted to active drug (0–1].
    t_end_h : Simulation duration (h).
    dt_h : Forward-Euler step size (h).

    Returns
    -------
    ProdrugActivationResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if not (0 < f_oral <= 1):
        raise ValueError(f"f_oral must be in (0, 1], got {f_oral}")
    if ka_absorption_per_h <= 0:
        raise ValueError(f"ka_absorption_per_h must be positive, got {ka_absorption_per_h}")
    if k_activation_per_h <= 0:
        raise ValueError(f"k_activation_per_h must be positive, got {k_activation_per_h}")
    if k_el_prodrug_per_h <= 0:
        raise ValueError(f"k_el_prodrug_per_h must be positive, got {k_el_prodrug_per_h}")
    if k_el_active_per_h <= 0:
        raise ValueError(f"k_el_active_per_h must be positive, got {k_el_active_per_h}")
    if vd_prodrug_L <= 0:
        raise ValueError(f"vd_prodrug_L must be positive, got {vd_prodrug_L}")
    if vd_active_L <= 0:
        raise ValueError(f"vd_active_L must be positive, got {vd_active_L}")
    if not (0 < fm_activation <= 1):
        raise ValueError(f"fm_activation must be in (0, 1], got {fm_activation}")

    # --- Initial conditions ---
    a_gut = dose_mg * f_oral  # absorbed prodrug mass in gut (mg)
    c_prod = 0.0  # prodrug plasma concentration (mg/L)
    c_act = 0.0  # active drug plasma concentration (mg/L)

    # Combined prodrug loss rate
    k_prod_total = k_activation_per_h + k_el_prodrug_per_h
    # Transfer factor: rate of active drug generation from prodrug compartment
    k_transfer = k_activation_per_h * fm_activation * vd_prodrug_L / vd_active_L

    times: list = [0.0]
    c_prodrug_list: list = [0.0]
    c_active_list: list = [0.0]

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    for _ in range(n_steps):
        # Forward Euler derivatives
        da_gut = -ka_absorption_per_h * a_gut
        dc_prod = ka_absorption_per_h * a_gut / vd_prodrug_L - k_prod_total * c_prod
        dc_act = k_transfer * c_prod - k_el_active_per_h * c_act

        # Update state
        a_gut = max(0.0, a_gut + da_gut * dt_h)
        c_prod = max(0.0, c_prod + dc_prod * dt_h)
        c_act = max(0.0, c_act + dc_act * dt_h)

        t += dt_h
        times.append(t)
        c_prodrug_list.append(c_prod)
        c_active_list.append(c_act)

    # --- Derived metrics ---
    cmax_prodrug = max(c_prodrug_list)
    cmax_active = max(c_active_list)

    tmax_prodrug = times[c_prodrug_list.index(cmax_prodrug)]
    tmax_active = times[c_active_list.index(cmax_active)]

    auc_prodrug = _trapezoidal_auc(times, c_prodrug_list)
    auc_active = _trapezoidal_auc(times, c_active_list)

    t_half_prodrug = math.log(2.0) / k_prod_total
    t_half_active = math.log(2.0) / k_el_active_per_h

    # Activation efficiency: fraction of absorbed dose appearing as active drug
    activation_efficiency = min(
        1.0,
        fm_activation * k_activation_per_h / k_prod_total,
    )

    notes = (
        f"Prodrug '{prodrug_name}' → active '{drug_name}'. "
        f"Dose {dose_mg} mg, F={f_oral:.2f}. "
        f"k_act={k_activation_per_h:.3f} h⁻¹, t½_prod={t_half_prodrug:.2f} h, "
        f"t½_act={t_half_active:.2f} h. "
        f"Activation efficiency: {activation_efficiency * 100:.1f}%."
    )

    return ProdrugActivationResult(
        drug_name=drug_name,
        prodrug_name=prodrug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_prodrug_mg_L=c_prodrug_list,
        c_active_mg_L=c_active_list,
        cmax_prodrug_mg_L=cmax_prodrug,
        cmax_active_mg_L=cmax_active,
        tmax_prodrug_h=tmax_prodrug,
        tmax_active_h=tmax_active,
        auc_prodrug_mg_h_per_L=auc_prodrug,
        auc_active_mg_h_per_L=auc_active,
        activation_efficiency=activation_efficiency,
        t_half_prodrug_h=t_half_prodrug,
        t_half_active_h=t_half_active,
        notes=notes,
    )


def compare_activation_rates(
    drug_name: str,
    prodrug_name: str,
    dose_mg: float,
    k_activation_values: list,
    vd_prodrug_L: float,
    vd_active_L: float,
    ka_absorption_per_h: float = 1.0,
    f_oral: float = 1.0,
    k_el_prodrug_per_h: float = 0.1,
    k_el_active_per_h: float = 0.2,
    fm_activation: float = 0.9,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list:
    """Compare prodrug simulations across multiple activation rate constants.

    Parameters
    ----------
    drug_name : Name of the active drug.
    prodrug_name : Name of the prodrug.
    dose_mg : Prodrug dose (mg).
    k_activation_values : List of activation rate constants (h⁻¹) to compare.
    vd_prodrug_L : Prodrug volume of distribution (L).
    vd_active_L : Active drug volume of distribution (L).
    ka_absorption_per_h : GI absorption rate constant (h⁻¹).
    f_oral : Oral bioavailability fraction.
    k_el_prodrug_per_h : Prodrug elimination rate constant (h⁻¹).
    k_el_active_per_h : Active drug elimination rate constant (h⁻¹).
    fm_activation : Fraction converted to active drug.
    t_end_h : Simulation duration (h).
    dt_h : Step size (h).

    Returns
    -------
    list of ProdrugActivationResult sorted by auc_active_mg_h_per_L descending.
    """
    results = []
    for k_act in k_activation_values:
        res = simulate_prodrug_activation(
            drug_name=drug_name,
            prodrug_name=prodrug_name,
            dose_mg=dose_mg,
            ka_absorption_per_h=ka_absorption_per_h,
            f_oral=f_oral,
            k_activation_per_h=k_act,
            k_el_prodrug_per_h=k_el_prodrug_per_h,
            k_el_active_per_h=k_el_active_per_h,
            vd_prodrug_L=vd_prodrug_L,
            vd_active_L=vd_active_L,
            fm_activation=fm_activation,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(res)

    results.sort(key=lambda r: r.auc_active_mg_h_per_L, reverse=True)
    return results
