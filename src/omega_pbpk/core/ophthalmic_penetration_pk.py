"""Ophthalmic drug penetration PK — Phase 733.

Simplified corneal/scleral penetration model for eye drop delivery.
Eye drop → Cornea → Aqueous Humor compartments using forward Euler ODEs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OphthalmicPenetrationResult:
    """Result of ophthalmic penetration PK simulation."""

    drug_name: str
    dose_mg: float
    logp: float
    mw_da: float
    times_h: list
    c_aq_mg_L: list
    a_surface_mg: list
    a_cornea_mg: list
    cmax_aq_mg_L: float
    tmax_aq_h: float
    auc_aq_mg_h_per_L: float
    corneal_permeability_index: float
    fraction_drained: float
    fraction_absorbed: float
    notes: str


def _compute_k_cornea(
    logp: float,
    mw_da: float,
    k_cornea_base_per_h: float,
) -> float:
    """Compute actual corneal absorption rate constant."""
    logp_factor = max(0.1, min(2.0, 1.0 + (logp - 1.5) * 0.2))
    mw_factor = max(0.1, 1.0 - (mw_da - 200.0) / 2000.0)
    return k_cornea_base_per_h * logp_factor * mw_factor


def simulate_ophthalmic_penetration(
    drug_name: str,
    dose_mg: float,
    logp: float = 1.5,
    mw_da: float = 300.0,
    k_cornea_base_per_h: float = 0.2,
    k_drain_per_h: float = 1.5,
    k_cornea_aq_per_h: float = 0.5,
    v_aq_L: float = 2.5e-4,
    ke_aq_per_h: float = 0.3,
    t_end_h: float = 6.0,
    dt_h: float = 0.01,
) -> OphthalmicPenetrationResult:
    """Simulate ophthalmic drug penetration after topical eye drop.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Eye drop dose (mg). Must be > 0.
    logp : float
        Octanol-water partition coefficient. Optimal ~1.5 for cornea.
    mw_da : float
        Molecular weight (Da). Smaller molecules penetrate better.
    k_cornea_base_per_h : float
        Base corneal absorption rate (1/h). Must be > 0.
    k_drain_per_h : float
        Lacrimal drainage rate (1/h). Must be > 0.
    k_cornea_aq_per_h : float
        Cornea to aqueous humor transfer rate (1/h).
    v_aq_L : float
        Aqueous humor volume (L). Must be > 0.
    ke_aq_per_h : float
        Aqueous humor elimination rate (1/h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler time step (h).

    Returns
    -------
    OphthalmicPenetrationResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if v_aq_L <= 0:
        raise ValueError("v_aq_L must be > 0")
    if k_cornea_base_per_h <= 0:
        raise ValueError("k_cornea_base_per_h must be > 0")
    if k_drain_per_h <= 0:
        raise ValueError("k_drain_per_h must be > 0")

    k_cornea = _compute_k_cornea(logp, mw_da, k_cornea_base_per_h)

    # State variables
    a_surface = dose_mg  # mg on ocular surface
    a_cornea = 0.0  # mg in corneal tissue
    c_aq = 0.0  # mg/L in aqueous humor

    times_h: list = []
    c_aq_list: list = []
    a_surface_list: list = []
    a_cornea_list: list = []

    n_steps = max(1, int(round(t_end_h / dt_h)))
    t = 0.0

    for i in range(n_steps + 1):
        times_h.append(t)
        c_aq_list.append(c_aq)
        a_surface_list.append(a_surface)
        a_cornea_list.append(a_cornea)

        if i < n_steps:
            # ODEs (Forward Euler)
            da_surface = -(k_cornea + k_drain_per_h) * a_surface
            da_cornea = k_cornea * a_surface - k_cornea_aq_per_h * a_cornea
            dc_aq = k_cornea_aq_per_h * a_cornea / v_aq_L - ke_aq_per_h * c_aq

            a_surface = max(0.0, a_surface + da_surface * dt_h)
            a_cornea = max(0.0, a_cornea + da_cornea * dt_h)
            c_aq = max(0.0, c_aq + dc_aq * dt_h)
            t += dt_h

    # PK metrics
    cmax_aq_mg_L = max(c_aq_list)
    tmax_idx = c_aq_list.index(cmax_aq_mg_L)
    tmax_aq_h = times_h[tmax_idx]

    # Trapezoidal AUC
    auc_aq = 0.0
    for i in range(len(times_h) - 1):
        auc_aq += 0.5 * (c_aq_list[i] + c_aq_list[i + 1]) * (times_h[i + 1] - times_h[i])

    total_fraction_left_surface = 1.0 - (a_surface_list[-1] / dose_mg)
    drain_rate_fraction = k_drain_per_h / (k_drain_per_h + k_cornea)
    cornea_rate_fraction = k_cornea / (k_drain_per_h + k_cornea)
    fraction_drained = drain_rate_fraction * total_fraction_left_surface
    fraction_absorbed = cornea_rate_fraction * total_fraction_left_surface

    # Notes
    if logp < 0.5:
        notes_str = "Very low logP — poor corneal penetration expected."
    elif logp > 3.0:
        notes_str = "High logP — potential corneal binding, reduced aqueous penetration."
    elif mw_da > 500:
        notes_str = "High MW — significantly reduced corneal permeability."
    else:
        notes_str = "Parameters within acceptable range for corneal penetration."

    return OphthalmicPenetrationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        mw_da=mw_da,
        times_h=times_h,
        c_aq_mg_L=c_aq_list,
        a_surface_mg=a_surface_list,
        a_cornea_mg=a_cornea_list,
        cmax_aq_mg_L=cmax_aq_mg_L,
        tmax_aq_h=tmax_aq_h,
        auc_aq_mg_h_per_L=auc_aq,
        corneal_permeability_index=k_cornea,
        fraction_drained=fraction_drained,
        fraction_absorbed=fraction_absorbed,
        notes=notes_str,
    )


def compare_ophthalmic_drugs(
    drugs_params: list,
) -> list:
    """Compare multiple drugs for ophthalmic penetration.

    Parameters
    ----------
    drugs_params : list of dict
        Each dict must contain 'drug_name', 'dose_mg', 'logp', 'mw_da'.
        Optional keys: k_cornea_base_per_h, k_drain_per_h, k_cornea_aq_per_h,
        v_aq_L, ke_aq_per_h, t_end_h, dt_h.

    Returns
    -------
    list of OphthalmicPenetrationResult sorted by auc_aq_mg_h_per_L descending
    """
    results = []
    for params in drugs_params:
        kwargs = {k: v for k, v in params.items() if k not in ("drug_name", "dose_mg")}
        result = simulate_ophthalmic_penetration(
            drug_name=params["drug_name"],
            dose_mg=params["dose_mg"],
            **kwargs,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_aq_mg_h_per_L, reverse=True)
    return results


__all__ = [
    "OphthalmicPenetrationResult",
    "simulate_ophthalmic_penetration",
    "compare_ophthalmic_drugs",
]
