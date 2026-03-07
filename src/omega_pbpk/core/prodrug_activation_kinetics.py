"""Prodrug Activation Kinetics — Phase 381.

Models the PK of a prodrug and its active metabolite including activation
kinetics, with consideration of site-specific activation (intestinal,
hepatic, or systemic).

References
----------
Rautio et al. (2008) Prodrugs: design and clinical applications.
Nat Rev Drug Discov 7, 255-270.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ProdrugActivationResult",
    "simulate_prodrug_activation",
    "compare_activation_sites",
]

_VALID_SITES = {"intestinal", "hepatic", "systemic"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProdrugActivationResult:
    """Result of prodrug activation kinetics simulation."""

    drug_name: str
    activation_site: str
    dose_mg: float
    times_h: list
    c_prodrug_mg_L: list
    c_active_mg_L: list
    auc_prodrug_mg_h_per_L: float
    auc_active_mg_h_per_L: float
    cmax_prodrug_mg_L: float
    cmax_active_mg_L: float
    tmax_active_h: float
    activation_efficiency_pct: float
    active_to_prodrug_auc_ratio: float
    notes: str


# ---------------------------------------------------------------------------
# Helper: trapezoidal AUC
# ---------------------------------------------------------------------------


def _trapz(x: list, y: list) -> float:
    """Manual trapezoidal integration."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_prodrug_activation(
    drug_name: str,
    dose_mg: float,
    activation_site: str = "hepatic",
    ka_absorption_per_h: float = 1.0,
    k_activation_per_h: float = 0.5,
    cl_prodrug_L_per_h: float = 2.0,
    cl_active_L_per_h: float = 5.0,
    vd_prodrug_L: float = 20.0,
    vd_active_L: float = 30.0,
    f_activation: float = 0.8,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> ProdrugActivationResult:
    """Simulate prodrug and active metabolite PK using Forward Euler.

    Parameters
    ----------
    drug_name:
        Name of the prodrug compound.
    dose_mg:
        Oral dose in mg. Must be > 0.
    activation_site:
        Where activation occurs: "intestinal", "hepatic", or "systemic".
    ka_absorption_per_h:
        First-order absorption rate constant (1/h). Must be > 0.
    k_activation_per_h:
        First-order activation rate constant (1/h). Must be > 0.
        Used only for hepatic/systemic sites.
    cl_prodrug_L_per_h:
        Total clearance of prodrug (L/h). Must be > 0.
    cl_active_L_per_h:
        Total clearance of active metabolite (L/h). Must be > 0.
    vd_prodrug_L:
        Volume of distribution of prodrug (L). Must be > 0.
    vd_active_L:
        Volume of distribution of active metabolite (L). Must be > 0.
    f_activation:
        Fraction of absorbed prodrug converted to active metabolite [0, 1].
    t_end_h:
        Simulation end time (h). Must be > 0.
    dt_h:
        Integration time step (h). Must be > 0.

    Returns
    -------
    ProdrugActivationResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if activation_site not in _VALID_SITES:
        raise ValueError(f"activation_site must be one of {_VALID_SITES}, got {activation_site!r}")
    if ka_absorption_per_h <= 0:
        raise ValueError(f"ka_absorption_per_h must be > 0, got {ka_absorption_per_h}")
    if k_activation_per_h <= 0:
        raise ValueError(f"k_activation_per_h must be > 0, got {k_activation_per_h}")
    if cl_prodrug_L_per_h <= 0:
        raise ValueError(f"cl_prodrug_L_per_h must be > 0, got {cl_prodrug_L_per_h}")
    if cl_active_L_per_h <= 0:
        raise ValueError(f"cl_active_L_per_h must be > 0, got {cl_active_L_per_h}")
    if vd_prodrug_L <= 0:
        raise ValueError(f"vd_prodrug_L must be > 0, got {vd_prodrug_L}")
    if vd_active_L <= 0:
        raise ValueError(f"vd_active_L must be > 0, got {vd_active_L}")
    if not (0.0 <= f_activation <= 1.0):
        raise ValueError(f"f_activation must be in [0, 1], got {f_activation}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # --- Rate constants ---
    ke_pd = cl_prodrug_L_per_h / vd_prodrug_L  # elimination rate of prodrug (1/h)
    ke_act = cl_active_L_per_h / vd_active_L  # elimination rate of active (1/h)

    # --- Initial conditions ---
    M_gut = dose_mg  # gut depot (mg)
    C_pd = 0.0  # prodrug plasma concentration (mg/L)
    C_act = 0.0  # active metabolite plasma concentration (mg/L)

    n_steps = max(1, int(round(t_end_h / dt_h)))

    times: list[float] = []
    c_prodrug_list: list[float] = []
    c_active_list: list[float] = []

    t = 0.0

    for step in range(n_steps + 1):
        times.append(t)
        c_prodrug_list.append(C_pd)
        c_active_list.append(C_act)

        if step == n_steps:
            break

        # Gut compartment: first-order absorption
        dM_gut = -ka_absorption_per_h * M_gut

        if activation_site == "intestinal":
            # Fraction f_activation goes directly to active during absorption
            dC_pd = ka_absorption_per_h * (1.0 - f_activation) * M_gut / vd_prodrug_L - ke_pd * C_pd
            dC_act = ka_absorption_per_h * f_activation * M_gut / vd_active_L - ke_act * C_act
        else:
            # hepatic or systemic: all absorbed as prodrug, then first-order activation
            dC_pd = (
                ka_absorption_per_h * M_gut / vd_prodrug_L
                - ke_pd * C_pd
                - k_activation_per_h * C_pd
            )
            # Mass balance: activation flux = k_act * C_pd * vd_pd -> into active compartment
            dC_act = k_activation_per_h * C_pd * (vd_prodrug_L / vd_active_L) - ke_act * C_act

        # Forward Euler update
        M_gut = max(0.0, M_gut + dM_gut * dt_h)
        C_pd = max(0.0, C_pd + dC_pd * dt_h)
        C_act = max(0.0, C_act + dC_act * dt_h)

        t = t + dt_h
        if t > t_end_h:
            t = t_end_h

    # --- Derived metrics ---
    auc_pd = _trapz(times, c_prodrug_list)
    auc_act = _trapz(times, c_active_list)

    cmax_pd = max(c_prodrug_list)
    cmax_act = max(c_active_list)

    tmax_act = times[c_active_list.index(cmax_act)] if cmax_act > 0 else 0.0

    # Activation efficiency
    if activation_site == "intestinal":
        activation_efficiency_pct = f_activation * 100.0
    else:
        total_auc = auc_act + auc_pd
        activation_efficiency_pct = (auc_act / total_auc * 100.0) if total_auc > 0 else 0.0

    # Active-to-prodrug AUC ratio
    active_to_prodrug_ratio = (auc_act / auc_pd) if auc_pd > 0 else 0.0

    # Notes
    if auc_act > auc_pd:
        dominance = "active metabolite dominant"
    else:
        dominance = "prodrug dominant"

    notes = (
        f"Site: {activation_site}; "
        f"AUC_prodrug={auc_pd:.3f} mg*h/L, AUC_active={auc_act:.3f} mg*h/L; "
        f"efficiency={activation_efficiency_pct:.1f}%; {dominance}"
    )

    return ProdrugActivationResult(
        drug_name=drug_name,
        activation_site=activation_site,
        dose_mg=dose_mg,
        times_h=times,
        c_prodrug_mg_L=c_prodrug_list,
        c_active_mg_L=c_active_list,
        auc_prodrug_mg_h_per_L=auc_pd,
        auc_active_mg_h_per_L=auc_act,
        cmax_prodrug_mg_L=cmax_pd,
        cmax_active_mg_L=cmax_act,
        tmax_active_h=tmax_act,
        activation_efficiency_pct=activation_efficiency_pct,
        active_to_prodrug_auc_ratio=active_to_prodrug_ratio,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison function
# ---------------------------------------------------------------------------


def compare_activation_sites(
    drug_name: str,
    dose_mg: float,
    ka_absorption_per_h: float = 1.0,
    k_activation_per_h: float = 0.5,
    cl_prodrug_L_per_h: float = 2.0,
    cl_active_L_per_h: float = 5.0,
) -> list:
    """Compare prodrug activation across all three sites.

    Parameters
    ----------
    drug_name:
        Name of the prodrug compound.
    dose_mg:
        Oral dose in mg. Must be > 0.
    ka_absorption_per_h:
        First-order absorption rate constant (1/h).
    k_activation_per_h:
        First-order activation rate constant (1/h).
    cl_prodrug_L_per_h:
        Clearance of prodrug (L/h).
    cl_active_L_per_h:
        Clearance of active metabolite (L/h).

    Returns
    -------
    list of ProdrugActivationResult
        Sorted by auc_active_mg_h_per_L descending (best activation first).
    """
    results = []
    for site in _VALID_SITES:
        result = simulate_prodrug_activation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            activation_site=site,
            ka_absorption_per_h=ka_absorption_per_h,
            k_activation_per_h=k_activation_per_h,
            cl_prodrug_L_per_h=cl_prodrug_L_per_h,
            cl_active_L_per_h=cl_active_L_per_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_active_mg_h_per_L, reverse=True)
    return results
