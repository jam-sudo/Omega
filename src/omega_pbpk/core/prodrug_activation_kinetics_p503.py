"""Phase 503 — Prodrug Activation Kinetics.

3-compartment prodrug -> active drug PK model with GI/plasma conversion.
Forward Euler ODE integration, IV bolus assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProdrugActivationResult:
    """Result of a prodrug activation kinetics simulation."""

    drug_name: str
    prodrug_name: str
    dose_mg: float

    times_h: list[float] = field(default_factory=list)
    c_prodrug_mg_L: list[float] = field(default_factory=list)
    c_active_mg_L: list[float] = field(default_factory=list)

    cmax_prodrug: float = 0.0
    tmax_prodrug_h: float = 0.0
    auc_prodrug: float = 0.0

    cmax_active: float = 0.0
    tmax_active_h: float = 0.0
    auc_active: float = 0.0

    conversion_efficiency_observed: float = 0.0
    t_half_active_h: float = 0.0
    notes: str = ""


def simulate_prodrug_activation(
    drug_name: str,
    prodrug_name: str,
    dose_mg: float,
    cl_prodrug_L_per_h: float,
    vd_prodrug_L: float,
    cl_active_L_per_h: float,
    vd_active_L: float,
    k_conv: float = 0.5,
    conv_efficiency: float = 0.85,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> ProdrugActivationResult:
    """Simulate prodrug activation kinetics via Forward Euler ODE.

    Parameters
    ----------
    drug_name:
        Name of the active drug.
    prodrug_name:
        Name of the prodrug.
    dose_mg:
        IV bolus dose in mg.
    cl_prodrug_L_per_h:
        Clearance of the prodrug (L/h).
    vd_prodrug_L:
        Volume of distribution of the prodrug (L).
    cl_active_L_per_h:
        Clearance of the active drug (L/h).
    vd_active_L:
        Volume of distribution of the active drug (L).
    k_conv:
        First-order conversion rate constant (1/h).
    conv_efficiency:
        Fraction of converted prodrug yielding active drug (0-1].
    t_end_h:
        Simulation end time (h).
    dt_h:
        Integration step size (h).

    Returns
    -------
    ProdrugActivationResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_prodrug_L_per_h <= 0:
        raise ValueError("cl_prodrug_L_per_h must be > 0")
    if vd_prodrug_L <= 0:
        raise ValueError("vd_prodrug_L must be > 0")
    if cl_active_L_per_h <= 0:
        raise ValueError("cl_active_L_per_h must be > 0")
    if vd_active_L <= 0:
        raise ValueError("vd_active_L must be > 0")
    if not (0 < conv_efficiency <= 1):
        raise ValueError("conv_efficiency must be in (0, 1]")

    # Derived elimination rate constants
    k_el_pro = cl_prodrug_L_per_h / vd_prodrug_L
    k_el_act = cl_active_L_per_h / vd_active_L

    # Initial conditions (IV bolus)
    A_prodrug = dose_mg
    A_active = 0.0

    times_h: list[float] = []
    c_prodrug_mg_L: list[float] = []
    c_active_mg_L: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        C_pro = A_prodrug / vd_prodrug_L
        C_act = A_active / vd_active_L

        times_h.append(t)
        c_prodrug_mg_L.append(C_pro)
        c_active_mg_L.append(C_act)

        # Forward Euler
        dA_prodrug = -(k_conv + k_el_pro) * A_prodrug
        dA_active = k_conv * A_prodrug * conv_efficiency - k_el_act * A_active

        A_prodrug = max(0.0, A_prodrug + dA_prodrug * dt_h)
        A_active = max(0.0, A_active + dA_active * dt_h)

        t = round(t + dt_h, 10)

    # --- Derived metrics ---
    # Cmax / Tmax
    cmax_prodrug = max(c_prodrug_mg_L)
    tmax_prodrug_h = times_h[c_prodrug_mg_L.index(cmax_prodrug)]

    cmax_active = max(c_active_mg_L)
    tmax_active_h = times_h[c_active_mg_L.index(cmax_active)]

    # AUC via manual trapezoidal
    auc_prodrug = sum(
        0.5 * (c_prodrug_mg_L[i] + c_prodrug_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_active = sum(
        0.5 * (c_active_mg_L[i] + c_active_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    total_auc = auc_active + auc_prodrug
    conv_eff_obs = auc_active / total_auc if total_auc > 0 else 0.0

    # t_half of active drug (analytical)
    t_half_active_h = 0.693 / k_el_act if k_el_act > 0 else float("inf")

    notes = (
        f"Prodrug: k_el={k_el_pro:.4f}/h, k_conv={k_conv:.4f}/h; "
        f"Active: k_el={k_el_act:.4f}/h; conv_efficiency={conv_efficiency}"
    )

    return ProdrugActivationResult(
        drug_name=drug_name,
        prodrug_name=prodrug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_prodrug_mg_L=c_prodrug_mg_L,
        c_active_mg_L=c_active_mg_L,
        cmax_prodrug=cmax_prodrug,
        tmax_prodrug_h=tmax_prodrug_h,
        auc_prodrug=auc_prodrug,
        cmax_active=cmax_active,
        tmax_active_h=tmax_active_h,
        auc_active=auc_active,
        conversion_efficiency_observed=conv_eff_obs,
        t_half_active_h=t_half_active_h,
        notes=notes,
    )


def compare_conversion_rates(
    drug_name: str,
    prodrug_name: str,
    dose_mg: float,
    cl_prodrug_L_per_h: float,
    vd_prodrug_L: float,
    cl_active_L_per_h: float,
    vd_active_L: float,
    k_conv_values: list,
) -> list:
    """Simulate prodrug activation for multiple k_conv values.

    Returns a list of ProdrugActivationResult objects sorted by auc_active
    descending.
    """
    results = []
    for k_conv in k_conv_values:
        res = simulate_prodrug_activation(
            drug_name=drug_name,
            prodrug_name=prodrug_name,
            dose_mg=dose_mg,
            cl_prodrug_L_per_h=cl_prodrug_L_per_h,
            vd_prodrug_L=vd_prodrug_L,
            cl_active_L_per_h=cl_active_L_per_h,
            vd_active_L=vd_active_L,
            k_conv=k_conv,
        )
        results.append(res)
    results.sort(key=lambda r: r.auc_active, reverse=True)
    return results
