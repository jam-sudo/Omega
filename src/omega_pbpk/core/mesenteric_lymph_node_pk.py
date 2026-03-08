"""Phase 523 — Mesenteric Lymph Node Drug Distribution.

Models drug distribution into mesenteric lymph nodes (MLN) via lymphatic
absorption following oral dosing. Lipophilic drugs preferentially enter
the lymphatic system, accumulating in MLN before reaching systemic circulation.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MesentericLymphNodeResult:
    """Result of a mesenteric lymph node PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    f_lymph: float
    f_portal: float

    # Time-series (list fields — plain dataclass)
    times_h: list
    a_portal_mg: list
    a_mln_mg: list
    c_systemic_mg_L: list

    # MLN peak
    cmax_mln_mg: float
    tmax_mln_h: float

    # Systemic peak and exposure
    cmax_systemic: float
    tmax_systemic_h: float
    auc_systemic: float

    # Derived percentages
    mln_peak_pct: float  # cmax_mln_mg / dose_mg * 100
    lymph_contribution_pct: float  # f_lymph * 100

    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_mesenteric_lymph_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    ka_per_h: float = 1.0,
    cl_L_per_h: float = 10.0,
    vd_sys_L: float = 50.0,
    f_abs: float = 0.9,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> MesentericLymphNodeResult:
    """Simulate mesenteric lymph node drug distribution after oral dosing.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    dose_mg:
        Oral dose in mg.
    logP:
        Octanol-water partition coefficient (log scale).
    ka_per_h:
        First-order GI absorption rate constant (h⁻¹).
    cl_L_per_h:
        Total systemic clearance (L/h).
    vd_sys_L:
        Systemic volume of distribution (L).
    f_abs:
        Fraction of dose absorbed from GI lumen (0–1).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler step size (h).

    Returns
    -------
    MesentericLymphNodeResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")

    # --- lymphatic fraction ---
    f_lymph = max(0.0, min(0.3, (logP - 2.0) * 0.1))
    f_portal = 1.0 - f_lymph

    # --- rate constants ---
    k_el = cl_L_per_h / vd_sys_L  # systemic elimination (h⁻¹)
    k_el_portal = 0.3  # portal blood turnover (h⁻¹)
    k_exit_mln = 0.05  # lymph node exit rate (h⁻¹)

    # --- initial conditions ---
    a_gi = dose_mg * f_abs
    a_portal = 0.0
    a_mln = 0.0
    c_sys = 0.0

    # --- storage ---
    times_h: list = [0.0]
    a_portal_mg: list = [a_portal]
    a_mln_mg: list = [a_mln]
    c_sys_list: list = [c_sys]

    # --- Forward Euler integration ---
    t = 0.0
    n_steps = int(round(t_end_h / dt_h))
    for _ in range(n_steps):
        d_a_gi = -ka_per_h * a_gi
        d_a_portal = ka_per_h * f_portal * a_gi - k_el_portal * a_portal
        d_a_mln = ka_per_h * f_lymph * a_gi - k_exit_mln * a_mln
        d_c_sys = (k_el_portal * a_portal + k_exit_mln * a_mln) / vd_sys_L - k_el * c_sys

        a_gi += d_a_gi * dt_h
        a_gi = max(0.0, a_gi)
        a_portal += d_a_portal * dt_h
        a_portal = max(0.0, a_portal)
        a_mln += d_a_mln * dt_h
        a_mln = max(0.0, a_mln)
        c_sys += d_c_sys * dt_h
        c_sys = max(0.0, c_sys)

        t += dt_h
        times_h.append(round(t, 10))
        a_portal_mg.append(a_portal)
        a_mln_mg.append(a_mln)
        c_sys_list.append(c_sys)

    # --- derived metrics: MLN ---
    cmax_mln_mg = max(a_mln_mg)
    tmax_mln_h = times_h[a_mln_mg.index(cmax_mln_mg)]

    # --- derived metrics: systemic ---
    cmax_systemic = max(c_sys_list)
    tmax_systemic_h = times_h[c_sys_list.index(cmax_systemic)]
    auc_systemic = sum(
        0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    mln_peak_pct = (cmax_mln_mg / dose_mg * 100.0) if dose_mg > 0 else 0.0
    lymph_contribution_pct = f_lymph * 100.0

    notes = (
        f"logP={logP:.1f}; lymphatic fraction={f_lymph:.3f} "
        f"({'significant' if f_lymph > 0.05 else 'negligible'} lymphatic absorption)"
    )

    return MesentericLymphNodeResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        f_lymph=f_lymph,
        f_portal=f_portal,
        times_h=times_h,
        a_portal_mg=a_portal_mg,
        a_mln_mg=a_mln_mg,
        c_systemic_mg_L=c_sys_list,
        cmax_mln_mg=cmax_mln_mg,
        tmax_mln_h=tmax_mln_h,
        cmax_systemic=cmax_systemic,
        tmax_systemic_h=tmax_systemic_h,
        auc_systemic=auc_systemic,
        mln_peak_pct=mln_peak_pct,
        lymph_contribution_pct=lymph_contribution_pct,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_logp_lymph(
    drug_name: str,
    dose_mg: float,
    logp_values: list,
    cl_L_per_h: float = 10.0,
    vd_sys_L: float = 50.0,
) -> list:
    """Simulate across a range of logP values and rank by lymphatic contribution.

    Parameters
    ----------
    drug_name:
        Base compound name (logP appended for each simulation).
    dose_mg:
        Oral dose in mg.
    logp_values:
        List of logP values to evaluate.
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution (L).

    Returns
    -------
    list of MesentericLymphNodeResult sorted by lymph_contribution_pct descending.
    """
    results = []
    for lp in logp_values:
        r = simulate_mesenteric_lymph_pk(
            drug_name=f"{drug_name}_logP{lp}",
            dose_mg=dose_mg,
            logP=lp,
            cl_L_per_h=cl_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(r)

    results.sort(key=lambda r: r.lymph_contribution_pct, reverse=True)
    return results
