"""
Phase 453 — Rectal drug administration PK model.

Models suppository and enema formulations, accounting for partial
first-pass bypass via inferior/middle hemorrhoidal veins.
"""

from dataclasses import dataclass


@dataclass
class RectalPKResult:
    """Result of rectal PK simulation."""

    drug_name: str
    dose_mg: float
    formulation: str
    times_h: list
    c_systemic_mg_L: list
    cmax_systemic_mg_L: float
    tmax_systemic_h: float
    auc_systemic_mg_h_per_L: float
    f_rectal: float
    first_pass_bypass_fraction: float
    t_half_systemic_h: float
    lag_time_h: float
    comparison_to_oral: str
    notes: str


def simulate_rectal_pk(
    drug_name: str,
    dose_mg: float,
    formulation: str,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    ka_per_h: float = 2.0,
    e_gut: float = 0.1,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> RectalPKResult:
    """
    Simulate rectal PK for suppository or enema formulation.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Administered dose in mg.
    formulation : str
        "suppository" or "enema".
    cl_sys_L_per_h : float
        Systemic clearance in L/h.
    vd_sys_L : float
        Volume of distribution in L.
    ka_per_h : float
        First-order absorption rate constant (per h).
    e_gut : float
        Gut extraction ratio (default 0.1).
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Integration time step in hours.

    Returns
    -------
    RectalPKResult
    """
    # --- Validation ---
    if formulation not in {"suppository", "enema"}:
        raise ValueError(f"formulation must be 'suppository' or 'enema', got '{formulation}'")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")

    # --- Pharmacokinetic parameters ---
    Q_liver = 90.0  # L/h
    f_hepatic = max(0.01, 1.0 - cl_sys_L_per_h / Q_liver)
    f_gut = 1.0 - e_gut
    # 50% of rectal dose bypasses portal (lower rectum), 50% goes via portal
    f_rectal = f_gut * (0.5 + 0.5 * f_hepatic)

    # Lag time and absorption rate depend on formulation
    # Enema (solution): shorter lag, faster absorption (1.5x ka) than suppository
    lag_time_h = 0.5 if formulation == "suppository" else 0.1
    if formulation == "enema":
        ka_per_h = ka_per_h * 1.5

    # Half-life
    t_half = 0.693 * vd_sys_L / cl_sys_L_per_h

    # --- Forward Euler ODE integration ---
    # State: M_rectal (mg remaining in rectum), C_sys (mg/L systemic)
    n_steps = int(t_end_h / dt_h) + 1
    times_h = [i * dt_h for i in range(n_steps)]
    c_systemic = [0.0] * n_steps

    M_rectal = dose_mg
    C_sys = 0.0
    c_systemic[0] = 0.0

    k_el = cl_sys_L_per_h / vd_sys_L  # elimination rate constant

    for i in range(1, n_steps):
        t = times_h[i - 1]
        # Only absorb after lag time
        if t >= lag_time_h:
            absorption_rate = ka_per_h * M_rectal  # mg/h leaving rectum
            # Fraction that actually reaches systemic = f_rectal
            dC_sys = (absorption_rate * f_rectal / vd_sys_L) - k_el * C_sys
            dM = -absorption_rate
        else:
            dC_sys = -k_el * C_sys
            dM = 0.0

        C_sys = max(0.0, C_sys + dC_sys * dt_h)
        M_rectal = max(0.0, M_rectal + dM * dt_h)
        c_systemic[i] = C_sys

    # --- Derived metrics ---
    cmax = max(c_systemic)
    tmax = times_h[c_systemic.index(cmax)]

    # Trapezoidal AUC
    auc = sum(
        0.5 * (c_systemic[i] + c_systemic[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Compare to oral: oral F = f_gut * f_hepatic
    f_oral_approx = f_gut * f_hepatic
    ratio = f_rectal / f_oral_approx if f_oral_approx > 0 else 1.0
    if ratio > 1.10:
        comparison_to_oral = "higher"
    elif ratio >= 0.90:
        comparison_to_oral = "similar"
    else:
        comparison_to_oral = "lower"

    notes = (
        f"{formulation.capitalize()} formulation; "
        f"f_rectal={f_rectal:.2f}; "
        f"50% of dose bypasses hepatic first-pass via inferior hemorrhoidal veins; "
        f"lag_time={lag_time_h} h; "
        f"bioavailability vs oral: {comparison_to_oral}."
    )

    return RectalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation=formulation,
        times_h=times_h,
        c_systemic_mg_L=c_systemic,
        cmax_systemic_mg_L=cmax,
        tmax_systemic_h=tmax,
        auc_systemic_mg_h_per_L=auc,
        f_rectal=f_rectal,
        first_pass_bypass_fraction=0.5,
        t_half_systemic_h=t_half,
        lag_time_h=lag_time_h,
        comparison_to_oral=comparison_to_oral,
        notes=notes,
    )


def compare_formulations_rectal(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
) -> list:
    """
    Compare suppository vs enema formulations for a given drug.

    Returns a list of RectalPKResult sorted by AUC descending.
    """
    results = []
    for form in ("suppository", "enema"):
        result = simulate_rectal_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation=form,
            cl_sys_L_per_h=cl_sys_L_per_h,
            vd_sys_L=vd_sys_L,
        )
        results.append(result)
    results.sort(key=lambda r: r.auc_systemic_mg_h_per_L, reverse=True)
    return results
