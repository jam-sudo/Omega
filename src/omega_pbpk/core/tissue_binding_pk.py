"""
Phase 782 — Tissue Binding PK
2-compartment model: plasma + tissue with saturable binding.
Tissue binding uses Bmax/Kd (Hill equation, n=1).
"""

from dataclasses import dataclass


@dataclass
class TissueBindingPKResult:
    drug_name: str
    dose_mg: float
    bmax_nmol_L: float
    kd_nmol_L: float
    route: str
    times_h: list[float]
    c_plasma_total_mg_L: list[float]
    c_tissue_free_nmol_L: list[float]
    c_tissue_bound_nmol_L: list[float]
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    occupancy_at_cmax_pct: float
    tissue_to_plasma_ratio_free: float
    notes: str


def _mg_L_to_nmol_L(c_mg_L: float, mw_g_mol: float) -> float:
    """Convert mg/L to nmol/L using molecular weight (g/mol)."""
    if mw_g_mol <= 0.0:
        return 0.0
    return c_mg_L / mw_g_mol * 1e6


def _compute_tissue_free(c_plasma_nmol_L: float, kp_free: float) -> float:
    """Tissue free concentration as fraction of plasma (simplified)."""
    return c_plasma_nmol_L * kp_free


def _compute_bound(c_free_nmol_L: float, bmax_nmol_L: float, kd_nmol_L: float) -> float:
    """Saturable binding: C_bound = Bmax * C_free / (Kd + C_free)."""
    if kd_nmol_L + c_free_nmol_L <= 0.0:
        return 0.0
    return bmax_nmol_L * c_free_nmol_L / (kd_nmol_L + c_free_nmol_L)


def simulate_tissue_binding_pk(
    drug_name: str,
    dose_mg: float,
    bmax_nmol_L: float = 100.0,
    kd_nmol_L: float = 10.0,
    cl_L_per_h: float = 5.0,
    vd_tissue_L: float = 200.0,
    vd_plasma_L: float = 5.0,
    fup: float = 0.1,
    route: str = "iv",
    f_oral: float = 0.8,
    ka_per_h: float = 1.0,
    mw_g_mol: float = 400.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> TissueBindingPKResult:
    """
    Simulate 1-cpt plasma PK with tissue binding at equilibrium.

    Plasma ODE (1-cpt):
      IV:   dC_plasma/dt = -ke * C_plasma
      Oral: dC_plasma/dt = ka * A_gut * F / Vd_plasma - ke * C_plasma

    ke = CL / (fup * Vd_plasma)  [effective elimination based on free drug]

    At each time step, tissue free concentration is derived from plasma
    using a Kp_free ratio (= Vd_plasma / Vd_tissue as a simplification),
    then bound tissue concentration computed from Hill equation.

    Occupancy = C_bound / Bmax * 100%
    """
    # Validation
    if dose_mg <= 0.0:
        raise ValueError("dose_mg must be > 0")
    if not (0.0 < fup <= 1.0):
        raise ValueError("fup must be in (0, 1]")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")
    if bmax_nmol_L < 0.0:
        raise ValueError("bmax_nmol_L must be >= 0")
    if kd_nmol_L <= 0.0:
        raise ValueError("kd_nmol_L must be > 0")
    if mw_g_mol <= 0.0:
        raise ValueError("mw_g_mol must be > 0")
    if t_end_h <= 0.0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0.0:
        raise ValueError("dt_h must be > 0")

    # Elimination rate constant based on free drug clearance
    # CL_eff = fup * cl_L_per_h acts on total plasma concentration
    ke = (fup * cl_L_per_h) / vd_plasma_L  # 1/h

    # Kp_free: tissue free / plasma (dimensionless partition)
    # Derived from apparent Vd: Vd_total = Vd_plasma + Vd_tissue * Kp_free
    # Approximation: Kp_free = (Vd_plasma / Vd_tissue) to stay < 1 for typical high Vd drugs
    if vd_tissue_L > 0.0:
        kp_free = vd_plasma_L / vd_tissue_L
    else:
        kp_free = 1.0

    # Initial conditions
    if route == "iv":
        c_plasma = dose_mg / vd_plasma_L
        a_gut = 0.0
    else:
        effective_dose = dose_mg * f_oral
        c_plasma = 0.0
        a_gut = effective_dose  # mg in gut

    times_h: list[float] = [0.0]
    c_plasma_list: list[float] = [c_plasma]

    # Compute initial tissue values
    c_plasma_nmol = _mg_L_to_nmol_L(c_plasma, mw_g_mol)
    c_tissue_free_0 = _compute_tissue_free(c_plasma_nmol, kp_free)
    c_tissue_bound_0 = _compute_bound(c_tissue_free_0, bmax_nmol_L, kd_nmol_L)
    c_tissue_free_list: list[float] = [c_tissue_free_0]
    c_tissue_bound_list: list[float] = [c_tissue_bound_0]

    t = 0.0
    n_steps = int(t_end_h / dt_h)

    for _ in range(n_steps):
        # Oral absorption term
        if route == "oral":
            absorption_rate = ka_per_h * a_gut  # mg/h
            d_a_gut = -ka_per_h * a_gut * dt_h
        else:
            absorption_rate = 0.0
            d_a_gut = 0.0

        # Plasma ODE
        d_c_plasma = (absorption_rate / vd_plasma_L) - ke * c_plasma

        # Euler update
        c_plasma += d_c_plasma * dt_h
        a_gut += d_a_gut
        if a_gut < 0.0:
            a_gut = 0.0
        if c_plasma < 0.0:
            c_plasma = 0.0

        t += dt_h

        # Tissue equilibrium at this plasma concentration
        c_plasma_nmol = _mg_L_to_nmol_L(c_plasma, mw_g_mol)
        c_tissue_free = _compute_tissue_free(c_plasma_nmol, kp_free)
        c_tissue_bound = _compute_bound(c_tissue_free, bmax_nmol_L, kd_nmol_L)

        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_tissue_free_list.append(c_tissue_free)
        c_tissue_bound_list.append(c_tissue_bound)

    # Cmax, Tmax, AUC for plasma
    cmax_plasma = 0.0
    tmax_plasma_h = 0.0
    for i, c in enumerate(c_plasma_list):
        if c > cmax_plasma:
            cmax_plasma = c
            tmax_plasma_h = times_h[i]

    auc_plasma = 0.0
    for i in range(1, len(times_h)):
        auc_plasma += (
            0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        )

    # Occupancy at Cmax
    tmax_idx = 0
    for i, t_val in enumerate(times_h):
        if abs(t_val - tmax_plasma_h) < dt_h / 2.0:
            tmax_idx = i
            break

    c_bound_at_tmax = c_tissue_bound_list[tmax_idx]
    if bmax_nmol_L > 0.0:
        occupancy_at_cmax_pct = min(100.0, c_bound_at_tmax / bmax_nmol_L * 100.0)
    else:
        occupancy_at_cmax_pct = 0.0

    # Tissue-to-plasma free ratio at Tmax
    c_free_tissue_at_tmax = c_tissue_free_list[tmax_idx]
    c_plasma_at_tmax_nmol = _mg_L_to_nmol_L(c_plasma_list[tmax_idx], mw_g_mol)
    if c_plasma_at_tmax_nmol > 0.0:
        tissue_to_plasma_ratio_free = c_free_tissue_at_tmax / c_plasma_at_tmax_nmol
    else:
        tissue_to_plasma_ratio_free = 0.0

    notes = (
        f"1-cpt plasma + tissue binding equilibrium. "
        f"ke={ke:.4f}/h, Kp_free={kp_free:.4f}, "
        f"Bmax={bmax_nmol_L:.1f} nmol/L, Kd={kd_nmol_L:.1f} nmol/L"
    )

    return TissueBindingPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        bmax_nmol_L=bmax_nmol_L,
        kd_nmol_L=kd_nmol_L,
        route=route,
        times_h=times_h,
        c_plasma_total_mg_L=c_plasma_list,
        c_tissue_free_nmol_L=c_tissue_free_list,
        c_tissue_bound_nmol_L=c_tissue_bound_list,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        occupancy_at_cmax_pct=occupancy_at_cmax_pct,
        tissue_to_plasma_ratio_free=tissue_to_plasma_ratio_free,
        notes=notes,
    )
