"""Phase 864 — Two-Pore Vascular Permeability Model.

Model vascular permeability for biologics using the two-pore model
(large pores for macromolecules, small pores for small molecules).
"""

from dataclasses import dataclass


def _get_mw_params(mw_kDa: float):
    """Return (ps_total_L_per_h, sigma_reflection) based on MW range."""
    if mw_kDa <= 1.0:
        return 2.0, 0.05
    elif mw_kDa <= 10.0:
        return 0.5, 0.3
    elif mw_kDa <= 150.0:
        return 0.05, 0.7
    else:
        return 0.01, 0.95


@dataclass
class TwoPoreVascularPKResult:
    """Result of two-pore vascular permeability simulation."""

    drug_name: str
    dose_mg: float
    mw_kDa: float
    sigma_reflection: float
    ps_total_L_per_h: float
    times_h: list
    c_plasma_mg_L: list
    c_tissue_mg_L: list
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    cmax_tissue: float
    tmax_tissue_h: float
    auc_tissue: float
    tissue_to_plasma_auc_ratio: float
    vss_L: float
    notes: str


def _trapezoidal_auc(times: list, values: list) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


def simulate_two_pore_vascular_pk(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float,
    vd_plasma_L: float = 3.0,
    vd_tissue_L: float = 12.0,
    cl_L_per_h: float = 0.5,
    lymph_flow_L_per_h: float = 0.1,
    ps_total_L_per_h: float | None = None,
    sigma_reflection: float | None = None,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> TwoPoreVascularPKResult:
    """Simulate two-pore vascular PK model for a biologic.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        IV bolus dose in mg.
    mw_kDa:
        Molecular weight in kDa.
    vd_plasma_L:
        Plasma volume in L (default 3.0).
    vd_tissue_L:
        Interstitial fluid volume in L (default 12.0).
    cl_L_per_h:
        Systemic clearance from plasma in L/h (default 0.5).
    lymph_flow_L_per_h:
        Lymph return flow in L/h (default 0.1).
    ps_total_L_per_h:
        Total PS product in L/h. If None, auto-assigned from MW.
    sigma_reflection:
        Staverman reflection coefficient (0=free, 1=impermeable).
        If None, auto-assigned from MW.
    t_end_h:
        Simulation end time in hours (default 168.0).
    dt_h:
        Forward Euler time step in hours (default 0.5).

    Returns
    -------
    TwoPoreVascularPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw_kDa <= 0:
        raise ValueError("mw_kDa must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")

    # Auto-assign MW-based parameters if not provided
    auto_ps, auto_sigma = _get_mw_params(mw_kDa)
    if ps_total_L_per_h is None:
        ps_total_L_per_h = auto_ps
    if sigma_reflection is None:
        sigma_reflection = auto_sigma

    # Elimination rate constant (from plasma only)
    ke = cl_L_per_h / vd_plasma_L

    # Effective permeability accounting for reflection coefficient
    ps_eff = ps_total_L_per_h * (1.0 - sigma_reflection)

    # Initial conditions: IV bolus → all drug in plasma
    c_plasma = dose_mg / vd_plasma_L
    c_tissue = 0.0

    # Simulation via Forward Euler
    n_steps = int(t_end_h / dt_h) + 1
    times = []
    c_plasma_list = []
    c_tissue_list = []

    t = 0.0
    for _ in range(n_steps):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_tissue_list.append(c_tissue)

        # ODE rates
        # dC_plasma/dt = -ps_eff*(C_plasma - C_tissue)/Vd_plasma
        #                + lymph_flow*C_tissue/Vd_plasma - ke*C_plasma
        dc_plasma = (
            -ps_eff * (c_plasma - c_tissue) / vd_plasma_L
            + lymph_flow_L_per_h * c_tissue / vd_plasma_L
            - ke * c_plasma
        )

        # dC_tissue/dt = ps_eff*(C_plasma - C_tissue)/Vd_tissue
        #               - lymph_flow*C_tissue/Vd_tissue
        dc_tissue = (
            ps_eff * (c_plasma - c_tissue) / vd_tissue_L
            - lymph_flow_L_per_h * c_tissue / vd_tissue_L
        )

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        c_tissue = max(0.0, c_tissue + dc_tissue * dt_h)

        t += dt_h

    # PK metrics — plasma
    cmax_plasma = max(c_plasma_list)
    tmax_plasma_h = times[c_plasma_list.index(cmax_plasma)]
    auc_plasma = _trapezoidal_auc(times, c_plasma_list)

    # PK metrics — tissue
    cmax_tissue = max(c_tissue_list)
    tmax_tissue_h = times[c_tissue_list.index(cmax_tissue)]
    auc_tissue = _trapezoidal_auc(times, c_tissue_list)

    # Tissue-to-plasma AUC ratio
    if auc_plasma > 0.0:
        tissue_to_plasma_auc_ratio = auc_tissue / auc_plasma
    else:
        tissue_to_plasma_auc_ratio = 0.0

    # Vss estimate: Vss = Vd_plasma + Vd_tissue * (AUC_tissue/AUC_plasma)
    vss_L = vd_plasma_L + vd_tissue_L * tissue_to_plasma_auc_ratio

    notes = (
        f"MW={mw_kDa} kDa, sigma_reflection={sigma_reflection:.2f}, "
        f"PS_total={ps_total_L_per_h:.3f} L/h. "
        f"Tissue peak at t={tmax_tissue_h:.1f}h. "
        f"Tissue/Plasma AUC ratio={tissue_to_plasma_auc_ratio:.3f}."
    )

    return TwoPoreVascularPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_kDa=mw_kDa,
        sigma_reflection=sigma_reflection,
        ps_total_L_per_h=ps_total_L_per_h,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_tissue_mg_L=c_tissue_list,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        cmax_tissue=cmax_tissue,
        tmax_tissue_h=tmax_tissue_h,
        auc_tissue=auc_tissue,
        tissue_to_plasma_auc_ratio=tissue_to_plasma_auc_ratio,
        vss_L=vss_L,
        notes=notes,
    )


def compare_mw_classes(
    drug_name: str,
    dose_mg: float,
    mw_classes: list[float],
) -> list[TwoPoreVascularPKResult]:
    """Simulate two-pore PK for multiple MW classes and return sorted by auc_tissue descending.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        IV bolus dose in mg.
    mw_classes:
        List of molecular weights in kDa to compare.

    Returns
    -------
    List of TwoPoreVascularPKResult sorted by auc_tissue descending.
    """
    results = []
    for mw in mw_classes:
        result = simulate_two_pore_vascular_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            mw_kDa=mw,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_tissue, reverse=True)
    return results
