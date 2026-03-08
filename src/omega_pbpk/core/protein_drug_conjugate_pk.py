"""Phase 315: Protein Drug Conjugate (PDC) PK Model.

Two-compartment model with slow dissociation releasing free drug from
protein-drug conjugates (e.g., albumin-drug, transferrin-drug conjugates).
"""

from dataclasses import dataclass


@dataclass
class PDCPKResult:
    """Result of a protein-drug conjugate PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    k_release_per_h: float
    times_h: list
    c_pdc_mg_L: list
    c_free_mg_L: list
    auc_pdc_mg_h_per_L: float
    cmax_pdc_mg_L: float
    auc_free_mg_h_per_L: float
    cmax_free_mg_L: float
    tmax_free_h: float
    sustained_release_index: float
    notes: str


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Compute AUC via manual trapezoidal rule."""
    return sum(
        0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


def simulate_pdc_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_protein_L_per_h: float,
    vd_protein_L: float,
    k_release_per_h: float,
    cl_free_drug_L_per_h: float,
    vd_free_drug_L: float,
    t_end_h: float = 168.0,
    dt_h: float = 0.1,
) -> PDCPKResult:
    """Simulate PK of a protein-drug conjugate.

    Parameters
    ----------
    drug_name:
        Name of the drug/conjugate.
    dose_mg:
        Administered dose in mg.
    route:
        Administration route: "iv_bolus" or "sc".
    cl_protein_L_per_h:
        Clearance of the protein-drug conjugate (L/h).
    vd_protein_L:
        Volume of distribution for the conjugate (L).
    k_release_per_h:
        First-order rate constant for drug release from protein (1/h).
    cl_free_drug_L_per_h:
        Clearance of free (released) drug (L/h).
    vd_free_drug_L:
        Volume of distribution for free drug (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    PDCPKResult
    """
    _validate_pdc_inputs(
        drug_name,
        dose_mg,
        route,
        cl_protein_L_per_h,
        vd_protein_L,
        k_release_per_h,
        cl_free_drug_L_per_h,
        vd_free_drug_L,
        t_end_h,
    )

    ke_protein = cl_protein_L_per_h / vd_protein_L
    ke_free = cl_free_drug_L_per_h / vd_free_drug_L

    # Initial conditions
    if route == "iv_bolus":
        a_pdc = dose_mg
        a_depot = 0.0
    else:  # sc
        a_pdc = 0.0
        a_depot = dose_mg * 0.85  # bioavailability 85%

    ka_sc = 0.05  # 1/h for protein conjugates SC absorption
    a_free = 0.0

    times: list = []
    c_pdc_list: list = []
    c_free_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        c_pdc = a_pdc / vd_protein_L
        c_free = a_free / vd_free_drug_L

        times.append(t)
        c_pdc_list.append(c_pdc)
        c_free_list.append(c_free)

        if t >= t_end_h:
            break

        # ODEs (Forward Euler)
        if route == "sc":
            d_depot = -ka_sc * a_depot
            input_pdc = ka_sc * a_depot
        else:
            d_depot = 0.0
            input_pdc = 0.0

        d_pdc = input_pdc - ke_protein * a_pdc - k_release_per_h * a_pdc
        d_free = k_release_per_h * a_pdc - ke_free * a_free

        a_depot += d_depot * dt_h
        a_pdc += d_pdc * dt_h
        a_free += d_free * dt_h

        # Clamp to non-negative
        if a_depot < 0.0:
            a_depot = 0.0
        if a_pdc < 0.0:
            a_pdc = 0.0
        if a_free < 0.0:
            a_free = 0.0

        t = round(t + dt_h, 10)

    auc_pdc = _trapezoidal_auc(times, c_pdc_list)
    auc_free = _trapezoidal_auc(times, c_free_list)

    cmax_pdc = max(c_pdc_list) if c_pdc_list else 0.0
    cmax_free = max(c_free_list) if c_free_list else 0.0

    idx_cmax_free = c_free_list.index(cmax_free)
    tmax_free = times[idx_cmax_free]

    if auc_pdc > 0.0:
        sustained_release_index = auc_free / auc_pdc
    else:
        sustained_release_index = 0.0

    notes = (
        f"PDC PK simulation for {drug_name}. "
        f"Route: {route}. "
        f"k_release={k_release_per_h:.4f}/h, "
        f"sustained_release_index={sustained_release_index:.3f}."
    )

    return PDCPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        k_release_per_h=k_release_per_h,
        times_h=times,
        c_pdc_mg_L=c_pdc_list,
        c_free_mg_L=c_free_list,
        auc_pdc_mg_h_per_L=auc_pdc,
        cmax_pdc_mg_L=cmax_pdc,
        auc_free_mg_h_per_L=auc_free,
        cmax_free_mg_L=cmax_free,
        tmax_free_h=tmax_free,
        sustained_release_index=sustained_release_index,
        notes=notes,
    )


def compare_release_rates(
    drug_name: str,
    dose_mg: float,
    k_release_list: list,
    cl_protein_L_per_h: float,
    vd_protein_L: float,
    cl_free_drug_L_per_h: float,
    vd_free_drug_L: float,
    route: str = "iv_bolus",
    t_end_h: float = 168.0,
    dt_h: float = 0.1,
) -> list:
    """Compare PDC PK across different release rates.

    Parameters
    ----------
    drug_name:
        Name of the drug/conjugate.
    dose_mg:
        Administered dose in mg.
    k_release_list:
        List of k_release values (1/h) to compare.
    cl_protein_L_per_h:
        Clearance of the protein-drug conjugate (L/h).
    vd_protein_L:
        Volume of distribution for the conjugate (L).
    cl_free_drug_L_per_h:
        Clearance of free drug (L/h).
    vd_free_drug_L:
        Volume of distribution for free drug (L).
    route:
        Administration route.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step (h).

    Returns
    -------
    list of PDCPKResult sorted by auc_free descending.
    """
    results = []
    for k_rel in k_release_list:
        result = simulate_pdc_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            cl_protein_L_per_h=cl_protein_L_per_h,
            vd_protein_L=vd_protein_L,
            k_release_per_h=k_rel,
            cl_free_drug_L_per_h=cl_free_drug_L_per_h,
            vd_free_drug_L=vd_free_drug_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)
    results.sort(key=lambda r: r.auc_free_mg_h_per_L, reverse=True)
    return results


def _validate_pdc_inputs(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_protein_L_per_h: float,
    vd_protein_L: float,
    k_release_per_h: float,
    cl_free_drug_L_per_h: float,
    vd_free_drug_L: float,
    t_end_h: float,
) -> None:
    """Validate inputs for PDC PK simulation."""
    valid_routes = {"iv_bolus", "sc"}
    if route not in valid_routes:
        raise ValueError(f"route must be one of {valid_routes}, got '{route}'")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_protein_L_per_h <= 0:
        raise ValueError(f"cl_protein_L_per_h must be > 0, got {cl_protein_L_per_h}")
    if vd_protein_L <= 0:
        raise ValueError(f"vd_protein_L must be > 0, got {vd_protein_L}")
    if k_release_per_h <= 0:
        raise ValueError(f"k_release_per_h must be > 0, got {k_release_per_h}")
    if cl_free_drug_L_per_h <= 0:
        raise ValueError(f"cl_free_drug_L_per_h must be > 0, got {cl_free_drug_L_per_h}")
    if vd_free_drug_L <= 0:
        raise ValueError(f"vd_free_drug_L must be > 0, got {vd_free_drug_L}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
