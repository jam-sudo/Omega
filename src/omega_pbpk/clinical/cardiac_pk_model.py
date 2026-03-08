"""
Phase 423: Cardiac Drug PK Model
Pharmacokinetics of antiarrhythmics and cardioactive drugs in myocardial tissue.
2-compartment model: plasma + myocardium.
Pure Python only (stdlib: math, dataclasses).
"""

from dataclasses import dataclass


@dataclass
class CardiacPKResult:
    """Result of cardiac PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list
    c_plasma_mg_L: list
    c_myocardium_mg_L: list
    cmax_plasma_mg_L: float
    cmax_myocardium_mg_L: float
    tmax_myocardium_h: float
    auc_plasma_mg_h_per_L: float
    auc_myocardium_mg_h_per_L: float
    myocardium_to_plasma_ratio: float
    t_half_myocardium_h: float
    cardiac_index: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    """Manual trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_cardiac_pk(
    drug_name: str,
    dose_mg: float,
    route: str,
    cl_sys_L_per_h: float,
    vd_plasma_L: float = 5.0,
    vd_myocardium_L: float = 0.35,
    q_cardiac_L_per_h: float = 0.4,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> CardiacPKResult:
    """
    Simulate cardiac PK using a 2-compartment model (plasma + myocardium).

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in mg (> 0).
    route : str
        "iv" or "oral".
    cl_sys_L_per_h : float
        Systemic clearance (L/h) (> 0).
    vd_plasma_L : float
        Volume of distribution of plasma compartment (L).
    vd_myocardium_L : float
        Volume of distribution of myocardial compartment (L).
    q_cardiac_L_per_h : float
        Cardiac blood flow to myocardium (L/h).
    ka_per_h : float
        Absorption rate constant for oral route (1/h).
    f_oral : float
        Oral bioavailability fraction (0 < f_oral <= 1).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Step size for Forward Euler (h).

    Returns
    -------
    CardiacPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if vd_myocardium_L <= 0:
        raise ValueError("vd_myocardium_L must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")

    # Rate constants
    k12 = q_cardiac_L_per_h / vd_plasma_L  # plasma -> myocardium
    k21 = q_cardiac_L_per_h / vd_myocardium_L  # myocardium -> plasma
    k_elim = cl_sys_L_per_h / vd_plasma_L  # elimination from plasma

    # Initial conditions
    if route == "iv":
        c_plasma = dose_mg / vd_plasma_L
        c_myocardium = 0.0
        m_gut = 0.0
    else:  # oral
        c_plasma = 0.0
        c_myocardium = 0.0
        m_gut = dose_mg * f_oral  # mg in gut

    # Time series storage
    n_steps = int(round(t_end_h / dt_h)) + 1
    times_h = []
    c_plasma_list = []
    c_myocardium_list = []

    t = 0.0
    for step in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        c_myocardium_list.append(c_myocardium)

        if step == n_steps - 1:
            break

        # Absorption input rate (mg/h per L of plasma = mg_absorbed/h / vd_plasma)
        if route == "oral":
            input_rate = ka_per_h * m_gut / vd_plasma_L  # mg/(L*h)
        else:
            input_rate = 0.0

        # ODEs (Forward Euler)
        dc_plasma = (
            input_rate
            - k_elim * c_plasma
            - k12 * c_plasma
            + k21 * c_myocardium * (vd_myocardium_L / vd_plasma_L)
        )
        dc_myocardium = k12 * (vd_plasma_L / vd_myocardium_L) * c_plasma - k21 * c_myocardium

        # Update gut depot
        if route == "oral":
            m_gut = max(0.0, m_gut - ka_per_h * m_gut * dt_h)

        # Update states
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        c_myocardium = max(0.0, c_myocardium + dc_myocardium * dt_h)

        t += dt_h

    # Derived metrics
    cmax_plasma = max(c_plasma_list)
    cmax_myocardium = max(c_myocardium_list)

    # tmax_myocardium
    idx_tmax_myo = c_myocardium_list.index(cmax_myocardium)
    tmax_myocardium_h = times_h[idx_tmax_myo]

    auc_plasma = _trapezoidal_auc(times_h, c_plasma_list)
    auc_myocardium = _trapezoidal_auc(times_h, c_myocardium_list)

    # Myocardium-to-plasma ratio (based on AUC)
    if auc_plasma > 0:
        myocardium_to_plasma_ratio = auc_myocardium / auc_plasma
    else:
        myocardium_to_plasma_ratio = 0.0

    # t_half of myocardium compartment
    if k21 > 0:
        t_half_myocardium_h = 0.693 / k21
    else:
        t_half_myocardium_h = float("inf")

    cardiac_index = myocardium_to_plasma_ratio * 100

    notes = (
        f"Route: {route}. "
        f"Myocardium/plasma AUC ratio: {myocardium_to_plasma_ratio:.3f}. "
        f"Cardiac index: {cardiac_index:.1f}. "
        f"k12={k12:.4f}/h, k21={k21:.4f}/h."
    )

    return CardiacPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_myocardium_mg_L=c_myocardium_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_myocardium_mg_L=cmax_myocardium,
        tmax_myocardium_h=tmax_myocardium_h,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_myocardium_mg_h_per_L=auc_myocardium,
        myocardium_to_plasma_ratio=myocardium_to_plasma_ratio,
        t_half_myocardium_h=t_half_myocardium_h,
        cardiac_index=cardiac_index,
        notes=notes,
    )


def compare_cardiac_drugs(
    drug_list: list,
    route: str = "iv",
) -> list[CardiacPKResult]:
    """
    Compare multiple cardiac drugs by myocardium penetration.

    Parameters
    ----------
    drug_list : list of dict
        Each dict has keys: drug_name, dose_mg, cl_sys_L_per_h,
        vd_plasma_L, vd_myocardium_L, q_cardiac_L_per_h.
    route : str
        "iv" or "oral" (applied to all drugs).

    Returns
    -------
    List[CardiacPKResult]
        Sorted by myocardium_to_plasma_ratio descending.
    """
    results = []
    for d in drug_list:
        result = simulate_cardiac_pk(
            drug_name=d["drug_name"],
            dose_mg=d["dose_mg"],
            route=route,
            cl_sys_L_per_h=d["cl_sys_L_per_h"],
            vd_plasma_L=d.get("vd_plasma_L", 5.0),
            vd_myocardium_L=d.get("vd_myocardium_L", 0.35),
            q_cardiac_L_per_h=d.get("q_cardiac_L_per_h", 0.4),
        )
        results.append(result)

    results.sort(key=lambda r: r.myocardium_to_plasma_ratio, reverse=True)
    return results
