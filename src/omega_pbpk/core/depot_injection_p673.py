"""Phase 673 - Drug Depot Injection PK."""

from dataclasses import dataclass, field

VALID_DEPOT_TYPES = ("oil_based_im", "microsphere_sc", "crystalline_suspension", "polymer_matrix")

DEPOT_PARAMS = {
    "oil_based_im": {"ka": 0.02, "f_depot": 0.95, "lag_h": 2.0},
    "microsphere_sc": {"ka": 0.05, "f_depot": 0.90, "lag_h": 0.0},
    "crystalline_suspension": {"ka": 0.01, "f_depot": 0.85, "lag_h": 0.0},
    "polymer_matrix": {"ka": 0.008, "f_depot": 0.98, "lag_h": 0.0},
}


@dataclass
class DepotInjectionResult:
    drug_name: str
    dose_mg: float
    depot_type: str
    times_h: list = field(default_factory=list)
    c_depot_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t_half_absorption_h: float = 0.0
    t_half_elimination_h: float = 0.0
    flip_flop_present: bool = False
    sustained_duration_h: float = 0.0
    bioavailability_pct: float = 0.0
    notes: str = ""


def simulate_depot_injection(
    drug_name: str,
    dose_mg: float,
    depot_type: str,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> DepotInjectionResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if depot_type not in VALID_DEPOT_TYPES:
        raise ValueError(f"depot_type must be one of {VALID_DEPOT_TYPES}")

    params = DEPOT_PARAMS[depot_type]
    ka = params["ka"]
    f_depot = params["f_depot"]
    lag_h = params["lag_h"]
    k_el = cl_sys_L_per_h / vd_sys_L

    a_depot = dose_mg * f_depot
    c_plasma = 0.0

    times_h = []
    c_depot_list = []
    c_plasma_list = []

    n_steps = int(t_end_h / dt_h)
    t = 0.0
    for _ in range(n_steps + 1):
        times_h.append(t)
        c_depot_list.append(a_depot)
        c_plasma_list.append(c_plasma)

        if t < t_end_h:
            if t >= lag_h:
                da_depot = -ka * a_depot * dt_h
                dc_plasma = (ka * a_depot / vd_sys_L - k_el * c_plasma) * dt_h
            else:
                da_depot = 0.0
                dc_plasma = -k_el * c_plasma * dt_h
            a_depot += da_depot
            c_plasma += dc_plasma
            if a_depot < 0:
                a_depot = 0.0
            if c_plasma < 0:
                c_plasma = 0.0
        t += dt_h

    # Cmax and tmax
    cmax = 0.0
    tmax = 0.0
    for i, cp in enumerate(c_plasma_list):
        if cp > cmax:
            cmax = cp
            tmax = times_h[i]

    # AUC by trapezoidal rule
    auc = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    t_half_abs = 0.693 / ka
    t_half_el = 0.693 / k_el
    flip_flop = ka < k_el

    # Sustained duration: time where C_plasma > 10% of Cmax
    threshold = 0.1 * cmax
    above = [times_h[i] for i in range(len(times_h)) if c_plasma_list[i] > threshold]
    if len(above) >= 2:
        sustained = above[-1] - above[0]
    elif len(above) == 1:
        sustained = 0.0
    else:
        sustained = 0.0

    return DepotInjectionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        depot_type=depot_type,
        times_h=times_h,
        c_depot_mg=c_depot_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        t_half_absorption_h=t_half_abs,
        t_half_elimination_h=t_half_el,
        flip_flop_present=flip_flop,
        sustained_duration_h=sustained,
        bioavailability_pct=f_depot * 100,
        notes=f"Depot type: {depot_type}, ka={ka}/h, lag={lag_h}h",
    )


def compare_depots(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
) -> list:
    results = []
    for dt in VALID_DEPOT_TYPES:
        r = simulate_depot_injection(drug_name, dose_mg, dt, cl_sys_L_per_h, vd_sys_L)
        results.append(r)
    results.sort(key=lambda x: x.auc_mg_h_per_L, reverse=True)
    return results
