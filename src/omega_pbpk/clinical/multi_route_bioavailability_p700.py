"""Phase 700 — Multi-Route Bioavailability Comparison."""

from dataclasses import dataclass, field

VALID_ROUTES = {"iv", "oral", "im", "sc", "sublingual"}
DEFAULT_ROUTES = ["iv", "oral", "im", "sc", "sublingual"]


@dataclass
class MultiRouteBioavailabilityResult:
    drug_name: str
    dose_mg: float
    routes_simulated: list = field(default_factory=list)
    times_h: list = field(default_factory=list)
    c_plasma_by_route: dict = field(default_factory=dict)
    cmax_by_route: dict = field(default_factory=dict)
    tmax_by_route: dict = field(default_factory=dict)
    auc_by_route: dict = field(default_factory=dict)
    f_relative_to_iv: dict = field(default_factory=dict)
    ranking: list = field(default_factory=list)
    best_route: str = ""
    notes: str = ""


def simulate_multi_route_bioavailability(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    routes: list = None,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> MultiRouteBioavailabilityResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")

    if routes is None:
        routes = list(DEFAULT_ROUTES)

    if len(routes) == 0:
        raise ValueError("routes must not be empty")

    route_set = set(routes)
    if not route_set.issubset(VALID_ROUTES):
        invalid = route_set - VALID_ROUTES
        raise ValueError(f"Invalid routes: {invalid}")

    ke = cl_sys_L_per_h / vd_sys_L

    # Route-specific parameters
    route_params = {}
    for r in routes:
        if r == "iv":
            route_params[r] = {"F": 1.0, "ka": None}
        elif r == "oral":
            f_oral = min(1.0, max(0.1, 0.6 + logP * 0.05 - mw_Da / 5000.0))
            route_params[r] = {"F": f_oral, "ka": 1.2}
        elif r == "im":
            route_params[r] = {"F": 0.85, "ka": 0.8}
        elif r == "sc":
            route_params[r] = {"F": 0.80, "ka": 0.3}
        elif r == "sublingual":
            f_sl = min(0.9, max(0.2, 0.4 + logP * 0.1))
            route_params[r] = {"F": f_sl, "ka": 3.0}

    # Build shared time grid
    n_steps = int(t_end_h / dt_h) + 1
    times_h = [i * dt_h for i in range(n_steps)]

    c_plasma_by_route = {}
    cmax_by_route = {}
    tmax_by_route = {}
    auc_by_route = {}

    for r in routes:
        params = route_params[r]
        conc = []

        if r == "iv":
            # IV bolus: C(0) = dose/Vd, then exponential decay
            c0 = dose_mg / vd_sys_L
            c_pl = c0
            for idx in range(n_steps):
                conc.append(max(0.0, c_pl))
                if idx < n_steps - 1:
                    dc = -ke * c_pl
                    c_pl += dc * dt_h
                    c_pl = max(0.0, c_pl)
        else:
            # 1-compartment with absorption
            ka = params["ka"]
            f_bio = params["F"]
            a_gut = dose_mg * f_bio
            c_pl = 0.0
            for idx in range(n_steps):
                conc.append(max(0.0, c_pl))
                if idx < n_steps - 1:
                    da_gut = -ka * a_gut
                    dc_pl = ka * a_gut / vd_sys_L - ke * c_pl
                    a_gut += da_gut * dt_h
                    c_pl += dc_pl * dt_h
                    a_gut = max(0.0, a_gut)
                    c_pl = max(0.0, c_pl)

        c_plasma_by_route[r] = conc

        # Cmax and tmax
        cmax = 0.0
        tmax = 0.0
        for idx, c in enumerate(conc):
            if c > cmax:
                cmax = c
                tmax = times_h[idx]
        cmax_by_route[r] = cmax
        tmax_by_route[r] = tmax

        # AUC (trapezoidal)
        auc = sum(
            0.5 * (conc[i] + conc[i - 1]) * (times_h[i] - times_h[i - 1])
            for i in range(1, len(times_h))
        )
        auc_by_route[r] = auc

    # Relative bioavailability to IV
    auc_iv = auc_by_route.get("iv", None)
    f_relative_to_iv = {}
    for r in routes:
        if auc_iv and auc_iv > 0:
            f_relative_to_iv[r] = auc_by_route[r] / auc_iv
        else:
            f_relative_to_iv[r] = 0.0

    # Ranking by AUC descending
    ranking = sorted(routes, key=lambda r: auc_by_route[r], reverse=True)
    best_route = ranking[0]

    notes_parts = [f"{r}: F={route_params[r]['F']:.2f}" for r in routes]

    return MultiRouteBioavailabilityResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        routes_simulated=list(routes),
        times_h=times_h,
        c_plasma_by_route=c_plasma_by_route,
        cmax_by_route=cmax_by_route,
        tmax_by_route=tmax_by_route,
        auc_by_route=auc_by_route,
        f_relative_to_iv=f_relative_to_iv,
        ranking=ranking,
        best_route=best_route,
        notes="; ".join(notes_parts),
    )
