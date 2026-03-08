"""Phase 697 — Drug Concentration in Tears."""

from dataclasses import dataclass, field


@dataclass
class TearFluidPKResult:
    drug_name: str
    dose_mg: float
    route: str
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_tear_mg_L: list = field(default_factory=list)
    tear_volume_mL: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    cmax_tear_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    auc_tear_mg_h_per_L: float = 0.0
    tear_to_plasma_ratio: float = 0.0
    k_drainage_per_h: float = 0.0
    notes: str = ""


def simulate_tear_fluid_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    fu_plasma: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    route: str = "systemic",
    tear_volume_mL: float = 0.007,
    t_end_h: float = 8.0,
    dt_h: float = 0.02,
) -> TearFluidPKResult:
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if route not in ("systemic", "topical"):
        raise ValueError("route must be 'systemic' or 'topical'")
    if tear_volume_mL <= 0:
        raise ValueError("tear_volume_mL must be > 0")

    k_drainage = 30.0  # /h
    k_sec = 0.05 * fu_plasma / max(1.0, mw_Da / 300.0)
    k_sec = max(0.001, min(0.5, k_sec))

    ke = cl_sys_L_per_h / vd_sys_L
    ka = 1.2  # oral absorption rate
    f_bio = 0.85

    # Numerical stability
    dt_int = min(dt_h, 0.4 / max(k_drainage, k_sec, 1.2, ke))

    # Build time grid
    n_steps = int(t_end_h / dt_int) + 1
    step = dt_int

    # State
    if route == "systemic":
        a_gut = dose_mg * f_bio
        c_plasma = 0.0
        c_tear = 0.0
    else:
        # Topical
        a_gut = 0.0
        c_plasma = 0.0
        tear_vol_L = tear_volume_mL / 1000.0
        c_tear = dose_mg / tear_vol_L  # very high initial

    times = []
    cp_list = []
    ct_list = []

    t = 0.0
    for _i in range(n_steps):
        times.append(t)
        if route == "systemic":
            cp_list.append(c_plasma)
        else:
            cp_list.append(0.0)
        ct_list.append(max(0.0, c_tear))

        # Euler step
        if route == "systemic":
            da_gut = -ka * a_gut
            dc_plasma = ka * a_gut / vd_sys_L - ke * c_plasma
            dc_tear = k_sec * c_plasma - k_drainage * c_tear

            a_gut += da_gut * step
            c_plasma += dc_plasma * step
            c_tear += dc_tear * step
            c_plasma = max(0.0, c_plasma)
            c_tear = max(0.0, c_tear)
        else:
            # Topical: rapid drainage, no systemic
            dc_tear = -k_drainage * c_tear
            c_tear += dc_tear * step
            c_tear = max(0.0, c_tear)

        t += step

    # Downsample to dt_h grid
    out_times = []
    out_cp = []
    out_ct = []
    sample_interval = max(1, int(dt_h / dt_int + 0.5))
    for i in range(0, len(times), sample_interval):
        out_times.append(times[i])
        out_cp.append(cp_list[i])
        out_ct.append(ct_list[i])
    # Ensure last point
    if out_times[-1] < times[-1]:
        out_times.append(times[-1])
        out_cp.append(cp_list[-1])
        out_ct.append(ct_list[-1])

    # Cmax
    cmax_plasma = max(out_cp) if out_cp else 0.0
    cmax_tear = max(out_ct) if out_ct else 0.0

    # AUC (trapezoidal)
    auc_plasma = sum(
        0.5 * (out_cp[i] + out_cp[i - 1]) * (out_times[i] - out_times[i - 1])
        for i in range(1, len(out_times))
    )
    auc_tear = sum(
        0.5 * (out_ct[i] + out_ct[i - 1]) * (out_times[i] - out_times[i - 1])
        for i in range(1, len(out_times))
    )

    if auc_plasma > 0:
        tear_to_plasma_ratio = auc_tear / auc_plasma
    else:
        tear_to_plasma_ratio = 0.0

    notes_parts = [
        f"Route: {route}",
        f"k_sec={k_sec:.4f}/h",
        f"k_drainage={k_drainage:.1f}/h",
    ]

    return TearFluidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=out_times,
        c_plasma_mg_L=out_cp,
        c_tear_mg_L=out_ct,
        tear_volume_mL=tear_volume_mL,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_tear_mg_L=cmax_tear,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_tear_mg_h_per_L=auc_tear,
        tear_to_plasma_ratio=tear_to_plasma_ratio,
        k_drainage_per_h=k_drainage,
        notes="; ".join(notes_parts),
    )
