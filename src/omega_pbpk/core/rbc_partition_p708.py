"""Phase 708 — Drug Partition into Red Blood Cells model."""

from dataclasses import dataclass


@dataclass
class RBCPartitionResult:
    drug_name: str
    dose_mg: float
    hematocrit: float
    kp_rbc: float
    blood_to_plasma_ratio: float
    fu_blood: float
    cl_blood_L_per_h: float
    cl_plasma_L_per_h: float
    times_h: list
    c_plasma_mg_L: list
    c_rbc_mg_L: list
    c_whole_blood_mg_L: list
    cmax_plasma_mg_L: float
    auc_plasma_mg_h_per_L: float
    auc_blood_mg_h_per_L: float
    notes: str


def simulate_rbc_partition(
    drug_name: str,
    dose_mg: float,
    logP: float,
    fu_plasma: float,
    hematocrit: float = 0.45,
    cl_blood_L_per_h: float = None,
    vd_sys_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> RBCPartitionResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be positive")
    if not (0 < hematocrit < 1):
        raise ValueError("hematocrit must be between 0 and 1 (exclusive)")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if cl_blood_L_per_h is not None and cl_blood_L_per_h <= 0:
        raise ValueError("cl_blood_L_per_h must be positive")

    H = hematocrit

    # Kp_rbc
    kp_rbc = 1.0 + max(0.0, logP) * 0.8 - fu_plasma * 2.0 + 0.5
    kp_rbc = max(0.1, min(10.0, kp_rbc))

    # Blood-to-plasma ratio
    Rb = 1.0 - H + H * kp_rbc

    # Free fraction in blood
    fu_blood = fu_plasma / Rb

    # Clearance
    cl_blood_used = cl_blood_L_per_h if cl_blood_L_per_h is not None else 5.0
    cl_plasma = cl_blood_used / Rb

    # PK parameters
    ka = 1.2
    bioavail = 0.85
    ke = cl_plasma / vd_sys_L

    # Simulate
    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_plasma_list = []
    c_rbc_list = []
    c_blood_list = []

    A_gut = dose_mg * bioavail
    C_plasma = 0.0

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)
        c_plasma_list.append(C_plasma)
        c_rbc_list.append(C_plasma * kp_rbc)
        c_blood_list.append(C_plasma * Rb)

        # Forward Euler
        dA_gut = -ka * A_gut
        dC = ka * A_gut / vd_sys_L - ke * C_plasma
        A_gut += dA_gut * dt_h
        C_plasma += dC * dt_h
        if A_gut < 0:
            A_gut = 0.0
        if C_plasma < 0:
            C_plasma = 0.0

    cmax_plasma = max(c_plasma_list)

    # AUC (trapezoidal)
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_blood = sum(
        0.5 * (c_blood_list[i] + c_blood_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    return RBCPartitionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        hematocrit=H,
        kp_rbc=kp_rbc,
        blood_to_plasma_ratio=Rb,
        fu_blood=fu_blood,
        cl_blood_L_per_h=cl_blood_used,
        cl_plasma_L_per_h=cl_plasma,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_rbc_mg_L=c_rbc_list,
        c_whole_blood_mg_L=c_blood_list,
        cmax_plasma_mg_L=cmax_plasma,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_blood_mg_h_per_L=auc_blood,
        notes="RBC partitioning model with blood-to-plasma ratio and equilibrium distribution",
    )
