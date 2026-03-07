"""
Phase 781 — Portal Circulation PK
3-compartment model: gut lumen -> portal vein -> liver -> systemic
Well-stirred hepatic first-pass extraction.
"""

from dataclasses import dataclass


@dataclass
class PortalCirculationResult:
    drug_name: str
    dose_mg: float
    f_gut_wall: float
    times_h: list[float]
    c_portal_mg_L: list[float]
    c_liver_mg_L: list[float]
    c_systemic_mg_L: list[float]
    cmax_systemic: float
    tmax_systemic_h: float
    auc_systemic: float
    f_hepatic: float
    f_oral_overall: float
    hepatic_extraction_ratio: float
    notes: str


def compute_hepatic_extraction(
    cl_int_L_per_h: float,
    fup: float,
    q_portal_L_per_h: float = 75.0,
) -> float:
    """
    Well-stirred model hepatic extraction ratio.
    Er = fup * CLint / (Qh + fup * CLint)
    """
    if fup <= 0.0 or fup > 1.0:
        raise ValueError("fup must be in (0, 1]")
    if cl_int_L_per_h < 0.0:
        raise ValueError("cl_int_L_per_h must be >= 0")
    if q_portal_L_per_h <= 0.0:
        raise ValueError("q_portal_L_per_h must be > 0")
    numerator = fup * cl_int_L_per_h
    return numerator / (q_portal_L_per_h + numerator)


def simulate_portal_pk(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float = 1.0,
    f_gut_wall: float = 0.9,
    cl_int_L_per_h: float = 30.0,
    fup: float = 0.1,
    q_portal_L_per_h: float = 75.0,
    vd_portal_L: float = 1.5,
    vd_liver_L: float = 1.5,
    vd_sys_L: float = 50.0,
    cl_renal_L_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> PortalCirculationResult:
    """
    Simulate portal circulation and hepatic first-pass.

    Compartments:
      A_gut  : drug mass in gut lumen (mg), absorbed at rate ka
      C_portal: portal vein concentration (mg/L)
      C_liver : liver concentration (mg/L)
      C_sys   : systemic concentration (mg/L)

    ODEs (Forward Euler):
      dA_gut/dt = -ka * A_gut
      Input to portal from gut = ka * A_gut * f_gut_wall   [mg/h]
      dC_portal/dt = (input_gut - Q_pv*(C_portal - C_liver)) / V_portal
      dC_liver/dt  = (Q_pv*(C_portal - C_liver) - CL_h*C_liver) / V_liver
      dC_sys/dt    = (CL_h*C_liver - CL_renal*C_sys) / V_sys

    where CL_h = Q_pv * Er  (well-stirred, based on C_liver free)
    """
    # Validation
    if dose_mg <= 0.0:
        raise ValueError("dose_mg must be > 0")
    if not (0.0 < fup <= 1.0):
        raise ValueError("fup must be in (0, 1]")
    if f_gut_wall < 0.0 or f_gut_wall > 1.0:
        raise ValueError("f_gut_wall must be in [0, 1]")
    if ka_per_h < 0.0:
        raise ValueError("ka_per_h must be >= 0")
    if t_end_h <= 0.0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0.0:
        raise ValueError("dt_h must be > 0")

    # Hepatic extraction ratio (well-stirred)
    er = compute_hepatic_extraction(cl_int_L_per_h, fup, q_portal_L_per_h)
    cl_h = q_portal_L_per_h * er  # hepatic clearance (L/h)
    f_hepatic = 1.0 - er

    # Initial conditions
    a_gut = dose_mg  # all drug in gut lumen
    c_portal = 0.0
    c_liver = 0.0
    c_sys = 0.0

    times_h: list[float] = [0.0]
    c_portal_list: list[float] = [0.0]
    c_liver_list: list[float] = [0.0]
    c_sys_list: list[float] = [0.0]

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _ in range(n_steps):
        # Gut absorption rate (mg/h), scaled by fg
        absorption_rate = ka_per_h * a_gut * f_gut_wall  # mg/h into portal

        # Portal: inflow from gut, outflow to liver
        d_c_portal = (absorption_rate - q_portal_L_per_h * (c_portal - c_liver)) / vd_portal_L

        # Liver: inflow from portal, outflow via hepatic clearance (well-stirred)
        d_c_liver = (q_portal_L_per_h * (c_portal - c_liver) - cl_h * c_liver) / vd_liver_L

        # Systemic: inflow from hepatic outflow, outflow via renal clearance
        # Hepatic outflow = Q_pv * (1-Er) * C_liver = Q_pv * C_liver - CL_h * C_liver
        hepatic_outflow = q_portal_L_per_h * (1.0 - er) * c_liver  # mg/h
        d_c_sys = (hepatic_outflow - cl_renal_L_per_h * c_sys) / vd_sys_L

        # Euler update
        a_gut += -ka_per_h * a_gut * dt_h
        c_portal += d_c_portal * dt_h
        c_liver += d_c_liver * dt_h
        c_sys += d_c_sys * dt_h

        # Clamp negatives
        if a_gut < 0.0:
            a_gut = 0.0
        if c_portal < 0.0:
            c_portal = 0.0
        if c_liver < 0.0:
            c_liver = 0.0
        if c_sys < 0.0:
            c_sys = 0.0

        t += dt_h

        times_h.append(t)
        c_portal_list.append(c_portal)
        c_liver_list.append(c_liver)
        c_sys_list.append(c_sys)

    # Compute Cmax, Tmax, AUC for systemic
    cmax_systemic = 0.0
    tmax_systemic_h = 0.0
    for i, c in enumerate(c_sys_list):
        if c > cmax_systemic:
            cmax_systemic = c
            tmax_systemic_h = times_h[i]

    # Trapezoidal AUC
    auc_systemic = 0.0
    for i in range(1, len(times_h)):
        auc_systemic += 0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times_h[i] - times_h[i - 1])

    f_oral_overall = f_gut_wall * f_hepatic

    notes = (
        f"Well-stirred model: Er={er:.3f}, CLh={cl_h:.2f} L/h, "
        f"Fh={f_hepatic:.3f}, Fg={f_gut_wall:.3f}, F_oral={f_oral_overall:.3f}"
    )

    return PortalCirculationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_gut_wall=f_gut_wall,
        times_h=times_h,
        c_portal_mg_L=c_portal_list,
        c_liver_mg_L=c_liver_list,
        c_systemic_mg_L=c_sys_list,
        cmax_systemic=cmax_systemic,
        tmax_systemic_h=tmax_systemic_h,
        auc_systemic=auc_systemic,
        f_hepatic=f_hepatic,
        f_oral_overall=f_oral_overall,
        hepatic_extraction_ratio=er,
        notes=notes,
    )
