"""Hepatic first-pass model using the well-stirred model."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HepaticFirstPassResult:
    """Result of a hepatic first-pass simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_portal_mg_L: list
    c_hepatic_vein_mg_L: list
    c_systemic_mg_L: list
    f_hepatic: float
    f_oral: float
    cmax_systemic_mg_L: float
    auc_systemic_mg_h_per_L: float
    extraction_ratio: float
    hepatic_clearance_L_per_h: float
    notes: str


def simulate_hepatic_first_pass(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float = 1.0,
    Q_h_L_per_h: float = 90.0,
    fu_plasma: float = 0.1,
    cl_int_L_per_h: float = 50.0,
    vd_sys_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> HepaticFirstPassResult:
    """Simulate hepatic first-pass extraction using the well-stirred model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in mg.
    ka_per_h:
        First-order absorption rate constant (h^-1).
    Q_h_L_per_h:
        Hepatic blood flow (L/h), default 90 L/h (~1.5 L/min).
    fu_plasma:
        Fraction unbound in plasma (0 < fu <= 1).
    cl_int_L_per_h:
        Hepatic intrinsic clearance (L/h).
    vd_sys_L:
        Volume of distribution for systemic compartment (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    HepaticFirstPassResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if Q_h_L_per_h <= 0:
        raise ValueError("Q_h_L_per_h must be > 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if cl_int_L_per_h <= 0:
        raise ValueError("cl_int_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")

    # --- Well-stirred model hepatic clearance ---
    # CL_h = Q_h * fu * CL_int / (Q_h + fu * CL_int)
    fu_cl_int = fu_plasma * cl_int_L_per_h
    cl_h = Q_h_L_per_h * fu_cl_int / (Q_h_L_per_h + fu_cl_int)
    f_hepatic = 1.0 - cl_h / Q_h_L_per_h
    extraction_ratio = cl_h / Q_h_L_per_h  # E = CL_h / Q_h

    # For the simulation we assume fa (intestinal absorption) = 1.0
    # so f_oral = f_intestinal * f_hepatic = 1.0 * f_hepatic
    f_oral = f_hepatic

    # --- ODE state variables ---
    # M_gi : mass of drug in GI tract (mg)
    # A_portal: mass of drug in portal compartment (mg)  -- small plug-flow approximation
    # A_sys  : amount of drug in systemic compartment (mg)
    #
    # The portal compartment is modelled as a very small volume so that
    # concentration can be inferred, but its dynamics follow the
    # absorption-extraction balance.  We use a lumped portal "reservoir"
    # with a fixed volume V_portal = Q_h * tau_portal, where tau_portal is
    # a short transit delay (0.1 h).

    tau_portal = 0.1  # h  — short transit through portal vein
    V_portal = Q_h_L_per_h * tau_portal  # L
    k_portal_drain = 1.0 / tau_portal  # h^-1  — portal drainage rate

    # Non-hepatic/renal clearance: assume small compared to hepatic
    # Use CL_renal = 0 for simplicity; systemic elimination purely renal = 0
    # Total systemic clearance = CL_h (recirculation handled via extraction)
    # After first-pass, systemic is a 1-cpt model with CL_sys
    # We set CL_sys = 0 so only hepatic first-pass removes drug
    # This keeps the model focused on first-pass; user can adjust by
    # modifying vd_sys_L which sets lambda_z.
    cl_sys = 0.0  # hepatic re-clearance on recirculation (simplified: ignore)

    m_gi = dose_mg
    a_portal = 0.0  # mg
    a_sys = 0.0  # mg

    times_h = []
    c_portal_list = []
    c_hv_list = []
    c_sys_list = []

    t = 0.0
    n_steps = int(math.ceil(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        # Record current state
        times_h.append(t)
        c_portal = a_portal / V_portal if V_portal > 0 else 0.0
        # Concentration arriving at liver = portal conc; hepatic vein = F_h * portal
        c_hv = f_hepatic * c_portal
        c_sys = a_sys / vd_sys_L

        c_portal_list.append(c_portal)
        c_hv_list.append(c_hv)
        c_sys_list.append(c_sys)

        # Derivatives
        # GI: first-order absorption
        d_mgi = -ka_per_h * m_gi

        # Portal: receives absorbed drug (ka*M_gi) and drains to liver
        d_aportal = ka_per_h * m_gi - k_portal_drain * a_portal

        # Systemic: input = hepatic vein flow * c_hv = Q_h * c_hv
        # Q_h * c_hv = Q_h * f_hepatic * c_portal = Q_h * f_hepatic * a_portal/V_portal
        # = f_hepatic * k_portal_drain * a_portal
        d_asys = f_hepatic * k_portal_drain * a_portal - (cl_sys / vd_sys_L) * a_sys

        if t >= t_end_h:
            break

        # Forward Euler
        m_gi += d_mgi * dt_h
        a_portal += d_aportal * dt_h
        a_sys += d_asys * dt_h
        t += dt_h

    # --- Derived PK metrics ---
    cmax_systemic = max(c_sys_list)

    # AUC by manual trapezoidal rule
    auc = sum(
        0.5 * (c_sys_list[i] + c_sys_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    notes = (
        f"Well-stirred model: E={extraction_ratio:.3f}, "
        f"CL_h={cl_h:.2f} L/h, F_h={f_hepatic:.3f}. "
        f"Portal transit tau={tau_portal} h."
    )

    return HepaticFirstPassResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_portal_mg_L=c_portal_list,
        c_hepatic_vein_mg_L=c_hv_list,
        c_systemic_mg_L=c_sys_list,
        f_hepatic=f_hepatic,
        f_oral=f_oral,
        cmax_systemic_mg_L=cmax_systemic,
        auc_systemic_mg_h_per_L=auc,
        extraction_ratio=extraction_ratio,
        hepatic_clearance_L_per_h=cl_h,
        notes=notes,
    )
