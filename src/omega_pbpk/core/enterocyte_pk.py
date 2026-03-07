"""Enterocyte PK: gut wall first-pass metabolism and transport during oral absorption.

Models drug metabolism by CYP3A4 and P-gp efflux in enterocytes, computing the
fg (gut bioavailability) component of F = fa * fg * fh.
"""

from dataclasses import dataclass


@dataclass
class EnterocytePKResult:
    drug_name: str
    dose_mg: float
    times_h: list
    a_lumen_mg: list  # drug amount in gut lumen
    c_enterocyte_mg_L: list  # drug concentration in enterocyte
    c_portal_mg_L: list  # drug concentration in portal blood
    fa: float  # fraction absorbed from lumen
    fg: float  # gut wall bioavailability (1 - fraction metabolized in gut)
    fa_fg: float  # product: overall gut contribution to F
    cmax_portal: float
    tmax_portal_h: float
    auc_portal: float
    cyp3a4_metabolized_mg: float  # total drug metabolized by gut CYP3A4
    pgp_effluxed_mg: float  # total drug effluxed back by P-gp
    notes: str


def simulate_enterocyte_pk(
    drug_name: str,
    dose_mg: float,
    ka_per_h: float = 1.0,
    k_cyp3a4_per_h: float = 0.5,
    k_pgp_per_h: float = 0.2,
    k_transport_per_h: float = 2.0,
    v_enterocyte_L: float = 1.0,
    v_portal_L: float = 3.0,
    cl_portal_L_per_h: float = 10.0,
    dissolution_rate_per_h: float = 2.0,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> EnterocytePKResult:
    """Simulate enterocyte PK using Forward Euler ODE integration.

    ODEs:
        dA_lumen/dt = -ka * A_lumen + k_pgp * C_ec * V_ec
        dC_ec/dt    = ka * A_lumen / V_ec - (k_cyp + k_pgp + k_trans) * C_ec
        dC_portal/dt = k_trans * C_ec * V_ec / V_portal - (cl_portal / V_portal) * C_portal
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_per_h <= 0:
        raise ValueError("ka_per_h must be > 0")
    if v_enterocyte_L <= 0:
        raise ValueError("v_enterocyte_L must be > 0")
    if v_portal_L <= 0:
        raise ValueError("v_portal_L must be > 0")
    if cl_portal_L_per_h <= 0:
        raise ValueError("cl_portal_L_per_h must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")

    # Initial conditions
    a_lumen = dose_mg
    c_ec = 0.0
    c_portal = 0.0

    times = [0.0]
    a_lumen_list = [a_lumen]
    c_ec_list = [c_ec]
    c_portal_list = [c_portal]

    cyp_integral = 0.0
    pgp_integral = 0.0

    t = 0.0
    k_total_ec = k_cyp3a4_per_h + k_pgp_per_h + k_transport_per_h

    n_steps = int(round(t_end_h / dt_h))
    actual_dt = t_end_h / n_steps

    for _ in range(n_steps):
        # Rates
        da_lumen = -ka_per_h * a_lumen + k_pgp_per_h * c_ec * v_enterocyte_L
        dc_ec = (ka_per_h * a_lumen / v_enterocyte_L) - k_total_ec * c_ec
        dc_portal = (
            k_transport_per_h * c_ec * v_enterocyte_L / v_portal_L
            - (cl_portal_L_per_h / v_portal_L) * c_portal
        )

        # Accumulate metabolized/effluxed integrals (trapezoidal approximation via Euler)
        cyp_integral += k_cyp3a4_per_h * c_ec * v_enterocyte_L * actual_dt
        pgp_integral += k_pgp_per_h * c_ec * v_enterocyte_L * actual_dt

        # Forward Euler update
        a_lumen += da_lumen * actual_dt
        c_ec += dc_ec * actual_dt
        c_portal += dc_portal * actual_dt

        # Clamp negatives due to numerical issues
        if a_lumen < 0.0:
            a_lumen = 0.0
        if c_ec < 0.0:
            c_ec = 0.0
        if c_portal < 0.0:
            c_portal = 0.0

        t += actual_dt
        times.append(t)
        a_lumen_list.append(a_lumen)
        c_ec_list.append(c_ec)
        c_portal_list.append(c_portal)

    # Compute fa: fraction absorbed = 1 - remaining in lumen / dose
    fa = 1.0 - a_lumen_list[-1] / dose_mg
    fa = max(0.0, min(1.0, fa))

    # Compute fg from rate constants
    fg = estimate_fg(k_cyp3a4_per_h, k_pgp_per_h, k_transport_per_h)

    fa_fg = fa * fg

    # Cmax and Tmax for portal blood
    cmax_portal = max(c_portal_list)
    tmax_portal_h = times[c_portal_list.index(cmax_portal)]

    # AUC portal (trapezoidal)
    auc_portal = 0.0
    for i in range(len(times) - 1):
        auc_portal += (c_portal_list[i] + c_portal_list[i + 1]) / 2.0 * (times[i + 1] - times[i])

    notes = (
        f"fa={fa:.3f}, fg={fg:.3f}, fa_fg={fa_fg:.3f}; "
        f"CYP3A4 metabolized {cyp_integral:.2f} mg; "
        f"P-gp effluxed {pgp_integral:.2f} mg"
    )

    return EnterocytePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        a_lumen_mg=a_lumen_list,
        c_enterocyte_mg_L=c_ec_list,
        c_portal_mg_L=c_portal_list,
        fa=fa,
        fg=fg,
        fa_fg=fa_fg,
        cmax_portal=cmax_portal,
        tmax_portal_h=tmax_portal_h,
        auc_portal=auc_portal,
        cyp3a4_metabolized_mg=cyp_integral,
        pgp_effluxed_mg=pgp_integral,
        notes=notes,
    )


def estimate_fg(
    k_cyp3a4_per_h: float,
    k_pgp_per_h: float,
    k_transport_per_h: float,
) -> float:
    """Estimate gut bioavailability fg from rate constants.

    fg = k_transport / (k_cyp + k_pgp + k_transport)
    """
    total = k_cyp3a4_per_h + k_pgp_per_h + k_transport_per_h
    if total <= 0.0:
        return 1.0
    return k_transport_per_h / total
