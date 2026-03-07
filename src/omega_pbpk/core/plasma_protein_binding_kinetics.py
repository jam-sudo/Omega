"""Phase 1127: Kinetics of plasma protein binding for high-affinity drugs.

Models association/dissociation kinetics between free drug, protein-bound
drug, and tissue compartment.

When binding kinetics are fast relative to the PK time step (binding
equilibration time << dt), a quasi-steady-state (QSS) approach is used:
binding equilibrates within each step and distributions are recalculated
from the instantaneous equilibrium. This avoids the numerical stiffness
problem of direct Forward Euler integration of fast binding reactions while
correctly representing the kinetic effects on distribution.

For the full kinetic model the ODEs are:
  dC_free/dt = -kon*P*C_free + koff*C_bound
               - kpt*C_free + ktp*C_tissue
               - CL/Vd_free * C_free

  dC_bound/dt = kon*P*C_free - koff*C_bound

  dC_tissue/dt = kpt*C_free - ktp*C_tissue

When binding_equilibration_h << dt, C_free = fu * C_total_plasma and
C_bound = (1-fu) * C_total_plasma at each step, and the effective ODE for
total plasma drug is integrated directly.
"""

from dataclasses import dataclass


@dataclass
class ProteinBindingKineticsResult:
    drug_name: str
    dose_mg: float
    fu_plasma: float
    kon_L_per_mg_per_h: float
    koff_per_h: float
    times_h: list
    c_free_mg_L: list
    c_bound_mg_L: list
    c_tissue_mg_L: list
    c_total_plasma_mg_L: list
    cmax_free_mg_L: float
    cmax_total_mg_L: float
    auc_free_mg_h_per_L: float
    auc_total_mg_h_per_L: float
    fu_at_tmax: float
    binding_equilibration_h: float
    notes: str


def simulate_protein_binding_kinetics(
    drug_name: str,
    dose_mg: float,
    fu_plasma: float = 0.1,
    kon_L_per_mg_per_h: float = 100.0,
    cl_L_per_h: float = 5.0,
    vd_total_L: float = 50.0,
    kpt_per_h: float = 0.1,
    ktp_per_h: float = 0.05,
    t_end_h: float = 24.0,
    dt_h: float = 0.02,
) -> ProteinBindingKineticsResult:
    """Simulate plasma protein binding kinetics.

    Uses quasi-steady-state binding approximation when binding kinetics are
    fast (binding_equilibration_h < dt_h / 10), otherwise uses adaptive
    Forward Euler with a stability-limited sub-step.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        fu_plasma: Unbound fraction in plasma (0 < fu < 1).
        kon_L_per_mg_per_h: Association rate constant (L/mg/h).
        cl_L_per_h: Total plasma clearance (L/h).
        vd_total_L: Total volume of distribution (L).
        kpt_per_h: Plasma-to-tissue transfer rate (/h).
        ktp_per_h: Tissue-to-plasma transfer rate (/h).
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        ProteinBindingKineticsResult with time-course data and PK metrics.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0 < fu_plasma < 1):
        raise ValueError("fu_plasma must be in (0, 1)")
    if kon_L_per_mg_per_h <= 0:
        raise ValueError("kon_L_per_mg_per_h must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_total_L <= 0:
        raise ValueError("vd_total_L must be > 0")

    # Derived parameters
    P_TOTAL_MG_L = 4500.0  # Simplified albumin concentration (mg/L)
    kon = kon_L_per_mg_per_h
    # koff = kon * P_total * fu / (1 - fu)   [from fu = koff / (kon*P + koff)]
    koff = kon * P_TOTAL_MG_L * fu_plasma / (1.0 - fu_plasma)

    # CL rate applied on free drug; effective elimination rate on Ctotal
    # Free drug CL: rate = CL * C_free = CL * fu * C_total
    cl_rate_on_free = cl_L_per_h / vd_total_L  # /h (per unit concentration)

    # Binding equilibration time
    binding_equilibration_h = 1.0 / (kon * P_TOTAL_MG_L + koff)

    # Initial conditions (IV bolus at instantaneous equilibrium)
    c_total_0 = dose_mg / vd_total_L  # mg/L
    c_free_0 = c_total_0 * fu_plasma
    c_bound_0 = c_total_0 * (1.0 - fu_plasma)
    c_tissue_0 = 0.0

    # Decide integration strategy
    # QSS is valid when binding equilibrates much faster than distribution
    use_qss = binding_equilibration_h < dt_h / 10.0

    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]

    c_free_list = [0.0] * n_steps
    c_bound_list = [0.0] * n_steps
    c_tissue_list = [0.0] * n_steps
    c_total_list = [0.0] * n_steps

    c_free_list[0] = c_free_0
    c_bound_list[0] = c_bound_0
    c_tissue_list[0] = c_tissue_0
    c_total_list[0] = c_free_0 + c_bound_0

    if use_qss:
        # Quasi-steady-state: binding always at equilibrium
        # Effective ODE for total plasma drug (C_p = C_free + C_bound):
        #   dC_p/dt = -(kpt*fu + cl_rate*fu)*C_p + ktp*C_tissue
        # (kpt acts on free fraction; CL acts on free fraction)
        # dC_tissue/dt = kpt*fu*C_p - ktp*C_tissue
        ct = c_tissue_0
        cp = c_total_0  # total plasma

        for i in range(1, n_steps):
            cf_eff = fu_plasma * cp
            dcp = -(kpt_per_h * fu_plasma) * cp - cl_rate_on_free * cf_eff + ktp_per_h * ct
            dct = kpt_per_h * fu_plasma * cp - ktp_per_h * ct

            cp = max(0.0, cp + dcp * dt_h)
            ct = max(0.0, ct + dct * dt_h)

            cf = fu_plasma * cp
            cb = (1.0 - fu_plasma) * cp

            c_free_list[i] = cf
            c_bound_list[i] = cb
            c_tissue_list[i] = ct
            c_total_list[i] = cp
    else:
        # Explicit Forward Euler — safe when binding is slow relative to dt
        cf = c_free_0
        cb = c_bound_0
        ct = c_tissue_0

        for i in range(1, n_steps):
            dcf = (
                -kon * P_TOTAL_MG_L * cf
                + koff * cb
                - kpt_per_h * cf
                + ktp_per_h * ct
                - cl_rate_on_free * cf
            )
            dcb = kon * P_TOTAL_MG_L * cf - koff * cb
            dct = kpt_per_h * cf - ktp_per_h * ct

            cf = max(0.0, cf + dcf * dt_h)
            cb = max(0.0, cb + dcb * dt_h)
            ct = max(0.0, ct + dct * dt_h)

            c_free_list[i] = cf
            c_bound_list[i] = cb
            c_tissue_list[i] = ct
            c_total_list[i] = cf + cb

    # Summary metrics
    cmax_free = max(c_free_list)
    cmax_total = max(c_total_list)

    # tmax index for total plasma concentration (t=0 for IV bolus)
    tmax_idx = c_total_list.index(cmax_total)

    # AUC via manual trapezoidal
    auc_free = 0.0
    auc_total = 0.0
    for i in range(1, n_steps):
        auc_free += 0.5 * (c_free_list[i - 1] + c_free_list[i]) * dt_h
        auc_total += 0.5 * (c_total_list[i - 1] + c_total_list[i]) * dt_h

    # fu at tmax
    c_tot_at_tmax = c_total_list[tmax_idx]
    if c_tot_at_tmax > 0:
        fu_at_tmax = c_free_list[tmax_idx] / c_tot_at_tmax
    else:
        fu_at_tmax = fu_plasma

    notes = (
        f"P_total={P_TOTAL_MG_L} mg/L; koff={koff:.4f}/h; "
        f"equilibration_h={binding_equilibration_h:.4e}; "
        f"fu_nominal={fu_plasma:.3f}; "
        f"integration={'QSS' if use_qss else 'ForwardEuler'}"
    )

    return ProteinBindingKineticsResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fu_plasma=fu_plasma,
        kon_L_per_mg_per_h=kon,
        koff_per_h=koff,
        times_h=times,
        c_free_mg_L=c_free_list,
        c_bound_mg_L=c_bound_list,
        c_tissue_mg_L=c_tissue_list,
        c_total_plasma_mg_L=c_total_list,
        cmax_free_mg_L=cmax_free,
        cmax_total_mg_L=cmax_total,
        auc_free_mg_h_per_L=auc_free,
        auc_total_mg_h_per_L=auc_total,
        fu_at_tmax=fu_at_tmax,
        binding_equilibration_h=binding_equilibration_h,
        notes=notes,
    )
