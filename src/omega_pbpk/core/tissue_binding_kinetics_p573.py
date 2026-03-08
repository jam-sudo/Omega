"""Phase 573 - Drug Tissue Binding Kinetics.

Models drug binding to tissue components (phospholipids, proteins)
as a rate-dependent process using Forward Euler integration.
"""

import math
from dataclasses import dataclass


@dataclass
class TissueBindingKineticsResult:
    """Result of tissue binding kinetics simulation."""

    drug_name: str
    tissue: str
    times_h: list
    c_free_tissue_mg_L: list
    c_bound_tissue_mg_L: list
    c_total_tissue_mg_L: list
    kd_equilibrium_mg_L: float
    t_half_binding_h: float
    f_bound_at_equilibrium: float
    tissue_accumulation_ratio: float
    binding_class: str
    notes: str


def simulate_tissue_binding(
    drug_name: str,
    tissue: str,
    bmax_mg_L: float,
    kd_mg_L: float,
    kon_per_h_per_mg_L: float,
    initial_free_tissue_mg_L: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> TissueBindingKineticsResult:
    """Simulate drug binding to tissue components over time.

    Args:
        drug_name: Name of the drug.
        tissue: Target tissue name.
        bmax_mg_L: Total binding site capacity (mg/L).
        kd_mg_L: Dissociation constant (mg/L).
        kon_per_h_per_mg_L: Association rate constant (1/(h*mg/L)).
        initial_free_tissue_mg_L: Initial free drug concentration (mg/L).
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        TissueBindingKineticsResult with full time course and metrics.
    """
    if bmax_mg_L <= 0:
        raise ValueError("bmax_mg_L must be > 0")
    if kd_mg_L <= 0:
        raise ValueError("kd_mg_L must be > 0")
    if kon_per_h_per_mg_L <= 0:
        raise ValueError("kon_per_h_per_mg_L must be > 0")
    if initial_free_tissue_mg_L <= 0:
        raise ValueError("initial_free_tissue_mg_L must be > 0")

    koff = kon_per_h_per_mg_L * kd_mg_L

    c_free = initial_free_tissue_mg_L
    c_bound = 0.0

    times_h = []
    c_free_list = []
    c_bound_list = []
    c_total_list = []

    n_steps = int(math.ceil(t_end_h / dt_h))

    # Estimate equilibrium bound for t_half calculation
    # At equilibrium: C_bound_eq = Bmax * C_free_eq / (Kd + C_free_eq)
    # Conservation: C_free_eq + C_bound_eq = initial_free_tissue_mg_L
    # Solve: C_free_eq + Bmax*C_free_eq/(Kd+C_free_eq) = C0
    # Use final simulation values instead for accuracy
    t_half_binding = 0.0

    for i in range(n_steps + 1):
        t = i * dt_h
        times_h.append(t)
        c_free_list.append(c_free)
        c_bound_list.append(c_bound)
        c_total_list.append(c_free + c_bound)

        if i < n_steps:
            d_bound = kon_per_h_per_mg_L * c_free * (bmax_mg_L - c_bound) - koff * c_bound
            c_bound_new = c_bound + d_bound * dt_h
            c_free_new = c_free - d_bound * dt_h

            # Clamp to physical bounds
            if c_bound_new < 0:
                c_bound_new = 0.0
            if c_free_new < 0:
                c_free_new = 0.0

            c_bound = c_bound_new
            c_free = c_free_new

    # Final values
    c_free_final = c_free_list[-1]
    c_bound_final = c_bound_list[-1]

    # Equilibrium bound estimate from final values
    eq_bound = c_bound_final

    # Find t_half_binding: time for C_bound to reach 50% of equilibrium value
    half_target = 0.5 * eq_bound
    t_half_binding = t_end_h  # default if not reached
    for i in range(len(c_bound_list)):
        if c_bound_list[i] >= half_target:
            t_half_binding = times_h[i]
            break

    # Fraction bound at equilibrium
    total_final = c_free_final + c_bound_final
    f_bound_at_equilibrium = c_bound_final / total_final if total_final > 0 else 0.0

    # Tissue accumulation ratio
    tissue_accumulation_ratio = total_final / initial_free_tissue_mg_L

    # Binding class
    if t_half_binding < 1.0:
        binding_class = "rapid"
    elif t_half_binding <= 6.0:
        binding_class = "intermediate"
    else:
        binding_class = "slow"

    notes = (
        f"Tissue binding simulation for {drug_name} in {tissue}. "
        f"Bmax={bmax_mg_L} mg/L, Kd={kd_mg_L} mg/L, "
        f"kon={kon_per_h_per_mg_L}/h/(mg/L), koff={koff:.4f}/h."
    )

    return TissueBindingKineticsResult(
        drug_name=drug_name,
        tissue=tissue,
        times_h=times_h,
        c_free_tissue_mg_L=c_free_list,
        c_bound_tissue_mg_L=c_bound_list,
        c_total_tissue_mg_L=c_total_list,
        kd_equilibrium_mg_L=kd_mg_L,
        t_half_binding_h=t_half_binding,
        f_bound_at_equilibrium=f_bound_at_equilibrium,
        tissue_accumulation_ratio=tissue_accumulation_ratio,
        binding_class=binding_class,
        notes=notes,
    )


def compare_tissues(
    drug_name: str,
    bmax_mg_L: float,
    kd_mg_L: float,
    initial_c_mg_L: float,
    tissues: list = None,
) -> list:
    """Compare tissue binding across multiple tissues.

    Uses same kon=0.1/h/mg_L for each tissue but scales Bmax by tissue factor.

    Args:
        drug_name: Name of the drug.
        bmax_mg_L: Base binding capacity (mg/L).
        kd_mg_L: Dissociation constant (mg/L).
        initial_c_mg_L: Initial free drug concentration (mg/L).
        tissues: List of tissue names. Defaults to standard set.

    Returns:
        List of TissueBindingKineticsResult sorted by f_bound_at_equilibrium descending.
    """
    if tissues is None:
        tissues = ["liver", "muscle", "adipose", "brain", "lung"]

    scale_factors = {
        "liver": 1.0,
        "muscle": 0.5,
        "adipose": 0.8,
        "brain": 0.3,
        "lung": 0.6,
    }

    kon = 0.1  # per h per mg/L

    results = []
    for tissue in tissues:
        sf = scale_factors.get(tissue, 1.0)
        result = simulate_tissue_binding(
            drug_name=drug_name,
            tissue=tissue,
            bmax_mg_L=bmax_mg_L * sf,
            kd_mg_L=kd_mg_L,
            kon_per_h_per_mg_L=kon,
            initial_free_tissue_mg_L=initial_c_mg_L,
        )
        results.append(result)

    results.sort(key=lambda r: r.f_bound_at_equilibrium, reverse=True)
    return results
