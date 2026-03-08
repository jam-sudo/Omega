"""Phase 667 - Tissue Binding Competition.

Models competitive binding between two drugs for the same tissue binding sites.
Uses Forward Euler integration for ODE solving.
"""

from dataclasses import dataclass


@dataclass
class TissueBindingCompetitionResult:
    """Result of tissue binding competition simulation."""

    drug_name_a: str
    drug_name_b: str
    tissue: str
    times_h: list
    c_free_a_mg_L: list
    c_bound_a_mg_L: list
    c_free_b_mg_L: list
    c_bound_b_mg_L: list
    bound_fraction_a: list
    bound_fraction_b: list
    competition_index: float
    displacement_pct_a: float
    displacement_pct_b: float
    winner_drug: str
    notes: str


def simulate_binding_competition(
    drug_name_a: str,
    drug_name_b: str,
    tissue: str,
    bmax_mg_L: float,
    kd_a_mg_L: float,
    kd_b_mg_L: float,
    initial_free_a_mg_L: float,
    initial_free_b_mg_L: float,
    kon_a_per_h_per_mg_L: float = 0.1,
    kon_b_per_h_per_mg_L: float = 0.1,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> TissueBindingCompetitionResult:
    """Simulate competitive binding of two drugs for tissue binding sites.

    Args:
        drug_name_a: Name of drug A.
        drug_name_b: Name of drug B.
        tissue: Target tissue name.
        bmax_mg_L: Total binding site capacity (mg/L).
        kd_a_mg_L: Dissociation constant for drug A (mg/L).
        kd_b_mg_L: Dissociation constant for drug B (mg/L).
        initial_free_a_mg_L: Initial free concentration of drug A (mg/L).
        initial_free_b_mg_L: Initial free concentration of drug B (mg/L).
        kon_a_per_h_per_mg_L: Association rate constant for drug A.
        kon_b_per_h_per_mg_L: Association rate constant for drug B.
        t_end_h: Simulation end time (h).
        dt_h: Time step (h).

    Returns:
        TissueBindingCompetitionResult with time-course and summary metrics.
    """
    if bmax_mg_L <= 0:
        raise ValueError("bmax_mg_L must be > 0")
    if kd_a_mg_L <= 0:
        raise ValueError("kd_a_mg_L must be > 0")
    if kd_b_mg_L <= 0:
        raise ValueError("kd_b_mg_L must be > 0")
    if initial_free_a_mg_L < 0:
        raise ValueError("initial_free_a_mg_L must be >= 0")
    if initial_free_b_mg_L < 0:
        raise ValueError("initial_free_b_mg_L must be >= 0")
    if initial_free_a_mg_L == 0 and initial_free_b_mg_L == 0:
        raise ValueError("At least one drug must have initial_free > 0")

    koff_a = kon_a_per_h_per_mg_L * kd_a_mg_L
    koff_b = kon_b_per_h_per_mg_L * kd_b_mg_L

    n_steps = int(t_end_h / dt_h)

    times_h = []
    c_free_a_mg_L_list = []
    c_bound_a_mg_L_list = []
    c_free_b_mg_L_list = []
    c_bound_b_mg_L_list = []
    bound_fraction_a_list = []
    bound_fraction_b_list = []

    c_free_a = initial_free_a_mg_L
    c_free_b = initial_free_b_mg_L
    b_a = 0.0
    b_b = 0.0

    for i in range(n_steps + 1):
        t = i * dt_h
        times_h.append(t)
        c_free_a_mg_L_list.append(c_free_a)
        c_bound_a_mg_L_list.append(b_a)
        c_free_b_mg_L_list.append(c_free_b)
        c_bound_b_mg_L_list.append(b_b)

        total_a = c_free_a + b_a
        total_b = c_free_b + b_b
        bf_a = b_a / total_a if total_a > 0 else 0.0
        bf_b = b_b / total_b if total_b > 0 else 0.0
        bound_fraction_a_list.append(bf_a)
        bound_fraction_b_list.append(bf_b)

        if i < n_steps:
            free_sites = max(0.0, bmax_mg_L - b_a - b_b)
            db_a = kon_a_per_h_per_mg_L * c_free_a * free_sites - koff_a * b_a
            db_b = kon_b_per_h_per_mg_L * c_free_b * free_sites - koff_b * b_b

            b_a_new = b_a + db_a * dt_h
            b_b_new = b_b + db_b * dt_h
            c_free_a_new = c_free_a - db_a * dt_h
            c_free_b_new = c_free_b - db_b * dt_h

            b_a = max(0.0, b_a_new)
            b_b = max(0.0, b_b_new)
            c_free_a = max(0.0, c_free_a_new)
            c_free_b = max(0.0, c_free_b_new)

            # Enforce total binding capacity
            if b_a + b_b > bmax_mg_L:
                excess = b_a + b_b - bmax_mg_L
                total_bound = b_a + b_b
                b_a -= excess * (b_a / total_bound)
                b_b -= excess * (b_b / total_bound)

    # Competition index: ratio of A's share * 2 (1 = equal, >1 = A wins)
    final_b_a = c_bound_a_mg_L_list[-1]
    final_b_b = c_bound_b_mg_L_list[-1]
    total_bound = final_b_a + final_b_b
    if total_bound > 0:
        competition_index = (final_b_a / total_bound) * 2.0
    else:
        competition_index = 1.0
    competition_index = max(0.0, min(2.0, competition_index))

    # Displacement: compare to solo equilibrium binding
    b_a_solo = (
        bmax_mg_L * initial_free_a_mg_L / (kd_a_mg_L + initial_free_a_mg_L)
        if initial_free_a_mg_L > 0
        else 0.0
    )
    b_b_solo = (
        bmax_mg_L * initial_free_b_mg_L / (kd_b_mg_L + initial_free_b_mg_L)
        if initial_free_b_mg_L > 0
        else 0.0
    )

    # Displacement only meaningful when competitor is present
    if b_a_solo > 0 and initial_free_b_mg_L > 0:
        displacement_pct_a = max(0.0, min(100.0, (1.0 - final_b_a / b_a_solo) * 100.0))
    else:
        displacement_pct_a = 0.0

    if b_b_solo > 0 and initial_free_a_mg_L > 0:
        displacement_pct_b = max(0.0, min(100.0, (1.0 - final_b_b / b_b_solo) * 100.0))
    else:
        displacement_pct_b = 0.0

    # Winner determination
    if final_b_a > final_b_b:
        winner_drug = "A"
    elif final_b_b > final_b_a:
        winner_drug = "B"
    else:
        winner_drug = "tie"

    notes = (
        f"Competitive binding simulation for {drug_name_a} vs {drug_name_b} "
        f"in {tissue}. Bmax={bmax_mg_L} mg/L, "
        f"Kd_A={kd_a_mg_L} mg/L, Kd_B={kd_b_mg_L} mg/L."
    )

    return TissueBindingCompetitionResult(
        drug_name_a=drug_name_a,
        drug_name_b=drug_name_b,
        tissue=tissue,
        times_h=times_h,
        c_free_a_mg_L=c_free_a_mg_L_list,
        c_bound_a_mg_L=c_bound_a_mg_L_list,
        c_free_b_mg_L=c_free_b_mg_L_list,
        c_bound_b_mg_L=c_bound_b_mg_L_list,
        bound_fraction_a=bound_fraction_a_list,
        bound_fraction_b=bound_fraction_b_list,
        competition_index=competition_index,
        displacement_pct_a=displacement_pct_a,
        displacement_pct_b=displacement_pct_b,
        winner_drug=winner_drug,
        notes=notes,
    )
