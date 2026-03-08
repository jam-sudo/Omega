"""Phase 414 — Drug Binding Kinetics to Target Receptor.

Models drug-receptor association and dissociation using a bimolecular
binding ODE integrated by Forward Euler.  Pure Python / stdlib only.
"""

import math
from dataclasses import dataclass


@dataclass
class ReceptorBindingResult:
    """Result of a receptor-binding kinetics simulation."""

    drug_name: str
    receptor_name: str
    times_s: list  # time points (seconds)
    c_free_drug_nM: list  # free drug concentration (nM)
    c_bound_drug_nM: list  # drug-receptor complex (nM)
    receptor_occupancy_pct: list  # % of receptors bound
    kd_nM: float  # koff / kon
    kon_per_nM_per_s: float
    koff_per_s: float
    equilibrium_occupancy_pct: float  # D / (D + Kd) * 100
    t_half_association_s: float  # time to 50% of eq. occupancy
    t_half_dissociation_s: float  # ln(2) / koff
    residence_time_s: float  # 1 / koff
    target_engagement_class: str  # "transient", "moderate", "prolonged"
    clinical_duration_class: str  # "short", "medium", "long"
    notes: str


def simulate_receptor_binding(
    drug_name: str,
    receptor_name: str,
    drug_conc_nM: float,
    receptor_conc_nM: float,
    kon_per_nM_per_s: float,
    koff_per_s: float,
    t_end_s: float = 3600.0,
    dt_s: float = 1.0,
) -> ReceptorBindingResult:
    """Simulate bimolecular drug-receptor binding kinetics via Forward Euler.

    ODEs
    ----
    dDR/dt =  kon * D * R - koff * DR
    dD/dt  = -kon * D * R + koff * DR
    dR/dt  = -kon * D * R + koff * DR

    Initial conditions: D = drug_conc_nM, R = receptor_conc_nM, DR = 0.

    Parameters
    ----------
    drug_name:
        Identifier of the drug.
    receptor_name:
        Identifier of the target receptor.
    drug_conc_nM:
        Initial free-drug concentration (nM).
    receptor_conc_nM:
        Total receptor concentration (nM).
    kon_per_nM_per_s:
        Association rate constant (nM^-1 s^-1).
    koff_per_s:
        Dissociation rate constant (s^-1).
    t_end_s:
        Simulation end time (seconds).
    dt_s:
        Integration step size (seconds).

    Returns
    -------
    ReceptorBindingResult
    """
    # --- validation ---
    if drug_conc_nM <= 0:
        raise ValueError("drug_conc_nM must be > 0")
    if receptor_conc_nM <= 0:
        raise ValueError("receptor_conc_nM must be > 0")
    if kon_per_nM_per_s <= 0:
        raise ValueError("kon_per_nM_per_s must be > 0")
    if koff_per_s <= 0:
        raise ValueError("koff_per_s must be > 0")
    if t_end_s <= 0:
        raise ValueError("t_end_s must be > 0")
    if dt_s <= 0:
        raise ValueError("dt_s must be > 0")

    # --- derived scalars ---
    kd_nM = koff_per_s / kon_per_nM_per_s
    eq_occ = drug_conc_nM / (drug_conc_nM + kd_nM) * 100.0
    t_half_diss = math.log(2.0) / koff_per_s
    residence_time = 1.0 / koff_per_s

    # --- Forward Euler integration ---
    n_steps = int(math.ceil(t_end_s / dt_s))
    times_s: list[float] = []
    c_free: list[float] = []
    c_bound: list[float] = []
    occ_pct: list[float] = []

    D = drug_conc_nM
    R = receptor_conc_nM
    DR = 0.0

    half_eq_DR = eq_occ / 100.0 * receptor_conc_nM  # 50% of eq occupancy in nM
    t_half_assoc = float("nan")

    for i in range(n_steps + 1):
        t = i * dt_s
        times_s.append(t)
        c_free.append(D)
        c_bound.append(DR)
        oc = DR / receptor_conc_nM * 100.0
        occ_pct.append(oc)

        # record first crossing of 50% equilibrium occupancy
        if math.isnan(t_half_assoc) and DR >= half_eq_DR and i > 0:
            t_half_assoc = t

        if i < n_steps:
            association = kon_per_nM_per_s * D * R
            dissociation = koff_per_s * DR
            dDR = association - dissociation
            D = max(0.0, D - dDR * dt_s)
            R = max(0.0, R - dDR * dt_s)
            DR = max(0.0, DR + dDR * dt_s)

    # If half-max was never reached within the simulation window use t_end
    if math.isnan(t_half_assoc):
        t_half_assoc = t_end_s

    # --- classification ---
    if residence_time < 60.0:
        eng_class = "transient"
    elif residence_time < 3600.0:
        eng_class = "moderate"
    else:
        eng_class = "prolonged"

    residence_h = residence_time / 3600.0
    if residence_h < 1.0:
        clin_class = "short"
    elif residence_h < 8.0:
        clin_class = "medium"
    else:
        clin_class = "long"

    notes = (
        f"Kd={kd_nM:.3g} nM; kon={kon_per_nM_per_s:.3g} nM^-1 s^-1; "
        f"koff={koff_per_s:.3g} s^-1; t_sim={t_end_s:.0f} s"
    )

    return ReceptorBindingResult(
        drug_name=drug_name,
        receptor_name=receptor_name,
        times_s=times_s,
        c_free_drug_nM=c_free,
        c_bound_drug_nM=c_bound,
        receptor_occupancy_pct=occ_pct,
        kd_nM=kd_nM,
        kon_per_nM_per_s=kon_per_nM_per_s,
        koff_per_s=koff_per_s,
        equilibrium_occupancy_pct=eq_occ,
        t_half_association_s=t_half_assoc,
        t_half_dissociation_s=t_half_diss,
        residence_time_s=residence_time,
        target_engagement_class=eng_class,
        clinical_duration_class=clin_class,
        notes=notes,
    )


def compare_koff_rates(
    drug_name: str,
    receptor_name: str,
    drug_conc_nM: float,
    receptor_conc_nM: float,
    kon_per_nM_per_s: float,
    koff_values: list[float] | None = None,
) -> list[ReceptorBindingResult]:
    """Run simulations for multiple koff values and return sorted results.

    Default koff_values: [1e-4, 1e-3, 1e-2, 0.1, 1.0] s^-1.
    Results sorted by *residence_time_s* descending (longest first).
    """
    if koff_values is None:
        koff_values = [1e-4, 1e-3, 1e-2, 0.1, 1.0]

    results = [
        simulate_receptor_binding(
            drug_name=drug_name,
            receptor_name=receptor_name,
            drug_conc_nM=drug_conc_nM,
            receptor_conc_nM=receptor_conc_nM,
            kon_per_nM_per_s=kon_per_nM_per_s,
            koff_per_s=koff,
        )
        for koff in koff_values
    ]

    results.sort(key=lambda r: r.residence_time_s, reverse=True)
    return results
