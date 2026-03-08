"""Phase 578 - Drug Metabolite Mass Balance Tracker.

Tracks mass balance of parent drug and metabolites during biotransformation
using Forward Euler integration.
"""

from dataclasses import dataclass


@dataclass
class MetaboliteMassBalanceResult:
    drug_name: str
    dose_mg: float
    n_metabolites: int
    times_h: list
    c_parent_mg_L: list
    c_metabolites_mg_L: list
    mass_parent_mg: list
    mass_recovered_pct: list
    auc_parent_mg_h_per_L: float
    auc_metabolites_mg_h_per_L: list
    fm_values: list
    mass_balance_final_pct: float
    notes: str


def _trapezoidal_auc(x: list, y: list) -> float:
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def simulate_metabolite_mass_balance(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    fm_values: list,
    cl_met_L_per_h_values: list,
    vd_met_L_values: list,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> MetaboliteMassBalanceResult:
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if sum(fm_values) > 1.0 + 1e-9:
        raise ValueError("sum of fm_values must be <= 1.0")
    n_met = len(fm_values)
    if len(cl_met_L_per_h_values) != n_met:
        raise ValueError("cl_met_L_per_h_values length must match fm_values")
    if len(vd_met_L_values) != n_met:
        raise ValueError("vd_met_L_values length must match fm_values")

    ka = 1.5
    f_oral = 0.9
    k_el = cl_L_per_h / vd_L

    # Initial conditions
    a_gi = dose_mg * f_oral  # amount in GI
    c_parent = 0.0
    c_met = [0.0] * n_met

    times = [0.0]
    c_parent_list = [0.0]
    c_met_lists = [[0.0] for _ in range(n_met)]
    mass_parent_list = [0.0]

    # Initial mass recovered: drug in GI + parent + metabolites
    initial_mass_pct = (
        (a_gi + c_parent * vd_L + sum(c_met[i] * vd_met_L_values[i] for i in range(n_met)))
        / dose_mg
        * 100.0
    )
    mass_recovered_list = [initial_mass_pct]

    n_steps = int(t_end_h / dt_h)

    for _ in range(n_steps):
        # GI absorption
        d_a_gi = -ka * a_gi
        dc_parent = ka * a_gi / vd_L - k_el * c_parent

        dc_met = []
        for i in range(n_met):
            k_met_el = cl_met_L_per_h_values[i] / vd_met_L_values[i]
            formation = fm_values[i] * k_el * c_parent * vd_L / vd_met_L_values[i]
            elimination = k_met_el * c_met[i]
            dc_met.append(formation - elimination)

        a_gi += d_a_gi * dt_h
        c_parent += dc_parent * dt_h
        for i in range(n_met):
            c_met[i] += dc_met[i] * dt_h

        t = times[-1] + dt_h
        times.append(t)
        c_parent_list.append(c_parent)
        for i in range(n_met):
            c_met_lists[i].append(c_met[i])

        mass_p = c_parent * vd_L
        mass_parent_list.append(mass_p)

        total_mass = a_gi + mass_p + sum(c_met[i] * vd_met_L_values[i] for i in range(n_met))
        mass_recovered_list.append(total_mass / dose_mg * 100.0)

    auc_parent = _trapezoidal_auc(times, c_parent_list)
    auc_mets = [_trapezoidal_auc(times, c_met_lists[i]) for i in range(n_met)]

    notes = (
        f"Parent CL={cl_L_per_h} L/h, Vd={vd_L} L. "
        f"{n_met} metabolite(s), fm_total={sum(fm_values):.2f}. "
        f"Oral absorption: ka={ka}/h, F={f_oral}."
    )

    return MetaboliteMassBalanceResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        n_metabolites=n_met,
        times_h=times,
        c_parent_mg_L=c_parent_list,
        c_metabolites_mg_L=c_met_lists,
        mass_parent_mg=mass_parent_list,
        mass_recovered_pct=mass_recovered_list,
        auc_parent_mg_h_per_L=auc_parent,
        auc_metabolites_mg_h_per_L=auc_mets,
        fm_values=fm_values,
        mass_balance_final_pct=mass_recovered_list[-1],
        notes=notes,
    )
