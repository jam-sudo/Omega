"""Phase 860 — mRNA PK/PD Model.

Model mRNA therapeutic pharmacokinetics and protein expression dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class mRNAPKPDResult:
    """Result of mRNA PK/PD simulation."""

    drug_name: str
    dose_mg: float
    route: str
    times_h: list[float]
    c_mrna_mg_L: list[float]
    c_protein_au: list[float]
    cmax_mrna_mg_L: float
    tmax_mrna_h: float
    cmax_protein_au: float
    tmax_protein_h: float
    auc_mrna_mg_h_per_L: float
    auc_protein_au_h: float
    protein_duration_above_half_max_h: float
    notes: str


_VALID_ROUTES = {"im", "iv", "sc"}


def simulate_mrna_pkpd(
    drug_name: str,
    dose_mg: float,
    route: str = "im",
    k_uptake_per_h: float = 0.3,
    t_half_mrna_h: float = 8.0,
    t_half_protein_h: float = 48.0,
    k_translation: float = 5.0,
    vd_mrna_L: float = 5.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> mRNAPKPDResult:
    """Simulate mRNA PK/PD including protein expression.

    Parameters
    ----------
    drug_name:
        Name of the mRNA drug.
    dose_mg:
        Administered dose in mg.
    route:
        Administration route: 'im', 'iv', or 'sc'.
    k_uptake_per_h:
        Cellular uptake rate constant from depot (1/h).
    t_half_mrna_h:
        mRNA half-life in hours.
    t_half_protein_h:
        Protein half-life in hours.
    k_translation:
        Translation rate (nM protein per mg/L mRNA per h).
    vd_mrna_L:
        mRNA distribution volume (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).

    Returns
    -------
    mRNAPKPDResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if t_half_mrna_h <= 0:
        raise ValueError("t_half_mrna_h must be > 0")
    if t_half_protein_h <= 0:
        raise ValueError("t_half_protein_h must be > 0")
    if vd_mrna_L <= 0:
        raise ValueError("vd_mrna_L must be > 0")
    if route not in _VALID_ROUTES:
        raise ValueError("route must be 'im', 'iv', or 'sc'")

    # Derived rate constants
    k_deg_mrna = math.log(2) / t_half_mrna_h
    k_deg_protein = math.log(2) / t_half_protein_h

    # Initial conditions
    if route == "iv":
        # IV: drug goes directly into mRNA plasma compartment, no depot
        a_mrna = 0.0
        c_mrna = dose_mg / vd_mrna_L
    else:
        # IM or SC: drug starts at injection site (depot)
        a_mrna = dose_mg
        c_mrna = 0.0
    c_protein = 0.0

    n_steps = int(math.ceil(t_end_h / dt_h))
    times_h: list[float] = []
    c_mrna_list: list[float] = []
    c_protein_list: list[float] = []

    t = 0.0
    for _ in range(n_steps + 1):
        times_h.append(t)
        c_mrna_list.append(c_mrna)
        c_protein_list.append(c_protein)

        if t >= t_end_h:
            break

        # ODEs
        d_a_mrna = -k_uptake_per_h * a_mrna
        d_c_mrna = k_uptake_per_h * a_mrna / vd_mrna_L - k_deg_mrna * c_mrna
        d_c_protein = k_translation * c_mrna - k_deg_protein * c_protein

        # Forward Euler
        a_mrna = max(0.0, a_mrna + d_a_mrna * dt_h)
        c_mrna = max(0.0, c_mrna + d_c_mrna * dt_h)
        c_protein = max(0.0, c_protein + d_c_protein * dt_h)

        t = min(t + dt_h, t_end_h)

    # mRNA PK metrics
    cmax_mrna_mg_L = max(c_mrna_list)
    tmax_mrna_h = times_h[c_mrna_list.index(cmax_mrna_mg_L)]

    # Protein PD metrics
    cmax_protein_au = max(c_protein_list)
    tmax_protein_h = times_h[c_protein_list.index(cmax_protein_au)]

    # Trapezoidal AUC for mRNA
    auc_mrna = 0.0
    for i in range(1, len(times_h)):
        auc_mrna += 0.5 * (c_mrna_list[i] + c_mrna_list[i - 1]) * (times_h[i] - times_h[i - 1])

    # Trapezoidal AUC for protein
    auc_protein = 0.0
    for i in range(1, len(times_h)):
        auc_protein += (
            0.5 * (c_protein_list[i] + c_protein_list[i - 1]) * (times_h[i] - times_h[i - 1])
        )

    # Duration protein above 50% of Cmax
    half_max_protein = 0.5 * cmax_protein_au
    duration_above = 0.0
    for i in range(1, len(times_h)):
        above_prev = c_protein_list[i - 1] >= half_max_protein
        above_curr = c_protein_list[i] >= half_max_protein
        if above_prev and above_curr:
            duration_above += times_h[i] - times_h[i - 1]
        elif above_prev or above_curr:
            # Partial interval via linear interpolation
            dt_interval = times_h[i] - times_h[i - 1]
            c_prev = c_protein_list[i - 1]
            c_curr = c_protein_list[i]
            if abs(c_curr - c_prev) > 1e-12:
                frac = (half_max_protein - c_prev) / (c_curr - c_prev)
                frac = max(0.0, min(1.0, frac))
                if above_prev:
                    duration_above += frac * dt_interval
                else:
                    duration_above += (1.0 - frac) * dt_interval

    notes = (
        f"mRNA PK/PD for {drug_name}: dose={dose_mg}mg, route={route}, "
        f"t_half_mRNA={t_half_mrna_h}h, t_half_protein={t_half_protein_h}h. "
        f"Peak protein at {tmax_protein_h:.1f}h."
    )

    return mRNAPKPDResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        times_h=times_h,
        c_mrna_mg_L=c_mrna_list,
        c_protein_au=c_protein_list,
        cmax_mrna_mg_L=cmax_mrna_mg_L,
        tmax_mrna_h=tmax_mrna_h,
        cmax_protein_au=cmax_protein_au,
        tmax_protein_h=tmax_protein_h,
        auc_mrna_mg_h_per_L=auc_mrna,
        auc_protein_au_h=auc_protein,
        protein_duration_above_half_max_h=duration_above,
        notes=notes,
    )


def compare_mrna_formulations(
    drug_name: str,
    dose_mg: float,
    formulations: list[dict],
) -> list[mRNAPKPDResult]:
    """Compare multiple mRNA formulations.

    Parameters
    ----------
    drug_name:
        Base name of the drug.
    dose_mg:
        Administered dose in mg.
    formulations:
        List of dicts with keys: 'name' (str), 't_half_mrna_h' (float),
        'k_uptake_per_h' (float).

    Returns
    -------
    List of mRNAPKPDResult sorted by auc_protein_au_h descending.
    """
    results = []
    for form in formulations:
        name = f"{drug_name}_{form['name']}"
        t_half_mrna = float(form["t_half_mrna_h"])
        k_uptake = float(form["k_uptake_per_h"])
        result = simulate_mrna_pkpd(
            drug_name=name,
            dose_mg=dose_mg,
            k_uptake_per_h=k_uptake,
            t_half_mrna_h=t_half_mrna,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_protein_au_h, reverse=True)
    return results
