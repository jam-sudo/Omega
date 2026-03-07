"""Metabolic pathway simulation — Phase 461.

Models multi-step sequential metabolic pathways:
    parent → M1 → M2 → ... → eliminated

Each species is modelled as a 1-compartment system connected sequentially.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MetabolicPathwayResult:
    """Result of a sequential metabolic pathway simulation."""

    n_steps: int
    step_names: list[str]  # ['parent', 'M1', 'M2', ...]
    times_h: list[float]
    concentrations: list[list[float]]  # one list per species (parent first)
    auc_values: list[float]  # AUC for parent + each metabolite
    metabolite_exposure_ratios: list[float]  # metabolite AUC / parent AUC
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(y: list[float], x: list[float]) -> float:
    """Pure-Python trapezoidal integration."""
    if len(y) < 2:
        return 0.0
    total = 0.0
    for i in range(len(y) - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def simulate_sequential_metabolism(
    parent_dose_mg: float,
    steps: list[dict],
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> MetabolicPathwayResult:
    """Simulate a sequential metabolic pathway with forward Euler integration.

    Parameters
    ----------
    parent_dose_mg : float
        IV bolus dose for the parent compound (mg). Must be > 0.
    steps : list[dict]
        List of step dicts, each containing:
        - 'name'      : str   — species name (e.g. 'M1')
        - 'cl_L_per_h': float — total clearance of this species (L/h)
        - 'vd_L'      : float — volume of distribution (L)
        - 'fm'        : float — fraction metabolised to the *next* species (0–1)
          (fm for the final species is ignored)
    t_end_h : float
        Simulation duration (h). Must be > 0.
    dt_h : float
        Time step (h).

    Returns
    -------
    MetabolicPathwayResult
    """
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if not steps:
        raise ValueError("steps must contain at least one metabolite step")
    if parent_dose_mg <= 0:
        raise ValueError("parent_dose_mg must be > 0")

    # Validate individual steps
    for i, s in enumerate(steps):
        if s.get("cl_L_per_h", 0) <= 0:
            raise ValueError(f"Step {i}: cl_L_per_h must be > 0")
        if s.get("vd_L", 0) <= 0:
            raise ValueError(f"Step {i}: vd_L must be > 0")
        fm = s.get("fm", 1.0)
        if not (0.0 <= fm <= 1.0):
            raise ValueError(f"Step {i}: fm must be in [0, 1]")

    # steps[0] describes the parent PK (cl_L_per_h, vd_L, fm → M1).
    # steps[1:] describe metabolite PK in sequential order.
    # step_names = ['parent', M1_name, M2_name, ...]
    # n_steps (returned) = number of metabolites produced.

    parent_step = steps[0]
    metabolite_steps = steps[1:]
    n_metabolites = len(metabolite_steps)

    # step_names: 'parent' + each metabolite name
    # Use metabolite_steps[i]['name'] for metabolites
    step_names_final: list[str] = ["parent"] + [s["name"] for s in metabolite_steps]
    n_species_total = 1 + n_metabolites  # parent + metabolites

    # Rate constants
    ke_list: list[float] = []
    vd_list: list[float] = []
    fm_list: list[float] = []

    # Parent
    vd_parent = parent_step["vd_L"]
    cl_parent = parent_step["cl_L_per_h"]
    fm_parent = parent_step.get("fm", 1.0)
    ke_list.append(cl_parent / vd_parent)
    vd_list.append(vd_parent)
    fm_list.append(fm_parent)

    for s in metabolite_steps:
        vd = s["vd_L"]
        cl = s["cl_L_per_h"]
        fm = s.get("fm", 1.0)
        ke_list.append(cl / vd)
        vd_list.append(vd)
        fm_list.append(fm)

    # Initial concentrations
    c: list[float] = [parent_dose_mg / vd_list[0]] + [0.0] * n_metabolites

    n_steps_sim = max(1, int(math.ceil(t_end_h / dt_h)))

    times: list[float] = []
    # One list per species
    conc_history: list[list[float]] = [[] for _ in range(n_species_total)]

    for i in range(n_steps_sim + 1):
        t = i * dt_h
        if t > t_end_h + 1e-12:
            break
        times.append(t)
        for k in range(n_species_total):
            conc_history[k].append(c[k])

        # Forward Euler: compute derivatives
        dc = [0.0] * n_species_total

        # Parent: dC0/dt = -ke0 * C0
        dc[0] = -ke_list[0] * c[0]

        # Metabolites: dCk/dt = ke_{k-1} * C_{k-1} * fm_{k-1} * (Vd_{k-1} / Vd_k) - ke_k * Ck
        for k in range(1, n_species_total):
            # Formation from precursor (species k-1)
            formation = ke_list[k - 1] * c[k - 1] * fm_list[k - 1] * (vd_list[k - 1] / vd_list[k])
            # Elimination of species k
            elimination = ke_list[k] * c[k]
            dc[k] = formation - elimination

        # Update
        c = [max(c[k] + dc[k] * dt_h, 0.0) for k in range(n_species_total)]

    # AUC by trapezoidal rule
    auc_values = [_trapz(conc_history[k], times) for k in range(n_species_total)]

    parent_auc = auc_values[0]
    metabolite_exposure_ratios = [
        auc_values[k] / parent_auc if parent_auc > 0 else 0.0 for k in range(1, n_species_total)
    ]

    notes = (
        f"Sequential {n_metabolites}-step metabolic pathway; "
        f"forward Euler dt={dt_h} h; "
        f"parent fm={fm_parent:.2f}"
    )

    return MetabolicPathwayResult(
        n_steps=n_metabolites,
        step_names=step_names_final,
        times_h=times,
        concentrations=conc_history,
        auc_values=auc_values,
        metabolite_exposure_ratios=metabolite_exposure_ratios,
        notes=notes,
    )


def calculate_metabolite_exposure_ratio(
    parent_auc: float,
    metabolite_auc: float,
) -> float:
    """Compute the metabolite-to-parent AUC ratio.

    Parameters
    ----------
    parent_auc : float
        AUC of the parent compound (any consistent unit).
    metabolite_auc : float
        AUC of the metabolite.

    Returns
    -------
    float
        Ratio metabolite_auc / parent_auc.

    Raises
    ------
    ValueError
        If parent_auc <= 0.
    """
    if parent_auc <= 0:
        raise ValueError("parent_auc must be > 0")
    return metabolite_auc / parent_auc


def predict_active_metabolite_contribution(
    parent_cl_L_per_h: float,
    fm: float,
    metabolite_potency_ratio: float,
    metabolite_cl_L_per_h: float,
    dose_mg: float,
    vd_L: float,
) -> dict:
    """Predict the pharmacological contribution of an active metabolite.

    Uses steady-state approximation:
        C_metabolite_ss = (fm * dose_mg/Vd_parent) / (ke_metabolite)
                         * (CL_parent / CL_metabolite)

    Effective contribution = C_metabolite_ss * potency_ratio / C_parent_ss

    Parameters
    ----------
    parent_cl_L_per_h : float
        Total clearance of the parent (L/h).
    fm : float
        Fraction of parent dose metabolised to the active metabolite (0–1).
    metabolite_potency_ratio : float
        Potency of metabolite relative to parent (e.g. 2 means twice as potent).
    metabolite_cl_L_per_h : float
        Total clearance of the metabolite (L/h).
    dose_mg : float
        Administered dose of parent (mg).
    vd_L : float
        Volume of distribution of parent (L).

    Returns
    -------
    dict with keys:
        'metabolite_auc_ratio'     : AUC_M / AUC_parent (fraction, unitless)
        'effective_contribution'   : potency-weighted contribution fraction
        'dominant_species'         : 'parent' or 'metabolite'
        'notes'                    : explanatory string
    """
    if parent_cl_L_per_h <= 0:
        raise ValueError("parent_cl_L_per_h must be > 0")
    if metabolite_cl_L_per_h <= 0:
        raise ValueError("metabolite_cl_L_per_h must be > 0")
    if not (0.0 <= fm <= 1.0):
        raise ValueError("fm must be in [0, 1]")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    # AUC_parent = dose / CL_parent  (1-cpt IV)
    auc_parent = dose_mg / parent_cl_L_per_h

    # AUC_metabolite = fm * dose / CL_metabolite
    auc_metabolite = fm * dose_mg / metabolite_cl_L_per_h

    metabolite_auc_ratio = auc_metabolite / auc_parent if auc_parent > 0 else 0.0

    # Potency-weighted contribution as fraction of combined activity
    parent_activity = auc_parent * 1.0  # relative potency = 1
    metabolite_activity = auc_metabolite * metabolite_potency_ratio
    total_activity = parent_activity + metabolite_activity
    effective_contribution = metabolite_activity / total_activity if total_activity > 0 else 0.0

    dominant_species = "metabolite" if effective_contribution > 0.5 else "parent"

    notes = (
        f"fm={fm:.2f}, potency_ratio={metabolite_potency_ratio:.2f}; "
        f"AUC_parent={auc_parent:.3g}, AUC_metabolite={auc_metabolite:.3g}; "
        f"metabolite accounts for {effective_contribution * 100:.1f}% of activity"
    )

    return {
        "metabolite_auc_ratio": metabolite_auc_ratio,
        "effective_contribution": effective_contribution,
        "dominant_species": dominant_species,
        "notes": notes,
    }
