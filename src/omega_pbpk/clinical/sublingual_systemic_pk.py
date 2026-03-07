"""Phase 914 — Sublingual to Systemic PK Simulator.

Models sublingual drug absorption directly to systemic circulation.
Focused on PK time-course for clinical use.
Pure Python / stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_VALID_DRUG_TYPES = {"small_molecule", "peptide", "opioid"}

_DRUG_TYPE_PARAMS = {
    "small_molecule": {"k_abs": 10.0, "f_sl": 0.75},
    "opioid": {"k_abs": 8.0, "f_sl": 0.55},
    "peptide": {"k_abs": 1.0, "f_sl": 0.15},
}


@dataclass
class SublingualSystemicPKResult:
    """Result of a sublingual-to-systemic PK simulation."""

    drug_name: str
    dose_mg: float
    drug_type: str
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    f_sublingual: float = 0.0
    time_to_onset_min: float = 0.0
    time_to_peak_min: float = 0.0
    duration_above_half_max_h: float = 0.0
    notes: str = ""


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i - 1] + concs[i]) * (times[i] - times[i - 1])
    return auc


def simulate_sublingual_systemic_pk(
    drug_name: str,
    dose_mg: float,
    drug_type: str = "small_molecule",
    cl_L_per_h: float = 15.0,
    vd_L: float = 100.0,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> SublingualSystemicPKResult:
    """Simulate sublingual drug absorption to systemic circulation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams (must be > 0).
    drug_type:
        One of "small_molecule", "opioid", "peptide".
    cl_L_per_h:
        Systemic clearance in L/h (must be > 0).
    vd_L:
        Volume of distribution in L (must be > 0).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward-Euler time step in hours.

    Returns
    -------
    SublingualSystemicPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {sorted(_VALID_DRUG_TYPES)}, got {drug_type!r}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    params = _DRUG_TYPE_PARAMS[drug_type]
    k_abs: float = params["k_abs"]
    f_sl: float = params["f_sl"]
    ke: float = cl_L_per_h / vd_L

    # --- Forward Euler ODE integration ---
    a_sl = dose_mg
    a_plasma = 0.0

    times: list[float] = []
    c_plasma_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        c = a_plasma / vd_L
        times.append(t)
        c_plasma_list.append(c)

        da_sl = -k_abs * a_sl
        da_plasma = k_abs * a_sl * f_sl - ke * a_plasma

        a_sl += da_sl * dt_h
        a_plasma += da_plasma * dt_h
        if a_sl < 0.0:
            a_sl = 0.0
        if a_plasma < 0.0:
            a_plasma = 0.0

        t += dt_h

    # --- Derived PK metrics ---
    cmax = max(c_plasma_list)
    tmax = times[c_plasma_list.index(cmax)]
    auc = _trapezoidal_auc(times, c_plasma_list)

    f_sublingual = (auc * cl_L_per_h) / dose_mg
    f_sublingual = max(0.0, min(1.0, f_sublingual))

    # Time to onset: first time c_plasma >= 10% of Cmax
    time_to_onset_min = 0.0
    threshold_onset = 0.1 * cmax
    for i, c in enumerate(c_plasma_list):
        if c >= threshold_onset:
            time_to_onset_min = times[i] * 60.0
            break

    time_to_peak_min = tmax * 60.0

    # Duration above 50% Cmax
    threshold_half = 0.5 * cmax
    duration_above_half_max_h = sum(dt_h for c in c_plasma_list if c >= threshold_half)

    # --- Notes ---
    if drug_type == "peptide":
        notes = "Poor sublingual absorption for peptides — consider buccal film or nasal route"
    elif time_to_onset_min < 5.0:
        notes = "Very rapid systemic onset — suitable for acute/emergency dosing"
    else:
        notes = "Standard sublingual kinetics"

    return SublingualSystemicPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        drug_type=drug_type,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_sublingual=f_sublingual,
        time_to_onset_min=time_to_onset_min,
        time_to_peak_min=time_to_peak_min,
        duration_above_half_max_h=duration_above_half_max_h,
        notes=notes,
    )


def compare_drug_types(
    drug_name: str,
    dose_mg: float,
    **kwargs,
) -> list[SublingualSystemicPKResult]:
    """Simulate all three drug types and return sorted by AUC (desc).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    **kwargs:
        Additional keyword arguments forwarded to
        :func:`simulate_sublingual_systemic_pk` (e.g. cl_L_per_h, vd_L).

    Returns
    -------
    List[SublingualSystemicPKResult] sorted by auc_mg_h_per_L descending.
    """
    results = []
    for drug_type in _VALID_DRUG_TYPES:
        result = simulate_sublingual_systemic_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            drug_type=drug_type,
            **kwargs,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
