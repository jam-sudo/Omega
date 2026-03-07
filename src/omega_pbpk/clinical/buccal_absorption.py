"""Phase 913 — Buccal Absorption Simulator.

Models drug absorption via buccal mucosa for fast-onset drugs
(sublingual nitroglycerin, buprenorphine, etc.).
Pure Python / stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_VALID_FORMULATIONS = {"film", "spray", "tablet", "gel"}

_FORMULATION_PARAMS = {
    "film": {"k_abs": 4.0, "f_absorbed": 0.70},
    "spray": {"k_abs": 6.0, "f_absorbed": 0.65},
    "tablet": {"k_abs": 1.5, "f_absorbed": 0.50},
    "gel": {"k_abs": 2.0, "f_absorbed": 0.60},
}

_FIRST_PASS_BYPASS = 0.95


@dataclass
class BuccalAbsorptionResult:
    """Result of a buccal absorption simulation."""

    drug_name: str
    dose_mg: float
    formulation: str
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    f_buccal: float = 0.0
    onset_time_min: float = 0.0
    first_pass_bypass_fraction: float = _FIRST_PASS_BYPASS
    compared_to_oral_auc_ratio: float = 1.0
    notes: str = ""


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i - 1] + concs[i]) * (times[i] - times[i - 1])
    return auc


def simulate_buccal_absorption(
    drug_name: str,
    dose_mg: float,
    formulation: str = "film",
    logp: float = 2.0,
    pka: float | None = None,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    t_end_h: float = 6.0,
    dt_h: float = 0.05,
) -> BuccalAbsorptionResult:
    """Simulate buccal drug absorption using a 1-compartment ODE model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams (must be > 0).
    formulation:
        One of "film", "spray", "tablet", "gel".
    logp:
        Lipophilicity (influences absorption but not used directly in this
        simplified model — reserved for future use).
    pka:
        pKa of the drug (optional; reserved for future extension).
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
    BuccalAbsorptionResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation must be one of {sorted(_VALID_FORMULATIONS)}, got {formulation!r}"
        )
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    params = _FORMULATION_PARAMS[formulation]
    k_abs: float = params["k_abs"]
    f_absorbed: float = params["f_absorbed"]
    ke: float = cl_L_per_h / vd_L

    # --- Forward Euler ODE integration ---
    a_buccal = dose_mg
    a_plasma = 0.0

    times: list[float] = []
    c_plasma_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        c = a_plasma / vd_L
        times.append(t)
        c_plasma_list.append(c)

        da_buccal = -k_abs * a_buccal
        da_plasma = k_abs * a_buccal * f_absorbed - ke * a_plasma

        a_buccal += da_buccal * dt_h
        a_plasma += da_plasma * dt_h
        if a_buccal < 0.0:
            a_buccal = 0.0
        if a_plasma < 0.0:
            a_plasma = 0.0

        t += dt_h

    # --- Derived PK metrics ---
    cmax = max(c_plasma_list)
    tmax = times[c_plasma_list.index(cmax)]
    auc = _trapezoidal_auc(times, c_plasma_list)

    f_buccal = (auc * cl_L_per_h) / dose_mg
    f_buccal = max(0.0, min(1.0, f_buccal))

    onset_time_min = tmax * 60.0 * 0.5

    compared_to_oral_auc_ratio = f_buccal / 0.5
    compared_to_oral_auc_ratio = max(0.1, min(5.0, compared_to_oral_auc_ratio))

    # --- Notes ---
    if onset_time_min < 5.0:
        notes = "Ultra-rapid onset — suitable for acute treatment"
    elif onset_time_min < 15.0:
        notes = "Rapid onset — suitable for PRN dosing"
    else:
        notes = "Moderate onset — consider spray formulation for faster absorption"

    return BuccalAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation=formulation,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_buccal=f_buccal,
        onset_time_min=onset_time_min,
        first_pass_bypass_fraction=_FIRST_PASS_BYPASS,
        compared_to_oral_auc_ratio=compared_to_oral_auc_ratio,
        notes=notes,
    )


def compare_buccal_formulations(
    drug_name: str,
    dose_mg: float,
    **kwargs,
) -> list[BuccalAbsorptionResult]:
    """Simulate all four buccal formulations and return sorted by AUC (desc).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    **kwargs:
        Additional keyword arguments forwarded to
        :func:`simulate_buccal_absorption` (e.g. cl_L_per_h, vd_L).

    Returns
    -------
    List[BuccalAbsorptionResult] sorted by auc_mg_h_per_L descending.
    """
    results = []
    for formulation in _VALID_FORMULATIONS:
        result = simulate_buccal_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation=formulation,
            **kwargs,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
