"""Phase 641 — Drug Gastric Retention Formulation.

Models drug release from gastric retentive formulations (floating tablets,
swellable matrices, bioadhesive systems). Tracks drug in gastric depot,
dissolved gastric fluid, and systemic plasma compartments via Forward Euler.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_FORMULATIONS = frozenset({"floating", "swellable", "bioadhesive"})

_FORMULATION_K_RELEASE: dict[str, float] = {
    "floating": 0.05,
    "swellable": 0.08,
    "bioadhesive": 0.03,
}

_FORMULATION_RETENTION_H: dict[str, float] = {
    "floating": 8.0,
    "swellable": 6.0,
    "bioadhesive": 12.0,
}

_V_GASTRIC_ML = 50.0
_KA_ABS_PER_H = 0.5


@dataclass
class GastricRetentionResult:
    """Result from gastric retention formulation simulation."""

    drug_name: str
    dose_mg: float
    formulation_type: str
    times_h: list
    c_gastric_depot_mg: list
    c_dissolved_mg_mL: list
    c_plasma_mg_L: list
    release_rate_mg_h: list
    fraction_released: list
    tmax_h: float
    cmax_mg_L: float
    auc_mg_h_per_L: float
    gastric_retention_time_h: float
    dose_release_sustained_h: float
    notes: str


def _validate_inputs(
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    formulation_type: str,
    solubility_mg_mL: float,
) -> None:
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0.0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0.0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if formulation_type not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation_type must be one of {sorted(_VALID_FORMULATIONS)}, "
            f"got '{formulation_type}'"
        )
    if solubility_mg_mL <= 0.0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")


def simulate_gastric_retention(
    drug_name: str,
    dose_mg: float,
    formulation_type: str,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    gastric_ph: float = 2.0,
    solubility_mg_mL: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> GastricRetentionResult:
    """Simulate drug release from a gastric retentive formulation.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Total dose loaded into the gastric retentive formulation (mg).
    formulation_type:
        One of "floating", "swellable", "bioadhesive".
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution (L).
    gastric_ph:
        Gastric pH (informational; affects solubility note).
    solubility_mg_mL:
        Drug solubility in gastric fluid (mg/mL).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    GastricRetentionResult
    """
    _validate_inputs(dose_mg, cl_sys_L_per_h, vd_sys_L, formulation_type, solubility_mg_mL)

    k_release = _FORMULATION_K_RELEASE[formulation_type]
    ke_sys = cl_sys_L_per_h / vd_sys_L  # elimination rate constant (1/h)
    c_dissolved_sat = solubility_mg_mL  # saturation concentration in gastric fluid

    # State variables
    a_depot = dose_mg  # drug in gastric depot (mg)
    c_dissolved = 0.0  # drug dissolved in gastric fluid (mg/mL)
    c_plasma = 0.0  # drug in systemic plasma (mg/L)

    n_steps = max(1, int(round(t_end_h / dt_h)))

    times_h: list = []
    c_gastric_depot_mg: list = []
    c_dissolved_mg_mL: list = []
    c_plasma_mg_L: list = []
    release_rate_mg_h: list = []
    fraction_released: list = []

    for i in range(n_steps + 1):
        t = i * dt_h

        # Record current state
        times_h.append(t)
        c_gastric_depot_mg.append(max(0.0, a_depot))
        c_dissolved_mg_mL.append(max(0.0, c_dissolved))
        c_plasma_mg_L.append(max(0.0, c_plasma))
        rr = k_release * max(0.0, a_depot)
        release_rate_mg_h.append(rr)
        fr = 1.0 - max(0.0, a_depot) / dose_mg
        fraction_released.append(fr)

        if i == n_steps:
            break

        # Forward Euler derivatives
        a_depot_curr = max(0.0, a_depot)
        c_dissolved_curr = max(0.0, c_dissolved)
        c_plasma_curr = max(0.0, c_plasma)

        # Release from depot (first-order)
        release = k_release * a_depot_curr

        # Dissolution in gastric fluid — limited by saturation
        # Rate of dissolution into gastric fluid
        if c_dissolved_curr < c_dissolved_sat:
            dissolve_rate = release / _V_GASTRIC_ML  # mg/mL/h
        else:
            dissolve_rate = 0.0

        # Absorption from gastric fluid into systemic
        abs_rate_mg_mL_h = _KA_ABS_PER_H * c_dissolved_curr  # mg/mL/h

        # Plasma input (mg/h) = abs_rate (mg/mL/h) * V_gastric (mL) / Vd (L) ... unit mgmL/h * mL / L
        # = abs_rate * V_gastric * (1/1000) [to convert mL to L] ... wait
        # C_plasma in mg/L, Vd in L
        # dC_plasma/dt = (ka_abs * C_dissolved * V_gastric) / Vd_L  - ke * C_plasma
        # Units: (1/h)(mg/mL)(mL) / L = mg/L/h  ✓  (since mL/L = 0.001, but V_gastric is explicit)
        # Actually V_gastric is in mL, Vd_sys is in L → need 1/1000 conversion
        plasma_input_mg_L_h = (_KA_ABS_PER_H * c_dissolved_curr * _V_GASTRIC_ML / 1000.0) / vd_sys_L

        d_a_depot = -release
        d_c_dissolved = dissolve_rate - abs_rate_mg_mL_h
        d_c_plasma = plasma_input_mg_L_h - ke_sys * c_plasma_curr

        a_depot = a_depot_curr + d_a_depot * dt_h
        c_dissolved = c_dissolved_curr + d_c_dissolved * dt_h
        c_plasma = c_plasma_curr + d_c_plasma * dt_h

        # Clamp negatives
        a_depot = max(0.0, a_depot)
        c_dissolved = max(0.0, c_dissolved)
        c_plasma = max(0.0, c_plasma)

    # Compute summary PK metrics
    cmax_mg_L = max(c_plasma_mg_L)
    tmax_h = times_h[c_plasma_mg_L.index(cmax_mg_L)]

    # Trapezoidal AUC
    auc = sum(
        0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    # Gastric retention time: first time fraction_released >= 0.9
    gastric_retention_time_h = t_end_h
    for t_val, fr_val in zip(times_h, fraction_released):
        if fr_val >= 0.9:
            gastric_retention_time_h = t_val
            break

    # Sustained release duration: cumulative time with C_plasma > 10% Cmax
    threshold = 0.1 * cmax_mg_L
    sustained_h = sum(dt_h for cp in c_plasma_mg_L if cp > threshold)

    notes = (
        f"Gastric pH={gastric_ph:.1f}; solubility={solubility_mg_mL:.2f} mg/mL; "
        f"k_release={k_release:.3f}/h; ke_sys={ke_sys:.3f}/h"
    )

    return GastricRetentionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        times_h=times_h,
        c_gastric_depot_mg=c_gastric_depot_mg,
        c_dissolved_mg_mL=c_dissolved_mg_mL,
        c_plasma_mg_L=c_plasma_mg_L,
        release_rate_mg_h=release_rate_mg_h,
        fraction_released=fraction_released,
        tmax_h=tmax_h,
        cmax_mg_L=cmax_mg_L,
        auc_mg_h_per_L=auc,
        gastric_retention_time_h=gastric_retention_time_h,
        dose_release_sustained_h=sustained_h,
        notes=notes,
    )
