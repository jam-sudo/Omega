"""Phase 260: Skin Wound Healing PK Model.

3-compartment ODE model (depot -> wound -> tissue; systemic absorption to
plasma) for topical drug delivery in wound healing.

ODEs are integrated with forward Euler. AUC is computed via manual trapezoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Formulation lookup table
# ---------------------------------------------------------------------------

FORMULATION_TYPES: dict[str, dict[str, float]] = {
    "hydrogel": {
        "f_wound": 0.8,
        "f_systemic": 0.1,
        "ka": 0.5,
        "release_duration_h": 12.0,
    },
    "cream": {
        "f_wound": 0.6,
        "f_systemic": 0.15,
        "ka": 0.3,
        "release_duration_h": 8.0,
    },
    "film": {
        "f_wound": 0.9,
        "f_systemic": 0.05,
        "ka": 0.2,
        "release_duration_h": 24.0,
    },
    "antibiotic_ointment": {
        "f_wound": 0.7,
        "f_systemic": 0.1,
        "ka": 0.4,
        "release_duration_h": 16.0,
    },
}

# Compartment volumes (L)
_V_WOUND = 0.01
_V_TISSUE = 0.1
_V_PLASMA = 3.0

# Rate constants (1/h)
_K_WT = 0.1  # wound -> tissue transfer
_K_ABS_SYS = 0.05  # wound -> plasma absorption
_K_TE = 0.05  # tissue elimination


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


# ---------------------------------------------------------------------------
# Result dataclass (plain, has list fields)
# ---------------------------------------------------------------------------


@dataclass
class WoundHealingPKResult:
    """Full PK simulation result for topical wound healing drug delivery."""

    drug_name: str
    dose_mg: float
    formulation_type: str
    times_h: list = field(default_factory=list)
    c_wound_mg_mL: list = field(default_factory=list)
    c_tissue_mg_mL: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_wound_mg_mL: float = 0.0
    tmax_wound_h: float = 0.0
    auc_wound_mg_h_per_mL: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    local_systemic_ratio: float = 0.0
    wound_coverage_score: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Simulation function
# ---------------------------------------------------------------------------


def simulate_wound_healing_pk(
    drug_name: str,
    dose_mg: float,
    formulation_type: str,
    cl_L_per_h: float = 5.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> WoundHealingPKResult:
    """Simulate wound bed and tissue PK for a topical formulation.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        Applied dose in milligrams (must be > 0).
    formulation_type:
        One of hydrogel, cream, film, antibiotic_ointment.
    cl_L_per_h:
        Systemic clearance in L/h (must be > 0).
    t_end_h:
        Simulation duration in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    WoundHealingPKResult
        Dataclass with concentration-time profiles and summary metrics.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if formulation_type not in FORMULATION_TYPES:
        raise ValueError(
            f"formulation_type must be one of {sorted(FORMULATION_TYPES)}, got '{formulation_type}'"
        )

    form = FORMULATION_TYPES[formulation_type]
    f_wound = form["f_wound"]
    f_systemic = form["f_systemic"]
    ka = form["ka"]
    release_duration_h = form["release_duration_h"]

    ke = cl_L_per_h / _V_PLASMA  # plasma elimination rate constant

    # --- State initialisation ---
    a_depot = dose_mg  # mg
    c_wound = 0.0  # mg/mL
    c_tissue = 0.0  # mg/mL
    c_plasma = 0.0  # mg/L

    times: list[float] = []
    c_wound_list: list[float] = []
    c_tissue_list: list[float] = []
    c_plasma_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h

        times.append(t)
        c_wound_list.append(c_wound)
        c_tissue_list.append(c_tissue)
        c_plasma_list.append(c_plasma)

        if step == n_steps:
            break

        # ka switches off after release_duration_h
        effective_ka = ka if t < release_duration_h else 0.0

        # ODEs
        da_depot = -effective_ka * a_depot

        dc_wound = (
            f_wound * effective_ka * a_depot / _V_WOUND - _K_WT * c_wound - _K_ABS_SYS * c_wound
        )

        dc_tissue = _K_WT * c_wound * _V_WOUND / _V_TISSUE - _K_TE * c_tissue

        dc_plasma = (
            f_systemic * effective_ka * a_depot / _V_PLASMA
            + _K_ABS_SYS * c_wound * _V_WOUND / _V_PLASMA
            - ke * c_plasma
        )

        # Forward Euler update
        a_depot = max(0.0, a_depot + da_depot * dt_h)
        c_wound = max(0.0, c_wound + dc_wound * dt_h)
        c_tissue = max(0.0, c_tissue + dc_tissue * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)

    # --- Summary metrics ---
    cmax_wound = max(c_wound_list)
    tmax_wound = times[c_wound_list.index(cmax_wound)]

    auc_wound = _trapz(times, c_wound_list)
    auc_plasma = _trapz(times, c_plasma_list)

    local_systemic_ratio = auc_wound / auc_plasma if auc_plasma > 0.0 else float("inf")

    # wound_coverage_score: scale AUC wound to 0-100
    # A wound AUC of 20 mg*h/mL would give score 100
    wound_coverage_score = min(100.0, auc_wound * 5.0)

    notes = (
        f"Wound healing PK simulation for {formulation_type} formulation. "
        f"Release duration: {release_duration_h} h. "
        f"f_wound={f_wound}, f_systemic={f_systemic}, ka={ka}/h. "
        f"Systemic CL={cl_L_per_h} L/h."
    )

    result = WoundHealingPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        times_h=times,
        c_wound_mg_mL=c_wound_list,
        c_tissue_mg_mL=c_tissue_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_wound_mg_mL=cmax_wound,
        tmax_wound_h=tmax_wound,
        auc_wound_mg_h_per_mL=auc_wound,
        auc_plasma_mg_h_per_L=auc_plasma,
        local_systemic_ratio=local_systemic_ratio,
        wound_coverage_score=wound_coverage_score,
        notes=notes,
    )
    return result


# ---------------------------------------------------------------------------
# Formulation comparison
# ---------------------------------------------------------------------------


def compare_wound_formulations(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
) -> list[WoundHealingPKResult]:
    """Compare all formulation types and sort by auc_wound descending.

    Returns a list of 4 WoundHealingPKResult objects, best first.
    """
    results = [
        simulate_wound_healing_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation_type=ft,
            cl_L_per_h=cl_L_per_h,
        )
        for ft in FORMULATION_TYPES
    ]
    return sorted(results, key=lambda r: r.auc_wound_mg_h_per_mL, reverse=True)
