"""Gastric-retentive drug delivery system PK model.

Models floating tablets, bioadhesive, and expandable formulations that extend
gastric residence time from ~2h to 8-12h, improving absorption for drugs with
narrow absorption windows in the upper GI.
"""

from __future__ import annotations

from dataclasses import dataclass

_FORMULATION_PARAMS: dict[str, dict[str, float]] = {
    "floating": {
        "retention_factor": 0.7,
        "k_release": 0.15,
        "gastric_residence_h": 8.0,
    },
    "bioadhesive": {
        "retention_factor": 0.6,
        "k_release": 0.12,
        "gastric_residence_h": 6.0,
    },
    "expandable": {
        "retention_factor": 0.8,
        "k_release": 0.08,
        "gastric_residence_h": 10.0,
    },
    "conventional": {
        "retention_factor": 0.0,
        "k_release": 0.5,
        "gastric_residence_h": 2.0,
    },
}

_VALID_FORMULATIONS = frozenset(_FORMULATION_PARAMS.keys())


@dataclass
class GastricRetentionResult:
    """Result from gastric-retentive PK simulation."""

    drug_name: str
    dose_mg: float
    formulation_type: str
    retention_factor: float
    times_h: list
    c_plasma_mg_L: list
    a_gastric_mg: list
    a_intestine_mg: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    gastric_residence_h: float
    bioavailability_estimate_pct: float
    notes: str


def _validate_inputs(
    drug_name: str,
    dose_mg: float,
    formulation_type: str,
    cl_L_per_h: float,
    vd_L: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if formulation_type not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation_type must be one of {sorted(_VALID_FORMULATIONS)}, "
            f"got '{formulation_type}'"
        )


def simulate_gastric_retention_pk(
    drug_name: str,
    dose_mg: float,
    formulation_type: str = "floating",
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    k_abs_per_h: float = 1.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> GastricRetentionResult:
    """Simulate gastric-retentive formulation PK using Forward Euler ODE.

    Compartments:
    - A_gastric (mg): drug in stomach (releases from retentive matrix)
    - A_intestine (mg): drug in proximal small intestine
    - C_plasma (mg/L): systemic plasma

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        Administered dose in mg.
    formulation_type:
        One of "floating", "bioadhesive", "expandable", "conventional".
    cl_L_per_h:
        Clearance in L/h.
    vd_L:
        Volume of distribution in L.
    k_abs_per_h:
        Small intestine absorption rate constant (1/h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    GastricRetentionResult
    """
    _validate_inputs(drug_name, dose_mg, formulation_type, cl_L_per_h, vd_L)

    params = _FORMULATION_PARAMS[formulation_type]
    retention_factor: float = params["retention_factor"]
    k_release: float = params["k_release"]
    gastric_residence_h: float = params["gastric_residence_h"]
    k_gastric_empty: float = 0.5  # baseline gastric emptying rate (1/h)

    # Initial conditions
    a_gastric = dose_mg
    a_intestine = 0.0
    c_plasma = 0.0

    times_h: list[float] = []
    c_plasma_list: list[float] = []
    a_gastric_list: list[float] = []
    a_intestine_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_gastric_list.append(a_gastric)
        a_intestine_list.append(a_intestine)

        # ODEs
        da_gastric = -k_release * a_gastric - k_gastric_empty * a_gastric * (1.0 - retention_factor)
        da_intestine = (
            k_gastric_empty * a_gastric * (1.0 - retention_factor) - k_abs_per_h * a_intestine
        )
        dc_plasma = (
            k_abs_per_h * a_intestine / vd_L
            + k_release * retention_factor * a_gastric / vd_L
            - (cl_L_per_h / vd_L) * c_plasma
        )

        # Forward Euler update
        a_gastric = max(0.0, a_gastric + da_gastric * dt_h)
        a_intestine = max(0.0, a_intestine + da_intestine * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)

        t += dt_h

    # Compute PK metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_h = times_h[c_plasma_list.index(cmax_mg_L)]

    # Manual trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])

    # Bioavailability estimate: AUC / (dose/CL) * 100
    auc_iv_reference = dose_mg / cl_L_per_h  # mg·h/L if 100% bioavailable
    bioavailability_pct = (auc / auc_iv_reference) * 100.0 if auc_iv_reference > 0 else 0.0

    notes = (
        f"Gastric-retentive {formulation_type} formulation; "
        f"retention_factor={retention_factor:.2f}; "
        f"k_release={k_release:.3f}/h; "
        f"gastric_residence={gastric_residence_h:.1f}h"
    )

    return GastricRetentionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        retention_factor=retention_factor,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_gastric_mg=a_gastric_list,
        a_intestine_mg=a_intestine_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        gastric_residence_h=gastric_residence_h,
        bioavailability_estimate_pct=bioavailability_pct,
        notes=notes,
    )


def compare_formulations(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
) -> list[GastricRetentionResult]:
    """Compare all four formulation types and return sorted by AUC descending.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        Administered dose in mg.
    cl_L_per_h:
        Clearance in L/h.
    vd_L:
        Volume of distribution in L.

    Returns
    -------
    list of GastricRetentionResult sorted by auc_mg_h_per_L descending.
    """
    results = [
        simulate_gastric_retention_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation_type=ft,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
        )
        for ft in _FORMULATION_PARAMS
    ]
    return sorted(results, key=lambda r: r.auc_mg_h_per_L, reverse=True)
