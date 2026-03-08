"""
Phase 417 — Dialysis PK: drug PK during hemodialysis / peritoneal dialysis.
"""

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DialysisPKResult:
    drug_name: str
    dialysis_type: str
    dose_mg: float
    cl_native_L_per_h: float
    cl_dialysis_L_per_h: float
    cl_total_L_per_h: float
    vd_L: float
    t_half_on_dialysis_h: float
    t_half_off_dialysis_h: float
    dose_removed_mg: float
    dose_removed_pct: float
    supplemental_dose_mg: float
    monitoring_flag: bool
    notes: str


def calculate_dialysis_pk(
    drug_name: str,
    dialysis_type: str,
    dose_mg: float,
    cl_native_L_per_h: float,
    vd_L: float,
    fu_plasma: float,
    mw_Da: float,
    session_duration_h: float = 4.0,
    sessions_per_week: int = 3,
    qb_L_per_h: float = 18.0,
) -> DialysisPKResult:
    """Calculate drug PK parameters during dialysis session.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dialysis_type : str
        Either "hemodialysis" or "peritoneal".
    dose_mg : float
        Administered dose in mg.
    cl_native_L_per_h : float
        Patient's residual (native) clearance in L/h.
    vd_L : float
        Volume of distribution in L.
    fu_plasma : float
        Unbound fraction in plasma (0 < fu_plasma <= 1).
    mw_Da : float
        Molecular weight in Daltons.
    session_duration_h : float
        Duration of dialysis session in hours (default 4.0).
    sessions_per_week : int
        Number of sessions per week (default 3, informational).
    qb_L_per_h : float
        Blood flow rate in L/h for hemodialysis (default 18.0).

    Returns
    -------
    DialysisPKResult
    """
    # Validation
    if dialysis_type not in {"hemodialysis", "peritoneal"}:
        raise ValueError(
            f"dialysis_type must be 'hemodialysis' or 'peritoneal', got '{dialysis_type}'"
        )
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_native_L_per_h < 0:
        raise ValueError(f"cl_native_L_per_h must be >= 0, got {cl_native_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")

    # Calculate dialysis clearance
    if dialysis_type == "hemodialysis":
        # Larger MW reduces dialyzability
        extraction_ratio = fu_plasma * min(1.0, 1.0 - (mw_Da / 10000.0) * 0.3)
        extraction_ratio = max(0.0, min(0.95, extraction_ratio))
        cl_dialysis = qb_L_per_h * extraction_ratio
    else:  # peritoneal
        cl_dialysis = fu_plasma * min(0.5, 1.0 - mw_Da / 20000.0) * 2.0
        cl_dialysis = max(0.0, cl_dialysis)

    cl_total = cl_native_L_per_h + cl_dialysis

    # Half-lives
    t_half_on = 0.693 * vd_L / cl_total
    t_half_off = 0.693 * vd_L / max(cl_native_L_per_h, 0.001)

    # Dose removed during session (single-compartment approximation)
    dose_removed = dose_mg * (1.0 - math.exp(-cl_dialysis / vd_L * session_duration_h))
    dose_removed_pct = dose_removed / dose_mg * 100.0

    # Supplemental dose recommendation
    if dose_removed_pct > 30.0:
        supplemental_dose_mg = dose_removed * 0.5
    else:
        supplemental_dose_mg = 0.0

    monitoring_flag = dose_removed_pct > 30.0

    # Build notes
    rec = (
        "Post-dialysis supplemental dose recommended."
        if monitoring_flag
        else "No supplemental dose needed."
    )
    notes = (
        f"{dialysis_type.capitalize()} session ({session_duration_h:.1f} h): "
        f"{dose_removed_pct:.1f}% of dose removed ({dose_removed:.2f} mg). {rec}"
    )

    return DialysisPKResult(
        drug_name=drug_name,
        dialysis_type=dialysis_type,
        dose_mg=dose_mg,
        cl_native_L_per_h=cl_native_L_per_h,
        cl_dialysis_L_per_h=cl_dialysis,
        cl_total_L_per_h=cl_total,
        vd_L=vd_L,
        t_half_on_dialysis_h=t_half_on,
        t_half_off_dialysis_h=t_half_off,
        dose_removed_mg=dose_removed,
        dose_removed_pct=dose_removed_pct,
        supplemental_dose_mg=supplemental_dose_mg,
        monitoring_flag=monitoring_flag,
        notes=notes,
    )


def compare_dialysis_drugs(
    drugs_list: list[dict[str, Any]],
    dialysis_type: str,
) -> list[DialysisPKResult]:
    """Compare multiple drugs under the same dialysis modality.

    Parameters
    ----------
    drugs_list : list of dict
        Each dict has keys matching calculate_dialysis_pk params (except dialysis_type).
    dialysis_type : str
        "hemodialysis" or "peritoneal".

    Returns
    -------
    list of DialysisPKResult sorted by dose_removed_pct descending.
    """
    results = []
    for drug in drugs_list:
        result = calculate_dialysis_pk(
            drug_name=drug["drug_name"],
            dialysis_type=dialysis_type,
            dose_mg=drug["dose_mg"],
            cl_native_L_per_h=drug["cl_native_L_per_h"],
            vd_L=drug["vd_L"],
            fu_plasma=drug["fu_plasma"],
            mw_Da=drug["mw_Da"],
            session_duration_h=drug.get("session_duration_h", 4.0),
            sessions_per_week=drug.get("sessions_per_week", 3),
            qb_L_per_h=drug.get("qb_L_per_h", 18.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.dose_removed_pct, reverse=True)
    return results
