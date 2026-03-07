"""Phase 916 — Drug Bone Marrow Distribution.

Model drug distribution in bone marrow — relevant for myelosuppressive
chemotherapy agents (cyclophosphamide, methotrexate) and bone marrow
targeting drugs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoneMarrowPKResult:
    """Result of bone marrow PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: tuple
    c_plasma_mg_L: tuple
    c_bone_marrow_mg_L: tuple
    cmax_plasma: float
    auc_plasma: float
    cmax_marrow: float
    auc_marrow: float
    marrow_plasma_kp: float
    myelosuppression_score: float
    myelosuppression_risk: str
    estimated_nadir_time_h: float
    notes: str


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return auc


def simulate_bone_marrow_pk(
    drug_name: str,
    dose_mg: float,
    k_abs_per_h: float = 1.0,
    k_marrow_uptake_per_h: float = 0.5,
    k_marrow_release_per_h: float = 0.1,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    v_marrow_L: float = 1.5,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> BoneMarrowPKResult:
    """Simulate bone marrow drug PK using forward Euler."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if v_marrow_L <= 0:
        raise ValueError("v_marrow_L must be > 0")

    ke_sys = cl_L_per_h / vd_L

    # Initial conditions
    a_gut = dose_mg
    a_plasma = 0.0
    a_marrow = 0.0

    times = [0.0]
    c_plasma_list = [0.0]
    c_marrow_list = [0.0]

    n_steps = int(round(t_end_h / dt_h))
    for _ in range(n_steps):
        d_gut = -k_abs_per_h * a_gut
        d_plasma = (
            k_abs_per_h * a_gut
            - ke_sys * a_plasma
            - k_marrow_uptake_per_h * a_plasma
            + k_marrow_release_per_h * a_marrow
        )
        d_marrow = k_marrow_uptake_per_h * a_plasma - k_marrow_release_per_h * a_marrow

        a_gut += d_gut * dt_h
        a_plasma += d_plasma * dt_h
        a_marrow += d_marrow * dt_h

        # Clamp negative amounts due to numerical noise
        a_gut = max(0.0, a_gut)
        a_plasma = max(0.0, a_plasma)
        a_marrow = max(0.0, a_marrow)

        t = times[-1] + dt_h
        times.append(round(t, 10))
        c_plasma_list.append(a_plasma / vd_L)
        c_marrow_list.append(a_marrow / v_marrow_L)

    # PK metrics
    cmax_plasma = max(c_plasma_list)
    cmax_marrow = max(c_marrow_list)
    auc_plasma = _trapz(times, c_plasma_list)
    auc_marrow = _trapz(times, c_marrow_list)

    marrow_plasma_kp = auc_marrow / auc_plasma if auc_plasma > 0.0 else 0.0

    # Myelosuppression score
    myelosuppression_score = min(100.0, max(0.0, marrow_plasma_kp * 30.0 + cmax_marrow * 5.0))

    # Risk level
    if myelosuppression_score > 60:
        myelosuppression_risk = "high"
    elif myelosuppression_score > 30:
        myelosuppression_risk = "moderate"
    else:
        myelosuppression_risk = "low"

    # Estimated nadir: time after cmax_marrow when conc falls to 50% of cmax
    tmax_marrow_idx = c_marrow_list.index(cmax_marrow)
    half_cmax_marrow = cmax_marrow * 0.5
    estimated_nadir_time_h = times[-1]
    for i in range(tmax_marrow_idx + 1, len(c_marrow_list)):
        if c_marrow_list[i] <= half_cmax_marrow:
            estimated_nadir_time_h = times[i]
            break

    # Notes
    if myelosuppression_risk == "high":
        notes = "High myelosuppression risk — schedule G-CSF support. Monitor CBC at nadir."
    elif myelosuppression_risk == "moderate":
        notes = "Moderate bone marrow exposure — routine CBC monitoring recommended"
    else:
        notes = "Low myelosuppression risk"

    return BoneMarrowPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=tuple(times),
        c_plasma_mg_L=tuple(c_plasma_list),
        c_bone_marrow_mg_L=tuple(c_marrow_list),
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        cmax_marrow=cmax_marrow,
        auc_marrow=auc_marrow,
        marrow_plasma_kp=marrow_plasma_kp,
        myelosuppression_score=myelosuppression_score,
        myelosuppression_risk=myelosuppression_risk,
        estimated_nadir_time_h=estimated_nadir_time_h,
        notes=notes,
    )


def compare_dosing_regimens(
    drug_name: str,
    dose_mg: float,
    k_abs_values: list,
) -> list:
    """Compare bone marrow PK across different absorption rate constants.

    Returns list of BoneMarrowPKResult sorted by myelosuppression_score descending.
    """
    results = []
    for k_abs in k_abs_values:
        result = simulate_bone_marrow_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            k_abs_per_h=k_abs,
        )
        results.append(result)

    results.sort(key=lambda r: r.myelosuppression_score, reverse=True)
    return results
