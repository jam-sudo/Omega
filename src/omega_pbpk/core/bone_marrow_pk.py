"""bone_marrow_pk.py — 2-compartment bone marrow PK model.

Bone marrow is the primary target/toxicity site for cytotoxic drugs.
Models drug penetration into marrow based on blood flow and lipophilicity.
Uses Forward Euler integration; no numpy/scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BoneMarrowPKResult:
    """Simulation result from simulate_bone_marrow_pk."""

    drug_name: str
    dose_mg: float
    logp: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_bm_mg_L: list[float]
    cmax_plasma: float
    cmax_bm: float
    auc_plasma: float
    auc_bm: float
    bm_to_plasma_auc_ratio: float
    kp_marrow: float
    hematotox_index: float
    hematotox_risk: str
    t_half_plasma_h: float
    notes: str


# ---------------------------------------------------------------------------
# Helper: Kp calculation
# ---------------------------------------------------------------------------


def _calc_kp_marrow(logp: float) -> float:
    """Compute bone marrow partition coefficient.

    Hydrophilic drugs (logP < 1): low partition.
    Lipophilic drugs (logP > 2): increasing partition.
    """
    if logp < 1.0:
        kp = 0.3 + logp * 0.1
    else:
        kp = 0.3 + 0.1 + (logp - 1.0) * 0.1  # linear between 1 and 2
        if logp > 2.0:
            # spec: Kp = 0.5 + (logP-2)*0.2 for logP > 2
            kp = 0.5 + (logp - 2.0) * 0.2
    return max(0.1, min(5.0, kp))


# ---------------------------------------------------------------------------
# Helper: trapezoidal AUC
# ---------------------------------------------------------------------------


def _trapz_auc(times: list[float], concs: list[float]) -> float:
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i - 1] + concs[i]) * (times[i] - times[i - 1])
    return auc


# ---------------------------------------------------------------------------
# Helper: hematotoxicity risk classification
# ---------------------------------------------------------------------------


def _classify_hematotox_risk(hematotox_index: float) -> str:
    """Classify hematotoxicity risk from hematotox_index.

    < 5   → "low"
    5–20  → "moderate"
    > 20  → "high"
    """
    if hematotox_index < 5.0:
        return "low"
    elif hematotox_index <= 20.0:
        return "moderate"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_bone_marrow_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 1.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    q_bm_L_per_h: float = 2.5,
    v_bm_L: float = 1.5,
    ke_bm_per_h: float = 0.05,
    drug_sensitivity_factor: float = 1.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> BoneMarrowPKResult:
    """Simulate 2-compartment bone marrow PK (plasma + bone marrow).

    IV bolus dosing.  Forward Euler integration.

    Parameters
    ----------
    drug_name : str
    dose_mg : float  — IV bolus dose in mg
    logp : float     — octanol-water partition coefficient (log)
    cl_L_per_h : float  — systemic clearance (L/h)
    vd_L : float        — plasma/central volume (L)
    q_bm_L_per_h : float — bone marrow blood flow (L/h); default 2.5
    v_bm_L : float       — bone marrow volume (L); default 1.5
    ke_bm_per_h : float  — marrow elimination rate constant (1/h); default 0.05
    drug_sensitivity_factor : float — multiplier for hematotoxicity index; default 1.0
    t_end_h : float  — simulation end time (h); default 72
    dt_h : float     — Forward Euler step (h); default 0.1

    Returns
    -------
    BoneMarrowPKResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if v_bm_L <= 0:
        raise ValueError(f"v_bm_L must be > 0, got {v_bm_L}")

    kp_marrow = _calc_kp_marrow(logp)
    ke = cl_L_per_h / vd_L  # plasma elimination rate constant

    # Initial conditions — IV bolus into plasma
    c_plasma = dose_mg / vd_L  # mg/L
    c_bm = 0.0

    times: list[float] = [0.0]
    cp_list: list[float] = [c_plasma]
    cbm_list: list[float] = [c_bm]

    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps):
        # --- ODEs ---
        # dC_plasma/dt = -ke × C_plasma - Q_bm × (C_plasma - C_bm/Kp_bm) / Vd
        bm_flux_plasma = q_bm_L_per_h * (c_plasma - c_bm / kp_marrow) / vd_L
        dc_plasma = -ke * c_plasma - bm_flux_plasma

        # dC_bm/dt = Q_bm × (C_plasma - C_bm/Kp_bm) / V_bm - ke_bm × C_bm
        bm_flux_bm = q_bm_L_per_h * (c_plasma - c_bm / kp_marrow) / v_bm_L
        dc_bm = bm_flux_bm - ke_bm_per_h * c_bm

        # Forward Euler
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        c_bm = max(0.0, c_bm + dc_bm * dt_h)

        t = times[-1] + dt_h
        times.append(t)
        cp_list.append(c_plasma)
        cbm_list.append(c_bm)

    # --- derived metrics ---
    cmax_plasma = max(cp_list)
    cmax_bm = max(cbm_list)
    auc_plasma = _trapz_auc(times, cp_list)
    auc_bm = _trapz_auc(times, cbm_list)

    bm_ratio = auc_bm / auc_plasma if auc_plasma > 0 else 0.0
    hematotox_index = auc_bm * drug_sensitivity_factor
    hematotox_risk = _classify_hematotox_risk(hematotox_index)
    t_half_plasma = math.log(2.0) / ke

    notes = (
        f"Kp_marrow={kp_marrow:.3f}; Q_BM={q_bm_L_per_h} L/h; "
        f"ke={ke:.4f} /h; t½_plasma={t_half_plasma:.2f} h; "
        f"hematotox_index={hematotox_index:.3f} ({hematotox_risk} risk)"
    )

    return BoneMarrowPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_bm_mg_L=cbm_list,
        cmax_plasma=cmax_plasma,
        cmax_bm=cmax_bm,
        auc_plasma=auc_plasma,
        auc_bm=auc_bm,
        bm_to_plasma_auc_ratio=bm_ratio,
        kp_marrow=kp_marrow,
        hematotox_index=hematotox_index,
        hematotox_risk=hematotox_risk,
        t_half_plasma_h=t_half_plasma,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------


def assess_hematotoxicity_risk(result: BoneMarrowPKResult) -> dict:
    """Assess hematotoxicity risk from a BoneMarrowPKResult.

    Uses hematotox_risk already stored on the result object.

    Returns
    -------
    dict with keys: risk_level, auc_bm, hematotox_index,
                    bm_to_plasma_auc_ratio, recommendation
    """
    risk_level = result.hematotox_risk

    if risk_level == "low":
        recommendation = "Bone marrow exposure is low.  Routine CBC monitoring is adequate."
    elif risk_level == "moderate":
        recommendation = (
            "Moderate bone marrow exposure.  "
            "Monitor CBC weekly; consider dose reduction in elderly or myelosuppressed patients."
        )
    else:
        recommendation = (
            "High bone marrow exposure predicted.  "
            "Intensive CBC monitoring required; prophylactic G-CSF should be considered."
        )

    return {
        "risk_level": risk_level,
        "auc_bm": result.auc_bm,
        "hematotox_index": result.hematotox_index,
        "bm_to_plasma_auc_ratio": result.bm_to_plasma_auc_ratio,
        "recommendation": recommendation,
    }
