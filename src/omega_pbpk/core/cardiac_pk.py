"""cardiac_pk.py — 3-compartment cardiac tissue PK model.

Models drug distribution to cardiac tissue, critical for antiarrhythmics
and cardiotoxic drugs.  Uses Forward Euler integration; no numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CardiacPKResult:
    """Simulation result from simulate_cardiac_pk."""

    drug_name: str
    dose_mg: float
    logp: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_cardiac_mg_L: list[float]
    cmax_plasma: float
    cmax_cardiac: float
    auc_plasma: float
    auc_cardiac: float
    cardiac_to_plasma_auc_ratio: float
    kp_cardiac: float
    cardiac_safety_margin: float
    notes: str


# ---------------------------------------------------------------------------
# Helper: Kp calculation
# ---------------------------------------------------------------------------


def _calc_kp_cardiac(logp: float, charge_at_ph74: float) -> float:
    """Compute cardiac tissue partition coefficient."""
    charge_effect = 0.5 if charge_at_ph74 > 0 else 0.0
    kp_base = 1.5 + logp * 0.3 - charge_effect
    return max(0.1, min(10.0, kp_base))


# ---------------------------------------------------------------------------
# Helper: trapezoidal AUC
# ---------------------------------------------------------------------------


def _trapz_auc(times: list[float], concs: list[float]) -> float:
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i - 1] + concs[i]) * (times[i] - times[i - 1])
    return auc


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_cardiac_pk(
    drug_name: str,
    dose_mg: float,
    logp: float = 2.0,
    charge_at_ph74: float = 0.0,
    cl_L_per_h: float = 5.0,
    vd_plasma_L: float = 5.0,
    cardiac_output_L_per_h: float = 300.0,
    v_cardiac_L: float = 0.3,
    v_periph_L: float = 30.0,
    q_periph_L_per_h: float = 80.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> CardiacPKResult:
    """Simulate 3-compartment cardiac PK (plasma, cardiac tissue, peripheral).

    IV bolus dosing.  Forward Euler integration.

    Parameters
    ----------
    drug_name : str
    dose_mg : float  — IV bolus dose in mg
    logp : float     — octanol-water partition coefficient (log)
    charge_at_ph74 : float — net molecular charge at physiological pH 7.4
    cl_L_per_h : float   — systemic clearance (L/h)
    vd_plasma_L : float  — plasma / central volume (L)
    cardiac_output_L_per_h : float — total cardiac output (L/h); default 300 = 5 L/min
    v_cardiac_L : float  — cardiac tissue volume (L); default 0.3
    v_periph_L : float   — peripheral volume (L); default 30
    q_periph_L_per_h : float — peripheral blood flow (L/h); default 80
    t_end_h : float  — simulation end time (h)
    dt_h : float     — Forward Euler step (h)

    Returns
    -------
    CardiacPKResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if vd_plasma_L <= 0:
        raise ValueError(f"vd_plasma_L must be > 0, got {vd_plasma_L}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if v_cardiac_L <= 0:
        raise ValueError(f"v_cardiac_L must be > 0, got {v_cardiac_L}")

    kp_cardiac = _calc_kp_cardiac(logp, charge_at_ph74)
    kp_periph = 1.0  # neutral partition for peripheral

    # Cardiac blood flow = 5 % of cardiac output
    q_cardiac = cardiac_output_L_per_h * 0.05  # 15 L/h at default CO

    # Initial conditions — IV bolus into plasma at t = 0
    c_plasma = dose_mg / vd_plasma_L  # mg/L
    c_cardiac = 0.0
    c_periph = 0.0

    times: list[float] = [0.0]
    cp_list: list[float] = [c_plasma]
    cc_list: list[float] = [c_cardiac]

    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps):
        # --- ODEs ---
        # Plasma
        elim = (cl_L_per_h / vd_plasma_L) * c_plasma
        cardiac_flux = q_cardiac * (c_plasma - c_cardiac / kp_cardiac) / vd_plasma_L
        periph_flux = q_periph_L_per_h * (c_plasma - c_periph / kp_periph) / vd_plasma_L
        dc_plasma = -(elim + cardiac_flux + periph_flux)

        # Cardiac tissue
        dc_cardiac = q_cardiac * (c_plasma - c_cardiac / kp_cardiac) / v_cardiac_L

        # Peripheral
        dc_periph = q_periph_L_per_h * (c_plasma - c_periph / kp_periph) / v_periph_L

        # Forward Euler
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        c_cardiac = max(0.0, c_cardiac + dc_cardiac * dt_h)
        c_periph = max(0.0, c_periph + dc_periph * dt_h)

        t = times[-1] + dt_h
        times.append(t)
        cp_list.append(c_plasma)
        cc_list.append(c_cardiac)

    # --- derived metrics ---
    cmax_plasma = max(cp_list)
    cmax_cardiac = max(cc_list)
    auc_plasma = _trapz_auc(times, cp_list)
    auc_cardiac = _trapz_auc(times, cc_list)

    ratio = auc_cardiac / auc_plasma if auc_plasma > 0 else 0.0

    safety_threshold_mg_L = 10.0
    cardiac_safety_margin = (
        safety_threshold_mg_L / cmax_cardiac if cmax_cardiac > 0 else float("inf")
    )

    notes = (
        f"Kp_cardiac={kp_cardiac:.3f}; Q_cardiac={q_cardiac:.2f} L/h "
        f"({100 * 0.05:.0f}% of CO); cardiac/plasma AUC ratio={ratio:.3f}"
    )

    return CardiacPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logp=logp,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_cardiac_mg_L=cc_list,
        cmax_plasma=cmax_plasma,
        cmax_cardiac=cmax_cardiac,
        auc_plasma=auc_plasma,
        auc_cardiac=auc_cardiac,
        cardiac_to_plasma_auc_ratio=ratio,
        kp_cardiac=kp_cardiac,
        cardiac_safety_margin=cardiac_safety_margin,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------


def assess_cardiac_concentration_risk(result: CardiacPKResult) -> dict:
    """Assess cardiac concentration risk from a CardiacPKResult.

    Risk stratification is based on cardiac-to-plasma AUC ratio:
      < 1.5  → "low"
      1.5–3.0 → "moderate"
      > 3.0  → "high"

    Returns
    -------
    dict with keys: risk_level, cardiac_to_plasma_auc_ratio,
                    cmax_cardiac, cardiac_safety_margin, recommendation
    """
    ratio = result.cardiac_to_plasma_auc_ratio

    if ratio < 1.5:
        risk_level = "low"
        recommendation = "Cardiac accumulation is minimal.  Standard monitoring is appropriate."
    elif ratio <= 3.0:
        risk_level = "moderate"
        recommendation = (
            "Moderate cardiac accumulation detected.  "
            "Consider ECG monitoring and dose adjustments in at-risk populations."
        )
    else:
        risk_level = "high"
        recommendation = (
            "High cardiac accumulation detected.  "
            "Intensive cardiac monitoring required; evaluate risk-benefit carefully."
        )

    return {
        "risk_level": risk_level,
        "cardiac_to_plasma_auc_ratio": ratio,
        "cmax_cardiac": result.cmax_cardiac,
        "cardiac_safety_margin": result.cardiac_safety_margin,
        "recommendation": recommendation,
    }
