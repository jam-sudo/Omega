"""
Phase 987 — Aminoglycoside PK model for therapeutic drug monitoring.

2-compartment IV infusion model for aminoglycosides (gentamicin, tobramycin).
Once-daily dosing: Cmax/MIC >= 8-10, trough < 1 mg/L.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AminoglycosidePKResult:
    """Result of aminoglycoside PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_central_mg_L: list
    cpeak_mg_L: float
    ctrough_mg_L: float
    auc_24h_mg_h_per_L: float
    cmax_mic_ratio: float
    auc_mic_ratio: float
    pk_pd_target_achieved: bool
    nephrotoxicity_risk: str
    dose_recommendation: str
    notes: str


def simulate_aminoglycoside_pk(
    drug_name: str = "Gentamicin",
    dose_mg: float = 320.0,
    patient_weight_kg: float = 70.0,
    gfr_mL_min: float = 90.0,
    mic_mg_L: float = 1.0,
    infusion_duration_h: float = 0.5,
    v1_L: float | None = None,
    v2_L: float | None = None,
    q_L_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> AminoglycosidePKResult:
    """
    Simulate aminoglycoside PK using a 2-compartment IV infusion model.

    Parameters
    ----------
    drug_name : str
        Name of the aminoglycoside drug.
    dose_mg : float
        Total dose in mg.
    patient_weight_kg : float
        Patient weight in kg.
    gfr_mL_min : float
        Glomerular filtration rate in mL/min.
    mic_mg_L : float
        Minimum inhibitory concentration in mg/L.
    infusion_duration_h : float
        Duration of IV infusion in hours.
    v1_L : float or None
        Central volume of distribution (L). If None, estimated from weight.
    v2_L : float or None
        Peripheral volume of distribution (L). If None, estimated from weight.
    q_L_per_h : float
        Inter-compartmental clearance (L/h).
    t_end_h : float
        End time of simulation in hours.
    dt_h : float
        Time step for Forward Euler integration in hours.

    Returns
    -------
    AminoglycosidePKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if patient_weight_kg <= 0:
        raise ValueError(f"patient_weight_kg must be positive, got {patient_weight_kg}")
    if gfr_mL_min <= 0:
        raise ValueError(f"gfr_mL_min must be positive, got {gfr_mL_min}")
    if mic_mg_L <= 0:
        raise ValueError(f"mic_mg_L must be positive, got {mic_mg_L}")
    if infusion_duration_h <= 0:
        raise ValueError(f"infusion_duration_h must be positive, got {infusion_duration_h}")

    # Volume of distribution defaults
    if v1_L is None:
        v1_L = 0.25 * patient_weight_kg
    if v2_L is None:
        v2_L = 0.15 * patient_weight_kg

    # Clearance (primarily renal)
    cl_L_per_h = gfr_mL_min * 60.0 / 1000.0 * 0.85

    # Rate constants
    ke10 = cl_L_per_h / v1_L
    ke12 = q_L_per_h / v1_L
    ke21 = q_L_per_h / v2_L

    # Infusion rate (mg/h)
    k_inf = dose_mg / infusion_duration_h

    # Forward Euler integration
    n_steps = int(t_end_h / dt_h) + 1
    times_h: list[float] = []
    c_central: list[float] = []

    c1 = 0.0  # central compartment concentration (mg/L)
    c2 = 0.0  # peripheral compartment concentration (mg/L)

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)
        c_central.append(c1)

        # Infusion active during t <= infusion_duration_h
        infusing = 1.0 if t < infusion_duration_h else 0.0

        dc1 = k_inf / v1_L * infusing - (ke10 + ke12) * c1 + ke21 * c2 * v2_L / v1_L
        dc2 = ke12 * c1 * v1_L / v2_L - ke21 * c2

        c1 = max(0.0, c1 + dc1 * dt_h)
        c2 = max(0.0, c2 + dc2 * dt_h)

    # Peak concentration: max during or shortly after infusion
    cpeak_mg_L = max(c_central)

    # Trough: concentration at t=24h (last time point if t_end_h=24)
    # Find the value at t closest to 24h
    target_trough_h = min(t_end_h, 24.0)
    trough_idx = min(
        range(len(times_h)),
        key=lambda i: abs(times_h[i] - target_trough_h),
    )
    ctrough_mg_L = c_central[trough_idx]

    # AUC 0-24h via trapezoidal rule
    auc_24h = 0.0
    end_idx = trough_idx
    for i in range(end_idx):
        auc_24h += 0.5 * (c_central[i] + c_central[i + 1]) * (times_h[i + 1] - times_h[i])

    # PK/PD indices
    cmax_mic_ratio = cpeak_mg_L / mic_mg_L
    auc_mic_ratio = auc_24h / mic_mg_L

    # PK/PD target: Cmax/MIC >= 8 AND trough < 1.0 mg/L
    pk_pd_target_achieved: bool = cmax_mic_ratio >= 8.0 and ctrough_mg_L < 1.0

    # Nephrotoxicity risk based on trough
    if ctrough_mg_L > 2.0:
        nephrotoxicity_risk = "high"
    elif ctrough_mg_L > 1.0:
        nephrotoxicity_risk = "moderate"
    else:
        nephrotoxicity_risk = "low"

    # Dose recommendation
    if cmax_mic_ratio < 8.0:
        dose_recommendation = "increase"
    elif ctrough_mg_L > 2.0:
        dose_recommendation = "decrease"
    else:
        dose_recommendation = "maintain"

    # Build notes
    notes_parts = [
        f"{drug_name} {dose_mg:.0f} mg IV infusion over {infusion_duration_h:.1f} h.",
        f"CL={cl_L_per_h:.2f} L/h (GFR={gfr_mL_min:.0f} mL/min).",
        f"Cpeak={cpeak_mg_L:.2f} mg/L, Ctrough={ctrough_mg_L:.3f} mg/L.",
        f"Cmax/MIC={cmax_mic_ratio:.1f}, AUC/MIC={auc_mic_ratio:.1f}.",
        f"Nephrotoxicity risk: {nephrotoxicity_risk}.",
    ]
    notes = " ".join(notes_parts)

    return AminoglycosidePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_central_mg_L=c_central,
        cpeak_mg_L=cpeak_mg_L,
        ctrough_mg_L=ctrough_mg_L,
        auc_24h_mg_h_per_L=auc_24h,
        cmax_mic_ratio=cmax_mic_ratio,
        auc_mic_ratio=auc_mic_ratio,
        pk_pd_target_achieved=pk_pd_target_achieved,
        nephrotoxicity_risk=nephrotoxicity_risk,
        dose_recommendation=dose_recommendation,
        notes=notes,
    )
