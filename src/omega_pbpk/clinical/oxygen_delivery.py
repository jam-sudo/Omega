"""Tissue oxygen delivery and hypoxia effect on drug PK — Phase 260.

Models how anemia or hypoxia alter organ blood flow, CYP activity,
and drug clearance. Relevant for oncology (tumor hypoxia) and
critical illness PK.

References
----------
- Upton RN et al., J Pharmacokinet Pharmacodyn. 2004
- Crawford JH et al., Blood. 2006;107(5):2070-8 (anemia vasodilation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "OxygenDeliveryResult",
    "HypoxiaPKResult",
    "assess_oxygen_delivery",
    "simulate_hypoxia_pk",
]


@dataclass(frozen=True)
class OxygenDeliveryResult:
    """Result from oxygen delivery assessment."""

    hemoglobin_g_dL: float
    spo2_pct: float
    cardiac_output_L_per_min: float
    do2_mL_per_min: float  # Oxygen delivery (DO2)
    do2_index: float  # DO2 / BSA
    cyp_activity_fraction: float  # Relative CYP activity (1.0 = normal)
    hepatic_cl_fraction: float  # Relative hepatic CL
    renal_cl_fraction: float  # Relative renal CL
    anemia_severity: str  # "none", "mild", "moderate", "severe"
    hypoxia_severity: str  # "none", "mild", "moderate", "severe"
    recommended_dose_fraction: float  # Dose adjustment
    notes: str


@dataclass(frozen=True)
class HypoxiaPKResult:
    """Result from hypoxia-adjusted PK simulation."""

    drug_name: str
    dose_mg: float
    hemoglobin_g_dL: float
    spo2_pct: float
    cl_adjusted_L_per_h: float
    vd_L: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    cmax: float
    auc: float
    t_half_h: float
    notes: str


def assess_oxygen_delivery(
    hemoglobin_g_dL: float,
    spo2_pct: float = 98.0,
    cardiac_output_L_per_min: float = 5.0,
    bsa_m2: float = 1.73,
) -> OxygenDeliveryResult:
    """Assess tissue oxygen delivery and pharmacokinetic consequences.

    DO2 = CO × Hb × SaO2 × 1.34  (mL O2/min)

    Parameters
    ----------
    hemoglobin_g_dL : Hemoglobin concentration (g/dL). Must be > 0.
    spo2_pct : Arterial oxygen saturation (%). Must be in (0, 100].
    cardiac_output_L_per_min : Cardiac output (L/min). Must be > 0.
    bsa_m2 : Body surface area (m²). Must be > 0.

    Returns
    -------
    OxygenDeliveryResult
    """
    if hemoglobin_g_dL <= 0:
        raise ValueError("hemoglobin_g_dL must be > 0")
    if not (0 < spo2_pct <= 100):
        raise ValueError("spo2_pct must be in (0, 100]")
    if cardiac_output_L_per_min <= 0:
        raise ValueError("cardiac_output_L_per_min must be > 0")
    if bsa_m2 <= 0:
        raise ValueError("bsa_m2 must be > 0")

    # DO2 calculation (mL O2/min): CO [L/min] * 10 (dL/L) * Hb * 1.34 * SaO2/100
    do2 = cardiac_output_L_per_min * 10.0 * hemoglobin_g_dL * 1.34 * spo2_pct / 100.0
    do2_index = do2 / bsa_m2

    # Anemia severity
    if hemoglobin_g_dL >= 12.0:
        anemia_sev = "none"
    elif hemoglobin_g_dL >= 10.0:
        anemia_sev = "mild"
    elif hemoglobin_g_dL >= 8.0:
        anemia_sev = "moderate"
    else:
        anemia_sev = "severe"

    # Hypoxia severity
    if spo2_pct >= 95.0:
        hypoxia_sev = "none"
    elif spo2_pct >= 90.0:
        hypoxia_sev = "mild"
    elif spo2_pct >= 85.0:
        hypoxia_sev = "moderate"
    else:
        hypoxia_sev = "severe"

    # DO2 normal ~1000 mL/min; critical <550 mL/min
    do2_fraction = min(1.0, do2 / 1000.0)

    # CYP activity reduced in hypoxia/anemia (O2-dependent oxidative metabolism)
    # Severe hypoxia (<85%) reduces CYP3A4 by up to 40%
    cyp_frac = max(0.3, do2_fraction * 0.7 + 0.3)

    # Hepatic CL: reduced by both hypoxia (lower CYP) and reduced hepatic blood flow in anemia
    hepatic_frac = max(0.2, cyp_frac * (1.0 - max(0.0, (12.0 - hemoglobin_g_dL) / 20.0)))

    # Renal CL: reduced in severe anemia (compensatory vasoconstriction)
    renal_frac = max(0.3, 1.0 - max(0.0, (10.0 - hemoglobin_g_dL) / 15.0))

    # Combined dose recommendation
    dose_frac = min(1.0, (hepatic_frac + renal_frac) / 2.0)

    notes = (
        f"Hb={hemoglobin_g_dL:.1f} g/dL, SpO2={spo2_pct:.0f}%, "
        f"DO2={do2:.0f} mL/min (index={do2_index:.0f}). "
        f"Anemia: {anemia_sev}, hypoxia: {hypoxia_sev}. "
        f"CYP={cyp_frac:.2f}x, hepatic_CL={hepatic_frac:.2f}x, renal_CL={renal_frac:.2f}x."
    )

    return OxygenDeliveryResult(
        hemoglobin_g_dL=hemoglobin_g_dL,
        spo2_pct=spo2_pct,
        cardiac_output_L_per_min=cardiac_output_L_per_min,
        do2_mL_per_min=do2,
        do2_index=do2_index,
        cyp_activity_fraction=cyp_frac,
        hepatic_cl_fraction=hepatic_frac,
        renal_cl_fraction=renal_frac,
        anemia_severity=anemia_sev,
        hypoxia_severity=hypoxia_sev,
        recommended_dose_fraction=dose_frac,
        notes=notes,
    )


def simulate_hypoxia_pk(
    drug_name: str,
    dose_mg: float,
    cl_normal_L_per_h: float,
    vd_L: float,
    hemoglobin_g_dL: float = 15.0,
    spo2_pct: float = 98.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> HypoxiaPKResult:
    """Simulate PK with hypoxia/anemia-adjusted clearance.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose (mg). Must be > 0.
    cl_normal_L_per_h : Normal clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    hemoglobin_g_dL : Hemoglobin (g/dL). Must be > 0.
    spo2_pct : SpO2 (%). Must be in (0, 100].
    route : 'iv' or 'oral'.
    ka_per_h : Oral absorption rate (h^-1).
    t_end_h : Simulation end time (h).
    dt_h : Time step (h).

    Returns
    -------
    HypoxiaPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_normal_L_per_h <= 0:
        raise ValueError("cl_normal_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")

    o2 = assess_oxygen_delivery(hemoglobin_g_dL, spo2_pct)
    cl_adj = cl_normal_L_per_h * o2.hepatic_cl_fraction
    ke = cl_adj / vd_L

    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]
    c_plasma = [0.0] * n_steps
    a_gut = 0.0

    if route == "iv":
        c_plasma[0] = dose_mg / vd_L
    else:
        a_gut = dose_mg

    for i in range(1, n_steps):
        cp = c_plasma[i - 1]
        if route == "oral":
            abs_r = ka_per_h * a_gut
            a_gut = max(0.0, a_gut - abs_r * dt_h)
        else:
            abs_r = 0.0
        dc = (abs_r - ke * cp * vd_L) / vd_L * dt_h
        c_plasma[i] = max(0.0, cp + dc)

    cmax = max(c_plasma)
    auc = sum(0.5 * (c_plasma[j - 1] + c_plasma[j]) * dt_h for j in range(1, n_steps))
    t_half = math.log(2) / ke

    notes = (
        f"{drug_name}: CL adjusted from {cl_normal_L_per_h:.1f} to {cl_adj:.2f} L/h "
        f"(hepatic_frac={o2.hepatic_cl_fraction:.2f}). "
        f"t\u00bd={t_half:.1f}h. Hb={hemoglobin_g_dL:.1f} g/dL, SpO2={spo2_pct:.0f}%."
    )

    return HypoxiaPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        hemoglobin_g_dL=hemoglobin_g_dL,
        spo2_pct=spo2_pct,
        cl_adjusted_L_per_h=cl_adj,
        vd_L=vd_L,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        cmax=cmax,
        auc=auc,
        t_half_h=t_half,
        notes=notes,
    )
